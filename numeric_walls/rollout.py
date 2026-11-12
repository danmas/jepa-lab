"""Зонд и открытая раскатка двадцати четырёх сетей в out/numeric_walls_rollout.json.

Файл пишется одним разом в конце. Неполный прогон его не оставляет.
"""
import json
import os

import numpy as np
import torch

from models import EMATarget, fit_ridge
from numeric_walls.data import (
    DT, PROBE_EPISODES, PROTOCOLS, ROLLOUT_N, ROLLOUT_SEED0, SPAN_CF, SPAN_TESTS,
    TRAIN_SEEDS, TRANSFER, TRANSFER_N, VARIANTS, WORLDS, probe_states,
    rollout_episode, scale_state, scale_walls, step_physics, transfer_states,
    walls_of,
)
from numeric_walls.models import LATENT, N_WALL, BallEncoder, WallPredictor, action_pairs
from numeric_walls.train import STEPS, ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_walls_rollout.json")
RIDGE_LAM = 1e-2
MARKS = (1, 10, 20, 30)
R2_NAMES = ("x", "y", "vx", "vy")
R2_FLOOR = 0.9
PROBE_TRAIN = 3072


def _r2(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot.clamp_min(1e-12)).cpu().numpy()


def _mean_std(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    if values.shape != (ROLLOUT_N,):
        raise RuntimeError(f"нужен вектор из {ROLLOUT_N}, пришло {values.shape}")
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(ddof=1)), 6),
    }


def _fit(ema, states: np.ndarray):
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states[:, 0:4], np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    if z.shape[1] != LATENT:
        raise RuntimeError(f"латент {z.shape[1]}")
    probe = fit_ridge(z[:PROBE_TRAIN], target[:PROBE_TRAIN], lam=RIDGE_LAM)
    if probe.fc.in_features != LATENT:
        raise RuntimeError("зонд собрал не тот вход")
    with torch.no_grad():
        pred = probe.fc(z[PROBE_TRAIN:])
    return probe, _r2(pred, target[PROBE_TRAIN:])


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("mode") != "jepa" or int(ck.get("steps", 0)) != STEPS:
        raise RuntimeError(path)
    variant = ck["variant"]
    encoder = BallEncoder()
    predictor = WallPredictor(N_WALL[variant])
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()
    return ck, encoder, predictor, ema


def _roll_one(encoder, predictor, ema, probe, states, held, walls, variant: str):
    """Ошибки на отметках и истинный сдвиг первого шага."""
    scaled = torch.from_numpy(scale_state(states))
    wall_t = torch.from_numpy(scale_walls(walls))[None, :] if variant == "vis" else None
    pos_err = {}
    vel_err = {}
    lat_err = {}
    read_pos = {}
    read_vel = {}
    with torch.no_grad():
        z = encoder(scaled[0:1])
        true_z = ema(scaled)
        for k in range(held.shape[0]):
            pair = action_pairs(torch.tensor([int(held[k])]), z)
            z = predictor(z, pair, wall_t)
            step = k + 1
            if step not in MARKS:
                continue
            pred = probe.fc(z)[0].cpu().numpy()
            truth = states[step, 0:4]
            pos_err[step] = float(np.linalg.norm(pred[0:2] - truth[0:2]))
            vel_err[step] = float(np.linalg.norm(pred[2:4] - truth[2:4]))
            lat_err[step] = float((z - true_z[step:step + 1]).abs().mean())
            read = probe.fc(true_z[step:step + 1])[0].cpu().numpy()
            read_pos[step] = float(np.linalg.norm(read[0:2] - truth[0:2]))
            read_vel[step] = float(np.linalg.norm(read[2:4] - truth[2:4]))
    shift = float(np.linalg.norm(states[1, 0:2] - states[0, 0:2]))
    return pos_err, vel_err, lat_err, read_pos, read_vel, shift


def _counterfactual(encoder, predictor, probe, variant: str):
    state = np.array([[39.0, 32.0, 60.0, 0.0, DT]], np.float64)
    out = []
    with torch.no_grad():
        z0 = encoder(torch.from_numpy(scale_state(state)))
        for span in SPAN_CF:
            walls = walls_of("W", span)
            wall_t = torch.from_numpy(scale_walls(np.asarray(walls)))[None, :]
            pair = action_pairs(torch.tensor([5]), z0)
            z = predictor(z0, pair, wall_t if variant == "vis" else None)
            pos = probe.fc(z)[0, 0:2].cpu().numpy()
            true, _vel, _n = step_physics([39.0, 32.0], [60.0, 0.0], 5, walls)
            out.append({
                "span": span,
                "pred_x": round(float(pos[0]), 6),
                "pred_y": round(float(pos[1]), 6),
                "true_x": round(float(true[0]), 6),
                "true_y": round(float(true[1]), 6),
            })
    if variant == "hid":
        if abs(out[0]["pred_x"] - out[1]["pred_x"]) > 1e-6:
            raise RuntimeError("скрытый контрфакт зависит от ширины")
        if abs(out[0]["pred_y"] - out[1]["pred_y"]) > 1e-6:
            raise RuntimeError("скрытый контрфакт зависит от ширины")
    return out


def _transfer_r2(ema, probe, world: str, protocol: str, span: float) -> dict:
    states = transfer_states(world, protocol, span)
    if len(states) != TRANSFER_N * 4:
        raise RuntimeError(len(states))
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states[:, 0:4], np.float32, copy=True))
    with torch.no_grad():
        pred = probe.fc(ema(scaled))
    r2 = _r2(pred, target)
    return {name: round(float(val), 6) for name, val in zip(R2_NAMES, r2)}


def main():
    cells = []
    for world in WORLDS:
        for protocol in PROTOCOLS:
            for variant in VARIANTS:
                for seed in TRAIN_SEEDS:
                    path = ckpt_path(world, protocol, variant, seed)
                    if not os.path.isfile(path):
                        raise SystemExit(
                            f"нет {os.path.basename(path)}. Сначала: "
                            "python -m numeric_walls.train --tag numeric-walls")
                    ck, encoder, predictor, ema = _load(path)
                    states = probe_states(world, protocol)
                    if len(states) != PROBE_EPISODES * 4:
                        raise RuntimeError(len(states))
                    probe, r2 = _fit(ema, states)
                    collapsed = bool(np.any(r2 < R2_FLOOR))
                    transfer = {
                        str(int(span)): _transfer_r2(ema, probe, world, protocol, span)
                        for span in SPAN_TESTS
                    }
                    spans = {}
                    for span in SPAN_TESTS:
                        bag = {m: {"pos": [], "vel": [], "lat": [], "read_pos": [], "read_vel": []}
                               for m in MARKS}
                        shifts = []
                        for i in range(ROLLOUT_N):
                            ep, held, walls = rollout_episode(ROLLOUT_SEED0 + i, world, span)
                            pos_e, vel_e, lat_e, rp, rv, shift = _roll_one(
                                encoder, predictor, ema, probe, ep, held, walls, variant)
                            shifts.append(shift)
                            for m in MARKS:
                                bag[m]["pos"].append(pos_e[m])
                                bag[m]["vel"].append(vel_e[m])
                                bag[m]["lat"].append(lat_e[m])
                                bag[m]["read_pos"].append(rp[m])
                                bag[m]["read_vel"].append(rv[m])
                        shift_mean = float(np.mean(shifts))
                        step1 = float(np.mean(bag[1]["pos"]))
                        marks = {}
                        for m in MARKS:
                            marks[str(m)] = {
                                "pos": _mean_std(bag[m]["pos"]),
                                "vel": _mean_std(bag[m]["vel"]),
                                "latent_l1": _mean_std(bag[m]["lat"]),
                                "read_pos": _mean_std(bag[m]["read_pos"]),
                                "read_vel": _mean_std(bag[m]["read_vel"]),
                            }
                        spans[str(int(span))] = {
                            "marks": marks,
                            "shift_mean": round(shift_mean, 6),
                            "step_ok": bool(step1 < 0.25 * shift_mean),
                        }
                    cells.append({
                        "world": world,
                        "protocol": protocol,
                        "variant": variant,
                        "seed": int(seed),
                        "file": os.path.basename(path),
                        "collapsed": collapsed,
                        "r2": {name: round(float(val), 6) for name, val in zip(R2_NAMES, r2)},
                        "transfer_r2": transfer,
                        "spans": spans,
                        "counterfactual": _counterfactual(encoder, predictor, probe, variant),
                        "channel": "unidentified" if protocol == "fixed" else "range",
                    })
                    print(f"готова {world} {protocol} {variant} s{seed}", flush=True)
    if len(cells) != 24:
        raise RuntimeError(len(cells))
    payload = {
        "dt": DT,
        "marks_steps": list(MARKS),
        "gap_x_px": "143/45",
        "gap_vx": "376/3",
        "cells": cells,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()

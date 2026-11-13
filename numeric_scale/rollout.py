"""Зонд и открытая раскатка двенадцати сетей в out/numeric_scale_rollout.json.

Файл пишется одним разом в конце. Неполный прогон его не оставляет.
"""
import json
import os

import numpy as np
import torch

from models import EMATarget, fit_ridge
from numeric_scale.data import (
    CF_H, CF_WIDTHS, DT, PROBE_EPISODES, RECT_SEED0, REPRS, ROLLOUT_N, ROLLOUT_SEED0,
    SQUARES, TRAIN_SEEDS, TRANSFER, TRANSFER_N, VARIANTS, probe_rows, probe_target,
    rollout_episode, scale_state, step_physics, transfer_rows, world_vector,
)
from numeric_scale.models import LATENT, N_WORLD, BallEncoder, ScalePredictor, action_pairs
from numeric_scale.train import STEPS, ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_scale_rollout.json")
RIDGE_LAM = 1e-2
MARKS = (1, 10, 20, 30)
R2_NAMES = ("x", "y", "vx", "vy")
R2_FLOOR = 0.9
PROBE_TRAIN = 3072
SIZE_RUNS = tuple(
    [(str(int(side)), side, side, ROLLOUT_SEED0) for side in SQUARES]
) + (("20x28", 20.0, 28.0, RECT_SEED0),)


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


def _pair_errors(pred, truth, width: float, height: float, repr_name: str):
    """Ошибки положения и скорости в пикселях и в долях. pred — выход зонда."""
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    width = float(width)
    height = float(height)
    if repr_name == "rel":
        true_f = np.array([
            truth[0] / width,
            truth[1] / height,
            truth[2] * DT / width,
            truth[3] * DT / height,
        ], np.float64)
        d = pred - true_f
        pos_frac = float(np.hypot(d[0], d[1]))
        pos_px = float(np.hypot(d[0] * width, d[1] * height))
        vel_frac = float(np.hypot(d[2], d[3]))
        vel_px = float(np.hypot(d[2] * width / DT, d[3] * height / DT))
    elif repr_name == "abs":
        d = pred - truth
        pos_px = float(np.hypot(d[0], d[1]))
        pos_frac = float(np.hypot(d[0] / width, d[1] / height))
        vel_px = float(np.hypot(d[2], d[3]))
        vel_frac = float(np.hypot(d[2] * DT / width, d[3] * DT / height))
    else:
        raise ValueError(repr_name)
    return pos_px, pos_frac, vel_px, vel_frac


def _fit(ema, states, widths, heights, repr_name: str):
    scaled = torch.from_numpy(scale_state(states, widths, heights, repr_name))
    target = torch.from_numpy(probe_target(states, widths, heights, repr_name))
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
    repr_name = ck["repr"]
    encoder = BallEncoder()
    predictor = ScalePredictor(N_WORLD[variant])
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()
    return ck, encoder, predictor, ema, repr_name


def _roll_one(encoder, predictor, ema, probe, states, held, width, height, variant, repr_name):
    scaled = torch.from_numpy(scale_state(states, width, height, repr_name))
    world_t = None
    if variant == "vis":
        world_t = torch.from_numpy(world_vector(np.array([width]), np.array([height])))
    found = {}
    with torch.no_grad():
        z = encoder(scaled[0:1])
        true_z = ema(scaled)
        for k in range(held.shape[0]):
            pair = action_pairs(torch.tensor([int(held[k])]), z)
            z = predictor(z, pair, world_t)
            step = k + 1
            if step not in MARKS:
                continue
            pred = probe.fc(z)[0].cpu().numpy()
            read = probe.fc(true_z[step:step + 1])[0].cpu().numpy()
            truth = states[step, 0:4]
            pos_px, pos_frac, vel_px, vel_frac = _pair_errors(
                pred, truth, width, height, repr_name)
            rpx, rfrac, rvx, rvfrac = _pair_errors(read, truth, width, height, repr_name)
            found[step] = {
                "pos_px": pos_px, "pos_frac": pos_frac,
                "vel_px": vel_px, "vel_frac": vel_frac,
                "lat": float((z - true_z[step:step + 1]).abs().mean()),
                "read_pos_px": rpx, "read_pos_frac": rfrac,
                "read_vel_px": rvx, "read_vel_frac": rvfrac,
            }
    if set(found) != set(MARKS):
        raise RuntimeError(sorted(found))
    dx = (states[1, 0] - states[0, 0]) / float(width)
    dy = (states[1, 1] - states[0, 1]) / float(height)
    shift = float(np.hypot(dx, dy))
    return found, shift


def _collect_roll(encoder, predictor, ema, probe, variant, repr_name, width, height, seed0):
    keys = ("pos_px", "pos_frac", "vel_px", "vel_frac", "lat",
            "read_pos_px", "read_pos_frac", "read_vel_px", "read_vel_frac")
    bag = {m: {k: [] for k in keys} for m in MARKS}
    shifts = []
    approaches = []
    reflected = 0
    for i in range(ROLLOUT_N):
        ep, held, w, h, approach, n_hit = rollout_episode(seed0 + i, width, height)
        found, shift = _roll_one(
            encoder, predictor, ema, probe, ep, held, w, h, variant, repr_name)
        shifts.append(shift)
        approaches.append(float(approach))
        reflected += int(n_hit)
        for m in MARKS:
            for k in keys:
                bag[m][k].append(found[m][k])
    shift_mean = float(np.mean(shifts))
    step1 = float(np.mean(bag[1]["pos_frac"]))
    marks = {}
    for m in MARKS:
        marks[str(m)] = {k: _mean_std(bag[m][k]) for k in keys}
    return {
        "width": float(width),
        "height": float(height),
        "seed0": int(seed0),
        "approach_share": round(float(np.mean(approaches)), 6),
        "reflect_share": round(float(reflected) / float(ROLLOUT_N * 30), 6),
        "marks": marks,
        "shift_frac_mean": round(shift_mean, 6),
        "step_ok": bool(step1 < 0.25 * shift_mean),
    }


def _frac_from_probe(pred, repr_name: str, width: float, height: float):
    pred = np.asarray(pred, np.float64)
    if repr_name == "rel":
        return float(pred[0]), float(pred[1]), float(pred[0] * width), float(pred[1] * height)
    return float(pred[0] / width), float(pred[1] / height), float(pred[0]), float(pred[1])


def _counterfactual(encoder, predictor, probe, variant: str, repr_name: str):
    out = []
    if repr_name == "rel":
        vec = np.array([[7.0 / 8.0, 0.25, 0.125, 0.0, 1.0]], np.float32)
        z0 = encoder(torch.from_numpy(vec))
        cases = []
        for width in CF_WIDTHS:
            vx = 0.125 * float(width) / DT
            cases.append((float(width), 14.0 if width == 16.0 else 28.0, vx))
    else:
        raw = np.array([[14.0, 8.0, 60.0, 0.0, DT]], np.float64)
        vec = torch.from_numpy(scale_state(raw, 16.0, CF_H, "abs"))
        z0 = encoder(vec)
        cases = [(float(width), 14.0, 60.0) for width in CF_WIDTHS]
    with torch.no_grad():
        for width, x0, vx0 in cases:
            world_t = None
            if variant == "vis":
                world_t = torch.from_numpy(
                    world_vector(np.array([width]), np.array([CF_H])))
            pair = action_pairs(torch.tensor([5]), z0)
            z = predictor(z0, pair, world_t)
            pred = probe.fc(z)[0].cpu().numpy()
            fx, fy, px, py = _frac_from_probe(pred, repr_name, width, CF_H)
            true, tvel, _n, _hit = step_physics(
                [x0, 8.0], [vx0, 0.0], 5, (0.0, width, 0.0, CF_H))
            out.append({
                "W": width,
                "H": CF_H,
                "pred_x_frac": round(fx, 6),
                "pred_y_frac": round(fy, 6),
                "pred_x_px": round(px, 6),
                "pred_y_px": round(py, 6),
                "true_x": round(float(true[0]), 6),
                "true_y": round(float(true[1]), 6),
                "true_vx": round(float(tvel[0]), 6),
                "true_x_frac": round(float(true[0]) / width, 6),
                "true_y_frac": round(float(true[1]) / CF_H, 6),
            })
    if variant == "hid":
        if repr_name == "rel":
            same = (abs(out[0]["pred_x_frac"] - out[1]["pred_x_frac"]) < 1e-5
                    and abs(out[0]["pred_y_frac"] - out[1]["pred_y_frac"]) < 1e-5)
        else:
            same = (abs(out[0]["pred_x_px"] - out[1]["pred_x_px"]) < 1e-5
                    and abs(out[0]["pred_y_px"] - out[1]["pred_y_px"]) < 1e-5)
        if not same:
            raise RuntimeError(f"скрытый контрфакт зависит от ширины: {repr_name}")
    return out


def _transfer_r2(ema, probe, repr_name: str, key: str) -> dict:
    states, widths, heights = transfer_rows(key)
    if len(states) != TRANSFER_N * 4:
        raise RuntimeError(len(states))
    scaled = torch.from_numpy(scale_state(states, widths, heights, repr_name))
    target = torch.from_numpy(probe_target(states, widths, heights, repr_name))
    with torch.no_grad():
        pred = probe.fc(ema(scaled))
    r2 = _r2(pred, target)
    return {name: round(float(val), 6) for name, val in zip(R2_NAMES, r2)}


def main():
    torch.set_num_threads(8)
    cells = []
    for repr_name in REPRS:
        for variant in VARIANTS:
            for seed in TRAIN_SEEDS:
                path = ckpt_path(repr_name, variant, seed)
                if not os.path.isfile(path):
                    raise SystemExit(
                        f"нет {os.path.basename(path)}. Сначала: "
                        "python -m numeric_scale.train --tag numeric-scale")
                _ck, encoder, predictor, ema, got = _load(path)
                if got != repr_name:
                    raise RuntimeError(path)
                states, widths, heights = probe_rows()
                if len(states) != PROBE_EPISODES * 4:
                    raise RuntimeError(len(states))
                probe, r2 = _fit(ema, states, widths, heights, repr_name)
                collapsed = bool(np.any(r2 < R2_FLOOR))
                transfer = {key: _transfer_r2(ema, probe, repr_name, key) for key in TRANSFER}
                sizes = {}
                for key, width, height, seed0 in SIZE_RUNS:
                    sizes[key] = _collect_roll(
                        encoder, predictor, ema, probe, variant, repr_name,
                        width, height, seed0)
                cells.append({
                    "repr": repr_name,
                    "variant": variant,
                    "seed": int(seed),
                    "file": os.path.basename(path),
                    "collapsed": collapsed,
                    "r2": {name: round(float(val), 6) for name, val in zip(R2_NAMES, r2)},
                    "transfer_r2": transfer,
                    "sizes": sizes,
                    "counterfactual": _counterfactual(
                        encoder, predictor, probe, variant, repr_name),
                })
                print(f"готова {repr_name} {variant} s{seed}", flush=True)
    if len(cells) != 12:
        raise RuntimeError(len(cells))
    payload = {
        "dt": DT,
        "dt_channel": 1,
        "abs_scale_xy": 64,
        "abs_scale_vel": 120,
        "marks_steps": list(MARKS),
        "gap_x_px": "16/45",
        "gap_vx": "136",
        "squares": [12, 24, 48],
        "rectangle": [20, 28],
        "cells": cells,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()

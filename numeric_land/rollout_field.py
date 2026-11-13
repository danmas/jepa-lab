"""Зонд, открытая раскатка и посадка большого поля.

Пишет out/numeric_land_field_rollout.json целиком в конце.

    python -m numeric_land.rollout_field
"""
import json
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.data import TRAIN_SEEDS, VARIANTS
from numeric_land.field import (
    GROUND, MARKS, PLAN_N, PLAN_SEED0, PROBE_EPISODES, PROBE_SEED0, PROBE_STEPS,
    RANGE_G, ROLLOUT_N, ROLLOUT_SEED0, SINGLE_TRANSFER_G, TRANSFER_N,
    held_episode, probe_rows, rollout_states, scale_state, start_and_actions,
    step_field, transfer_rows, world_vector,
)
from numeric_land.models import LATENT, N_WORLD, BallEncoder, LandPredictor, action_pairs, fit_readout
from numeric_land.plan_field import HORIZON, ITERS, SAMPLES, run_episode
from numeric_land.train_field import STEPS, TAG, ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_land_field_rollout.json")
RIDGE_LAM = 1e-2
PROBE_TRAIN = 3072
R2_NAMES = ("x", "y", "vx", "vy", "fuel")
R2_FLOOR = 0.9
RANGE_TRANSFER = ((2.0, 83000), (8.0, 84000), (20.0, 85000))
SINGLE_TRANSFER = ((4.0, 87000), (14.0, 88000))


def _r2(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot.clamp_min(1e-12)).cpu().numpy()


def _mean_std(values) -> dict:
    values = np.asarray(values, np.float64)
    if values.shape != (ROLLOUT_N,):
        raise RuntimeError(f"нужен вектор из {ROLLOUT_N}, пришло {values.shape}")
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(ddof=1)), 6),
    }


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("mode") != "jepa" or ck.get("tag") != TAG or int(ck.get("steps", 0)) != STEPS:
        raise RuntimeError(path)
    variant = ck["variant"]
    encoder = BallEncoder()
    predictor = LandPredictor(N_WORLD[variant])
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()
    return ck, encoder, predictor, ema


def _fit(ema, states: np.ndarray):
    if len(states) != PROBE_EPISODES * (PROBE_STEPS + 1):
        raise RuntimeError(len(states))
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states, np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    if z.shape[1] != LATENT or target.shape[1] != 5:
        raise RuntimeError((z.shape, target.shape))
    probe = fit_readout(z[:PROBE_TRAIN], target[:PROBE_TRAIN], lam=RIDGE_LAM)
    with torch.no_grad():
        pred = probe.fc(z[PROBE_TRAIN:])
    r2 = _r2(pred, target[PROBE_TRAIN:])
    return probe, {name: round(float(val), 6) for name, val in zip(R2_NAMES, r2)}


def _transfer(ema, probe, seed0: int, g: float) -> dict:
    states = transfer_rows(seed0, g)
    if len(states) != TRANSFER_N * (PROBE_STEPS + 1):
        raise RuntimeError(len(states))
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states, np.float32, copy=True))
    with torch.no_grad():
        pred = probe.fc(ema(scaled))
    r2 = _r2(pred, target)
    return {
        "g": float(g),
        "seed0": int(seed0),
        "n": TRANSFER_N,
        "r2": {name: round(float(val), 6) for name, val in zip(R2_NAMES, r2)},
    }


def _roll_one(encoder, predictor, probe, states, held, g, variant):
    scaled = torch.from_numpy(scale_state(states))
    world = None
    if variant == "vis":
        world = torch.from_numpy(world_vector(np.array([float(g)])))
    found = {}
    with torch.no_grad():
        z = encoder(scaled[0:1])
        for k in range(held.shape[0]):
            z = predictor(z, action_pairs(torch.tensor([int(held[k])]), z), world)
            step = k + 1
            if step not in MARKS:
                continue
            pred = probe.fc(z)[0].cpu().numpy()
            truth = states[step]
            found[step] = {
                "pos_px": float(np.hypot(pred[0] - truth[0], pred[1] - truth[1])),
                "vel_px": float(np.hypot(pred[2] - truth[2], pred[3] - truth[3])),
            }
    if set(found) != set(MARKS):
        raise RuntimeError(sorted(found))
    return found


def _collect_roll(encoder, predictor, probe, variant, g):
    bag = {m: {"pos_px": [], "vel_px": []} for m in MARKS}
    tape0 = None
    for i in range(ROLLOUT_N):
        states, held, actions = held_episode(ROLLOUT_SEED0 + i, g)
        if i == 0:
            tape0 = [int(a) for a in actions]
        found = _roll_one(encoder, predictor, probe, states, held, g, variant)
        for m in MARKS:
            bag[m]["pos_px"].append(found[m]["pos_px"])
            bag[m]["vel_px"].append(found[m]["vel_px"])
    return {
        "g": float(g),
        "seed0": ROLLOUT_SEED0,
        "n": ROLLOUT_N,
        "hold": 2,
        "horizon": 30,
        "tape0": tape0,
        "marks": {str(m): {k: _mean_std(bag[m][k]) for k in ("pos_px", "vel_px")} for m in MARKS},
    }


def _counterfactual(encoder, predictor, probe, variant):
    raw = np.array([[256.0, 464.0, 0.0, 0.0, 10.0]], np.float64)
    with torch.no_grad():
        z0 = encoder(torch.from_numpy(scale_state(raw)))
        rows = []
        pred_vys = []
        for g in (2.0, 15.0):
            world = None
            if variant == "vis":
                world = torch.from_numpy(world_vector(np.array([g])))
            z = predictor(z0, action_pairs(torch.tensor([7]), z0), world)
            pred = probe.fc(z)[0].cpu().numpy()
            true = step_field(256.0, 464.0, 0.0, 0.0, 10.0, False, 7, g)
            pred_vys.append(float(pred[3]))
            rows.append({
                "g": g,
                "pred_y": round(float(pred[1]), 6),
                "pred_vy": round(float(pred[3]), 6),
                "true_y": round(float(true[1]), 6),
                "true_vy": round(float(true[3]), 6),
            })
    gap_vy = rows[1]["true_vy"] - rows[0]["true_vy"]
    gap_y = rows[1]["true_y"] - rows[0]["true_y"]
    if abs(gap_vy - 13.0 / 30.0) > 1e-6 or abs(gap_y - 13.0 / 900.0) > 1e-6:
        raise RuntimeError(f"зазор контрфакта {gap_vy}, {gap_y}")
    if variant == "hid" and pred_vys[0] != pred_vys[1]:
        raise RuntimeError(f"скрытый контрфакт различил тяжести: {pred_vys}")
    return {
        "action": 7,
        "rows": rows,
        "true_vy_gap": round(gap_vy, 6),
        "true_y_gap": round(gap_y, 6),
        "hid_vy_match": variant != "vis",
    }


def _same_dict(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def _jobs():
    rows = []
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            rows.append({
                "protocol": "range", "variant": variant, "seed": seed, "g": None,
                "roll_g": RANGE_G, "own": ("range", None), "transfer": RANGE_TRANSFER,
                "counterfactual": True,
            })
    rows.append({
        "protocol": "fixed", "variant": "vis", "seed": 0, "g": 9.0,
        "roll_g": (4.0, 14.0, 9.0), "own": ("fixed", 9.0), "transfer": SINGLE_TRANSFER,
        "counterfactual": False,
    })
    for g in (4.0, 14.0):
        rows.append({
            "protocol": "tune", "variant": "vis", "seed": 0, "g": g,
            "roll_g": SINGLE_TRANSFER_G, "own": ("fixed", g), "transfer": SINGLE_TRANSFER,
            "counterfactual": False,
        })
    rows.append({
        "protocol": "scratch", "variant": "vis", "seed": 0, "g": 4.0,
        "roll_g": SINGLE_TRANSFER_G, "own": ("fixed", 4.0), "transfer": SINGLE_TRANSFER,
        "counterfactual": False,
    })
    return rows


def _train_row(ck, path):
    last = ck["log"][-1]
    return {
        "file": os.path.basename(path),
        "protocol": ck["protocol"],
        "variant": ck["variant"],
        "seed": int(ck["seed"]),
        "g": ck.get("g"),
        "steps": int(ck["steps"]),
        "seconds": ck.get("seconds"),
        "loss": last["loss"],
        "l1": last["l1"],
        "penalty": last["penalty"],
        "frozen_encoder": bool(ck.get("frozen_encoder", False)),
    }


def _plan_row(episodes):
    counts = {name: sum(1 for ep in episodes if ep["result"] == name)
              for name in ("мягкая", "удар", "не сел")}
    touched = [ep for ep in episodes if ep["result"] != "не сел"]
    missed = [ep for ep in episodes if ep["result"] == "не сел"]
    touch_mean = None
    if touched:
        touch_mean = round(sum(ep["steps"] for ep in touched) / len(touched), 3)
    miss_height = None
    if missed:
        miss_height = round(sum(GROUND - ep["y"] for ep in missed) / len(missed), 3)
    start_height = round(sum(GROUND - ep["start_y"] for ep in episodes) / len(episodes), 3)
    return {
        "file": "landf_range_vis_s0.pt",
        "g": float(episodes[0]["g"]),
        "horizon": HORIZON,
        "samples": SAMPLES,
        "iters": ITERS,
        "cap": 450,
        "of": PLAN_N,
        "soft": counts["мягкая"],
        "crash": counts["удар"],
        "miss": counts["не сел"],
        "touch_n": len(touched),
        "touch_steps_mean": touch_mean,
        "miss_height_mean": miss_height,
        "start_height_mean": start_height,
        "episodes": episodes,
    }


def main():
    torch.set_num_threads(8)
    fixed_path = ckpt_path("fixed", "vis", 0, 9)
    if not os.path.isfile(fixed_path):
        raise SystemExit("нет landf_fixed_g9_s0.pt")
    fixed_ck = torch.load(fixed_path, map_location="cpu", weights_only=False)
    nets = []
    tapes = {}
    for job in _jobs():
        path = ckpt_path(job["protocol"], job["variant"], job["seed"], job["g"])
        if not os.path.isfile(path):
            raise SystemExit(f"нет {os.path.basename(path)}")
        ck, encoder, predictor, ema = _load(path)
        print(f"зонд {os.path.basename(path)}", flush=True)
        mode, own_g = job["own"]
        states = probe_rows(mode, own_g)
        probe, r2 = _fit(ema, states)
        collapsed = any(r2[name] < R2_FLOOR for name in R2_NAMES)
        transfer = [_transfer(ema, probe, seed0, g) for g, seed0 in job["transfer"]]
        rolls = []
        for g in job["roll_g"]:
            print(f"  раскатка g={g:g}", flush=True)
            row = _collect_roll(encoder, predictor, probe, job["variant"], g)
            key = (job["protocol"], job["variant"], job["seed"])
            tapes.setdefault(key, row["tape0"])
            if row["tape0"] != tapes[key]:
                raise RuntimeError("лента сида 80000 разошлась")
            rolls.append(row)
        cell = {
            "train": _train_row(ck, path),
            "probe": {
                "own": "range" if mode == "range" else float(own_g),
                "seed0": PROBE_SEED0,
                "n": PROBE_EPISODES,
                "r2": r2,
                "collapsed": bool(collapsed),
                "transfer": transfer,
            },
            "rollout": rolls,
        }
        if job["counterfactual"]:
            cell["counterfactual"] = _counterfactual(encoder, predictor, probe, job["variant"])
        if job["protocol"] in ("tune", "scratch"):
            cell["encoder_matches_fixed"] = _same_dict(ck["encoder"], fixed_ck["encoder"])
            cell["predictor_matches_fixed"] = _same_dict(ck["predictor"], fixed_ck["predictor"])
            if not cell["encoder_matches_fixed"]:
                raise RuntimeError(f"кодировщик {os.path.basename(path)} сдвинулся")
            if job["protocol"] == "scratch" and cell["predictor_matches_fixed"]:
                raise RuntimeError("scratch скопировал предиктор g=9")
        nets.append(cell)

    print("план landf_range_vis_s0", flush=True)
    plan = []
    kept_keys = ("seed", "g", "result", "steps", "x", "y", "vy", "fuel", "start_y", "plan_s")
    for g in RANGE_G:
        episodes = []
        for seed in range(PLAN_SEED0, PLAN_SEED0 + PLAN_N):
            ep = run_episode(seed, g)
            episodes.append({k: ep[k] for k in kept_keys})
            print(f"  g={g:g} seed={seed} {ep['result']} шагов {ep['steps']}", flush=True)
        plan.append(_plan_row(episodes))
    heights = [row["start_height_mean"] for row in plan]
    if any(abs(h - heights[0]) > 1e-6 for h in heights):
        raise RuntimeError(f"высота старта зависит от g: {heights}")

    vis_tape = tapes[("range", "vis", 0)]
    hid_tape = tapes[("range", "hid", 0)]
    if vis_tape != hid_tape:
        raise RuntimeError("vis и hid разошлись в ленте 80000")
    x2, y2, vx2, _vy2, a2 = start_and_actions(ROLLOUT_SEED0, 15)
    x20, y20, vx20, _vy20, a20 = start_and_actions(ROLLOUT_SEED0, 15)
    if a2.tolist() != a20.tolist() or x2 != x20 or vx2 != vx20 or y2 != y20:
        raise RuntimeError("старт 80000 зависит от вызова")
    s2 = rollout_states(x2, y2, vx2, _vy2, np.repeat(a2, 2), 2.0)
    s20 = rollout_states(x20, y20, vx20, _vy20, np.repeat(a20, 2), 20.0)
    if s2[0, 0] != s20[0, 0] or s2[0, 2] != s20[0, 2]:
        raise RuntimeError("стартовые x и vx разошлись")

    payload = {
        "tag": TAG,
        "dt": 1.0 / 30.0,
        "ground": GROUND,
        "width_nominal": 512.0,
        "pad": [248.0, 264.0],
        "fuel0": 10.0,
        "vy_soft": 8.0,
        "episode_cap": 450,
        "cem_horizon": HORIZON,
        "r2_floor": R2_FLOOR,
        "start_height_mean": heights[0],
        "nets": nets,
        "tape_80000": vis_tape,
        "plan": plan,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()

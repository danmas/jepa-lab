"""Зонд, открытая раскатка, контрфакт цели и план с целевым предиктором.

Пишет out/numeric_land_goal_rollout.json целиком в конце.

    python -m numeric_land.rollout_goal
"""
import json
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.data import TRAIN_SEEDS, VARIANTS
from numeric_land.field import (
    GROUND, HOLD, MARKS, N_HOLD_ACTIONS, PROBE_STEPS, scale_state, start_and_actions, step_field,
)
from numeric_land.goal import (
    OPEN_PADS, OPEN_VY, PLAN_N, PLAN_PADS, PLAN_SEED0, PLAN_VY, PROBE_EPISODES, PROBE_SEED0,
    RANGE_G, ROLLOUT_N, ROLLOUT_SEED0, probe_states, world_vector,
)
from numeric_land.models import LATENT, BallEncoder, LandPredictor, action_pairs, fit_readout
from numeric_land.plan_goal import HORIZON, ITERS, SAMPLES, run_episode
from numeric_land.train_goal import SOURCE_NAME, STEPS, TAG, _load_source, _same, ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_land_goal_rollout.json")
RIDGE_LAM = 1e-2
PROBE_TRAIN = 3072
R2_NAMES = ("x", "y", "vx", "vy", "fuel")
R2_FLOOR = 0.9


def _r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot.clamp_min(1e-12)).cpu().numpy()


def _mean_std(values):
    values = np.asarray(values, np.float64)
    if values.shape != (ROLLOUT_N,):
        raise RuntimeError(f"нужен вектор из {ROLLOUT_N}, пришло {values.shape}")
    return {"mean": round(float(values.mean()), 6), "std": round(float(values.std(ddof=1)), 6)}


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("mode") != "jepa" or ck.get("tag") != TAG or int(ck.get("steps", 0)) != STEPS:
        raise RuntimeError(path)
    if ck.get("source") != SOURCE_NAME or not ck.get("frozen_encoder"):
        raise RuntimeError(path)
    variant = ck["variant"]
    encoder = BallEncoder()
    predictor = LandPredictor(int(ck["n_world"]))
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval(); predictor.eval(); ema.eval()
    return ck, encoder, predictor, ema


def _fit(ema, states):
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


def _held(seed, g):
    x, y, vx, vy, actions = start_and_actions(seed, N_HOLD_ACTIONS)
    held = np.repeat(actions, HOLD)
    states = [(x, y, vx, vy, 10.0)]
    fuel = 10.0
    grounded = False
    for action in held:
        x, y, vx, vy, fuel, grounded = step_field(x, y, vx, vy, fuel, grounded, int(action), g)
        states.append((x, y, vx, vy, fuel))
    return np.asarray(states, np.float64), held, actions


def _predict_marks(encoder, predictor, probe, states, held, g, pad, vy, variant):
    scaled = torch.from_numpy(scale_state(states))
    world = None
    if variant == "vis":
        world = torch.from_numpy(world_vector(np.array([float(g)]),
                                              np.array([float(pad)]), np.array([float(vy)])))
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
                "pred": [float(v) for v in pred],
            }
    if set(found) != set(MARKS):
        raise RuntimeError(sorted(found))
    return found


def _collect_roll(encoder, predictor, probe, variant, g):
    """Раскатка на тяжёстях g: ошибки по площадкам; предсказания hid при разных целях совпадают."""
    bag = {(pad, vy): {m: {"pos_px": [], "vel_px": []} for m in MARKS}
           for pad in OPEN_PADS for vy in OPEN_VY}
    tape0 = None
    for i in range(ROLLOUT_N):
        states, held, actions = _held(ROLLOUT_SEED0 + i, g)
        again, _h, _a = _held(ROLLOUT_SEED0 + i, g)
        if not np.array_equal(states, again):
            raise RuntimeError("истина раскатки не повторилась")
        if i == 0:
            tape0 = [int(a) for a in actions]
        # база для hid: цель (256, 0)
        base = None
        for pad in OPEN_PADS:
            for vy in OPEN_VY:
                found = _predict_marks(encoder, predictor, probe, states, held, g, pad, vy, variant)
                if base is None:
                    base = found
                elif variant == "hid":
                    for m in MARKS:
                        if found[m]["pred"] != base[m]["pred"]:
                            raise RuntimeError(f"скрытый различил цели на шаге {m}")
                for m in MARKS:
                    bag[(pad, vy)][m]["pos_px"].append(found[m]["pos_px"])
                    bag[(pad, vy)][m]["vel_px"].append(found[m]["vel_px"])
    cells = []
    for pad in OPEN_PADS:
        for vy in OPEN_VY:
            cells.append({
                "pad_x": float(pad),
                "desired_vy": float(vy),
                "marks": {str(m): {k: _mean_std(bag[(pad, vy)][m][k]) for k in ("pos_px", "vel_px")}
                          for m in MARKS},
            })
    return {"g": float(g), "seed0": ROLLOUT_SEED0, "n": ROLLOUT_N, "hold": 2, "horizon": 30,
            "tape0": tape0, "cells": cells}


def _counterfactual(encoder, predictor, probe, variant):
    """Одно состояние, действие 1, разные цели. vis различает, hid — нет."""
    raw = np.array([[256.0, 464.0, 0.0, 0.0, 10.0]], np.float64)
    goals = [(200.0, 0.0), (200.0, 6.0), (406.0, 0.0), (406.0, 6.0)]
    rows = []
    preds = []
    with torch.no_grad():
        z0 = encoder(torch.from_numpy(scale_state(raw)))
        for pad, vy in goals:
            world = None
            if variant == "vis":
                world = torch.from_numpy(world_vector(np.array([8.0]), np.array([pad]), np.array([vy])))
            z = predictor(z0, action_pairs(torch.tensor([1]), z0), world)
            pred = probe.fc(z)[0]
            pred_f = [round(float(pred[i]), 6) for i in range(5)]
            preds.append(pred_f)
            rows.append({"pad_x": pad, "desired_vy": vy, "g": 8.0, "pred": pred_f})
    if variant == "hid":
        for r in preds[1:]:
            if r != preds[0]:
                raise RuntimeError(f"скрытый различил цели: {preds}")
    rec = {
        "action": 1, "start": [256.0, 464.0, 0.0, 0.0, 10.0], "g": 8.0,
        "rows": rows, "hid_match": variant != "vis",
    }
    if variant == "vis":
        rec["vis_x_gap_pad"] = round(abs(preds[2][0] - preds[0][0]), 6)
        rec["vis_x_gap_vy"] = round(abs(preds[1][0] - preds[0][0]), 6)
    return rec


def _train_row(ck, path):
    last = ck["log"][-1]
    weight = ck["predictor"]["net.0.weight"]
    return {
        "file": os.path.basename(path), "protocol": ck["protocol"], "variant": ck["variant"],
        "seed": int(ck["seed"]), "steps": int(ck["steps"]), "seconds": ck.get("seconds"),
        "loss": last["loss"], "l1": last["l1"], "term": last.get("term"), "penalty": last["penalty"],
        "w_term": ck.get("w_term"), "frozen_encoder": True, "source": ck.get("source"),
        "n_world": int(ck["n_world"]), "predictor_in": int(weight.shape[1]),
    }


def _plan_row(episodes, variant, pad, g, vy):
    counts = {name: sum(1 for ep in episodes if ep["result"] == name)
              for name in ("мягкая", "удар", "не сел")}
    touched = [ep for ep in episodes if ep["result"] != "не сел"]
    missed = [ep for ep in episodes if ep["result"] == "не сел"]
    touch_mean = round(sum(ep["steps"] for ep in touched) / len(touched), 3) if touched else None
    miss_height = round(sum(GROUND - ep["y"] for ep in missed) / len(missed), 3) if missed else None
    return {
        "file": f"landg_range_{variant}_s0.pt", "variant": variant, "g": float(g),
        "pad_x": float(pad), "desired_vy": float(vy), "horizon": HORIZON, "samples": SAMPLES,
        "iters": ITERS, "cap": 450, "of": PLAN_N,
        "soft": counts["мягкая"], "crash": counts["удар"], "miss": counts["не сел"],
        "touch_n": len(touched), "touch_steps_mean": touch_mean, "miss_height_mean": miss_height,
        "episodes": episodes,
    }


def _partial_path():
    return OUT_JSON + ".episodes.jsonl"


def _load_partial():
    done = {}
    if not os.path.isfile(_partial_path()):
        return done
    with open(_partial_path(), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            key = (ep["variant"], float(ep["g"]), float(ep["pad_x"]), float(ep["desired_vy"]), int(ep["seed"]))
            done[key] = ep
    return done


def _append_partial(ep):
    with open(_partial_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(ep, ensure_ascii=False) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets-only", action="store_true",
                    help="только зонд/раскатка/контрфакт (быстрые секции), без плана")
    ap.add_argument("--seeds", type=int, default=10, help="сидов плана на клетку")
    ap.add_argument("--pads", default="200,406", help="центры плана")
    ap.add_argument("--gs", default="8,20", help="тяжести плана")
    ap.add_argument("--vys", default="0,6", help="цели desired_vy плана")
    args = ap.parse_args()
    plan_pads = [float(x) for x in args.pads.split(",") if x]
    plan_gs = [float(x) for x in args.gs.split(",") if x]
    plan_vys = [float(x) for x in args.vys.split(",") if x]
    plan_n = int(args.seeds)

    torch.set_num_threads(8)
    source = _load_source()
    states = probe_states()
    nets = []
    tapes = {}
    r2_seen = None
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            path = ckpt_path(variant, seed)
            if not os.path.isfile(path):
                raise SystemExit(f"нет {os.path.basename(path)}")
            ck, encoder, predictor, ema = _load(path)
            print(f"зонд {os.path.basename(path)}", flush=True)
            if not _same(ck["encoder"], source["encoder"]):
                raise RuntimeError(f"кодировщик {os.path.basename(path)} сдвинулся")
            if not _same(ck["ema"], source["ema"]):
                raise RuntimeError(f"EMA {os.path.basename(path)} сдвинулась")
            expect_in = 38 if variant == "vis" else 34
            if int(ck["predictor"]["net.0.weight"].shape[1]) != expect_in:
                raise RuntimeError("ширина предиктора")
            probe, r2 = _fit(ema, states)
            if r2_seen is None:
                r2_seen = r2
            elif r2 != r2_seen:
                raise RuntimeError(f"зонд разошёлся у {os.path.basename(path)}")
            collapsed = any(r2[name] < R2_FLOOR for name in R2_NAMES)
            rolls = []
            for g in RANGE_G:
                print(f"  раскатка g={g:g}", flush=True)
                row = _collect_roll(encoder, predictor, probe, variant, g)
                key = (variant, seed)
                tapes.setdefault(key, row["tape0"])
                if row["tape0"] != tapes[key]:
                    raise RuntimeError("лента сида 96500 разошлась")
                rolls.append(row)
            nets.append({
                "train": _train_row(ck, path),
                "encoder_matches_source": True,
                "ema_matches_source": True,
                "probe": {"own": "range", "seed0": PROBE_SEED0, "n": PROBE_EPISODES,
                          "r2": r2, "collapsed": bool(collapsed), "same_across_nets": True},
                "rollout": rolls,
                "counterfactual": _counterfactual(encoder, predictor, probe, variant),
            })

    payload_base = {
        "tag": TAG, "dt": 1.0 / 30.0, "ground": GROUND, "pad_half": 8.0,
        "train_pad": [156.0, 356.0], "desired_vy_set": list(OPEN_VY),
        "source_encoder": SOURCE_NAME, "open_pads": list(OPEN_PADS),
        "open_vy": list(OPEN_VY), "fuel0": 10.0, "vy_soft": 8.0, "cem_horizon": HORIZON,
        "r2_floor": R2_FLOOR, "probe_same_across_nets": True, "w_term": 0.25,
        "nets": nets,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    nets_json = OUT_JSON.replace(".json", "_nets.json")
    with open(nets_json, "w", encoding="utf-8") as handle:
        json.dump(payload_base, handle, ensure_ascii=False, indent=2)
    print(nets_json, flush=True)
    if args.nets_only:
        print("nets-only: план пропущен", flush=True)
        return

    print("план", flush=True)
    plan = []
    kept = ("seed", "g", "pad_x", "desired_vy", "result", "steps", "x", "y", "vy", "fuel",
            "start_x", "start_y", "plan_s")
    loaded = {}
    for variant in ("vis", "hid"):
        _ck, encoder, predictor, ema = _load(ckpt_path(variant, 0))
        probe, _r2 = _fit(ema, states)
        loaded[variant] = (encoder, predictor, probe)
    done = _load_partial()
    for variant in ("vis", "hid"):
        for pad in plan_pads:
            for g in plan_gs:
                for vy in plan_vys:
                    episodes = []
                    for seed in range(PLAN_SEED0, PLAN_SEED0 + plan_n):
                        key = (variant, float(g), float(pad), float(vy), int(seed))
                        if key in done:
                            ep = done[key]
                        else:
                            fresh = run_episode(seed, g, pad, vy, loaded[variant])
                            ep = {k: fresh[k] for k in kept}
                            ep["variant"] = variant
                            _append_partial(ep)
                        episodes.append({k: ep[k] for k in kept})
                    row = _plan_row(episodes, variant, pad, g, vy)
                    row["of"] = plan_n
                    plan.append(row)
                    print(f"  {variant} pad={pad:g} g={g:g} vy={vy:g}: "
                          f"мягкая {row['soft']} удар {row['crash']} не сел {row['miss']}", flush=True)

    payload = dict(payload_base)
    payload["plan_pads"] = plan_pads
    payload["plan_gs"] = plan_gs
    payload["plan_vy"] = plan_vys
    payload["plan_n"] = plan_n
    payload["plan"] = plan
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)
    if os.path.isfile(_partial_path()):
        os.remove(_partial_path())
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
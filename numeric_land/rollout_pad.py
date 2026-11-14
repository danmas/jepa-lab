"""Зонд, открытая раскатка и посадка с подвижным центром.

Пишет out/numeric_land_pad_rollout.json целиком в конце.

    python -m numeric_land.rollout_pad
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
from numeric_land.models import LATENT, BallEncoder, LandPredictor, action_pairs, fit_readout
from numeric_land.pad import (
    OPEN_PADS, PLAN_N, PLAN_PADS, PLAN_SEED0, PROBE_EPISODES, PROBE_SEED0, RANGE_G,
    ROLLOUT_N, ROLLOUT_SEED0, probe_states, world_vector,
)
from numeric_land.plan_pad import HORIZON, ITERS, SAMPLES, run_episode
from numeric_land.train_pad import SOURCE_NAME, STEPS, TAG, _load_source, _same, ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_land_pad_rollout.json")
PARTIAL = OUT_JSON + ".episodes.jsonl"
RIDGE_LAM = 1e-2
PROBE_TRAIN = 3072
R2_NAMES = ("x", "y", "vx", "vy", "fuel")
R2_FLOOR = 0.9


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
    if ck.get("source") != SOURCE_NAME or not ck.get("frozen_encoder"):
        raise RuntimeError(path)
    variant = ck["variant"]
    encoder = BallEncoder()
    predictor = LandPredictor(int(ck["n_world"]))
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


def _held(seed: int, g: float):
    x, y, vx, vy, actions = start_and_actions(seed, N_HOLD_ACTIONS)
    held = np.repeat(actions, HOLD)
    if len(held) != N_HOLD_ACTIONS * HOLD:
        raise RuntimeError(len(held))
    states = []
    fuel = 10.0
    grounded = False
    row0 = (x, y, vx, vy, fuel)
    states.append(row0)
    for action in held:
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, int(action), g)
        states.append((x, y, vx, vy, fuel))
    return np.asarray(states, np.float64), held, actions


def _predict_marks(encoder, predictor, probe, states, held, g, pad_x, variant):
    scaled = torch.from_numpy(scale_state(states))
    world = None
    if variant == "vis":
        world = torch.from_numpy(world_vector(np.array([float(g)]), np.array([float(pad_x)])))
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
    bag = {pad: {m: {"pos_px": [], "vel_px": []} for m in MARKS} for pad in OPEN_PADS}
    tape0 = None
    for i in range(ROLLOUT_N):
        states, held, actions = _held(ROLLOUT_SEED0 + i, g)
        again, _held2, _acts2 = _held(ROLLOUT_SEED0 + i, g)
        if not np.array_equal(states, again):
            raise RuntimeError("истина раскатки не повторилась")
        if i == 0:
            tape0 = [int(a) for a in actions]
        base = None
        for pad in OPEN_PADS:
            found = _predict_marks(
                encoder, predictor, probe, states, held, g, pad, variant)
            if base is None:
                base = found
            elif variant == "hid":
                for m in MARKS:
                    if found[m]["pred"] != base[m]["pred"]:
                        raise RuntimeError(f"скрытый различил центры на шаге {m}")
            for m in MARKS:
                bag[pad][m]["pos_px"].append(found[m]["pos_px"])
                bag[pad][m]["vel_px"].append(found[m]["vel_px"])
    pads = []
    for pad in OPEN_PADS:
        pads.append({
            "pad_x": float(pad),
            "marks": {
                str(m): {k: _mean_std(bag[pad][m][k]) for k in ("pos_px", "vel_px")}
                for m in MARKS
            },
        })
    return {
        "g": float(g),
        "seed0": ROLLOUT_SEED0,
        "n": ROLLOUT_N,
        "hold": 2,
        "horizon": 30,
        "tape0": tape0,
        "pads": pads,
    }


def _counterfactual(encoder, predictor, probe, variant):
    raw = np.array([[256.0, 464.0, 0.0, 0.0, 10.0]], np.float64)
    with torch.no_grad():
        z0 = encoder(torch.from_numpy(scale_state(raw)))
        pad_rows = []
        pad_preds = []
        for pad in (106.0, 406.0):
            world = None
            if variant == "vis":
                world = torch.from_numpy(world_vector(np.array([8.0]), np.array([pad])))
            z = predictor(z0, action_pairs(torch.tensor([1]), z0), world)
            pred = probe.fc(z)[0]
            true = step_field(256.0, 464.0, 0.0, 0.0, 10.0, False, 1, 8.0)
            pred_f = [float(pred[i]) for i in range(5)]
            pad_preds.append(pred_f)
            pad_rows.append({
                "pad_x": pad,
                "g": 8.0,
                "pred": [round(v, 6) for v in pred_f],
                "true": [round(float(true[i]), 6) for i in range(5)],
            })
        if pad_rows[0]["true"] != pad_rows[1]["true"]:
            raise RuntimeError("истина контрфакта центра разошлась")
        if variant == "hid" and pad_preds[0] != pad_preds[1]:
            raise RuntimeError(f"скрытый различил центры: {pad_preds}")
        g_rows = []
        g_vys = []
        for g in (2.0, 15.0):
            world = None
            if variant == "vis":
                world = torch.from_numpy(world_vector(np.array([g]), np.array([256.0])))
            z = predictor(z0, action_pairs(torch.tensor([7]), z0), world)
            pred = probe.fc(z)[0]
            true = step_field(256.0, 464.0, 0.0, 0.0, 10.0, False, 7, g)
            g_vys.append(float(pred[3]))
            g_rows.append({
                "g": g,
                "pad_x": 256.0,
                "pred_y": round(float(pred[1]), 6),
                "pred_vy": round(float(pred[3]), 6),
                "true_y": round(float(true[1]), 6),
                "true_vy": round(float(true[3]), 6),
            })
    gap_vy = g_rows[1]["true_vy"] - g_rows[0]["true_vy"]
    gap_y = g_rows[1]["true_y"] - g_rows[0]["true_y"]
    if abs(gap_vy - 13.0 / 30.0) > 1e-6 or abs(gap_y - 13.0 / 900.0) > 1e-6:
        raise RuntimeError(f"зазор контрфакта {gap_vy}, {gap_y}")
    if variant == "hid" and g_vys[0] != g_vys[1]:
        raise RuntimeError(f"скрытый контрфакт различил тяжести: {g_vys}")
    return {
        "pad": {
            "action": 1,
            "rows": pad_rows,
            "hid_match": variant != "vis",
        },
        "gravity": {
            "action": 7,
            "rows": g_rows,
            "true_vy_gap": round(gap_vy, 6),
            "true_y_gap": round(gap_y, 6),
            "hid_vy_match": variant != "vis",
        },
    }


def _train_row(ck, path):
    last = ck["log"][-1]
    weight = ck["predictor"]["net.0.weight"]
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
        "frozen_encoder": True,
        "source": ck.get("source"),
        "n_world": int(ck["n_world"]),
        "predictor_in": int(weight.shape[1]),
    }


def _plan_row(episodes, variant):
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
        "file": f"landp_range_{variant}_s0.pt",
        "variant": variant,
        "g": float(episodes[0]["g"]),
        "pad_x": float(episodes[0]["pad_x"]),
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


def _load_partial():
    done = {}
    if not os.path.isfile(PARTIAL):
        return done
    with open(PARTIAL, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            key = (ep["variant"], float(ep["g"]), float(ep["pad_x"]), int(ep["seed"]))
            done[key] = ep
    return done


def _append_partial(ep):
    with open(PARTIAL, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(ep, ensure_ascii=False) + "\n")


def main():
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
            expect_in = 37 if variant == "vis" else 34
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
                    raise RuntimeError("лента сида 93000 разошлась")
                rolls.append(row)
            cell = {
                "train": _train_row(ck, path),
                "encoder_matches_source": True,
                "ema_matches_source": True,
                "probe": {
                    "own": "range",
                    "seed0": PROBE_SEED0,
                    "n": PROBE_EPISODES,
                    "r2": r2,
                    "collapsed": bool(collapsed),
                    "same_across_nets": True,
                },
                "rollout": rolls,
                "counterfactual": _counterfactual(encoder, predictor, probe, variant),
            }
            nets.append(cell)

    if tapes[("vis", 0)] != tapes[("hid", 0)]:
        raise RuntimeError("vis и hid разошлись в ленте 93000")

    print("план landp_range_vis_s0 и landp_range_hid_s0", flush=True)
    plan = []
    kept = ("seed", "g", "pad_x", "result", "steps", "x", "y", "vy", "fuel",
            "start_x", "start_y", "plan_s")
    done = _load_partial()
    if done:
        print(f"  уже есть эпизодов: {len(done)}", flush=True)
    loaded = {}
    for variant in ("vis", "hid"):
        _ck, encoder, predictor, ema = _load(ckpt_path(variant, 0))
        probe, _r2 = _fit(ema, states)
        loaded[variant] = (encoder, predictor, probe)
    anchor = None
    for variant in ("vis", "hid"):
        for pad in PLAN_PADS:
            for g in RANGE_G:
                episodes = []
                for seed in range(PLAN_SEED0, PLAN_SEED0 + PLAN_N):
                    key = (variant, float(g), float(pad), int(seed))
                    if key in done:
                        ep = done[key]
                    else:
                        fresh = run_episode(seed, g, pad, loaded[variant])
                        ep = {k: fresh[k] for k in kept}
                        ep["variant"] = variant
                        _append_partial(ep)
                        print(
                            f"  {variant} g={g:g} pad={pad:g} seed={seed} "
                            f"{ep['result']} шагов {ep['steps']}",
                            flush=True)
                    episodes.append({k: ep[k] for k in kept})
                row = _plan_row(episodes, variant)
                if anchor is None:
                    anchor = (episodes[0]["start_x"], episodes[0]["start_y"], row["start_height_mean"])
                else:
                    if (episodes[0]["start_x"], episodes[0]["start_y"]) != anchor[:2]:
                        raise RuntimeError("старт 94000 зависит от клетки")
                    if abs(row["start_height_mean"] - anchor[2]) > 1e-6:
                        raise RuntimeError("высота старта зависит от клетки")
                plan.append(row)

    payload = {
        "tag": TAG,
        "dt": 1.0 / 30.0,
        "ground": GROUND,
        "pad_half": 8.0,
        "train_pad": [156.0, 356.0],
        "train_pad_high_open": True,
        "source_encoder": SOURCE_NAME,
        "open_pads": list(OPEN_PADS),
        "plan_pads": list(PLAN_PADS),
        "fuel0": 10.0,
        "vy_soft": 8.0,
        "episode_cap": 450,
        "cem_horizon": HORIZON,
        "r2_floor": R2_FLOOR,
        "probe_same_across_nets": True,
        "start_height_mean": anchor[2],
        "nets": nets,
        "tape_93000": tapes[("vis", 0)],
        "plan": plan,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)
    if os.path.isfile(PARTIAL):
        os.remove(PARTIAL)
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()

"""Замер трёх семейств предикторов: без учителя (goal), правило v1 (expert), наведение (hover).

Метрики на эпизод: результат, промах |finish_x − pad_x|, попадание в окно ±8,
«мягко и в окне». Сравнение по одинаковым клеткам.

    python -m numeric_land.rollout_hover [--seeds 6] [--pads 200,406] [--gs 8,20] [--vys 0,6]
    python -m numeric_land.rollout_hover --nets-only      # без плана: только зонд и раскатка
"""
import argparse
import json
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.field import GROUND, PROBE_STEPS, scale_state
from numeric_land.goal import PLAN_SEED0, probe_states
from numeric_land.models import LATENT, BallEncoder, LandPredictor, fit_readout
from numeric_land.plan_goal import HORIZON, ITERS, SAMPLES, run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_land_hover_rollout.json")
OUT_NETS = os.path.join(ROOT, "out", "numeric_land_hover_rollout_nets.json")
RIDGE_LAM = 1e-2
PROBE_TRAIN = 3072
R2_NAMES = ("x", "y", "vx", "vy", "fuel")

# Семейство -> (файл, тег). Логика эпизода общая, различается только чекпоинт.
FAMILIES = {
    "goal": ("landg_range_vis_s0.pt", "numeric-land-goal"),
    "expert": ("lande_range_vis_s0.pt", "numeric-land-expert"),
    "hover": ("landh_range_vis_s0.pt", "numeric-land-hover"),
}


def _load(name):
    fname, tag = FAMILIES[name]
    path = os.path.join(ROOT, "out", fname)
    if not os.path.isfile(path):
        raise SystemExit(f"нет {fname}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != tag:
        raise RuntimeError(f"{fname}: тег {ck.get('tag')} != {tag}")
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


def _r2(target, pred):
    ss_res = ((target - pred) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot.clamp_min(1e-12)).cpu().numpy()


def _probe_and_rollout(ema, predictor, encoder):
    """Зонд качества латента плюс раскатка по 2/4/8/40 шагов."""
    states = probe_states()
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states, np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    probe = fit_readout(z[:PROBE_TRAIN], target[:PROBE_TRAIN], lam=RIDGE_LAM)
    with torch.no_grad():
        pred = probe.fc(z[PROBE_TRAIN:])
    r2 = {n: round(float(v), 6) for n, v in zip(R2_NAMES, _r2(target[PROBE_TRAIN:], pred))}
    return probe, r2


def _run(models, name, seed, g, pad, vy):
    ep = run_episode(seed, g, pad, vy, models, max_steps=450)
    touch = ep["result"] != "не сел"
    miss = abs(ep["x"] - float(pad))
    return {
        "family": name, "seed": int(seed), "g": float(g), "pad_x": float(pad),
        "desired_vy": float(vy), "result": ep["result"], "steps": ep["steps"],
        "finish_x": round(float(ep["x"]), 4), "finish_vy": round(float(ep["vy"]), 4),
        "miss_center": round(float(miss), 4), "touch": touch,
        "soft": ep["result"] == "мягкая",
        "center": bool(touch and miss <= 8.0),
        "both": bool(ep["result"] == "мягкая" and miss <= 8.0),
    }


def _agg(rows):
    n = len(rows)
    touched = [r for r in rows if r["touch"]]
    return {
        "n": n,
        "soft": sum(1 for r in rows if r["soft"]),
        "center": sum(1 for r in rows if r["center"]),
        "both": sum(1 for r in rows if r["both"]),
        "touch": len(touched),
        "miss_center_mean": round(float(np.mean([r["miss_center"] for r in touched])), 3) if touched else None,
        "finish_vy_mean": round(float(np.mean([r["finish_vy"] for r in touched])), 3) if touched else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--pads", default="200,406")
    ap.add_argument("--gs", default="8,20")
    ap.add_argument("--vys", default="0,6")
    ap.add_argument("--nets-only", action="store_true",
                    help="без плана: только зонд и раскатка (быстро)")
    args = ap.parse_args()
    pads = [float(x) for x in args.pads.split(",") if x]
    gs = [float(x) for x in args.gs.split(",") if x]
    vys = [float(x) for x in args.vys.split(",") if x]
    torch.set_num_threads(8)

    nets = {}
    for name in FAMILIES:
        ck, enc, pred, ema = _load(name)
        probe, r2 = _probe_and_rollout(ema, pred, enc)
        nets[name] = {"tag": ck["tag"], "n_world": int(ck["n_world"]),
                      "r2": r2, "probe": probe, "enc": enc, "pred": pred}
        print(f"{name:7s} тег {ck['tag']:20s} n_world={ck['n_world']} "
              f"R2 x={r2['x']} y={r2['y']} fuel={r2['fuel']}", flush=True)

    if args.nets_only:
        payload = {"families": FAMILIES, "nets_only": True,
                   "nets": {k: {"tag": v["tag"], "n_world": v["n_world"], "r2": v["r2"]}
                            for k, v in nets.items()}}
        with open(OUT_NETS, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("\n" + OUT_NETS, flush=True)
        return

    cells = []
    totals = {name: [] for name in nets}
    for name in nets:
        models = (nets[name]["enc"], nets[name]["pred"], nets[name]["probe"])
        for pad in pads:
            for g in gs:
                for vy in vys:
                    rows = [_run(models, name, PLAN_SEED0 + i, g, pad, vy)
                            for i in range(args.seeds)]
                    totals[name].extend(rows)
                    a = _agg(rows)
                    cells.append({"family": name, "pad_x": pad, "g": g, "desired_vy": vy, **a})
                    print(f"  {name:7s} pad={pad:5.0f} g={g:4.0f} vy={vy:.0f}: "
                          f"мягкая {a['soft']}/{a['n']} центр {a['center']}/{a['n']} "
                          f"оба {a['both']}/{a['n']} промах {a['miss_center_mean']} "
                          f"vy {a['finish_vy_mean']}", flush=True)

    print("\n=== ИТОГ ===", flush=True)
    summary = {}
    for name, rows in totals.items():
        a = _agg(rows)
        summary[name] = a
        print(f"{name:7s}: мягкая {a['soft']}/{a['n']}, центр {a['center']}/{a['n']}, "
              f"оба {a['both']}/{a['n']}, коснулись {a['touch']}/{a['n']}, "
              f"промах ср {a['miss_center_mean']}, vy ср {a['finish_vy_mean']}", flush=True)

    payload = {
        "pads": pads, "gs": gs, "vys": vys, "seeds": args.seeds, "seed0": PLAN_SEED0,
        "cem": {"samples": SAMPLES, "iters": ITERS, "horizon": HORIZON, "cap": 450},
        "families": FAMILIES, "nets": {k: {"tag": v["tag"], "n_world": v["n_world"], "r2": v["r2"]}
                                       for k, v in nets.items()},
        "summary": summary, "cells": cells,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()

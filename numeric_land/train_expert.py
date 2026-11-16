"""Шесть предикторов на смеси случайных окон и траекторий правила (expert demos).

    python -m numeric_land.train_expert --tag numeric-land-expert [--only range-vis-0] [--expert-frac 0.5]

Архитектура и лосс — как в numeric-land-goal (цель — вход предиктора, кодировщик
заморожен из out/landf_range_vis_s0.pt). Отличие одно: часть батча берётся из пула
траекторий правила (см. numeric_land/expert.py). Тег numeric-land-expert, файлы lande_*.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_land.data import TRAIN_SEEDS, VARIANTS
from numeric_land.expert import POOL_DEFAULT, make_batch_mixed, expert_pool
from numeric_land.field import GROUND, scale_state
from numeric_land.goal import N_WORLD
from numeric_land.goal import world_vector
from numeric_land.models import BallEncoder, LandPredictor
from numeric_land.train_goal import (
    EMA_MOMENTUM, LR, SOURCE_NAME, STEPS, BATCH, THREADS, WEIGHT_DECAY, W_TERM,
    _load_source, _same, goal_loss, source_path,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
TAG = "numeric-land-expert"


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "обучение экспертного пакета: нужен тег numeric-land-expert. "
            "numeric-land, numeric-land-field, numeric-land-pad, numeric-land-goal, "
            "пустой тег, v4, v5-*, numeric, numeric-dt, numeric-walls и numeric-scale запрещены.")


def ckpt_path(variant: str, seed: int) -> str:
    if variant not in VARIANTS or int(seed) not in TRAIN_SEEDS:
        raise ValueError((variant, seed))
    return os.path.join(OUT, f"lande_range_{variant}_s{int(seed)}.pt")


def _save(path, payload):
    if not os.path.basename(path).startswith("lande_"):
        raise RuntimeError(os.path.basename(path))
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _loop(encoder, predictor, ema, opt, rng, pool, label, expert_frac):
    log = []
    t0 = time.perf_counter()
    print(f"[{label}] steps={STEPS} batch={BATCH} expert_frac={expert_frac} "
          f"pool={pool['states'].shape[0]} (landed {pool['landed_frac']:.2f}) w_term={W_TERM}",
          flush=True)
    for step in range(1, STEPS + 1):
        states, actions, gs, pads, vys = make_batch_mixed(
            rng, BATCH, pool, expert_frac=expert_frac)
        states_t = torch.from_numpy(scale_state(states))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        height0 = GROUND - states[:, 0, 1]
        world_t = None
        if predictor.n_world:
            world_t = torch.from_numpy(world_vector(gs, pads, vys))
        loss, parts = goal_loss(encoder, predictor, ema, states_t, actions_t,
                                world_t, pads, vys, height0, freeze_encoder=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5),
                   **{k: round(v, 5) for k, v in parts.items()}}
            log.append(row)
            el = time.perf_counter() - t0
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} l1={row['l1']:.4f} "
                  f"term={row.get('term', 0):.4f} pen={row['penalty']:.4f} "
                  f"{el / step * 1000:.0f} ms/step", flush=True)
    return log, time.perf_counter() - t0


def train_one(variant, seed, tag, pool, expert_frac):
    refuse_tag(tag)
    source = _load_source()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder = BallEncoder()
    encoder.load_state_dict(source["encoder"])
    encoder.requires_grad_(False)
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    ema.load_state_dict(source["ema"])
    for p in ema.parameters():
        p.requires_grad_(False)
    predictor = LandPredictor(N_WORLD[variant])
    opt = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    label = f"expert {variant} s{seed}"
    log, seconds = _loop(encoder, predictor, ema, opt, rng, pool, label, expert_frac)
    if not _same(encoder.state_dict(), source["encoder"]):
        raise RuntimeError("кодировщик сдвинулся")
    if not _same(ema.state_dict(), source["ema"]):
        raise RuntimeError("EMA сдвинулась")
    path = ckpt_path(variant, seed)
    payload = {
        "mode": "jepa", "protocol": "range", "variant": variant, "seed": int(seed),
        "g": None, "tag": tag, "steps": STEPS, "batch": BATCH, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "ema_momentum": EMA_MOMENTUM,
        "n_world": N_WORLD[variant], "goal_dim": 2, "w_term": W_TERM,
        "frozen_encoder": True, "source": SOURCE_NAME,
        "expert_frac": float(expert_frac), "expert_pool": int(pool["states"].shape[0]),
        "expert_landed_frac": float(pool["landed_frac"]),
        "seconds": round(float(seconds), 3),
        "encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
        "ema": ema.state_dict(), "log": log,
    }
    _save(path, payload)
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--expert-frac", type=float, default=0.5)
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    args = ap.parse_args()
    refuse_tag(args.tag)
    if not 0.0 <= args.expert_frac <= 1.0:
        raise SystemExit("expert-frac вне 0…1")
    torch.set_num_threads(THREADS)
    print(f"строю пул экспертных окон: {args.pool} из траекторий правила", flush=True)
    pool = expert_pool(args.pool)
    print(f"пул: {pool['states'].shape[0]} окон, эпизодов {pool['episodes_seen']}, "
          f"доля коснувшихся {pool['landed_frac']:.3f}", flush=True)
    jobs = []
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            key = f"range-{variant}-{seed}"
            if not args.only or args.only == key:
                jobs.append((variant, seed))
    if not jobs:
        raise SystemExit(f"неизвестный --only {args.only}")
    for variant, seed in jobs:
        train_one(variant, seed, args.tag, pool, args.expert_frac)


if __name__ == "__main__":
    main()
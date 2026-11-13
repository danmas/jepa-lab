"""Обучение четырёх взглядов на одних физических окнах.

    python -m numeric_scale.train --tag numeric-scale
    python -m numeric_scale.train --tag numeric-scale --repr rel --variant vis --seed 0

Пустые --repr, --variant и --seed -1 учат все 12 сетей.
Пустой тег, v4, v5-*, numeric, numeric-dt и numeric-walls отклоняются до оптимизатора.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_scale.data import (
    HORIZON, REPRS, TRAIN_SEEDS, VARIANTS, make_batch, scale_state, world_vector,
)
from numeric_scale.models import N_WORLD, BallEncoder, ScalePredictor, jepa_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STEPS = 3000
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8


def refuse_tag(tag: str) -> None:
    if tag != "numeric-scale":
        raise SystemExit(
            "обучение масштаба: нужен тег numeric-scale. "
            "Пустой тег, v4, v5-*, numeric, numeric-dt и numeric-walls запрещены.")


def ckpt_path(repr_name: str, variant: str, seed: int) -> str:
    if repr_name not in REPRS or variant not in VARIANTS or int(seed) not in TRAIN_SEEDS:
        raise ValueError((repr_name, variant, seed))
    name = f"scale_{repr_name}_{variant}_s{int(seed)}.pt"
    return os.path.join(OUT, name)


def train_one(repr_name: str, variant: str, seed: int, tag: str) -> str:
    refuse_tag(tag)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder = BallEncoder()
    predictor = ScalePredictor(N_WORLD[variant])
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    log = []
    t0 = time.perf_counter()
    label = f"{repr_name} {variant} s{seed}"
    print(f"[{label}] steps={STEPS} batch={BATCH} lr={LR}", flush=True)
    for step in range(1, STEPS + 1):
        states, actions, widths, heights = make_batch(rng, BATCH, HORIZON)
        states_t = torch.from_numpy(scale_state(states, widths, heights, repr_name))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        world_t = None
        if variant == "vis":
            world_t = torch.from_numpy(world_vector(widths, heights))
        loss, parts = jepa_loss(encoder, predictor, ema, states_t, actions_t, world_t)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(encoder)
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5),
                   **{k: round(v, 5) for k, v in parts.items()}}
            log.append(row)
            el = time.perf_counter() - t0
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} pen={row['penalty']:.4f} "
                  f"{el / step * 1000:.0f} ms/step", flush=True)
    path = ckpt_path(repr_name, variant, seed)
    os.makedirs(OUT, exist_ok=True)
    torch.save({
        "mode": "jepa",
        "repr": repr_name,
        "variant": variant,
        "seed": int(seed),
        "tag": tag,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "n_world": N_WORLD[variant],
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "log": log,
    }, path)
    print(f"[{label}] {path}  ({(time.perf_counter() - t0) / 60:.1f} мин)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--repr", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    reprs = (args.repr,) if args.repr else REPRS
    variants = (args.variant,) if args.variant else VARIANTS
    seeds = (args.seed,) if args.seed >= 0 else TRAIN_SEEDS
    for repr_name in reprs:
        for variant in variants:
            for seed in seeds:
                train_one(repr_name, variant, int(seed), args.tag)


if __name__ == "__main__":
    main()

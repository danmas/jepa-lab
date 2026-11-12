"""Обучение JEPA со стенками в предикторе или без них.

    python -m numeric_walls.train --tag numeric-walls
    python -m numeric_walls.train --tag numeric-walls --world W --protocol range --variant vis --seed 0

Пустые --world, --protocol, --variant и --seed -1 учат все 24 сети.
Пустой тег, v4, v5-*, numeric и numeric-dt отклоняются до оптимизатора.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_walls.data import (
    HORIZON, PROTOCOLS, TRAIN_SEEDS, VARIANTS, WORLDS, make_batch, scale_state,
    scale_walls,
)
from numeric_walls.models import N_WALL, BallEncoder, WallPredictor, jepa_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STEPS = 3000
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8


def refuse_tag(tag: str) -> None:
    if tag != "numeric-walls":
        raise SystemExit(
            "обучение стенок: нужен тег numeric-walls. "
            "Пустой тег, v4, v5-*, numeric и numeric-dt запрещены.")


def ckpt_path(world: str, protocol: str, variant: str, seed: int) -> str:
    if (world not in WORLDS or protocol not in PROTOCOLS
            or variant not in VARIANTS or int(seed) not in TRAIN_SEEDS):
        raise ValueError((world, protocol, variant, seed))
    name = f"walls_{world}_{protocol}_{variant}_s{int(seed)}.pt"
    return os.path.join(OUT, name)


def train_one(world: str, protocol: str, variant: str, seed: int, tag: str) -> str:
    refuse_tag(tag)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder = BallEncoder()
    predictor = WallPredictor(N_WALL[variant])
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    log = []
    t0 = time.perf_counter()
    label = f"{world} {protocol} {variant} s{seed}"
    print(f"[{label}] steps={STEPS} batch={BATCH} lr={LR}", flush=True)
    for step in range(1, STEPS + 1):
        states, actions, walls, _spans = make_batch(rng, world, protocol, BATCH, HORIZON)
        states_t = torch.from_numpy(scale_state(states))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        walls_t = torch.from_numpy(scale_walls(walls)) if variant == "vis" else None
        loss, parts = jepa_loss(encoder, predictor, ema, states_t, actions_t, walls_t)
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
    path = ckpt_path(world, protocol, variant, seed)
    os.makedirs(OUT, exist_ok=True)
    torch.save({
        "mode": "jepa",
        "world": world,
        "protocol": protocol,
        "variant": variant,
        "seed": int(seed),
        "tag": tag,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "n_wall": N_WALL[variant],
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
    ap.add_argument("--world", default="")
    ap.add_argument("--protocol", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    worlds = (args.world,) if args.world else WORLDS
    protocols = (args.protocol,) if args.protocol else PROTOCOLS
    variants = (args.variant,) if args.variant else VARIANTS
    seeds = (args.seed,) if args.seed >= 0 else TRAIN_SEEDS
    for world in worlds:
        for protocol in protocols:
            for variant in variants:
                for seed in seeds:
                    train_one(world, protocol, variant, int(seed), args.tag)


if __name__ == "__main__":
    main()

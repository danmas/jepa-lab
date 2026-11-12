"""Обучение JEPA с явным dt.

    python -m numeric_dt.train --tag numeric-dt
    python -m numeric_dt.train --tag numeric-dt --arm B --variant with-dt --seed 0

Пустой тег, v4, v5-* и numeric отклоняются до оптимизатора.
Имена файлов от тега не зависят.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_dt.data import HORIZON, TRAIN_SEEDS, make_batch, scale_state
from numeric_dt.models import N_IN, DtEncoder, DtPredictor, jepa_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STEPS = 3000
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8
ARMS = ("A", "B")
VARIANTS = ("with-dt", "no-dt")


def refuse_tag(tag: str) -> None:
    if tag != "numeric-dt":
        raise SystemExit(
            "обучение dt: нужен тег numeric-dt. "
            "Пустой тег, v4, v5-* и numeric запрещены.")


def ckpt_path(arm: str, variant: str, seed: int) -> str:
    if arm not in ARMS or variant not in VARIANTS or int(seed) not in TRAIN_SEEDS:
        raise ValueError((arm, variant, seed))
    kind = "jepa" if variant == "with-dt" else "nodt"
    return os.path.join(OUT, f"numeric_dt{arm}_{kind}_s{int(seed)}.pt")


def train_jepa(arm: str, variant: str, seed: int, tag: str) -> str:
    refuse_tag(tag)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder = DtEncoder(N_IN[variant])
    predictor = DtPredictor()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    log = []
    t0 = time.perf_counter()
    label = f"{arm} {variant} s{seed}"
    print(f"[{label}] steps={STEPS} batch={BATCH} lr={LR}", flush=True)
    for step in range(1, STEPS + 1):
        states, actions, _dts = make_batch(rng, arm, variant, BATCH, HORIZON)
        states_t = torch.from_numpy(scale_state(states, variant))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        loss, parts = jepa_loss(encoder, predictor, ema, states_t, actions_t)
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
    path = ckpt_path(arm, variant, seed)
    os.makedirs(OUT, exist_ok=True)
    torch.save({
        "mode": "jepa",
        "arm": arm,
        "variant": variant,
        "seed": int(seed),
        "tag": tag,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "n_in": N_IN[variant],
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "log": log,
    }, path)
    print(f"[{label}] {path}  ({(time.perf_counter() - t0) / 60:.1f} мин)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="сторож: принимается только numeric-dt")
    ap.add_argument("--arm", default="", help="A или B; пусто — обе руки")
    ap.add_argument("--variant", default="", help="with-dt или no-dt; пусто — оба")
    ap.add_argument("--seed", type=int, default=-1, help="0, 1 или 2; -1 — все три")
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    arms = (args.arm,) if args.arm else ARMS
    variants = (args.variant,) if args.variant else VARIANTS
    seeds = (args.seed,) if args.seed >= 0 else TRAIN_SEEDS
    for arm in arms:
        for variant in variants:
            for seed in seeds:
                train_jepa(arm, variant, int(seed), args.tag)


if __name__ == "__main__":
    main()

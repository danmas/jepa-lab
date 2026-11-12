"""Обучение числовой JEPA и прямого предиктора.

    python -m numeric.train --tag numeric

Пустой тег, v4 и v5-* отклоняются до оптимизатора. Имена файлов от тега не
зависят: out/numeric_jepa_s0.pt … s2 и out/numeric_direct_s0.pt … s2.
Пиксельные чекпоинты этот скрипт не открывает.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric.data import HORIZON, TRAIN_SEEDS, make_batch, scale_state
from numeric.models import DirectPredictor, NumericEncoder, NumericPredictor, direct_loss, jepa_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STEPS = 3000
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8


def refuse_tag(tag: str) -> None:
    """Запрещённые теги не доходят до оптимизатора и не открывают чужие файлы."""
    if not tag or tag == "v4" or str(tag).startswith("v5-"):
        raise SystemExit(
            "числовое обучение: пустой тег, v4 и v5-* запрещены. "
            "Пиксельные чекпоинты не открываются.")


def ckpt_path(mode: str, seed: int) -> str:
    if mode not in ("jepa", "direct"):
        raise ValueError(mode)
    if int(seed) not in TRAIN_SEEDS:
        raise ValueError(seed)
    return os.path.join(OUT, f"numeric_{mode}_s{int(seed)}.pt")


def _batch_tensors(rng):
    states, actions = make_batch(rng, BATCH, HORIZON)
    return (
        torch.from_numpy(scale_state(states)),
        torch.from_numpy(actions.astype(np.int64)),
    )


def train_jepa(seed: int, tag: str) -> str:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder = NumericEncoder()
    predictor = NumericPredictor()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    log = []
    t0 = time.perf_counter()
    print(f"[jepa s{seed}] steps={STEPS} batch={BATCH} lr={LR}")
    for step in range(1, STEPS + 1):
        states, actions = _batch_tensors(rng)
        loss, parts = jepa_loss(encoder, predictor, ema, states, actions)
        if not torch.isfinite(loss):
            raise RuntimeError(f"jepa s{seed}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(encoder)
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5),
                   **{k: round(v, 5) for k, v in parts.items()}}
            log.append(row)
            el = time.perf_counter() - t0
            print(f"  jepa s{seed} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} pen={row['penalty']:.4f} "
                  f"{el / step * 1000:.0f} ms/step")
    path = ckpt_path("jepa", seed)
    torch.save({
        "mode": "jepa",
        "seed": int(seed),
        "tag": tag,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "log": log,
    }, path)
    print(f"[jepa s{seed}] {path}  ({(time.perf_counter() - t0) / 60:.1f} мин)")
    return path


def train_direct(seed: int, tag: str) -> str:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    predictor = DirectPredictor()
    opt = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    log = []
    t0 = time.perf_counter()
    print(f"[direct s{seed}] steps={STEPS} batch={BATCH} lr={LR}")
    for step in range(1, STEPS + 1):
        states, actions = _batch_tensors(rng)
        loss, parts = direct_loss(predictor, states, actions)
        if not torch.isfinite(loss):
            raise RuntimeError(f"direct s{seed}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5),
                   **{k: round(v, 5) for k, v in parts.items()}}
            log.append(row)
            el = time.perf_counter() - t0
            print(f"  direct s{seed} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} {el / step * 1000:.0f} ms/step")
    path = ckpt_path("direct", seed)
    torch.save({
        "mode": "direct",
        "seed": int(seed),
        "tag": tag,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "predictor": predictor.state_dict(),
        "log": log,
    }, path)
    print(f"[direct s{seed}] {path}  ({(time.perf_counter() - t0) / 60:.1f} мин)")
    return path


def train_all(tag: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    for seed in TRAIN_SEEDS:
        train_jepa(seed, tag)
        train_direct(seed, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="сторож: пустой, v4 и v5-* запрещены")
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    train_all(args.tag)


if __name__ == "__main__":
    main()

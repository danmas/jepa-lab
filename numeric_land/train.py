"""Десять сетей посадки.

    python -m numeric_land.train --tag numeric-land

Пустой тег, v4, v5-*, numeric, numeric-dt, numeric-walls и numeric-scale
отклоняются до оптимизатора. Чужие чекпоинты на запись не открываются.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_land.data import TRAIN_SEEDS, VARIANTS, make_batch, scale_state, world_vector
from numeric_land.models import N_WORLD, BallEncoder, LandPredictor, jepa_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STEPS = 3000
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8
TAG = "numeric-land"


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "обучение посадки: нужен тег numeric-land. "
            "Пустой тег, v4, v5-*, numeric, numeric-dt, numeric-walls "
            "и numeric-scale запрещены.")


def ckpt_path(protocol: str, variant: str, seed: int, g=None) -> str:
    if protocol == "range":
        if variant not in VARIANTS or int(seed) not in TRAIN_SEEDS:
            raise ValueError((protocol, variant, seed))
        name = f"land_range_{variant}_s{int(seed)}.pt"
    elif protocol == "fixed":
        name = "land_fixed_g9_s0.pt"
    elif protocol == "tune":
        name = f"land_tune_g{int(g)}_s0.pt"
    elif protocol == "scratch":
        name = "land_scratch_g4_s0.pt"
    else:
        raise ValueError(protocol)
    return os.path.join(OUT, name)


def _save(path, payload):
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _loop(encoder, predictor, ema, opt, rng, protocol, g, label, freeze_encoder):
    log = []
    t0 = time.perf_counter()
    print(f"[{label}] steps={STEPS} batch={BATCH} lr={LR}", flush=True)
    for step in range(1, STEPS + 1):
        states, actions, gs = make_batch(rng, BATCH, protocol, g)
        states_t = torch.from_numpy(scale_state(states))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        world_t = None
        if predictor.n_world:
            world_t = torch.from_numpy(world_vector(gs))
        loss, parts = jepa_loss(
            encoder, predictor, ema, states_t, actions_t, world_t, freeze_encoder)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not freeze_encoder:
            ema.update(encoder)
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5),
                   **{k: round(v, 5) for k, v in parts.items()}}
            log.append(row)
            el = time.perf_counter() - t0
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} pen={row['penalty']:.4f} "
                  f"{el / step * 1000:.0f} ms/step", flush=True)
    return log, time.perf_counter() - t0


def _payload(protocol, variant, seed, tag, g, log, seconds, encoder, predictor, ema, frozen):
    return {
        "mode": "jepa",
        "protocol": protocol,
        "variant": variant,
        "seed": int(seed),
        "g": None if g is None else float(g),
        "tag": tag,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "n_world": N_WORLD[variant],
        "frozen_encoder": bool(frozen),
        "seconds": round(float(seconds), 3),
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "log": log,
    }


def train_joint(protocol: str, variant: str, seed: int, g, tag: str) -> str:
    refuse_tag(tag)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder = BallEncoder()
    predictor = LandPredictor(N_WORLD[variant])
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    label = f"{protocol} {variant} s{seed}" if g is None else f"{protocol} g{g:g} s{seed}"
    log, seconds = _loop(encoder, predictor, ema, opt, rng, protocol, g, label, False)
    path = ckpt_path(protocol, variant, seed, g)
    _save(path, _payload(
        protocol, variant, seed, tag, g, log, seconds, encoder, predictor, ema, False))
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def _load_fixed():
    path = ckpt_path("fixed", "vis", 0, 9)
    if not os.path.isfile(path):
        raise SystemExit(f"нет {os.path.basename(path)}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != TAG or ck.get("protocol") != "fixed":
        raise RuntimeError(path)
    return ck


def train_frozen(protocol: str, g: float, tag: str) -> str:
    """Кодировщик fixed заморожен. scratch получает новый предиктор."""
    refuse_tag(tag)
    ck = _load_fixed()
    encoder = BallEncoder()
    encoder.load_state_dict(ck["encoder"])
    encoder.requires_grad_(False)
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    ema.load_state_dict(ck["ema"])
    for p in ema.parameters():
        p.requires_grad_(False)
    predictor = LandPredictor(N_WORLD["vis"])
    if protocol == "tune":
        predictor.load_state_dict(ck["predictor"])
    elif protocol != "scratch":
        raise ValueError(protocol)
    opt = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(0)
    label = f"{protocol} g{g:g} s0"
    log, seconds = _loop(encoder, predictor, ema, opt, rng, "fixed", g, label, True)
    path = ckpt_path(protocol, "vis", 0, g)
    payload = _payload("vis", "vis", 0, tag, g, log, seconds, encoder, predictor, ema, True)
    payload["protocol"] = protocol
    payload["source"] = "land_fixed_g9_s0.pt"
    _save(path, payload)
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="range-vis-0, fixed, tune-4, tune-14, scratch")
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    only = args.only
    jobs = []
    if not only or only.startswith("range"):
        for variant in VARIANTS:
            for seed in TRAIN_SEEDS:
                key = f"range-{variant}-{seed}"
                if not only or only == key:
                    jobs.append(("joint", "range", variant, seed, None))
    if not only or only == "fixed":
        jobs.append(("joint", "fixed", "vis", 0, 9.0))
    if not only or only == "tune-4":
        jobs.append(("frozen", "tune", 4.0))
    if not only or only == "tune-14":
        jobs.append(("frozen", "tune", 14.0))
    if not only or only == "scratch":
        jobs.append(("frozen", "scratch", 4.0))
    if not jobs:
        raise SystemExit(f"неизвестный --only {only}")
    for job in jobs:
        if job[0] == "joint":
            _kind, protocol, variant, seed, g = job
            train_joint(protocol, variant, seed, g, args.tag)
        else:
            _kind, protocol, g = job
            train_frozen(protocol, g, args.tag)


if __name__ == "__main__":
    main()

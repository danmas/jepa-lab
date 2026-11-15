"""Шесть предикторов цели. Кодировщик поля 512 заморожен.

    python -m numeric_land.train_goal --tag numeric-land-goal

Чужие теги отклоняются до оптимизатора. land_*.pt, landf_*.pt и landp_*.pt на запись не открываются.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import EMATarget
from numeric_land.data import TRAIN_SEEDS, VARIANTS
from numeric_land.field import GROUND, scale_state
from numeric_land.goal import N_WORLD, make_batch, target_state, world_vector
from numeric_land.models import (
    BallEncoder, LandPredictor, action_pairs, collapse_penalty, predictor_input,
)
from numeric_land.train_field import ckpt_path as field_ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STEPS = 3000
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8
TAG = "numeric-land-goal"
SOURCE_NAME = "landf_range_vis_s0.pt"
W_TERM = 0.25          # слабый терминальный член
LOSS_WEIGHTS = (1.0, 0.5, 0.25)
COLLAPSE_WEIGHT = 0.1  # как в models.jepa_loss


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "обучение цели: нужен тег numeric-land-goal. "
            "numeric-land, numeric-land-field, numeric-land-pad, пустой тег, v4, v5-*, "
            "numeric, numeric-dt, numeric-walls и numeric-scale запрещены.")


def ckpt_path(variant: str, seed: int) -> str:
    if variant not in VARIANTS or int(seed) not in TRAIN_SEEDS:
        raise ValueError((variant, seed))
    name = f"landg_range_{variant}_s{int(seed)}.pt"
    return os.path.join(OUT, name)


def source_path() -> str:
    return field_ckpt_path("range", "vis", 0)


def _save(path, payload):
    name = os.path.basename(path)
    if not name.startswith("landg_"):
        raise RuntimeError(name)
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _same(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def _load_source():
    path = source_path()
    if not os.path.isfile(path) or os.path.basename(path) != SOURCE_NAME:
        raise SystemExit(f"нет {SOURCE_NAME}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != "numeric-land-field" or ck.get("variant") != "vis":
        raise RuntimeError(path)
    if int(ck.get("seed", -1)) != 0 or ck.get("protocol") != "range":
        raise RuntimeError(path)
    return ck


def goal_loss(encoder, predictor, ema, states, actions, world, pads, vys,
              height0, freeze_encoder=True):
    """JEPA-L1 по трём шагам плюс слабый терминальный член.

    states: (B, 4, 5) уже поделены. actions: (B, 3). world у hid — None.
    Терминал: латент конца горизонта ближе к латенту цели s* = (pad, GROUND−h0, 0, vy, fuel0).
    """
    b, t, dim = states.shape
    flat = encoder(states.reshape(b * t, dim)).detach()
    penalty = collapse_penalty(flat)
    z = flat.view(b, t, 32)[:, 0]
    pred_loss = states.new_zeros(())
    parts = {}
    world_step = world if predictor.n_world else None
    for k, weight in enumerate(LOSS_WEIGHTS):
        z = predictor(z, action_pairs(actions[:, k], z), world_step)
        with torch.no_grad():
            target = ema(states[:, k + 1])
        step = F.l1_loss(z, target)
        pred_loss = pred_loss + weight * step
        parts[f"l{k + 1}"] = float(step.detach())
    # терминальный член: ещё один шаг к целевому состоянию
    z_H = predictor(z, action_pairs(actions[:, -1], z), world_step)
    tgt_raw = np.concatenate([
        target_state(pads[i], vys[i], height0[i]) for i in range(b)], axis=0)
    with torch.no_grad():
        z_goal = ema(torch.from_numpy(scale_state(tgt_raw)))
    term = F.l1_loss(z_H, z_goal)
    total = pred_loss + COLLAPSE_WEIGHT * penalty + W_TERM * term
    parts["penalty"] = float(penalty.detach())
    parts["term"] = float(term.detach())
    return total, parts


def _loop(encoder, predictor, ema, opt, rng, label):
    log = []
    t0 = time.perf_counter()
    print(f"[{label}] steps={STEPS} batch={BATCH} lr={LR} w_term={W_TERM}", flush=True)
    for step in range(1, STEPS + 1):
        states, actions, gs, pads, vys = make_batch(rng, BATCH, "range")
        states_t = torch.from_numpy(scale_state(states))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        height0 = GROUND - states[:, 0, 1]   # высота старта окна
        world_t = None
        if predictor.n_world:
            world_t = torch.from_numpy(world_vector(gs, pads, vys))
        loss, parts = goal_loss(
            encoder, predictor, ema, states_t, actions_t, world_t, pads, vys,
            height0, freeze_encoder=True)
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
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} term={row.get('term', 0):.4f} "
                  f"pen={row['penalty']:.4f} {el / step * 1000:.0f} ms/step",
                  flush=True)
    return log, time.perf_counter() - t0


def train_one(variant: str, seed: int, tag: str) -> str:
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
    label = f"goal {variant} s{seed}"
    log, seconds = _loop(encoder, predictor, ema, opt, rng, label)
    if not _same(encoder.state_dict(), source["encoder"]):
        raise RuntimeError("кодировщик сдвинулся")
    if not _same(ema.state_dict(), source["ema"]):
        raise RuntimeError("EMA сдвинулась")
    path = ckpt_path(variant, seed)
    payload = {
        "mode": "jepa",
        "protocol": "range",
        "variant": variant,
        "seed": int(seed),
        "g": None,
        "tag": tag,
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "n_world": N_WORLD[variant],
        "goal_dim": 2,
        "w_term": W_TERM,
        "frozen_encoder": True,
        "source": SOURCE_NAME,
        "seconds": round(float(seconds), 3),
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "log": log,
    }
    _save(path, payload)
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="range-vis-0 … range-hid-2")
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    jobs = []
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            key = f"range-{variant}-{seed}"
            if not args.only or args.only == key:
                jobs.append((variant, seed))
    if not jobs:
        raise SystemExit(f"неизвестный --only {args.only}")
    for variant, seed in jobs:
        train_one(variant, seed, args.tag)


if __name__ == "__main__":
    main()
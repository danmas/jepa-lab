"""Навыки на замороженном landr_scratch_s0. Предиктор с новой инициализации.

    python -m numeric_land.train_skills --tag numeric-land-decision

Чужой тег отклоняется до пула и до оптимизатора. landf, landh и landm на запись не открываются.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_land.expert import make_batch_mixed
from numeric_land.field import GROUND
from numeric_land.models import LandPredictor
from numeric_land.relenc import (
    OUT, TAG, collect_pool, refuse_tag, require_soft, scale_rel,
    scan_range, tensors_equal, train_seeds,
)
from numeric_land.skills import N_WORLD, skill_loss, world_of
from numeric_land.train_goal import BATCH, EMA_MOMENTUM, LR, STEPS, THREADS, W_TERM, WEIGHT_DECAY
from numeric_land.train_relenc import ckpt_path as relenc_ckpt_path

EXPERT_FRAC = 0.5
SOURCE_NAME = "landr_scratch_s0.pt"


def refuse_foreign(tag: str) -> None:
    refuse_tag(tag)


def ckpt_path(phase: str, seed: int) -> str:
    seed = int(seed)
    if phase in ("approach", "land"):
        if seed not in (0, 1, 2):
            raise ValueError((phase, seed))
    elif phase == "shared":
        if seed != 0:
            raise ValueError(seed)
    else:
        raise ValueError(phase)
    return os.path.join(OUT, f"landk_{phase}_s{seed}.pt")


def _save(path, payload):
    name = os.path.basename(path)
    if not name.startswith("landk_"):
        raise RuntimeError(name)
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _load_source():
    path = relenc_ckpt_path("scratch", 0)
    if not os.path.isfile(path) or os.path.basename(path) != SOURCE_NAME:
        raise SystemExit(f"нет {SOURCE_NAME}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != TAG or ck.get("kind") != "scratch" or int(ck.get("seed", -1)) != 0:
        raise RuntimeError(path)
    if float(ck.get("w_probe")) != 0.1 or int(ck.get("n_in")) != 7:
        raise RuntimeError(path)
    return ck


def _frozen(source):
    from numeric_land.relenc import RelEncoder
    encoder = RelEncoder()
    encoder.load_state_dict(source["encoder"])
    encoder.requires_grad_(False)
    encoder.eval()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    ema.load_state_dict(source["ema"])
    for param in ema.parameters():
        param.requires_grad_(False)
    ema.eval()
    return encoder, ema


def _loop(encoder, predictor, ema, opt, rng, pool, label):
    log = []
    t0 = time.perf_counter()
    print(f"[{label}] steps={STEPS} batch={BATCH} expert_frac={EXPERT_FRAC} "
          f"pool={pool['states'].shape[0]} w_term={W_TERM}", flush=True)
    for step in range(1, STEPS + 1):
        states, actions, gs, pads, vys = make_batch_mixed(
            rng, BATCH, pool, expert_frac=EXPERT_FRAC)
        states_t = torch.from_numpy(np.ascontiguousarray(scale_rel(states, pads)))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        height0 = GROUND - states[:, 0, 1]
        world_t = torch.from_numpy(np.ascontiguousarray(world_of(gs, pads, vys)))
        loss, parts = skill_loss(
            encoder, predictor, ema, states_t, actions_t, world_t, pads, vys, height0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5),
                   **{key: round(value, 5) for key, value in parts.items()}}
            log.append(row)
            elapsed = time.perf_counter() - t0
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} term={row['term']:.4f} "
                  f"pen={row['penalty']:.4f} {elapsed / step * 1000:.0f} ms/step",
                  flush=True)
    return log, time.perf_counter() - t0


def train_one(phase: str, seed: int, tag: str, pool, source) -> str:
    refuse_tag(tag)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder, ema = _frozen(source)
    # Предиктор новый: ни из landr, ни из landh.
    predictor = LandPredictor(N_WORLD)
    if predictor.n_world != N_WORLD:
        raise RuntimeError(predictor.n_world)
    opt = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    label = f"{phase} s{seed}"
    log, seconds = _loop(encoder, predictor, ema, opt, rng, pool, label)
    if not tensors_equal(encoder.state_dict(), source["encoder"]):
        raise RuntimeError("кодировщик навыка сдвинулся")
    if not tensors_equal(ema.state_dict(), source["ema"]):
        raise RuntimeError("EMA навыка сдвинулась")
    path = ckpt_path(phase, seed)
    payload = {
        "tag": TAG,
        "phase": phase,
        "seed": int(seed),
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "expert_frac": EXPERT_FRAC,
        "w_term": W_TERM,
        "n_world": N_WORLD,
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


def jobs_of(only: str):
    jobs = []
    for phase in ("approach", "land"):
        for seed in (0, 1, 2):
            key = f"{phase}-{seed}"
            if not only or only == key:
                jobs.append((phase, seed))
    if not only or only == "shared-0":
        jobs.append(("shared", 0))
    if only and not jobs:
        raise SystemExit(f"неизвестный --only {only}")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    seed0, n = train_seeds()
    episodes = scan_range(seed0, n, "навыки")
    require_soft(episodes, 256, "навыки")
    pools = {
        "approach": collect_pool(episodes, "approach"),
        "land": collect_pool(episodes, "land"),
        "shared": collect_pool(episodes, "shared"),
    }
    for phase, pool in pools.items():
        print(f"пул {phase}: {pool['states'].shape[0]} окон", flush=True)
    source = _load_source()
    # Кодировщик читается, предиктор источника в инициализацию не копируется.
    del source["predictor"]
    for phase, seed in jobs_of(args.only):
        train_one(phase, seed, args.tag, pools[phase], source)


if __name__ == "__main__":
    main()

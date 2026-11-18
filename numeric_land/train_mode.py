"""Голова режима на метке правила с наведением. Кодировщик заморожен.

    python -m numeric_land.train_mode --tag numeric-land-mode

Чужие теги отклоняются до оптимизатора. Сети landf/landp/landg/landh на запись не открываются.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_land.mode import (
    KINDS, THRESHOLD, TRAIN_N, TRAIN_SEED0, TRAIN_SEEDS,
    ModeHead, class_weight, collect_split, encode_rows, features_of, n_in_of,
)
from numeric_land.models import BallEncoder
from numeric_land.train_field import ckpt_path as field_ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

STEPS = 1000
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8
TAG = "numeric-land-mode"
SOURCE_NAME = "landf_range_vis_s0.pt"
# Пакеты, которые этот тег не имеет права учить. Отказ — до оптимизатора.
_FORBIDDEN = (
    "", "numeric-land", "numeric-land-field", "numeric-land-pad",
    "numeric-land-goal", "numeric-land-hover",
)


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "обучение головы режима: нужен тег numeric-land-mode. "
            "Чужие теги, включая numeric-land-hover, запрещены.")


def ckpt_path(kind: str, seed: int) -> str:
    if kind not in KINDS:
        raise ValueError(kind)
    seed = int(seed)
    if kind == "zblind":
        if seed != 0:
            raise ValueError(seed)
    elif seed not in TRAIN_SEEDS:
        raise ValueError(seed)
    return os.path.join(OUT, f"landm_{kind}_s{seed}.pt")


def source_path() -> str:
    return field_ckpt_path("range", "vis", 0)


def _save(path, payload):
    name = os.path.basename(path)
    if not name.startswith("landm_"):
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


def _encoder_of(source):
    encoder = BallEncoder()
    encoder.load_state_dict(source["encoder"])
    encoder.requires_grad_(False)
    encoder.eval()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    ema.load_state_dict(source["ema"])
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return encoder, ema


def _loop(head, opt, features, labels, pos_weight, label):
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    feat = torch.from_numpy(np.asarray(features, np.float32))
    target = torch.from_numpy(np.asarray(labels, np.float32))
    # Один и тот же порядок батчей у всех сидов. Сид головы сюда не входит.
    rng = np.random.default_rng(0)
    n = len(target)
    log = []
    t0 = time.perf_counter()
    print(f"[{label}] steps={STEPS} batch={BATCH} pos_weight={pos_weight:.4f}", flush=True)
    for step in range(1, STEPS + 1):
        pick = rng.integers(0, n, size=BATCH)
        logit = head(feat[pick])
        loss = loss_fn(logit, target[pick])
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5)}
            log.append(row)
            el = time.perf_counter() - t0
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"{el / step * 1000:.0f} ms/step", flush=True)
    return log, time.perf_counter() - t0


def train_one(kind: str, seed: int, tag: str, split, z, pos_weight: float) -> str:
    refuse_tag(tag)
    source = _load_source()
    torch.manual_seed(int(seed))
    encoder, ema = _encoder_of(source)
    head = ModeHead(n_in_of(kind))
    # Дальше генератор torch не кормит выбор батча.
    torch.manual_seed(0)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    feats = features_of(kind, split["state"], split["pad_x"], z)
    if feats.shape[1] != n_in_of(kind):
        raise RuntimeError((kind, feats.shape))
    name = f"{kind} s{seed}"
    log, seconds = _loop(head, opt, feats, split["label"], pos_weight, name)
    if not _same(encoder.state_dict(), source["encoder"]):
        raise RuntimeError("кодировщик сдвинулся")
    if not _same(ema.state_dict(), source["ema"]):
        raise RuntimeError("EMA сдвинулась")
    path = ckpt_path(kind, seed)
    payload = {
        "tag": tag,
        "kind": kind,
        "seed": int(seed),
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "pos_weight": float(pos_weight),
        "n_in": n_in_of(kind),
        "frozen_encoder": True,
        "source": SOURCE_NAME,
        "seconds": round(float(seconds), 3),
        "encoder": encoder.state_dict(),
        "ema": ema.state_dict(),
        "head": head.state_dict(),
        "log": log,
    }
    _save(path, payload)
    print(f"[{name}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def jobs_of(only: str):
    jobs = []
    for kind in ("latent", "state"):
        for seed in TRAIN_SEEDS:
            key = f"{kind}-{seed}"
            if not only or only == key:
                jobs.append((kind, seed))
    if not only or only == "zblind-0":
        jobs.append(("zblind", 0))
    if only and not jobs:
        raise SystemExit(f"неизвестный --only {only}")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="latent-0 … state-2 или zblind-0")
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    print(f"сбор учителя {TRAIN_SEED0}…{TRAIN_SEED0 + TRAIN_N - 1}", flush=True)
    split = collect_split(TRAIN_SEED0, TRAIN_N)
    pos_weight = class_weight(split["label"])
    n_land = int(np.sum(split["label"] == 1))
    n_approach = int(np.sum(split["label"] == 0))
    print(f"строк {len(split['label'])}  наведение {n_approach}  посадка {n_land}  "
          f"вес {pos_weight:.4f}", flush=True)
    source = _load_source()
    encoder, _ema = _encoder_of(source)
    z = encode_rows(encoder, split["state"])
    for kind, seed in jobs_of(args.only):
        train_one(kind, seed, args.tag, split, z, pos_weight)


if __name__ == "__main__":
    main()

"""Голова решения на успешных посадках. Старая голова режима не читается.

    python -m numeric_land.train_decision --tag numeric-land-decision

Чужой тег отклоняется до пула и до оптимизатора.
"""
import argparse
import os
import time

import numpy as np
import torch

from numeric_land.decision import KINDS, THRESHOLD, DecisionHead, features_of, n_in_of
from numeric_land.mode import class_weight
from numeric_land.relenc import (
    OUT, TAG, MIN_HELD_SOFT, MIN_TRAIN_SOFT, encode_ball, encode_rel,
    file_stamp, frames_of, held_seeds, landf_path, refuse_tag, require_soft,
    scan_range, tensors_equal, train_seeds,
)
from numeric_land.train_relenc import ckpt_path as relenc_ckpt_path

STEPS = 1000
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
THREADS = 8
SOURCE_NEW = "landr_scratch_s0.pt"
SOURCE_OLD = "landf_range_vis_s0.pt"
_ALLOWED = {SOURCE_NEW, SOURCE_OLD}


def load_weights(path: str):
    """Чтение только двух разрешённых файлов. Старая голова сюда не проходит."""
    name = os.path.basename(path)
    if name not in _ALLOWED:
        raise RuntimeError(name)
    stamp = file_stamp(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if file_stamp(path) != stamp:
        raise RuntimeError(f"чтение изменило {name}")
    return ck, stamp


def ckpt_path(kind: str, seed: int) -> str:
    if kind not in KINDS:
        raise ValueError(kind)
    seed = int(seed)
    if kind == "latent":
        if seed not in (0, 1, 2):
            raise ValueError(seed)
    elif seed != 0:
        raise ValueError((kind, seed))
    return os.path.join(OUT, f"landd_{kind}_s{seed}.pt")


def _save(path, payload):
    name = os.path.basename(path)
    if not name.startswith("landd_"):
        raise RuntimeError(name)
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _loop(head, opt, features, labels, pos_weight, label):
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    feat = torch.from_numpy(np.asarray(features, np.float32))
    target = torch.from_numpy(np.asarray(labels, np.float32))
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
            elapsed = time.perf_counter() - t0
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"{elapsed / step * 1000:.0f} ms/step", flush=True)
    return log, time.perf_counter() - t0


def train_one(kind, seed, tag, split, z_new, z_old, pos_weight, sources) -> str:
    refuse_tag(tag)
    source_name = SOURCE_OLD if kind == "oldz" else SOURCE_NEW
    source, _stamp = sources[source_name]
    z = z_old if kind == "oldz" else z_new
    torch.manual_seed(int(seed))
    head = DecisionHead(n_in_of(kind))
    torch.manual_seed(0)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    feats = features_of(kind, split["pad_x"], split["g"], z)
    if feats.shape[1] != n_in_of(kind):
        raise RuntimeError((kind, feats.shape))
    label = f"{kind} s{seed}"
    log, seconds = _loop(head, opt, feats, split["label"], pos_weight, label)
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
        "source": source_name,
        "seconds": round(float(seconds), 3),
        "encoder": {key: value.detach().cpu().clone() for key, value in source["encoder"].items()},
        "ema": {key: value.detach().cpu().clone() for key, value in source["ema"].items()},
        "head": head.state_dict(),
        "log": log,
    }
    _save(path, payload)
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def jobs_of(only: str):
    jobs = []
    for seed in (0, 1, 2):
        key = f"latent-{seed}"
        if not only or only == key:
            jobs.append(("latent", seed))
    for kind in ("zblind", "g", "oldz"):
        key = f"{kind}-0"
        if not only or only == key:
            jobs.append((kind, 0))
    if only and not jobs:
        raise SystemExit(f"неизвестный --only {only}")
    return jobs


def _encoder_of(kind_name, state):
    if kind_name == SOURCE_OLD:
        from numeric_land.models import BallEncoder
        encoder = BallEncoder()
    else:
        from numeric_land.relenc import RelEncoder
        encoder = RelEncoder()
    encoder.load_state_dict(state)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    refuse_tag(args.tag)
    torch.set_num_threads(THREADS)
    seed0, n = train_seeds()
    episodes = scan_range(seed0, n, "модель решения, обучение")
    require_soft(episodes, MIN_TRAIN_SOFT, "модель решения, обучение")
    held0, held_n = held_seeds()
    held = scan_range(held0, held_n, "модель решения, отложенные")
    require_soft(held, MIN_HELD_SOFT, "модель решения, отложенные")
    del held
    split = frames_of(episodes, soft_only=True)
    pos_weight = class_weight(split["label"])
    n_land = int(np.sum(split["label"] == 1))
    n_approach = int(np.sum(split["label"] == 0))
    print(f"строк {len(split['label'])}  наведение {n_approach}  посадка {n_land}  "
          f"вес {pos_weight:.4f}", flush=True)
    scratch, stamp_new = load_weights(relenc_ckpt_path("scratch", 0))
    if scratch.get("tag") != TAG or scratch.get("kind") != "scratch":
        raise RuntimeError(SOURCE_NEW)
    old, stamp_old = load_weights(landf_path())
    if old.get("tag") != "numeric-land-field":
        raise RuntimeError(SOURCE_OLD)
    z_new = encode_rel(_encoder_of(SOURCE_NEW, scratch["encoder"]), split["state"], split["pad_x"])
    z_old = encode_ball(_encoder_of(SOURCE_OLD, old["encoder"]), split["state"])
    sources = {SOURCE_NEW: (scratch, stamp_new), SOURCE_OLD: (old, stamp_old)}
    for kind, seed in jobs_of(args.only):
        train_one(kind, seed, args.tag, split, z_new, z_old, pos_weight, sources)
    if file_stamp(relenc_ckpt_path("scratch", 0)) != stamp_new:
        raise RuntimeError("landr_scratch_s0 изменился")
    if file_stamp(landf_path()) != stamp_old:
        raise RuntimeError("landf изменился")
    # Повторное чтение: веса кодировщика на диске те же, что лежат в голове.
    scratch_again, _stamp = load_weights(relenc_ckpt_path("scratch", 0))
    if not tensors_equal(scratch_again["encoder"], scratch["encoder"]):
        raise RuntimeError("кодировщик после обучения не совпал с landr_scratch_s0")


if __name__ == "__main__":
    main()

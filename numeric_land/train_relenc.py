"""Семь каналов с нуля и одно дообучение первого слоя.

    python -m numeric_land.train_relenc --tag numeric-land-decision

Чужой тег отклоняется до пула и до оптимизатора. ``landr_scratch_*`` файл поля не читает.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_land.data import world_vector
from numeric_land.expert import make_batch_mixed
from numeric_land.models import LandPredictor
from numeric_land.relenc import (
    N_IN, OUT, TAG, W_PROBE, RelEncoder, RelProbe,
    collect_pool, expand_ema, expand_encoder, file_stamp, landf_path,
    probe_targets, refuse_tag, rel_loss, require_soft, scale_rel, scan_range,
    tensors_equal, train_seeds,
)

STEPS = 3000
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
THREADS = 8
EXPERT_FRAC = 0.5
N_WORLD = 2


def ckpt_path(kind: str, seed: int) -> str:
    seed = int(seed)
    if kind == "scratch":
        if seed not in (0, 1, 2):
            raise ValueError(seed)
        name = f"landr_scratch_s{seed}.pt"
    elif kind == "tune":
        if seed != 0:
            raise ValueError(seed)
        name = "landr_tune_s0.pt"
    else:
        raise ValueError(kind)
    return os.path.join(OUT, name)


def _save(path, payload):
    name = os.path.basename(path)
    if not name.startswith("landr_"):
        raise RuntimeError(name)
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _load_landf_readonly():
    """Только дообучение. Боевой scratch эту функцию не вызывает."""
    path = landf_path()
    stamp = file_stamp(path)
    if stamp is None or os.path.basename(path) != "landf_range_vis_s0.pt":
        raise SystemExit(f"нет {os.path.basename(path)}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if file_stamp(path) != stamp:
        raise RuntimeError("чтение landf изменило файл")
    if ck.get("tag") != "numeric-land-field" or ck.get("variant") != "vis":
        raise RuntimeError(path)
    if int(ck.get("seed", -1)) != 0 or ck.get("protocol") != "range":
        raise RuntimeError(path)
    if int(ck.get("n_world", -1)) != N_WORLD:
        raise RuntimeError(ck.get("n_world"))
    return ck, stamp


def _loop(encoder, predictor, probe, ema, opt, rng, pool, label):
    log = []
    t0 = time.perf_counter()
    print(f"[{label}] steps={STEPS} batch={BATCH} expert_frac={EXPERT_FRAC} "
          f"pool={pool['states'].shape[0]} w_probe={W_PROBE}", flush=True)
    for step in range(1, STEPS + 1):
        states, actions, gs, pads, _vys = make_batch_mixed(
            rng, BATCH, pool, expert_frac=EXPERT_FRAC)
        states_t = torch.from_numpy(np.ascontiguousarray(scale_rel(states, pads)))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        world_t = torch.from_numpy(np.ascontiguousarray(world_vector(gs)))
        targets_t = torch.from_numpy(np.ascontiguousarray(probe_targets(states, pads)))
        loss, parts = rel_loss(
            encoder, predictor, probe, ema, states_t, actions_t, world_t, targets_t)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(encoder)
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": round(float(loss.detach()), 5),
                   **{key: round(value, 5) for key, value in parts.items()}}
            log.append(row)
            elapsed = time.perf_counter() - t0
            print(f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} probe={row['probe']:.4f} "
                  f"pen={row['penalty']:.4f} {elapsed / step * 1000:.0f} ms/step",
                  flush=True)
    return log, time.perf_counter() - t0


def _payload(kind, seed, log, seconds, encoder, predictor, ema, probe, source):
    return {
        "tag": TAG,
        "kind": kind,
        "seed": int(seed),
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "expert_frac": EXPERT_FRAC,
        "w_probe": W_PROBE,
        "n_in": N_IN,
        "n_world": N_WORLD,
        "frozen_encoder": False,
        "source": source,
        "seconds": round(float(seconds), 3),
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "probe": probe.state_dict(),
        "log": log,
    }


def train_scratch(seed: int, tag: str, pool) -> str:
    """С нуля. Чужой чекпоинт не читается."""
    refuse_tag(tag)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder = RelEncoder()
    predictor = LandPredictor(N_WORLD)
    probe = RelProbe()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()) + list(probe.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    label = f"scratch s{seed}"
    log, seconds = _loop(encoder, predictor, probe, ema, opt, rng, pool, label)
    path = ckpt_path("scratch", seed)
    _save(path, _payload("scratch", seed, log, seconds, encoder, predictor, ema, probe, None))
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def train_tune(tag: str, pool) -> str:
    """Дообучение: первый слой расширен, два новых столбца — нули. В зачёт не входит."""
    refuse_tag(tag)
    source, stamp = _load_landf_readonly()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    encoder = RelEncoder()
    expand_encoder(encoder, source["encoder"])
    predictor = LandPredictor(N_WORLD)
    predictor.load_state_dict({key: value.detach().clone() for key, value in source["predictor"].items()})
    probe = RelProbe()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    expand_ema(ema, source["ema"])
    if file_stamp(landf_path()) != stamp:
        raise RuntimeError("дообучение задело landf")
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()) + list(probe.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    label = "tune s0"
    log, seconds = _loop(encoder, predictor, probe, ema, opt, rng, pool, label)
    if file_stamp(landf_path()) != stamp:
        raise RuntimeError("дообучение задело landf")
    # Исходные столбцы после обучения уже не нули: это дообучение, не копия scratch.
    path = ckpt_path("tune", 0)
    payload = _payload("tune", 0, log, seconds, encoder, predictor, ema, probe, "landf_range_vis_s0.pt")
    payload["landf_mtime_ns"] = stamp[0]
    payload["landf_size"] = stamp[1]
    _save(path, payload)
    if not tensors_equal(encoder.state_dict(), payload["encoder"]):
        raise RuntimeError("снимок tune не совпал")
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин)", flush=True)
    return path


def jobs_of(only: str):
    jobs = []
    for seed in (0, 1, 2):
        key = f"scratch-{seed}"
        if not only or only == key:
            jobs.append(("scratch", seed))
    if not only or only == "tune-0":
        jobs.append(("tune", 0))
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
    stamp = file_stamp(landf_path())
    if stamp is None:
        raise SystemExit("нет landf_range_vis_s0.pt")
    seed0, n = train_seeds()
    episodes = scan_range(seed0, n, "обучение кодировщика")
    require_soft(episodes, 256, "обучение кодировщика")
    pool = collect_pool(episodes, "encoder")
    print(f"экспертный пул {pool['states'].shape[0]} окон, включая стык", flush=True)
    for kind, seed in jobs_of(args.only):
        if kind == "scratch":
            train_scratch(seed, args.tag, pool)
        else:
            train_tune(args.tag, pool)
    if file_stamp(landf_path()) != stamp:
        raise RuntimeError("landf изменился")


if __name__ == "__main__":
    main()

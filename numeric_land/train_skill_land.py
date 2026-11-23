"""Навык посадки над площадкой на замороженном landr_scratch_s0.

    python -m numeric_land.train_skill_land --tag numeric-land-skill-land

Чужой тег отклоняется до пула и до оптимизатора.
Пишет только out/landk_pad_s0.pt, out/landk_pad_s1.pt и отчёт JSON.
landk_land_s0.pt и landr_scratch_s0.pt открываются на чтение.
"""
import argparse
import os
import time

import numpy as np
import torch

from models import EMATarget
from numeric_land.expert import make_batch_mixed
from numeric_land.models import LandPredictor
from numeric_land.relenc import (
    MIN_TRAIN_SOFT, N_IN, OUT, POOL_CAP, W_PROBE, RelEncoder, collect_pool,
    file_stamp, require_soft, scale_rel, scan_range, tensors_equal, train_seeds,
    write_json,
)
from numeric_land.relenc import TAG as ENCODER_TAG
from numeric_land.skill_land import (
    EXPERT_FRAC, GOAL_FUEL_TRAIN, GOAL_HEIGHT0, GOAL_VY, GRID_N, HORIZON, LINE,
    N_SEQ, TAG, fly_grid, line_ok, terminal_state, train_height_and_vy,
)
from numeric_land.skills import N_WORLD, skill_loss, world_of
from numeric_land.train_goal import (
    BATCH, COLLAPSE_WEIGHT, EMA_MOMENTUM, LOSS_WEIGHTS, LR, STEPS, THREADS,
    W_TERM, WEIGHT_DECAY,
)
from numeric_land.train_relenc import ckpt_path as relenc_ckpt_path
from numeric_land.train_skills import ckpt_path as land_ckpt_path

SOURCE_NAME = "landr_scratch_s0.pt"
BASELINE_NAME = "landk_land_s0.pt"
JSON_NAME = "numeric_land_skill_land.json"
PAD_NAMES = ("landk_pad_s0.pt", "landk_pad_s1.pt")


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "нужен тег numeric-land-skill-land. Другие теги запрещены.")


def scratch_path() -> str:
    path = relenc_ckpt_path("scratch", 0)
    if os.path.basename(path) != SOURCE_NAME:
        raise RuntimeError(path)
    return path


def baseline_path() -> str:
    path = land_ckpt_path("land", 0)
    if os.path.basename(path) != BASELINE_NAME:
        raise RuntimeError(path)
    return path


def pad_path(seed: int) -> str:
    seed = int(seed)
    if seed not in (0, 1):
        raise ValueError(seed)
    name = f"landk_pad_s{seed}.pt"
    if name not in PAD_NAMES:
        raise RuntimeError(name)
    return os.path.join(OUT, name)


def json_path() -> str:
    return os.path.join(OUT, JSON_NAME)


def _save(path, payload) -> None:
    name = os.path.basename(path)
    if name not in PAD_NAMES:
        raise RuntimeError(name)
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _assert_schedule() -> None:
    if TAG != "numeric-land-skill-land" or ENCODER_TAG == TAG:
        raise SystemExit(TAG)
    if STEPS != 3000 or BATCH != 128 or THREADS != 8:
        raise SystemExit("расписание обучения сдвинуто")
    if LR != 1e-3 or WEIGHT_DECAY != 1e-4 or EMA_MOMENTUM != 0.99:
        raise SystemExit("оптимизатор сдвинут")
    if W_TERM != 0.25 or COLLAPSE_WEIGHT != 0.1 or LOSS_WEIGHTS != (1.0, 0.5, 0.25):
        raise SystemExit("терминал или штраф сдвинуты")
    if EXPERT_FRAC != 1.0 or int(round(BATCH * EXPERT_FRAC)) != BATCH:
        raise SystemExit("в батч попали бы случайные окна")
    if MIN_TRAIN_SOFT != 256 or POOL_CAP != 20_000:
        raise SystemExit("фильтр пула сдвинут")
    if N_IN != 7 or W_PROBE != 0.1 or int(N_WORLD) != 4:
        raise SystemExit("кодировщик или канал мира")
    if HORIZON != 3 or N_SEQ != 729 or LINE != 114 or GRID_N != 162:
        raise SystemExit("поиск или линия сдвинуты")
    if GOAL_VY != 6.0 or GOAL_HEIGHT0 != 0.0 or GOAL_FUEL_TRAIN != 10.0:
        raise SystemExit("терминал сдвинут")


def _load_scratch(path):
    if not os.path.isfile(path) or os.path.basename(path) != SOURCE_NAME:
        raise SystemExit(f"нет {SOURCE_NAME}")
    stamp = file_stamp(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if file_stamp(path) != stamp:
        raise RuntimeError("чтение кодировщика изменило файл")
    if ck.get("tag") != ENCODER_TAG or ck.get("kind") != "scratch" or int(ck.get("seed", -1)) != 0:
        raise RuntimeError(path)
    if float(ck.get("w_probe")) != 0.1 or int(ck.get("n_in")) != 7:
        raise RuntimeError(path)
    return ck


def _load_baseline(path, source):
    if not os.path.isfile(path) or os.path.basename(path) != BASELINE_NAME:
        raise SystemExit(f"нет {BASELINE_NAME}")
    stamp = file_stamp(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if file_stamp(path) != stamp:
        raise RuntimeError("чтение базы изменило файл")
    if ck.get("tag") != ENCODER_TAG or ck.get("phase") != "land" or int(ck.get("seed", -1)) != 0:
        raise RuntimeError(path)
    if int(ck.get("n_world", -1)) != int(N_WORLD):
        raise RuntimeError(ck.get("n_world"))
    if not tensors_equal(ck["encoder"], source["encoder"]):
        raise RuntimeError("кодировщик базы не совпал с landr_scratch_s0")
    if not tensors_equal(ck["ema"], source["ema"]):
        raise RuntimeError("EMA базы не совпала с landr_scratch_s0")
    return ck


def _frozen(source):
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


def _predictor_from(state):
    predictor = LandPredictor(N_WORLD)
    predictor.load_state_dict(state)
    predictor.requires_grad_(False)
    predictor.eval()
    return predictor


def _loop(encoder, predictor, ema, opt, rng, pool, label):
    log = []
    t0 = time.perf_counter()
    print(
        f"[{label}] steps={STEPS} batch={BATCH} expert_frac={EXPERT_FRAC} "
        f"pool={pool['states'].shape[0]} w_term={W_TERM} goal_vy={GOAL_VY} "
        f"height0={GOAL_HEIGHT0}",
        flush=True,
    )
    for step in range(1, STEPS + 1):
        states, actions, gs, pads, _pool_vys = make_batch_mixed(
            rng, BATCH, pool, expert_frac=EXPERT_FRAC)
        height0, vys = train_height_and_vy(BATCH)
        if height0.shape != (BATCH,) or not np.all(height0 == 0.0) or not np.all(vys == GOAL_VY):
            raise RuntimeError("высота окна или vy пула попали в терминал")
        if step == 1:
            sample = terminal_state(float(pads[0]), GOAL_FUEL_TRAIN)
            if float(sample[0, 1]) != 512.0 or float(sample[0, 3]) != GOAL_VY:
                raise RuntimeError(sample)
            if float(sample[0, 4]) != GOAL_FUEL_TRAIN or float(sample[0, 2]) != 0.0:
                raise RuntimeError(sample)
        states_t = torch.from_numpy(np.ascontiguousarray(scale_rel(states, pads)))
        actions_t = torch.from_numpy(actions.astype(np.int64))
        world_np = world_of(gs, pads, vys)
        if not np.allclose(world_np[:, 3], GOAL_VY / 120.0):
            raise RuntimeError(world_np[0])
        world_t = torch.from_numpy(np.ascontiguousarray(world_np))
        loss, parts = skill_loss(
            encoder, predictor, ema, states_t, actions_t, world_t, pads, vys, height0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "loss": round(float(loss.detach()), 5),
                **{key: round(value, 5) for key, value in parts.items()},
            }
            log.append(row)
            elapsed = time.perf_counter() - t0
            print(
                f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                f"l1={row['l1']:.4f} term={row['term']:.4f} "
                f"pen={row['penalty']:.4f} {elapsed / step * 1000:.0f} ms/step",
                flush=True,
            )
    return log, time.perf_counter() - t0


def train_one(seed, source, baseline_state, pool):
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    encoder, ema = _frozen(source)
    predictor = LandPredictor(N_WORLD)
    if predictor.n_world != int(N_WORLD):
        raise RuntimeError(predictor.n_world)
    if tensors_equal(predictor.state_dict(), baseline_state):
        raise RuntimeError("инициализация скопирована из landk_land")
    opt = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    label = f"pad s{seed}"
    log, seconds = _loop(encoder, predictor, ema, opt, rng, pool, label)
    if not log:
        raise RuntimeError(label)
    if not tensors_equal(encoder.state_dict(), source["encoder"]):
        raise RuntimeError("кодировщик навыка сдвинулся")
    if not tensors_equal(ema.state_dict(), source["ema"]):
        raise RuntimeError("EMA навыка сдвинулась")
    if tensors_equal(predictor.state_dict(), baseline_state):
        raise RuntimeError("предиктор совпал с landk_land")
    path = pad_path(seed)
    payload = {
        "tag": TAG,
        "phase": "pad",
        "seed": int(seed),
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "ema_momentum": EMA_MOMENTUM,
        "expert_frac": EXPERT_FRAC,
        "w_term": W_TERM,
        "n_world": int(N_WORLD),
        "frozen_encoder": True,
        "source": SOURCE_NAME,
        "goal_vy": GOAL_VY,
        "goal_height0": GOAL_HEIGHT0,
        "goal_fuel_train": GOAL_FUEL_TRAIN,
        "horizon": HORIZON,
        "seconds": round(float(seconds), 3),
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "log": log,
    }
    _save(path, payload)
    print(f"[{label}] {path}  ({seconds / 60:.1f} мин) loss={log[-1]['loss']}", flush=True)
    return path, encoder, ema, predictor, log, seconds


def _watched():
    names = (
        SOURCE_NAME,
        BASELINE_NAME,
        "landk_approach_s0.pt",
        "landd_latent_s0.pt",
        "landd_early_s0.pt",
        "landm_latent_s0.pt",
        "landr_scratch_s1.pt",
        "landr_scratch_s2.pt",
    )
    paths = {name: os.path.join(OUT, name) for name in names}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths["expert_hover.py"] = os.path.join(root, "numeric_land", "expert_hover.py")
    return {key: file_stamp(path) for key, path in paths.items()}


def _row(name, flown, scored):
    row = {
        "name": name,
        "scored": bool(scored),
        "soft": int(flown["soft"]),
        "crash": int(flown["crash"]),
        "miss": int(flown["miss"]),
        "of": int(GRID_N),
        "cells": flown["cells"],
    }
    if scored:
        row["line_ok"] = bool(line_ok(row["soft"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    refuse_tag(args.tag)
    _assert_schedule()
    torch.set_num_threads(THREADS)
    report = json_path()
    if os.path.isfile(report):
        raise SystemExit(f"уже есть {report}, второй прогон отчёт не затирает")
    before = _watched()
    if before[SOURCE_NAME] is None or before[BASELINE_NAME] is None:
        raise SystemExit("нет кодировщика или базы до обучения")
    if before["expert_hover.py"] is None:
        raise SystemExit("нет expert_hover.py")
    source = _load_scratch(scratch_path())
    baseline = _load_baseline(baseline_path(), source)
    baseline_state = baseline["predictor"]
    if "predictor" in source:
        del source["predictor"]
    if _watched() != before:
        raise RuntimeError("чтение сдвинуло файл")
    _probe_encoder, _probe_ema = _frozen(source)
    _probe_pred = LandPredictor(N_WORLD)
    del _probe_encoder, _probe_ema, _probe_pred
    seed0, n = train_seeds()
    if int(seed0) != 98000 or int(n) != 512:
        raise SystemExit((seed0, n))
    episodes = scan_range(seed0, n, "посадка над площадкой")
    n_soft = require_soft(episodes, MIN_TRAIN_SOFT, "посадка над площадкой")
    pool = collect_pool(episodes, "land", POOL_CAP)
    pool_n = int(pool["states"].shape[0])
    print(f"пул land: {pool_n} окон, мягких эпизодов {n_soft}", flush=True)
    if pool_n < 1 or pool_n > POOL_CAP:
        raise RuntimeError(pool_n)
    _pad0, encoder, ema, predictor, log0, _sec0 = train_one(0, source, baseline_state, pool)
    _pad1, _enc1, _ema1, pred1, log1, _sec1 = train_one(1, source, baseline_state, pool)
    del pred1
    if _watched() != before:
        raise RuntimeError("обучение сдвинуло чужой файл")
    print(f"полёт landk_pad_s0, {GRID_N} стартов", flush=True)
    pad_flown = fly_grid("landk_pad_s0", predictor, encoder, ema)
    base_pred = _predictor_from(baseline_state)
    print(f"полёт landk_land_s0, {GRID_N} стартов", flush=True)
    base_flown = fly_grid("landk_land_s0", base_pred, encoder, ema)
    if _watched() != before:
        raise RuntimeError("полёт сдвинул чужой файл")
    pad_row = _row("landk_pad_s0", pad_flown, True)
    pad_row["final_loss"] = float(log0[-1]["loss"])
    base_row = _row("landk_land_s0", base_flown, False)
    if "line_ok" in base_row:
        raise RuntimeError("база закрыла линию")
    if pad_row["soft"] + pad_row["crash"] + pad_row["miss"] != GRID_N:
        raise RuntimeError(pad_row)
    if bool(pad_row["line_ok"]) != bool(line_ok(pad_row["soft"])):
        raise RuntimeError("линия разошлась")
    payload = {
        "tag": TAG,
        "horizon": HORIZON,
        "n_sequences": N_SEQ,
        "cem": False,
        "horizon_40": False,
        "goal_vy": GOAL_VY,
        "goal_height0": GOAL_HEIGHT0,
        "goal_fuel_train": GOAL_FUEL_TRAIN,
        "goal_fuel_loop": "current",
        "expert_frac": EXPERT_FRAC,
        "grid_n": GRID_N,
        "line": LINE,
        "decision_loaded": False,
        "approach_loaded": False,
        "hover_action": False,
        "uses_start_and_actions": False,
        "encoder": SOURCE_NAME,
        "pool_windows": pool_n,
        "pool_cap": int(POOL_CAP),
        "soft_episodes": int(n_soft),
        "train_seeds": [int(seed0), int(seed0) + int(n) - 1],
        "seed1_trained": True,
        "seed1_flown": False,
        "seed1_final_loss": float(log1[-1]["loss"]),
        "passed": bool(pad_row["line_ok"]),
        "rows": [pad_row, base_row],
        "stamps_unchanged": True,
    }
    write_json(report, payload)
    print(
        f"итог landk_pad_s0 {pad_row['soft']}/{GRID_N} линия {LINE} "
        f"{'пройдена' if payload['passed'] else 'не пройдена'}; "
        f"landk_land_s0 {base_row['soft']}/{GRID_N} в линию не входит",
        flush=True,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

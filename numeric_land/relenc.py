"""Относительный кодировщик: семь каналов и probe только на время обучения.

Боевой кодировщик учится с нуля. ``BallEncoder()`` по-прежнему принимает пять чисел.
Седьмой канал равен ``1 − y/512`` и высоту сам почти не добавляет: это делает probe.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numeric_land.data import g_of_window
from numeric_land.expert_hover import hover_episode, hover_mode
from numeric_land.field import GROUND, SCALE_XY, TRANSITIONS, scale_state
from numeric_land.goal import goal_of_window, outcome_at
from numeric_land.mode import (
    HELD_N, HELD_SEED0, TRAIN_N, TRAIN_SEED0,
)
from numeric_land.models import (
    COLLAPSE_WEIGHT, HIDDEN, LATENT, LOSS_WEIGHTS, action_pairs, collapse_penalty,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

TAG = "numeric-land-decision"
W_PROBE = 0.1
PROBE_RATIO = 0.9
POOL_CAP = 20_000
MIN_TRAIN_SOFT = 256
MIN_HELD_SOFT = 64
N_IN = 7
LANDF_NAME = "landf_range_vis_s0.pt"
# Отличие от landf — два канала и probe вместе. Вес 0.1 в одиночку причиной не называется.
PACKAGE_SENTENCE = (
    "Отличие боевого кодировщика от landf — два добавленных канала и probe вместе: "
    "старый кодировщик площадку при обучении не видел. Вес 0.1 не единственная причина разрыва."
)


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "нужен тег numeric-land-decision. "
            "numeric-land-field, numeric-land-hover, numeric-land-mode, "
            "пустой тег и прочие запрещены.")


def landf_path() -> str:
    return os.path.join(OUT, LANDF_NAME)


def file_stamp(path: str):
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return (int(st.st_mtime_ns), int(st.st_size))


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    import json
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class RelEncoder(nn.Module):
    """Семь каналов, скрытый слой 64, латент 32, единичная норма."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, LATENT),
        )

    def forward(self, state):
        return F.normalize(self.net(state), dim=-1)


class RelProbe(nn.Module):
    """Линейный probe 32 → 2. В инференсе, в голове и в контуре не вызывается."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, 2)

    def forward(self, z):
        return self.fc(z)


def scale_rel(states, pad_x) -> np.ndarray:
    """Пять чисел ``scale_state`` плюс знаковый ``(x − pad_x)/512`` и ``1 − y/512``.

    Седьмой канал собран из уже посчитанного ``y/512``, поэтому он равен ``1 − y/512``
    в той же точности. Это не новая высота сверх ``y``.
    """
    base = scale_state(states)
    raw = np.asarray(states, np.float64)
    x = raw[..., 0]
    pad = np.asarray(pad_x, np.float64)
    while pad.ndim < x.ndim:
        pad = np.expand_dims(pad, axis=-1)
    signed = ((x - pad) / float(SCALE_XY)).astype(np.float32)
    height = (np.float32(1.0) - base[..., 1]).astype(np.float32)
    return np.concatenate(
        [base, signed[..., None], height[..., None]], axis=-1).astype(np.float32)


def probe_targets(states, pad_x) -> np.ndarray:
    """Две цели: модуль знакового отношения и та же высота, что седьмой канал."""
    rel = scale_rel(states, pad_x)
    dist = np.abs(rel[..., 5])
    height = rel[..., 6]
    return np.stack([dist, height], axis=-1).astype(np.float32)


def _copy_linear(dst_weight, dst_bias, src_weight, src_bias, n_old: int) -> None:
    if src_weight.shape[1] != n_old or dst_weight.shape[1] <= n_old:
        raise RuntimeError((src_weight.shape, dst_weight.shape))
    dst_weight.zero_()
    dst_weight[:, :n_old] = src_weight.detach().to(dst_weight.dtype).clone()
    dst_bias.copy_(src_bias.detach().to(dst_bias.dtype).clone())


def expand_encoder(encoder: RelEncoder, old_state: dict) -> None:
    """Первый слой 5 → 7. Старые столбцы и смещение копируются, два новых — нули."""
    fresh = encoder.state_dict()
    _copy_linear(
        fresh["net.0.weight"], fresh["net.0.bias"],
        old_state["net.0.weight"], old_state["net.0.bias"], 5)
    for key, value in old_state.items():
        if key.startswith("net.0."):
            continue
        fresh[key] = value.detach().clone()
    encoder.load_state_dict(fresh)


def expand_ema(ema, old_state: dict) -> None:
    """То же расширение для EMA. Копия онлайн-кодировщика сюда не подставляется."""
    fresh = ema.state_dict()
    _copy_linear(
        fresh["module.net.0.weight"], fresh["module.net.0.bias"],
        old_state["module.net.0.weight"], old_state["module.net.0.bias"], 5)
    for key, value in old_state.items():
        if key in ("module.net.0.weight", "module.net.0.bias"):
            continue
        fresh[key] = value.detach().clone()
    ema.load_state_dict(fresh)


def rel_loss(encoder, predictor, probe, ema, states, actions, world, targets):
    """JEPA без терминала плюс средний L1 probe по двум целям и четырём кадрам.

    Градиент probe идёт в онлайн-кодировщик. В EMA и в предиктор он не отвечает:
    предиктор probe не читает, EMA обновляется копией весов.
    """
    b, t, dim = states.shape
    flat = encoder(states.reshape(b * t, dim))
    penalty = collapse_penalty(flat)
    z = flat.view(b, t, LATENT)[:, 0]
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
    pred_probe = probe(flat)
    probe_l1 = F.l1_loss(pred_probe, targets.reshape(b * t, 2))
    total = pred_loss + COLLAPSE_WEIGHT * penalty + W_PROBE * probe_l1
    parts["penalty"] = float(penalty.detach())
    parts["probe"] = float(probe_l1.detach())
    return total, parts


def probe_target_flag(scratch_l1: float, landf_l1: float) -> bool:
    """Истина, только если новый L1 не выше 0.9 от старого и старый L1 больше нуля."""
    if float(landf_l1) <= 0.0:
        return False
    return float(scratch_l1) <= PROBE_RATIO * float(landf_l1)


def probe_pass(flag_distance: bool, flag_height: bool) -> bool:
    return bool(flag_distance) and bool(flag_height)


def l1_ratio(new: float, old: float):
    if old is None or float(old) <= 0.0:
        return None
    return float(new) / float(old)


def is_soft_touch(grounded: bool, x, y, vy, pad_x) -> bool:
    if not grounded:
        return False
    return outcome_at(x, y, vy, True, pad_x) == "мягкая"


def bucket_of(modes) -> str | None:
    if any(mode is None for mode in modes):
        return None
    if all(mode == "approach" for mode in modes):
        return "approach"
    if all(mode == "land" for mode in modes):
        return "land"
    return "mixed"


def pools_of(modes) -> frozenset:
    bucket = bucket_of(modes)
    if bucket is None:
        return frozenset()
    if bucket == "approach":
        return frozenset({"approach", "shared"})
    if bucket == "land":
        return frozenset({"land", "shared"})
    return frozenset({"shared"})


def accepted(bucket, pool_name: str) -> bool:
    if bucket is None:
        return False
    if pool_name in ("encoder", "shared"):
        return True
    if pool_name == "approach":
        return bucket == "approach"
    if pool_name == "land":
        return bucket == "land"
    raise ValueError(pool_name)


def _mode_of_row(row, g, pad_x, v_target):
    state = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
    return hover_mode(state, g, pad_x, v_target)


def scan_range(seed0: int, n: int, label: str):
    """Эпизоды правила с наведением. Успех — ``outcome_at`` == «мягкая» на касании."""
    episodes = []
    n_soft = 0
    print(f"сбор {label}: сиды {seed0}…{seed0 + n - 1}", flush=True)
    for i, seed in enumerate(range(int(seed0), int(seed0) + int(n))):
        g = float(g_of_window(seed))
        pad_x, v_target = goal_of_window(seed)
        states, actions, grounded = hover_episode(seed, g, pad_x, v_target)
        soft = False
        if grounded and len(states):
            soft = is_soft_touch(
                True, states[-1, 0], states[-1, 1], states[-1, 3], pad_x)
        if soft:
            n_soft += 1
        episodes.append({
            "seed": int(seed),
            "g": float(g),
            "pad_x": float(pad_x),
            "v_target": float(v_target),
            "states": states,
            "actions": actions,
            "grounded": bool(grounded),
            "soft": bool(soft),
        })
        if (i + 1) % 64 == 0 or i + 1 == n:
            print(f"  {label} {i + 1}/{n}  мягких {n_soft}", flush=True)
    return episodes


def count_soft(episodes) -> int:
    return int(sum(1 for ep in episodes if ep["soft"]))


def require_soft(episodes, minimum: int, label: str) -> int:
    n_soft = count_soft(episodes)
    if n_soft < int(minimum):
        raise SystemExit(
            f"{label}: мягких {n_soft}, нужно не меньше {minimum}. Фильтр не ослабляется.")
    return n_soft


def iter_windows(episodes, soft_only: bool = True):
    """Окна по сиду и шагу. Пустой режим выбрасывает окно. Стык остаётся ``mixed``."""
    for ep in episodes:
        if soft_only and not ep["soft"]:
            continue
        states = ep["states"]
        actions = ep["actions"]
        n_act = int(actions.shape[0])
        if n_act < TRANSITIONS:
            continue
        for start in range(n_act - TRANSITIONS + 1):
            chunk = states[start:start + TRANSITIONS + 1]
            modes = []
            dropped = False
            for row in chunk:
                mode = _mode_of_row(row, ep["g"], ep["pad_x"], ep["v_target"])
                if mode is None:
                    dropped = True
                    break
                modes.append(mode)
            if dropped:
                continue
            yield {
                "states": np.asarray(chunk, np.float64),
                "actions": np.asarray(actions[start:start + TRANSITIONS], np.int64),
                "g": float(ep["g"]),
                "pad_x": float(ep["pad_x"]),
                "v_target": float(ep["v_target"]),
                "bucket": bucket_of(modes),
                "seed": int(ep["seed"]),
                "step": int(start),
            }


def pack_windows(rows) -> dict:
    if not rows:
        raise RuntimeError("пул пуст")
    return {
        "states": np.asarray([row["states"] for row in rows], np.float64),
        "actions": np.asarray([row["actions"] for row in rows], np.int64),
        "gs": np.asarray([row["g"] for row in rows], np.float64),
        "pads": np.asarray([row["pad_x"] for row in rows], np.float64),
        "vys": np.asarray([row["v_target"] for row in rows], np.float64),
    }


def collect_pool(episodes, pool_name: str, cap: int = POOL_CAP) -> dict:
    rows = []
    for window in iter_windows(episodes, soft_only=True):
        if not accepted(window["bucket"], pool_name):
            continue
        rows.append(window)
        if len(rows) >= int(cap):
            break
    pool = pack_windows(rows)
    pool["pool"] = pool_name
    pool["cap"] = int(cap)
    return pool


def frames_of(episodes, soft_only: bool) -> dict:
    """Кадры с непустым режимом. Пустой режим в строки не входит."""
    states, pads, labels, seeds, steps, heights, dxs, gs = [], [], [], [], [], [], [], []
    for ep in episodes:
        if soft_only and not ep["soft"]:
            continue
        for step_i, row in enumerate(ep["states"]):
            mode = _mode_of_row(row, ep["g"], ep["pad_x"], ep["v_target"])
            if mode is None:
                continue
            state = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
            states.append(state)
            pads.append(float(ep["pad_x"]))
            labels.append(1 if mode == "land" else 0)
            seeds.append(int(ep["seed"]))
            steps.append(int(step_i))
            heights.append(float(GROUND - state[1]))
            dxs.append(abs(float(ep["pad_x"]) - state[0]))
            gs.append(float(ep["g"]))
    if not states:
        raise RuntimeError("пустой набор кадров")
    return {
        "state": np.asarray(states, np.float64),
        "pad_x": np.asarray(pads, np.float64),
        "label": np.asarray(labels, np.int64),
        "seed": np.asarray(seeds, np.int64),
        "step_i": np.asarray(steps, np.int64),
        "height": np.asarray(heights, np.float64),
        "dx": np.asarray(dxs, np.float64),
        "g": np.asarray(gs, np.float64),
    }


def encode_matrix(encoder, scaled: np.ndarray, batch: int = 4096) -> np.ndarray:
    pieces = []
    tensor = torch.from_numpy(np.ascontiguousarray(scaled.astype(np.float32)))
    with torch.no_grad():
        for start in range(0, len(tensor), batch):
            pieces.append(encoder(tensor[start:start + batch]).cpu())
    return torch.cat(pieces, dim=0).numpy().astype(np.float32)


def encode_rel(encoder, states: np.ndarray, pads: np.ndarray, batch: int = 4096) -> np.ndarray:
    return encode_matrix(encoder, scale_rel(states, pads), batch)


def encode_ball(encoder, states: np.ndarray, batch: int = 4096) -> np.ndarray:
    return encode_matrix(encoder, scale_state(states), batch)


def tensors_equal(left, right) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def train_seeds():
    return TRAIN_SEED0, TRAIN_N


def held_seeds():
    return HELD_SEED0, HELD_N

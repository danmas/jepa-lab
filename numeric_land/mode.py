"""Голова режима. Учитель — ветка правила с наведением, не физика шага.

Боевой вход — латент или пять чисел состояния, плюс pad_x/512.
Тяжесть и desired_vy в голову не входят: в условии учителя их нет.
"""
import numpy as np
import torch
import torch.nn as nn

from numeric_land.data import g_of_window
from numeric_land.expert_hover import HOVER_CAP, hover_episode, hover_mode
from numeric_land.field import GROUND, SCALE_XY, scale_state
from numeric_land.goal import goal_of_window

TRAIN_SEED0 = 98000
TRAIN_N = 512
HELD_SEED0 = 98800
HELD_N = 128
LOOP_SEED0 = 99000
LOOP_N = 30
THRESHOLD = 0.5
V_TARGET_LOOP = 6.0
PASS_G = (2.0, 8.0, 15.0)
PASS_PADS = (256.0, 200.0)
EXTRA_G = 20.0
EXTRA_PAD = 406.0
HEAD_HIDDEN = 32
KINDS = ("latent", "state", "zblind")
TRAIN_SEEDS = (0, 1, 2)


class ModeHead(nn.Module):
    """Один скрытый слой. Выход — логит класса «посадка»."""

    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = int(n_in)
        self.net = nn.Sequential(
            nn.Linear(self.n_in, HEAD_HIDDEN),
            nn.SiLU(),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def n_in_of(kind: str) -> int:
    if kind == "latent":
        return 33
    if kind == "state":
        return 6
    if kind == "zblind":
        return 32
    raise ValueError(kind)


def _rows_of_episode(seed: int):
    """Строки одного эпизода учителя. Пустой режим в список не входит."""
    g = float(g_of_window(int(seed)))
    pad_x, v_target = goal_of_window(int(seed))
    states, _actions, _grounded = hover_episode(int(seed), g, pad_x, v_target, HOVER_CAP)
    kept = []
    for step_i, row in enumerate(states):
        state = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        mode = hover_mode(state, g, pad_x, v_target)
        if mode is None:
            continue
        kept.append({
            "state": state,
            "pad_x": float(pad_x),
            "label": 1 if mode == "land" else 0,
            "seed": int(seed),
            "step_i": int(step_i),
            "height": float(GROUND - state[1]),
            "dx": abs(float(pad_x) - state[0]),
            "g": g,
        })
    return kept


def collect_split(seed0: int, n: int):
    """Эпизоды подряд. Тяжесть и цель — из сида окна, не из отдельного генератора."""
    rows = []
    for seed in range(int(seed0), int(seed0) + int(n)):
        rows.extend(_rows_of_episode(seed))
    if not rows:
        raise RuntimeError("пустой сбор")
    states = np.asarray([r["state"] for r in rows], np.float64)
    return {
        "state": states,
        "pad_x": np.asarray([r["pad_x"] for r in rows], np.float64),
        "label": np.asarray([r["label"] for r in rows], np.int64),
        "seed": np.asarray([r["seed"] for r in rows], np.int64),
        "step_i": np.asarray([r["step_i"] for r in rows], np.int64),
        "height": np.asarray([r["height"] for r in rows], np.float64),
        "dx": np.asarray([r["dx"] for r in rows], np.float64),
        "g": np.asarray([r["g"] for r in rows], np.float64),
    }


def class_weight(labels: np.ndarray) -> float:
    """Число наведений / число посадок. Считается только по обучающим строкам."""
    n_land = int(np.sum(labels == 1))
    n_approach = int(np.sum(labels == 0))
    if n_land <= 0 or n_approach <= 0:
        raise RuntimeError(f"пустой класс: approach={n_approach} land={n_land}")
    return float(n_approach) / float(n_land)


def pad_channel(pads: np.ndarray) -> np.ndarray:
    return (np.asarray(pads, np.float32) / np.float32(SCALE_XY)).reshape(-1, 1)


def feat_state(states: np.ndarray, pads: np.ndarray) -> np.ndarray:
    scaled = scale_state(states)
    return np.concatenate([scaled, pad_channel(pads)], axis=1).astype(np.float32)


def feat_latent(z: np.ndarray, pads: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(z, np.float32), pad_channel(pads)], axis=1)


def feat_zblind(z: np.ndarray) -> np.ndarray:
    return np.asarray(z, np.float32)


def features_of(kind: str, states: np.ndarray, pads: np.ndarray, z: np.ndarray) -> np.ndarray:
    if kind == "state":
        return feat_state(states, pads)
    if kind == "latent":
        return feat_latent(z, pads)
    if kind == "zblind":
        return feat_zblind(z)
    raise ValueError(kind)


def encode_rows(encoder, states: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Замороженный кодировщик. Градиент не считается."""
    scaled = torch.from_numpy(scale_state(states))
    pieces = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch):
            pieces.append(encoder(scaled[start:start + batch]).cpu())
    return torch.cat(pieces, dim=0).numpy().astype(np.float32)

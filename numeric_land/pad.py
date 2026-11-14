"""Центр площадки — параметр окна. Физику считает step_field, площадку он не видит.

Учебный центр равномерный на [156, 356). Старт ракеты за площадкой не следует.
"""
import numpy as np

from numeric_land.data import g_of_window, world_vector as world_g
from numeric_land.field import (
    PROBE_STEPS, SCALE_XY, VY_SOFT,
    start_and_actions, step_field, window_of,
)


def physics_step(x, y, vx, vy, fuel, grounded, action, g):
    """Тот же шаг, что у поля 512. Центра площадки в аргументах нет."""
    return step_field(x, y, vx, vy, fuel, grounded, action, g)

# Полуширина прежняя. Имя без числа 8 в поле field, чтобы не путать с порогом скорости.
PAD_HALF = 8.0
TRAIN_PAD_LO = 156.0
TRAIN_PAD_HI = 356.0
PROBE_SEED0 = 91000
PROBE_EPISODES = 1024
ROLLOUT_SEED0 = 93000
ROLLOUT_N = 64
PLAN_SEED0 = 94000
PLAN_N = 30
OPEN_PADS = (256.0, 200.0, 356.0, 106.0, 406.0)
PLAN_PADS = (256.0, 200.0, 406.0)
RANGE_G = (2.0, 8.0, 20.0)
N_WORLD = {"vis": 3, "hid": 0}


def pad_of_window(window_seed: int) -> float:
    """Центр из [156, 356). Поток старта и поток тяжести это не ест."""
    rng = np.random.default_rng(np.random.SeedSequence([int(window_seed), 19]))
    return float(rng.uniform(TRAIN_PAD_LO, TRAIN_PAD_HI))


def outcome_at(x, y, vy, grounded, pad_x) -> str:
    """Касание внутри [pad_x − 8, pad_x + 8] и vy < 8. Порог включает равенство как удар."""
    del y
    if not grounded:
        return "не сел"
    lo = float(pad_x) - PAD_HALF
    hi = float(pad_x) + PAD_HALF
    if lo <= float(x) <= hi and float(vy) < VY_SOFT:
        return "мягкая"
    return "удар"


def world_vector(gs, pads) -> np.ndarray:
    """(g/9, 1, pad_x/512). Первые два канала — прежняя пара мира."""
    base = world_g(gs)
    pad = np.asarray(pads, np.float32) / np.float32(SCALE_XY)
    pad = np.reshape(pad, base.shape[:-1] + (1,))
    return np.concatenate([base, pad], axis=-1).astype(np.float32)


def make_batch(rng, batch: int = 128, protocol: str = "range", g=None):
    """Окна поля 512 плюс свой центр на окно. range берёт g из сида окна."""
    if protocol == "fixed":
        if g is None:
            raise ValueError("fixed требует g")
    elif protocol != "range":
        raise ValueError(protocol)
    from numeric_land.field import TRANSITIONS

    states = np.empty((batch, TRANSITIONS + 1, 5), np.float64)
    actions = np.empty((batch, TRANSITIONS), np.int64)
    gs = np.empty(batch, np.float64)
    pads = np.empty(batch, np.float64)
    for i in range(batch):
        window_seed = int(rng.integers(100_000, 2**31 - 1))
        gi = g_of_window(window_seed) if protocol == "range" else float(g)
        packed, acts = window_of(window_seed, gi)
        states[i] = packed
        actions[i] = acts
        gs[i] = gi
        pads[i] = pad_of_window(window_seed)
    return states, actions, gs, pads


def probe_states():
    """1024 эпизода, 4 состояния. Центр в состояние не входит."""
    rows = []
    for seed in range(PROBE_SEED0, PROBE_SEED0 + PROBE_EPISODES):
        packed, _acts = window_of(seed, g_of_window(seed), PROBE_STEPS)
        rows.append(packed)
    out = np.concatenate(rows, axis=0)
    if len(out) != PROBE_EPISODES * (PROBE_STEPS + 1):
        raise RuntimeError(len(out))
    return out


def same_start(seed: int):
    """Старт ракеты. От центра площадки и от g не зависит."""
    return start_and_actions(int(seed), 0)

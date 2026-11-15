"""Цель посадки как вход предиктора. Физику считает step_field, цель он не видит.

Цель окна: (pad_x, desired_vy). pad_x равномерно на [156, 356), desired_vy из {0, 3, 6}.
Кодировщик и EMA заморожены из out/landf_range_vis_s0.pt.
"""
import numpy as np

from numeric_land.data import g_of_window, world_vector as world_g
from numeric_land.field import (
    FUEL0, GROUND, PROBE_STEPS, SCALE_VEL, SCALE_XY,
    start_and_actions, step_field, window_of,
)

# Тройка мира поля 512 плюс канал цели — вместе четвёрка для видимого предиктора.
PAD_HALF = 8.0
TRAIN_PAD_LO = 156.0
TRAIN_PAD_HI = 356.0
DESIRED_VY = (0.0, 3.0, 6.0)
PROBE_SEED0 = 95000
PROBE_EPISODES = 1024
ROLLOUT_SEED0 = 96500
ROLLOUT_N = 64
PLAN_SEED0 = 97000
PLAN_N = 30
OPEN_PADS = (256.0, 200.0, 356.0, 106.0, 406.0)
PLAN_PADS = (256.0, 200.0, 406.0)
OPEN_VY = (0.0, 3.0, 6.0)
PLAN_VY = (0.0, 3.0, 6.0)
RANGE_G = (2.0, 8.0, 20.0)
# vis получает четвёрку (g/9, 1, pad_x/512, desired_vy/120); hid — ничего.
N_WORLD = {"vis": 4, "hid": 0}
GOAL_DIM = 2  # длина вектора цели (pad_x, desired_vy)


def goal_of_window(window_seed: int):
    """Цель окна. Ни генератор старта, ни поток тяжести это не едят."""
    rng_pad = np.random.default_rng(np.random.SeedSequence([int(window_seed), 19]))
    pad_x = float(rng_pad.uniform(TRAIN_PAD_LO, TRAIN_PAD_HI))
    rng_vy = np.random.default_rng(np.random.SeedSequence([int(window_seed), 23]))
    desired_vy = float(DESIRED_VY[int(rng_vy.integers(0, len(DESIRED_VY)))])
    return pad_x, desired_vy


def world_vector(gs, pad_xs, desired_vys) -> np.ndarray:
    """Четвёрка (g/9, 1, pad_x/512, desired_vy/120) для видимого предиктора."""
    base = world_g(gs)
    pad = np.asarray(pad_xs, np.float32) / np.float32(SCALE_XY)
    vy = np.asarray(desired_vys, np.float32) / np.float32(SCALE_VEL)
    pad = np.reshape(pad, base.shape[:-1] + (1,))
    vy = np.reshape(vy, base.shape[:-1] + (1,))
    return np.concatenate([base, pad, vy], axis=-1).astype(np.float32)


def target_state(pad_x, desired_vy, height0, fuel0=FUEL0):
    """Целевое состояние под терминальный лосс: над площадкой, желаемая vy вниз.

    y растёт вниз: над землёй на height0 высота y = GROUND - height0.
    desired_vy — величина скорости вниз (положительное число = движение к земле).
    """
    x = float(pad_x)
    y = float(GROUND) - float(height0)
    vx = 0.0
    vy = float(desired_vy)
    return np.array([[x, y, vx, vy, float(fuel0)]], np.float64)


def make_batch(rng, batch: int = 128, protocol: str = "range", g=None, pad=None):
    """Окна поля 512 плюс свой центр и желаемая скорость на окно."""
    if protocol == "fixed":
        if g is None or pad is None:
            raise ValueError("fixed требует g и pad")
    elif protocol != "range":
        raise ValueError(protocol)
    from numeric_land.field import TRANSITIONS

    states = np.empty((batch, TRANSITIONS + 1, 5), np.float64)
    actions = np.empty((batch, TRANSITIONS), np.int64)
    gs = np.empty(batch, np.float64)
    pads = np.empty(batch, np.float64)
    vys = np.empty(batch, np.float64)
    for i in range(batch):
        window_seed = int(rng.integers(100_000, 2**31 - 1))
        if protocol == "range":
            gi = g_of_window(window_seed)
            pad_i, vy_i = goal_of_window(window_seed)
        else:
            gi, pad_i, vy_i = float(g), float(pad), float(DESIRED_VY[0])
        packed, acts = window_of(window_seed, gi)
        states[i] = packed
        actions[i] = acts
        gs[i] = gi
        pads[i] = pad_i
        vys[i] = vy_i
    return states, actions, gs, pads, vys


def probe_states():
    """1024 эпизода, 4 состояния. Цель в состояние не входит."""
    rows = []
    for seed in range(PROBE_SEED0, PROBE_SEED0 + PROBE_EPISODES):
        packed, _acts = window_of(seed, g_of_window(seed), PROBE_STEPS)
        rows.append(packed)
    out = np.concatenate(rows, axis=0)
    if len(out) != PROBE_EPISODES * (PROBE_STEPS + 1):
        raise RuntimeError(len(out))
    return out


def same_start(seed: int):
    """Старт ракеты. От цели и от g не зависит."""
    return start_and_actions(int(seed), 0)


def physics_step(x, y, vx, vy, fuel, grounded, action, g):
    """Тот же шаг, что у поля 512. Цели в аргументах нет."""
    return step_field(x, y, vx, vy, fuel, grounded, action, g)


def outcome_at(x, y, vy, grounded, pad_x) -> str:
    """Касание внутри [pad_x − 8, pad_x + 8] и vy < 8."""
    del y
    if not grounded:
        return "не сел"
    lo = float(pad_x) - PAD_HALF
    hi = float(pad_x) + PAD_HALF
    if lo <= float(x) <= hi and float(vy) < 8.0:
        return "мягкая"
    return "удар"
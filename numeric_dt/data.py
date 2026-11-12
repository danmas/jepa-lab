"""Интегратор со скоростью в пикселях в секунду. step_state не вызывается.

Шаг: v ← v + (a + g)·h, затем x ← x + v·h. Множителя 0.6 нет.
"""
import math

import numpy as np

from data import N_ACTIONS, RADIUS, SIZE

ACCEL = 240.0
GRAVITY_ACCEL = 9.0
DT_REF = 1.0 / 30.0
DT_A = DT_REF
DT_B_LO = 1.0 / 40.0
DT_B_HI = 1.0 / 24.0
SCALE_XY = float(SIZE)
SCALE_VEL = 120.0
SCALE_FRAME = 4.0
SCALE_DT = DT_REF
HORIZON = 6
TRAIN_SEEDS = (0, 1, 2)
PROBE_SEED0 = 20000
PROBE_EPISODES = 1024
PROBE_STEPS = 3
PAGE_DT = {"1/60": 1.0 / 60.0, "1/30": 1.0 / 30.0, "1/15": 1.0 / 15.0}
# Индекс (dy+1)*3+(dx+1), пары (dx, dy) из {-1, 0, +1}.
PAIRS = np.array(
    [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.float64)
LO = float(RADIUS)
HI = float(SIZE) - 1.0 - float(RADIUS)


def action_accel(action: int) -> np.ndarray:
    dx, dy = PAIRS[int(action)]
    return np.array([dx * ACCEL, dy * ACCEL], np.float64)


def substep_count(vel, a_vec, dt: float) -> int:
    speed = float(np.hypot(vel[0], vel[1]))
    a_mag = float(np.hypot(a_vec[0], a_vec[1]))
    bound = (speed + (a_mag + GRAVITY_ACCEL) * float(dt)) * float(dt)
    return max(1, math.ceil(bound / 1.0))


def _reflect(pos, vel):
    for ax in range(2):
        if pos[ax] < LO:
            pos[ax] = LO + (LO - pos[ax])
            vel[ax] *= -1.0
        if pos[ax] > HI:
            pos[ax] = HI - (pos[ax] - HI)
            vel[ax] *= -1.0
    return pos, vel


def step_physics(pos, vel, action: int, dt: float):
    """Один полный шаг. Внутри — подшаги не длиннее 1 px той же формулы."""
    pos = np.array(pos, np.float64, copy=True)
    vel = np.array(vel, np.float64, copy=True)
    a_vec = action_accel(int(action))
    g_vec = np.array([0.0, GRAVITY_ACCEL], np.float64)
    n = substep_count(vel, a_vec, dt)
    h = float(dt) / n
    for _ in range(n):
        vel = vel + (a_vec + g_vec) * h
        pos = pos + vel * h
        pos, vel = _reflect(pos, vel)
    return pos, vel, n


def start_pair(rng: np.random.Generator):
    pos = rng.uniform(LO + 2.0, HI - 2.0, size=2).astype(np.float64)
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    spd = float(rng.uniform(20.0, 60.0))
    vel = np.array([np.cos(ang) * spd, np.sin(ang) * spd], np.float64)
    return pos, vel


def dt_of_window(window_seed: int, arm: str) -> float:
    if arm == "A":
        return DT_A
    if arm != "B":
        raise ValueError(arm)
    side = np.random.default_rng(np.random.SeedSequence([int(window_seed), 11]))
    return float(side.uniform(DT_B_LO, DT_B_HI))


def follow(pos, vel, actions, dt: float) -> np.ndarray:
    """(T, 4) физические x, y, vx, vy. T = len(actions)+1. dt на всех шагах один."""
    states = np.empty((len(actions) + 1, 4), np.float64)
    pos = np.array(pos, np.float64)
    vel = np.array(vel, np.float64)
    states[0, 0:2] = pos
    states[0, 2:4] = vel
    for i, action in enumerate(actions):
        pos, vel, _n = step_physics(pos, vel, int(action), dt)
        states[i + 1, 0:2] = pos
        states[i + 1, 2:4] = vel
    return states


def pack_state(phys: np.ndarray, dt: float, variant: str) -> np.ndarray:
    if variant == "with-dt":
        out = np.empty((len(phys), 5), np.float32)
        out[:, 0:4] = phys
        out[:, 4] = np.float32(dt)
        return out
    if variant == "no-dt":
        out = np.empty((len(phys), 4), np.float32)
        out[:, 0:2] = phys[:, 0:2]
        out[:, 2:4] = phys[:, 2:4] * np.float32(dt)
        return out
    raise ValueError(variant)


def episode_window(window_seed: int, arm: str, steps: int, variant: str):
    rng = np.random.default_rng(int(window_seed))
    pos, vel = start_pair(rng)
    actions = rng.integers(0, N_ACTIONS, size=int(steps)).astype(np.int64)
    dt = dt_of_window(window_seed, arm)
    phys = follow(pos, vel, actions, dt)
    return pack_state(phys, dt, variant), actions, dt


def make_batch(rng: np.random.Generator, arm: str, variant: str,
               batch: int = 128, horizon: int = HORIZON):
    dim = 5 if variant == "with-dt" else 4
    states = np.empty((batch, horizon, dim), np.float32)
    actions = np.empty((batch, horizon - 1), np.int64)
    dts = np.empty(batch, np.float64)
    for i in range(batch):
        window_seed = int(rng.integers(100_000, 2**31 - 1))
        packed, acts, dt = episode_window(window_seed, arm, horizon - 1, variant)
        states[i] = packed
        actions[i] = acts
        dts[i] = dt
    return states, actions, dts


def probe_states(arm: str = "B", variant: str = "with-dt") -> np.ndarray:
    rows = []
    for seed in range(PROBE_SEED0, PROBE_SEED0 + PROBE_EPISODES):
        packed, _acts, _dt = episode_window(seed, arm, PROBE_STEPS, variant)
        rows.append(packed)
    return np.concatenate(rows, axis=0)


def page_start(seed: int):
    return start_pair(np.random.default_rng(int(seed)))


def scale_state(states: np.ndarray, variant: str) -> np.ndarray:
    out = np.array(states, np.float32, copy=True)
    out[..., 0:2] /= np.float32(SCALE_XY)
    if variant == "with-dt":
        out[..., 2:4] /= np.float32(SCALE_VEL)
        out[..., 4] /= np.float32(SCALE_DT)
    elif variant == "no-dt":
        out[..., 2:4] /= np.float32(SCALE_FRAME)
    else:
        raise ValueError(variant)
    return out

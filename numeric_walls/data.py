"""Интегратор numeric_dt со стенками в аргументе. dt всегда 1/30.

Шаг: v ← v + (a + g)·h, затем x ← x + v·h. step_state не вызывается.
"""
import math

import numpy as np

from numeric_dt.data import ACCEL, DT_REF, GRAVITY_ACCEL, PAIRS, SCALE_VEL, SCALE_XY

LO = 5.0
HI = 58.0
CENTER = (LO + HI) / 2.0
DT = DT_REF
SPAN_LO = 16.0
SPAN_HI = 24.0
SPAN_FIXED = 20.0
SPAN_TESTS = (12.0, 20.0, 32.0)
SPAN_CF = (16.0, 24.0)
WORLDS = ("W", "H")
PROTOCOLS = ("range", "fixed")
VARIANTS = ("vis", "hid")
HORIZON = 6
TRAIN_SEEDS = (0, 1, 2)
ROLLOUT_SEED0 = 50000
ROLLOUT_N = 64
PROBE_SEED0 = 51000
PROBE_EPISODES = 1024
PROBE_STEPS = 3
TRANSFER = {12.0: 53000, 20.0: 54000, 32.0: 55000}
TRANSFER_N = 256
HOLD = 2
N_HOLD_ACTIONS = 15


def walls_of(world: str, span: float):
    span = float(span)
    half = span / 2.0
    if world == "W":
        return (CENTER - half, CENTER + half, LO, HI)
    if world == "H":
        return (LO, HI, CENTER - half, CENTER + half)
    raise ValueError(world)


def span_of_window(window_seed: int, protocol: str) -> float:
    if protocol == "fixed":
        return SPAN_FIXED
    if protocol != "range":
        raise ValueError(protocol)
    rng = np.random.default_rng(np.random.SeedSequence([int(window_seed), 11]))
    return float(rng.uniform(SPAN_LO, SPAN_HI))


def _reflect(pos, vel, walls):
    bounds = ((walls[0], walls[1]), (walls[2], walls[3]))
    for ax in range(2):
        lo, hi = bounds[ax]
        if pos[ax] < lo:
            pos[ax] = lo + (lo - pos[ax])
            vel[ax] *= -1.0
        if pos[ax] > hi:
            pos[ax] = hi - (pos[ax] - hi)
            vel[ax] *= -1.0
    return pos, vel


def step_physics(pos, vel, action: int, walls, dt: float = DT):
    """Один полный шаг. Подшаги не длиннее 1 px. dt по умолчанию 1/30."""
    pos = np.array(pos, np.float64, copy=True)
    vel = np.array(vel, np.float64, copy=True)
    dx, dy = PAIRS[int(action)]
    a_vec = np.array([dx * ACCEL, dy * ACCEL], np.float64)
    g_vec = np.array([0.0, GRAVITY_ACCEL], np.float64)
    speed = float(np.hypot(vel[0], vel[1]))
    a_mag = float(np.hypot(a_vec[0], a_vec[1]))
    bound = (speed + (a_mag + GRAVITY_ACCEL) * float(dt)) * float(dt)
    n = max(1, math.ceil(bound / 1.0))
    h = float(dt) / n
    for _ in range(n):
        vel = vel + (a_vec + g_vec) * h
        pos = pos + vel * h
        pos, vel = _reflect(pos, vel, walls)
    return pos, vel, n


def _draw_start(rng, walls, world: str):
    left, right, top, bottom = walls
    if float(rng.random()) < 0.5:
        pos = np.array([
            rng.uniform(left + 2.0, right - 2.0),
            rng.uniform(top + 2.0, bottom - 2.0),
        ], np.float64)
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        spd = float(rng.uniform(20.0, 60.0))
        vel = np.array([np.cos(ang) * spd, np.sin(ang) * spd], np.float64)
        return pos, vel
    depth = float(rng.uniform(0.5, 3.0))
    spd = float(rng.uniform(30.0, 60.0))
    side = float(rng.uniform(-20.0, 20.0))
    if world == "W":
        y = float(rng.uniform(top + 2.0, bottom - 2.0))
        if float(rng.random()) < 0.5:
            return np.array([left + depth, y]), np.array([-spd, side])
        return np.array([right - depth, y]), np.array([spd, side])
    x = float(rng.uniform(left + 2.0, right - 2.0))
    if float(rng.random()) < 0.5:
        return np.array([x, top + depth]), np.array([side, -spd])
    return np.array([x, bottom - depth]), np.array([side, spd])


def follow(pos, vel, actions, walls) -> np.ndarray:
    states = np.empty((len(actions) + 1, 4), np.float64)
    pos = np.array(pos, np.float64, copy=True)
    vel = np.array(vel, np.float64, copy=True)
    states[0, 0:2] = pos
    states[0, 2:4] = vel
    for i, action in enumerate(actions):
        pos, vel, _n = step_physics(pos, vel, int(action), walls)
        states[i + 1, 0:2] = pos
        states[i + 1, 2:4] = vel
    return states


def episode_window(window_seed: int, world: str, protocol: str, steps: int):
    span = span_of_window(window_seed, protocol)
    walls = walls_of(world, span)
    rng = np.random.default_rng(int(window_seed))
    pos, vel = _draw_start(rng, walls, world)
    actions = rng.integers(0, 9, size=int(steps)).astype(np.int64)
    phys = follow(pos, vel, actions, walls)
    packed = np.empty((len(phys), 5), np.float64)
    packed[:, 0:4] = phys
    packed[:, 4] = DT
    return packed, actions, np.asarray(walls, np.float64), float(span)


def make_batch(rng, world: str, protocol: str, batch: int = 128, horizon: int = HORIZON):
    states = np.empty((batch, horizon, 5), np.float32)
    actions = np.empty((batch, horizon - 1), np.int64)
    walls = np.empty((batch, 4), np.float32)
    spans = np.empty(batch, np.float64)
    for i in range(batch):
        window_seed = int(rng.integers(100_000, 2**31 - 1))
        packed, acts, wall, span = episode_window(
            window_seed, world, protocol, horizon - 1)
        states[i] = packed
        actions[i] = acts
        walls[i] = wall
        spans[i] = span
    return states, actions, walls, spans


def probe_states(world: str, protocol: str) -> np.ndarray:
    rows = []
    for seed in range(PROBE_SEED0, PROBE_SEED0 + PROBE_EPISODES):
        packed, _acts, _wall, _span = episode_window(seed, world, protocol, PROBE_STEPS)
        rows.append(packed)
    return np.concatenate(rows, axis=0)


def transfer_states(world: str, protocol: str, span: float) -> np.ndarray:
    """Те же монета и действия, размер задан аргументом и поток сида не сдвигает."""
    rows = []
    seed0 = TRANSFER[float(span)]
    for seed in range(seed0, seed0 + TRANSFER_N):
        walls = walls_of(world, span)
        rng = np.random.default_rng(int(seed))
        pos, vel = _draw_start(rng, walls, world)
        actions = rng.integers(0, 9, size=PROBE_STEPS).astype(np.int64)
        phys = follow(pos, vel, actions, walls)
        packed = np.empty((len(phys), 5), np.float64)
        packed[:, 0:4] = phys
        packed[:, 4] = DT
        rows.append(packed)
    return np.concatenate(rows, axis=0)


def demo_start(seed: int, world: str, span: float):
    """Старт страницы: та же монета и тот же ящик, что у окна обучения.

    Размер — аргумент и поток сида не сдвигает.
    """
    walls = walls_of(world, span)
    rng = np.random.default_rng(int(seed))
    pos, vel = _draw_start(rng, walls, world)
    return pos, vel, np.asarray(walls, np.float64)


def rollout_episode(seed: int, world: str, span: float):
    walls = walls_of(world, span)
    rng = np.random.default_rng(int(seed))
    pos, vel = _draw_start(rng, walls, world)
    actions = rng.integers(0, 9, size=N_HOLD_ACTIONS).astype(np.int64)
    held = np.repeat(actions, HOLD)
    phys = follow(pos, vel, held, walls)
    packed = np.empty((len(phys), 5), np.float64)
    packed[:, 0:4] = phys
    packed[:, 4] = DT
    return packed, held, np.asarray(walls, np.float64)


def scale_state(states: np.ndarray) -> np.ndarray:
    out = np.array(states, np.float32, copy=True)
    out[..., 0:2] /= np.float32(SCALE_XY)
    out[..., 2:4] /= np.float32(SCALE_VEL)
    out[..., 4] /= np.float32(DT)
    return out


def scale_walls(walls: np.ndarray) -> np.ndarray:
    return np.array(walls, np.float32, copy=True) / np.float32(SCALE_XY)

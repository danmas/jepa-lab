"""Интегратор в коробке [0, W] × [0, H]. dt всегда 1/30.

Шаг: v ← v + (a + g)·h, затем x ← x + v·h. Силы от размера не зависят.
step_state не вызывается.
"""
import math

import numpy as np

from numeric_dt.data import ACCEL, DT_REF, GRAVITY_ACCEL, PAIRS, SCALE_VEL, SCALE_XY

DT = DT_REF
W_LO = 16.0
W_HI = 32.0
MARGIN = 2.0
REPRS = ("rel", "abs")
VARIANTS = ("vis", "hid")
HORIZON = 6
TRAIN_SEEDS = (0, 1, 2)
ROLLOUT_SEED0 = 60000
ROLLOUT_N = 64
RECT_SEED0 = 66000
PROBE_SEED0 = 61000
PROBE_EPISODES = 1024
PROBE_STEPS = 3
TRANSFER_N = 256
HOLD = 2
N_HOLD_ACTIONS = 15
SQUARES = (12.0, 24.0, 48.0)
RECT = (20.0, 28.0)
TRANSFER = {
    "12": (12.0, 12.0, 63000),
    "24": (24.0, 24.0, 64000),
    "48": (48.0, 48.0, 65000),
    "20x28": (20.0, 28.0, 67000),
}
CF_WIDTHS = (16.0, 32.0)
CF_H = 32.0


def size_of_window(window_seed: int):
    """W, затем H, оба из [16, 32). Поток старта это не потребляет."""
    rng = np.random.default_rng(np.random.SeedSequence([int(window_seed), 11]))
    width = float(rng.uniform(W_LO, W_HI))
    height = float(rng.uniform(W_LO, W_HI))
    return width, height


def _reflect(pos, vel, walls):
    hit = False
    bounds = ((walls[0], walls[1]), (walls[2], walls[3]))
    for ax in range(2):
        lo, hi = bounds[ax]
        if pos[ax] < lo:
            pos[ax] = lo + (lo - pos[ax])
            vel[ax] *= -1.0
            hit = True
        elif pos[ax] > hi:
            pos[ax] = hi - (pos[ax] - hi)
            vel[ax] *= -1.0
            hit = True
    return pos, vel, hit


def step_physics(pos, vel, action: int, walls, dt: float = DT):
    """Один полный шаг. Подшаги не длиннее 1 px. Возвращает pos, vel, n, был ли отскок."""
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
    reflected = False
    for _ in range(n):
        vel = vel + (a_vec + g_vec) * h
        pos = pos + vel * h
        pos, vel, hit = _reflect(pos, vel, walls)
        reflected = reflected or hit
    return pos, vel, n, reflected


def walls_of(width: float, height: float):
    return (0.0, float(width), 0.0, float(height))


def _draw_start(rng, width: float, height: float):
    """Монета, затем внутренний старт или подход к одной из четырёх сторон."""
    approach = float(rng.random()) < 0.5
    if not approach:
        pos = np.array([
            rng.uniform(MARGIN, float(width) - MARGIN),
            rng.uniform(MARGIN, float(height) - MARGIN),
        ], np.float64)
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        spd = float(rng.uniform(20.0, 60.0))
        vel = np.array([np.cos(ang) * spd, np.sin(ang) * spd], np.float64)
        return pos, vel, False
    side = int(rng.integers(0, 4))
    depth = float(rng.uniform(0.5, 3.0))
    spd = float(rng.uniform(30.0, 60.0))
    tangent = float(rng.uniform(-20.0, 20.0))
    if side == 0:
        y = float(rng.uniform(MARGIN, float(height) - MARGIN))
        return np.array([depth, y]), np.array([-spd, tangent]), True
    if side == 1:
        y = float(rng.uniform(MARGIN, float(height) - MARGIN))
        return np.array([float(width) - depth, y]), np.array([spd, tangent]), True
    if side == 2:
        x = float(rng.uniform(MARGIN, float(width) - MARGIN))
        return np.array([x, depth]), np.array([tangent, -spd]), True
    x = float(rng.uniform(MARGIN, float(width) - MARGIN))
    return np.array([x, float(height) - depth]), np.array([tangent, spd]), True


def follow(pos, vel, actions, walls):
    """Физика (T, 4) и число шагов с отражением. T = len(actions)+1."""
    states = np.empty((len(actions) + 1, 4), np.float64)
    pos = np.array(pos, np.float64, copy=True)
    vel = np.array(vel, np.float64, copy=True)
    states[0, 0:2] = pos
    states[0, 2:4] = vel
    reflected = 0
    for i, action in enumerate(actions):
        pos, vel, _n, hit = step_physics(pos, vel, int(action), walls)
        states[i + 1, 0:2] = pos
        states[i + 1, 2:4] = vel
        reflected += int(hit)
    return states, reflected


def _pack(phys: np.ndarray) -> np.ndarray:
    packed = np.empty((len(phys), 5), np.float64)
    packed[:, 0:4] = phys
    packed[:, 4] = DT
    return packed


def episode_window(window_seed: int, steps: int, width=None, height=None):
    """Окно. Размер из сида, если аргументы пусты. Поток старта размер не ест."""
    if width is None or height is None:
        width, height = size_of_window(window_seed)
    width = float(width)
    height = float(height)
    walls = walls_of(width, height)
    rng = np.random.default_rng(int(window_seed))
    pos, vel, approach = _draw_start(rng, width, height)
    actions = rng.integers(0, 9, size=int(steps)).astype(np.int64)
    phys, reflected = follow(pos, vel, actions, walls)
    return _pack(phys), actions, width, height, bool(approach), int(reflected)


def make_batch(rng, batch: int = 128, horizon: int = HORIZON):
    states = np.empty((batch, horizon, 5), np.float64)
    actions = np.empty((batch, horizon - 1), np.int64)
    widths = np.empty(batch, np.float64)
    heights = np.empty(batch, np.float64)
    for i in range(batch):
        window_seed = int(rng.integers(100_000, 2**31 - 1))
        packed, acts, width, height, _ap, _rf = episode_window(window_seed, horizon - 1)
        states[i] = packed
        actions[i] = acts
        widths[i] = width
        heights[i] = height
    return states, actions, widths, heights


def _rows(seed0: int, n: int, steps: int, width, height):
    states = []
    widths = []
    heights = []
    for seed in range(int(seed0), int(seed0) + int(n)):
        if width is None:
            packed, _a, w, h, _ap, _rf = episode_window(seed, steps)
        else:
            packed, _a, w, h, _ap, _rf = episode_window(seed, steps, width, height)
        states.append(packed)
        widths.append(np.full(len(packed), w, np.float64))
        heights.append(np.full(len(packed), h, np.float64))
    return (np.concatenate(states, axis=0),
            np.concatenate(widths),
            np.concatenate(heights))


def probe_rows():
    """Свой генератор: размер из сида, 1024 эпизода по 4 состояния."""
    return _rows(PROBE_SEED0, PROBE_EPISODES, PROBE_STEPS, None, None)


def transfer_rows(key: str):
    width, height, seed0 = TRANSFER[key]
    return _rows(seed0, TRANSFER_N, PROBE_STEPS, width, height)


def rollout_episode(seed: int, width: float, height: float):
    rng = np.random.default_rng(int(seed))
    pos, vel, approach = _draw_start(rng, float(width), float(height))
    actions = rng.integers(0, 9, size=N_HOLD_ACTIONS).astype(np.int64)
    held = np.repeat(actions, HOLD)
    phys, reflected = follow(pos, vel, held, walls_of(width, height))
    return _pack(phys), held, float(width), float(height), bool(approach), int(reflected)


def scale_state(states: np.ndarray, widths, heights, repr_name: str) -> np.ndarray:
    """Вход кодировщика. rel — доли поля. abs — деление на 64 и 120."""
    if repr_name not in REPRS:
        raise ValueError(repr_name)
    raw = np.array(states, np.float32, copy=True)
    w = np.asarray(widths, np.float32)
    h = np.asarray(heights, np.float32)
    if raw.ndim == 3 and w.ndim == 1:
        w = w[:, None]
        h = h[:, None]
    dt = raw[..., 4]
    out = np.empty_like(raw)
    if repr_name == "rel":
        out[..., 0] = raw[..., 0] / w
        out[..., 1] = raw[..., 1] / h
        out[..., 2] = raw[..., 2] * dt / w
        out[..., 3] = raw[..., 3] * dt / h
    else:
        out[..., 0:2] = raw[..., 0:2] / np.float32(SCALE_XY)
        out[..., 2:4] = raw[..., 2:4] / np.float32(SCALE_VEL)
    out[..., 4] = dt / np.float32(DT)
    return out


def world_vector(widths, heights) -> np.ndarray:
    """Тройка (W/64, H/64, 1). Последнее число константа и временем не является."""
    w = np.asarray(widths, np.float32) / np.float32(SCALE_XY)
    h = np.asarray(heights, np.float32) / np.float32(SCALE_XY)
    ones = np.ones_like(w)
    return np.stack([w, h, ones], axis=-1).astype(np.float32)


def probe_target(states: np.ndarray, widths, heights, repr_name: str) -> np.ndarray:
    scaled = scale_state(states, widths, heights, repr_name)
    if repr_name == "rel":
        return np.array(scaled[..., 0:4], np.float32, copy=True)
    return np.array(states[..., 0:4], np.float32, copy=True)

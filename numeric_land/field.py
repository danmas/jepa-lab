"""Большое поле посадки. Земля y = 512, по x стены нет, бак 10.

Формула шага та же, что у поля 64. Функция того поля отсюда для шага не вызывается.
"""
import math

import numpy as np

from numeric_land.data import DT, action_parts, g_of_window, world_vector

GROUND = 512.0
FIELD_W = 512.0
PAD_CENTER = 256.0
PAD_LO = 248.0
PAD_HI = 264.0
VY_SOFT = 8.0
FUEL0 = 10.0
SCALE_XY = 512.0
SCALE_VEL = 120.0
SCALE_FUEL = 10.0
PROBE_SEED0 = 81000
PROBE_EPISODES = 1024
PROBE_STEPS = 3
TRANSFER_N = 256
ROLLOUT_SEED0 = 80000
ROLLOUT_N = 64
HOLD = 2
N_HOLD_ACTIONS = 15
PLAN_SEED0 = 89000
PLAN_N = 30
EPISODE_CAP = 450
RANGE_G = (2.0, 8.0, 20.0)
SINGLE_TRANSFER_G = (4.0, 14.0)
MARKS = (1, 10, 20, 30)
TRANSITIONS = 3


def step_field(x, y, vx, vy, fuel, grounded, action, g):
    """Один полный шаг. x не зажимается. Касание земли замирает."""
    if grounded:
        return float(x), float(y), float(vx), float(vy), float(fuel), True
    g = float(g)
    fuel = float(fuel)
    ax, thrust_y, frac = action_parts(action, fuel)
    x = float(x)
    y = float(y)
    vx = float(vx)
    vy = float(vy)
    ay = g - thrust_y
    speed = math.hypot(vx, vy)
    a_mag = math.hypot(ax, ay)
    bound = (speed + a_mag * DT) * DT
    n = max(1, math.ceil(bound / 1.0))
    h = DT / n
    hit = False
    for _ in range(n):
        vx += ax * h
        vy += ay * h
        x += vx * h
        y += vy * h
        if y >= GROUND:
            y = GROUND
            vx = 0.0
            hit = True
            break
    if fuel > 0.0 and frac >= 1.0:
        fuel = max(0.0, fuel - 0.02)
    elif fuel > 0.0 and frac >= 0.5:
        fuel = max(0.0, fuel - 0.01)
    return x, y, vx, vy, fuel, hit


def start_and_actions(seed: int, n_actions: int):
    """Порядок: x, y, vx, vy, затем индексы. Топливо всегда 10 и из генератора не берётся."""
    rng = np.random.default_rng(int(seed))
    x = float(rng.uniform(240.0, 272.0))
    y = float(rng.uniform(464.0, 488.0))
    vx = float(rng.uniform(-6.0, 6.0))
    vy = float(rng.uniform(-4.0, 8.0))
    actions = rng.integers(0, 9, size=int(n_actions)).astype(np.int64)
    return x, y, vx, vy, actions


def rollout_states(x, y, vx, vy, actions, g) -> np.ndarray:
    """(T, 5): x, y, vx, vy, fuel. Бак начинается с 10."""
    fuel = FUEL0
    grounded = False
    rows = np.empty((len(actions) + 1, 5), np.float64)
    rows[0] = (x, y, vx, vy, fuel)
    for i, action in enumerate(actions):
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, int(action), g)
        rows[i + 1] = (x, y, vx, vy, fuel)
    return rows


def window_of(window_seed: int, g: float, n_actions: int = TRANSITIONS):
    x, y, vx, vy, actions = start_and_actions(window_seed, n_actions)
    states = rollout_states(x, y, vx, vy, actions, g)
    return states, actions


def make_batch(rng, batch: int = 128, protocol: str = "range", g=None):
    """Окна большого поля. range берёт своё g на окно. fixed держит одно g."""
    if protocol == "fixed":
        if g is None:
            raise ValueError("fixed требует g")
    elif protocol != "range":
        raise ValueError(protocol)
    states = np.empty((batch, TRANSITIONS + 1, 5), np.float64)
    actions = np.empty((batch, TRANSITIONS), np.int64)
    gs = np.empty(batch, np.float64)
    for i in range(batch):
        window_seed = int(rng.integers(100_000, 2**31 - 1))
        gi = g_of_window(window_seed) if protocol == "range" else float(g)
        packed, acts = window_of(window_seed, gi)
        states[i] = packed
        actions[i] = acts
        gs[i] = gi
    return states, actions, gs


def scale_state(states: np.ndarray) -> np.ndarray:
    """x/512, y/512, vx/120, vy/120, fuel/10. 512 — эта земля, 10 — начальный бак."""
    raw = np.array(states, np.float32, copy=True)
    out = np.empty_like(raw)
    out[..., 0:2] = raw[..., 0:2] / np.float32(SCALE_XY)
    out[..., 2:4] = raw[..., 2:4] / np.float32(SCALE_VEL)
    out[..., 4] = raw[..., 4] / np.float32(SCALE_FUEL)
    return out


def _rows(seed0: int, n: int, g_mode: str, g=None):
    states = []
    for seed in range(int(seed0), int(seed0) + int(n)):
        gi = g_of_window(seed) if g_mode == "range" else float(g)
        packed, _acts = window_of(seed, gi, PROBE_STEPS)
        states.append(packed)
    return np.concatenate(states, axis=0)


def probe_rows(g_mode: str = "range", g=None):
    """1024 эпизода, 4 состояния."""
    rows = _rows(PROBE_SEED0, PROBE_EPISODES, g_mode, g)
    if len(rows) != PROBE_EPISODES * (PROBE_STEPS + 1):
        raise RuntimeError(len(rows))
    return rows


def transfer_rows(seed0: int, g: float):
    rows = _rows(int(seed0), TRANSFER_N, "fixed", g)
    if len(rows) != TRANSFER_N * (PROBE_STEPS + 1):
        raise RuntimeError(len(rows))
    return rows


def held_episode(seed: int, g: float):
    """15 индексов, каждый держится два шага. Лента от g не зависит."""
    x, y, vx, vy, actions = start_and_actions(seed, N_HOLD_ACTIONS)
    held = np.repeat(actions, HOLD)
    if len(held) != N_HOLD_ACTIONS * HOLD:
        raise RuntimeError(len(held))
    states = rollout_states(x, y, vx, vy, held, g)
    return states, held, actions


def outcome(x, y, vy, grounded) -> str:
    if not grounded:
        return "не сел"
    if PAD_LO <= float(x) <= PAD_HI and float(vy) < VY_SOFT:
        return "мягкая"
    return "удар"

"""Траектории плоского мяча. Шаг только через data.step_state, своего интегратора нет.

Состояние — (x, y, vx, vy). Времени в векторе нет. На диск траектории не пишутся.
"""
import numpy as np

from data import ACTIONS, IMPULSE, N_ACTIONS, RADIUS, SIZE, step_state

# Спека фиксирует делители 64 и 4. Это SIZE и IMPULSE, не статистика батча.
assert int(SIZE) == 64
assert float(IMPULSE) == 4.0

SCALE_XY = float(SIZE)
SCALE_VEL = float(IMPULSE)
N_STATE = 4
HORIZON = 6
TRAIN_SEEDS = (0, 1, 2)
ROLLOUT_SEED0 = 10000
ROLLOUT_N = 64
ROLLOUT_STEPS = 20
PROBE_SEED0 = 20000
PROBE_STATES = 4096
PROBE_TRAIN = 3072
PROBE_STEPS = 3  # старт + 3 толчка = 4 состояния, 1024 сида дают 4096 строк

ACTION_PAIR = (ACTIONS / np.float32(IMPULSE)).astype(np.float32)


def start_state(rng: np.random.Generator):
    """Тот же ящик, угол и скорость, что у плоского World до первого толчка."""
    pos = rng.uniform(RADIUS + 2, SIZE - 1 - RADIUS - 2, size=2).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    spd = rng.uniform(0.3, 1.0)
    vel = np.array([np.cos(ang) * spd, np.sin(ang) * spd], np.float32)
    return pos, vel


def follow(pos, vel, actions) -> np.ndarray:
    """(T, 4), T = len(actions) + 1. Каждый переход — step_state."""
    states = np.empty((len(actions) + 1, N_STATE), np.float32)
    pos = np.array(pos, np.float32)
    vel = np.array(vel, np.float32)
    states[0, 0:2] = pos
    states[0, 2:4] = vel
    for i, action in enumerate(actions):
        pos, vel = step_state(pos, vel, int(action))
        states[i + 1, 0:2] = pos
        states[i + 1, 2:4] = vel
    return states


def episode(seed: int, steps: int):
    """Состояния (steps+1, 4) и действия (steps,). Сид фиксирует и старт, и толчки."""
    rng = np.random.default_rng(int(seed))
    pos, vel = start_state(rng)
    actions = rng.integers(0, N_ACTIONS, size=int(steps)).astype(np.int64)
    return follow(pos, vel, actions), actions


def make_batch(rng: np.random.Generator, batch: int = 128, horizon: int = HORIZON):
    """Обучающее окно из живого генератора. -> states (B,H,4), actions (B,H-1)."""
    states = np.empty((batch, horizon, N_STATE), np.float32)
    actions = np.empty((batch, horizon - 1), np.int64)
    for i in range(batch):
        pos, vel = start_state(rng)
        actions[i] = rng.integers(0, N_ACTIONS, size=horizon - 1)
        states[i] = follow(pos, vel, actions[i])
    return states, actions


def scale_state(states: np.ndarray) -> np.ndarray:
    out = np.array(states, np.float32, copy=True)
    out[..., 0:2] /= np.float32(SCALE_XY)
    out[..., 2:4] /= np.float32(SCALE_VEL)
    return out


def unscale_state(states: np.ndarray) -> np.ndarray:
    out = np.array(states, np.float32, copy=True)
    out[..., 0:2] *= np.float32(SCALE_XY)
    out[..., 2:4] *= np.float32(SCALE_VEL)
    return out


def probe_states() -> np.ndarray:
    """4096 истинных состояний. Сиды 20000…21023, по 4 состояния, в порядке сида.

    Первые 3072 строк — 768 первых сидов целиком, последние 1024 — следующие 256.
    Оценочные сиды 10000…10063 в этот набор не входят.
    """
    per = PROBE_STEPS + 1
    if PROBE_STATES % per != 0:
        raise RuntimeError("число состояний зонда не делится на длину окна")
    n = PROBE_STATES // per
    rows = np.empty((PROBE_STATES, N_STATE), np.float32)
    for i in range(n):
        states, _ = episode(PROBE_SEED0 + i, PROBE_STEPS)
        rows[i * per:(i + 1) * per] = states
    return rows

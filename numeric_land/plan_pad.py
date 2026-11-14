"""CEM площадки. Стоимость смотрит в текущий центр. Топлива в сумме нет.

В мир уходит первое действие. Успех решает симулятор. Потолок 450, горизонт 40.
"""
import time

import numpy as np
import torch

from numeric_land.field import EPISODE_CAP, FUEL0, GROUND, SCALE_VEL, SCALE_XY, scale_state, step_field
from numeric_land.models import action_pairs
from numeric_land.pad import outcome_at, same_start, world_vector

N_ACTIONS = 9
SAMPLES = 128
ITERS = 8
ELITE = 16
HORIZON = 40


def step_cost(pred: torch.Tensor, pad_x: float) -> torch.Tensor:
    """((x-pad_x)/512)² + ((y-512)/512)² + (vy/120)². Топливо и vx не читаются."""
    x = pred[:, 0]
    y = pred[:, 1]
    vy = pred[:, 3]
    return ((x - float(pad_x)) ** 2 + (y - GROUND) ** 2) / (SCALE_XY ** 2) + (vy / SCALE_VEL) ** 2


def _state_tensor(x, y, vx, vy, fuel) -> torch.Tensor:
    raw = np.array([[x, y, vx, vy, fuel]], np.float64)
    return torch.from_numpy(scale_state(raw))


@torch.no_grad()
def cem_plan(encoder, predictor, probe, x, y, vx, vy, fuel, g, pad_x,
             horizon=HORIZON, samples=SAMPLES, iters=ITERS, elite=ELITE, seed=0):
    t0 = time.perf_counter()
    z0 = encoder(_state_tensor(x, y, vx, vy, fuel))
    world = None
    if predictor.n_world:
        world = torch.from_numpy(world_vector(np.array([float(g)]), np.array([float(pad_x)])))
    probs = torch.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    gen = torch.Generator().manual_seed(int(seed))
    best_seq, best_cost = None, float("inf")
    for _ in range(iters):
        u = torch.multinomial(probs, samples, replacement=True, generator=gen).T.contiguous()
        z = z0.expand(samples, -1).contiguous()
        fed = None if world is None else world.expand(samples, -1)
        cost = torch.zeros(samples)
        for k in range(horizon):
            z = predictor(z, action_pairs(u[:, k], z), fed)
            cost = cost + step_cost(probe.fc(z), pad_x)
        i = int(cost.argmin())
        if float(cost[i]) < best_cost:
            best_cost, best_seq = float(cost[i]), u[i].clone()
        idx = cost.argsort()[:elite]
        counts = torch.zeros_like(probs)
        for e in idx.tolist():
            counts[torch.arange(horizon), u[e]] += 1
        dist = (counts + 0.5) / (elite + 0.5 * N_ACTIONS)
        probs = 0.7 * probs + 0.3 * dist
    return best_seq, best_cost, time.perf_counter() - t0


def _frame(x, y, vx, vy, fuel, grounded, action):
    return {
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "vx": round(float(vx), 4),
        "vy": round(float(vy), 4),
        "fuel": round(float(fuel), 4),
        "grounded": bool(grounded),
        "action": int(action),
    }


def run_episode(seed: int, g: float, pad_x: float, models, max_steps: int = EPISODE_CAP):
    """MPC от текущего центра. models — (encoder, predictor, probe), без подмены на True."""
    if models is None or models is True:
        raise ValueError("нужна тройка моделей")
    encoder, predictor, probe = models
    x, y, vx, vy, _empty = same_start(int(seed))
    start_x = float(x)
    start_y = float(y)
    fuel = FUEL0
    grounded = False
    frames = [_frame(x, y, vx, vy, fuel, False, 1)]
    plan_s = 0.0
    for t in range(int(max_steps)):
        if grounded:
            break
        seq, _cost, elapsed = cem_plan(
            encoder, predictor, probe, x, y, vx, vy, fuel, g, pad_x,
            seed=int(seed) + t)
        plan_s += float(elapsed)
        action = int(seq[0])
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, action, g)
        frames.append(_frame(x, y, vx, vy, fuel, grounded, action))
    result = outcome_at(x, y, vy, grounded, pad_x)
    return {
        "seed": int(seed),
        "g": float(g),
        "pad_x": float(pad_x),
        "result": result,
        "steps": len(frames) - 1,
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "vy": round(float(vy), 4),
        "fuel": round(float(fuel), 4),
        "start_x": round(start_x, 4),
        "start_y": round(start_y, 4),
        "plan_s": round(plan_s, 3),
        "frames": frames,
    }

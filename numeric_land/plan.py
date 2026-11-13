"""CEM посадки. В мир уходит первое действие, успех решает симулятор.

Только land_range_vis_s0. Топлива в стоимости нет.
"""
import os
import threading
import time

import numpy as np
import torch

from models import EMATarget
from numeric_land.data import (
    EPISODE_CAP, SCALE_VEL, SCALE_XY, outcome, probe_rows, scale_state,
    start_and_actions, step_lander, world_vector,
)
from numeric_land.models import LATENT, N_WORLD, BallEncoder, LandPredictor, action_pairs, fit_readout
from numeric_land.train import STEPS, TAG, ckpt_path

_LOCK = threading.Lock()
_BUNDLE = None
RIDGE_LAM = 1e-2
PROBE_TRAIN = 3072
N_ACTIONS = 9
SAMPLES = 128
ITERS = 8
ELITE = 16
HORIZON = 40


def step_cost(pred: torch.Tensor) -> torch.Tensor:
    """(x-32)²/64² + (y-64)²/64² + (vy/120)². Канал топлива не читается."""
    x = pred[:, 0]
    y = pred[:, 1]
    vy = pred[:, 3]
    return (x - 32.0) ** 2 / (SCALE_XY ** 2) + (y - 64.0) ** 2 / (SCALE_XY ** 2) + (vy / SCALE_VEL) ** 2


def _load_vis():
    path = ckpt_path("range", "vis", 0)
    if not os.path.isfile(path):
        raise ValueError(
            f"нет {os.path.basename(path)}. Сначала: python -m numeric_land.train --tag numeric-land")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck.get("mode") != "jepa" or ck.get("tag") != TAG or ck.get("protocol") != "range"
            or ck.get("variant") != "vis" or int(ck.get("seed", -1)) != 0
            or int(ck.get("steps", 0)) != STEPS or int(ck.get("n_world", -1)) != N_WORLD["vis"]):
        raise ValueError(path)
    encoder = BallEncoder()
    predictor = LandPredictor(N_WORLD["vis"])
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()
    return encoder, predictor, ema


def _fit(ema):
    states = probe_rows("range")
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states, np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    if z.shape[1] != LATENT or target.shape[1] != 5:
        raise RuntimeError((z.shape, target.shape))
    probe = fit_readout(z[:PROBE_TRAIN], target[:PROBE_TRAIN], lam=RIDGE_LAM)
    if probe.fc.in_features != LATENT or probe.fc.out_features != 5:
        raise RuntimeError("зонд собрал не тот размер")
    return probe


def bundle():
    global _BUNDLE
    with _LOCK:
        if _BUNDLE is None:
            torch.set_num_threads(8)
            encoder, predictor, ema = _load_vis()
            _BUNDLE = (encoder, predictor, _fit(ema))
        return _BUNDLE


def _state_tensor(x, y, vx, vy, fuel) -> torch.Tensor:
    raw = np.array([[x, y, vx, vy, fuel]], np.float64)
    return torch.from_numpy(scale_state(raw))


@torch.no_grad()
def cem_plan(encoder, predictor, probe, x, y, vx, vy, fuel, g,
             horizon=HORIZON, samples=SAMPLES, iters=ITERS, elite=ELITE, seed=0):
    t0 = time.perf_counter()
    z0 = encoder(_state_tensor(x, y, vx, vy, fuel))
    world = torch.from_numpy(world_vector(np.array([float(g)])))
    probs = torch.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    gen = torch.Generator().manual_seed(int(seed))
    best_seq, best_cost = None, float("inf")
    for _ in range(iters):
        u = torch.multinomial(probs, samples, replacement=True, generator=gen).T.contiguous()
        z = z0.expand(samples, -1).contiguous()
        fed = world.expand(samples, -1)
        cost = torch.zeros(samples)
        for k in range(horizon):
            z = predictor(z, action_pairs(u[:, k], z), fed)
            cost = cost + step_cost(probe.fc(z))
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


def run_episode(seed: int, g: float, models=None, max_steps: int = EPISODE_CAP):
    """MPC: первое действие плана, затем переплан. Кадры — симулятор."""
    if models is None:
        models = bundle()
    encoder, predictor, probe = models
    x, y, vx, vy, _empty = start_and_actions(int(seed), 0)
    fuel = 1.0
    grounded = False
    frames = [_frame(x, y, vx, vy, fuel, False, 1)]
    plan_s = 0.0
    for t in range(int(max_steps)):
        if grounded:
            break
        seq, _cost, elapsed = cem_plan(
            encoder, predictor, probe, x, y, vx, vy, fuel, g, seed=int(seed) + t)
        plan_s += float(elapsed)
        action = int(seq[0])
        x, y, vx, vy, fuel, grounded = step_lander(
            x, y, vx, vy, fuel, grounded, action, g)
        frames.append(_frame(x, y, vx, vy, fuel, grounded, action))
    result = outcome(x, y, vy, grounded)
    return {
        "seed": int(seed),
        "g": float(g),
        "result": result,
        "steps": len(frames) - 1,
        "x": round(float(x), 4),
        "y": round(float(y), 4),
        "vy": round(float(vy), 4),
        "fuel": round(float(fuel), 4),
        "plan_s": round(plan_s, 3),
        "frames": frames,
    }

"""План JEPA с явным dt: CEM в латенте, в мир уходит первое действие.

Чекпоинт один — рука B, вариант с dt, сид обучения 0.
Кадр в сеть не подаётся. step_state не вызывается.
"""
import os
import threading
import time

import numpy as np
import torch

from data import ACTION_NAMES, N_ACTIONS, ZONE_CENTERS, ZONES, World, in_zone
from models import EMATarget, fit_ridge
from numeric_dt.data import PAGE_DT, page_start, probe_states, scale_state, step_physics
from numeric_dt.models import LATENT, N_IN, DtEncoder, DtPredictor, action_pairs
from numeric_dt.train import STEPS, ckpt_path

_LOCK = threading.Lock()
_BUNDLE = None
RIDGE_LAM = 1e-2
PROBE_TRAIN = 3072


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("arm") != "B" or ck.get("variant") != "with-dt":
        raise ValueError(path)
    if int(ck.get("steps", 0)) != STEPS or int(ck.get("n_in", 0)) != N_IN["with-dt"]:
        raise ValueError(path)
    encoder = DtEncoder(N_IN["with-dt"])
    predictor = DtPredictor()
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()
    return encoder, predictor, ema


def _fit_probe(ema):
    states = probe_states("B", "with-dt")
    scaled = torch.from_numpy(scale_state(states, "with-dt"))
    target = torch.from_numpy(np.array(states[:, 0:4], np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    if z.shape[1] != LATENT:
        raise RuntimeError(f"латент {z.shape[1]}, ожидался {LATENT}")
    probe = fit_ridge(z[:PROBE_TRAIN], target[:PROBE_TRAIN], lam=RIDGE_LAM)
    if probe.fc.in_features != LATENT:
        raise RuntimeError("fit_ridge собрал не тот вход: нужен один латент 32")
    return probe


def bundle():
    global _BUNDLE
    with _LOCK:
        if _BUNDLE is None:
            path = ckpt_path("B", "with-dt", 0)
            if not os.path.exists(path):
                raise ValueError(
                    "нет out/numeric_dtB_jepa_s0.pt. "
                    "Сначала: python -m numeric_dt.train --tag numeric-dt")
            encoder, predictor, ema = _load(path)
            _BUNDLE = (encoder, predictor, _fit_probe(ema))
        return _BUNDLE


def _state_tensor(pos, vel, dt: float) -> torch.Tensor:
    raw = np.array([[pos[0], pos[1], vel[0], vel[1], dt]], np.float32)
    return torch.from_numpy(scale_state(raw, "with-dt"))


@torch.no_grad()
def cem_plan(encoder, predictor, probe, pos, vel, dt, goal_xy, horizon=10,
             samples=128, iters=8, elite=16, seed=0):
    t0 = time.perf_counter()
    z0 = encoder(_state_tensor(pos, vel, dt))
    probs = torch.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    g = torch.Generator().manual_seed(int(seed))
    goal = torch.tensor(goal_xy, dtype=torch.float32)
    best_seq, best_cost = None, float("inf")
    for _ in range(iters):
        u = torch.multinomial(probs, samples, replacement=True, generator=g).T.contiguous()
        z = z0.expand(samples, -1).contiguous()
        dists = []
        for k in range(horizon):
            z = predictor(z, action_pairs(u[:, k], z))
            xy = probe.fc(z)[:, :2]
            dists.append(((xy - goal) ** 2).sum(1))
        cost = dists[-1] + 0.3 * torch.stack(dists[-3:]).mean(0)
        i = int(cost.argmin())
        if float(cost[i]) < best_cost:
            best_cost, best_seq = float(cost[i]), u[i].clone()
        idx = cost.argsort()[:elite]
        counts = torch.zeros_like(probs)
        for e in idx:
            counts[torch.arange(horizon), u[e]] += 1
        dist = (counts + 0.5) / (elite + 0.5 * N_ACTIONS)
        probs = 0.7 * probs + 0.3 * dist
    return best_seq, best_cost, time.perf_counter() - t0


def _paint(world, pos, vel):
    world.pos = np.array(pos, np.float32)
    world.vel = np.array(vel, np.float32)
    return world.frame()[0], world.pos.astype(float).copy()


def execute(zone_index, seed=1, dt_key="1/30", horizon=10, max_steps=20,
            samples=128, iters=8):
    """MPC. Кадры рисует плоский World, шаг считает интегратор dt."""
    if dt_key not in PAGE_DT:
        raise ValueError("dt: 1/60, 1/30 или 1/15")
    dt = PAGE_DT[dt_key]
    encoder, predictor, probe = bundle()
    pos, vel = page_start(int(seed))
    world = World(0, world="flat")
    frames = []
    ball = []
    frame, point = _paint(world, pos, vel)
    frames.append(frame)
    ball.append(point)
    pos, vel, _n = step_physics(pos, vel, 4, dt)
    frame, point = _paint(world, pos, vel)
    frames.append(frame)
    ball.append(point)
    cx, cy = ZONE_CENTERS[zone_index]
    steps = []
    success = False
    for t in range(max_steps):
        seq, cost, elapsed = cem_plan(
            encoder, predictor, probe, pos, vel, dt, (cx, cy),
            horizon=horizon, samples=samples, iters=iters, seed=seed)
        action = int(seq[0])
        pos, vel, _n = step_physics(pos, vel, action, dt)
        frame, point = _paint(world, pos, vel)
        frames.append(frame)
        ball.append(point)
        steps.append({
            "step": t + 1,
            "action": ACTION_NAMES[action],
            "cost": round(cost, 4),
            "plan_time": round(elapsed, 3),
            "pos": [round(float(v), 1) for v in pos],
        })
        if in_zone(pos, zone_index):
            success = True
            break
    miss = float(np.hypot(pos[0] - cx, pos[1] - cy))
    return {
        "zone": zone_index,
        "zone_name": ZONES[zone_index]["name"],
        "seed": int(seed),
        "success": success,
        "steps_used": len(steps),
        "miss_px": round(miss, 2),
        "steps": steps,
        "final_pos": [round(float(v), 1) for v in pos],
        "frames": frames,
        "ball": ball,
        "dt": dt_key,
        "arm": "B",
        "train_seed": 0,
    }

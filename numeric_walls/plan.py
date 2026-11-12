"""План JEPA со стенками: CEM в латенте, в мир уходит первое действие.

На страницу выходят четыре сети: протокол range, сид обучения 0,
оси W и H, варианты vis и hid. Кадр в сеть не подаётся.
step_state не вызывается. dt всегда 1/30.
"""
import os
import threading
import time

import numpy as np
import torch

from data import ACTION_NAMES, N_ACTIONS, ZONE_CENTERS, ZONE_HALF, ZONES, World, in_zone
from models import EMATarget, fit_ridge
from numeric_walls.data import (
    DT, demo_start, probe_states, scale_state, scale_walls, step_physics,
)
from numeric_walls.models import LATENT, N_WALL, BallEncoder, WallPredictor, action_pairs
from numeric_walls.train import STEPS, ckpt_path

_LOCK = threading.Lock()
_BUNDLE = {}
RIDGE_LAM = 1e-2
PROBE_TRAIN = 3072
SPANS = (12, 20, 32)


def wall_input(predictor, walls_1x4, n: int):
    """Скрытый предиктор стенки не получает. Видимый — ту же четвёрку на каждый образец."""
    if predictor.n_wall == 0:
        return None
    return walls_1x4.expand(n, -1)


def _load(path, axis: str, variant: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck.get("mode") != "jepa" or ck.get("world") != axis
            or ck.get("protocol") != "range" or ck.get("variant") != variant
            or int(ck.get("seed", -1)) != 0 or int(ck.get("steps", 0)) != STEPS
            or int(ck.get("n_wall", -1)) != N_WALL[variant]):
        raise ValueError(path)
    encoder = BallEncoder()
    predictor = WallPredictor(N_WALL[variant])
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()
    return encoder, predictor, ema


def _fit_probe(ema, axis: str):
    states = probe_states(axis, "range")
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states[:, 0:4], np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    if z.shape[1] != LATENT:
        raise RuntimeError(f"латент {z.shape[1]}, ожидался {LATENT}")
    probe = fit_ridge(z[:PROBE_TRAIN], target[:PROBE_TRAIN], lam=RIDGE_LAM)
    if probe.fc.in_features != LATENT:
        raise RuntimeError("fit_ridge собрал не тот вход: нужен один латент 32")
    return probe


def bundle(axis: str, variant: str):
    if axis not in ("W", "H") or variant not in ("vis", "hid"):
        raise ValueError((axis, variant))
    key = (axis, variant)
    with _LOCK:
        if key not in _BUNDLE:
            path = ckpt_path(axis, "range", variant, 0)
            if not os.path.exists(path):
                raise ValueError(
                    f"нет {os.path.basename(path)}. "
                    "Сначала: python -m numeric_walls.train --tag numeric-walls")
            encoder, predictor, ema = _load(path, axis, variant)
            _BUNDLE[key] = (encoder, predictor, _fit_probe(ema, axis))
        return _BUNDLE[key]


def _state_tensor(pos, vel) -> torch.Tensor:
    raw = np.array([[pos[0], pos[1], vel[0], vel[1], DT]], np.float64)
    return torch.from_numpy(scale_state(raw))


def _walls_tensor(walls) -> torch.Tensor:
    return torch.from_numpy(scale_walls(np.asarray(walls, np.float64)))[None, :]


@torch.no_grad()
def cem_plan(encoder, predictor, probe, pos, vel, walls, goal_xy, horizon=10,
             samples=128, iters=8, elite=16, seed=0):
    t0 = time.perf_counter()
    z0 = encoder(_state_tensor(pos, vel))
    wall_1 = _walls_tensor(walls)
    probs = torch.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    g = torch.Generator().manual_seed(int(seed))
    goal = torch.tensor(goal_xy, dtype=torch.float32)
    best_seq, best_cost = None, float("inf")
    for _ in range(iters):
        u = torch.multinomial(probs, samples, replacement=True, generator=g).T.contiguous()
        z = z0.expand(samples, -1).contiguous()
        dists = []
        fed = wall_input(predictor, wall_1, samples)
        for k in range(horizon):
            z = predictor(z, action_pairs(u[:, k], z), fed)
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


def zone_outside(zone_index: int, walls) -> bool:
    cx, cy = ZONE_CENTERS[zone_index]
    half = float(ZONE_HALF)
    left, right, top, bottom = (float(v) for v in walls)
    overlap_x = min(cx + half, right) - max(cx - half, left)
    overlap_y = min(cy + half, bottom) - max(cy - half, top)
    return overlap_x <= 0.0 or overlap_y <= 0.0


def execute(zone_index, seed=1, axis="W", variant="vis", span=20,
            horizon=10, max_steps=20, samples=128, iters=8):
    """MPC. Кадры рисует плоский World, шаг считает интегратор со стенками."""
    if axis not in ("W", "H"):
        raise ValueError("axis: W или H")
    if variant not in ("vis", "hid"):
        raise ValueError("walls_variant: vis или hid")
    if int(span) not in SPANS:
        raise ValueError("span: 12, 20 или 32")
    encoder, predictor, probe = bundle(axis, variant)
    pos, vel, walls = demo_start(int(seed), axis, int(span))
    world = World(0, world="flat")
    frames = []
    ball = []
    frame, point = _paint(world, pos, vel)
    frames.append(frame)
    ball.append(point)
    pos, vel, _n = step_physics(pos, vel, 4, walls, DT)
    frame, point = _paint(world, pos, vel)
    frames.append(frame)
    ball.append(point)
    cx, cy = ZONE_CENTERS[zone_index]
    steps = []
    success = False
    for t in range(max_steps):
        seq, cost, elapsed = cem_plan(
            encoder, predictor, probe, pos, vel, walls, (cx, cy),
            horizon=horizon, samples=samples, iters=iters, seed=seed)
        action = int(seq[0])
        pos, vel, _n = step_physics(pos, vel, action, walls, DT)
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
        "axis": axis,
        "walls_variant": variant,
        "span": int(span),
        "walls": [round(float(v), 4) for v in walls],
        "zone_outside": zone_outside(zone_index, walls),
        "protocol": "range",
        "train_seed": 0,
        "dt": "1/30",
    }

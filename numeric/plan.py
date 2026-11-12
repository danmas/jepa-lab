"""План числовой JEPA на плоском мире: CEM в латенте, в мир уходит первое действие.

Чекпоинт один — сид обучения 0. Зонд тот же, что у раскатки: читает положение
из E_ema(s). Кадр в сеть не подаётся.
"""
import os
import threading
import time

import numpy as np
import torch

from data import ACTION_NAMES, N_ACTIONS, ZONE_CENTERS, ZONES, World, in_zone
from numeric.data import scale_state
from numeric.models import action_pairs
from numeric.rollout import fit_probe
from numeric.train import ckpt_path

_LOCK = threading.Lock()
_BUNDLE = None


def bundle():
    """Энкодер, предиктор и замороженный зонд. Грузится один раз."""
    global _BUNDLE
    with _LOCK:
        if _BUNDLE is None:
            path = ckpt_path("jepa", 0)
            if not os.path.exists(path):
                raise ValueError(
                    "нет out/numeric_jepa_s0.pt. Сначала: python -m numeric.train --tag numeric")
            from numeric.rollout import _load_jepa
            from numeric.data import probe_states
            encoder, predictor, ema = _load_jepa(path)
            probe, _r2 = fit_probe(ema, probe_states())
            _BUNDLE = (encoder, predictor, probe)
        return _BUNDLE


def _state_tensor(pos, vel) -> torch.Tensor:
    raw = np.array([pos[0], pos[1], vel[0], vel[1]], np.float32)
    return torch.from_numpy(scale_state(raw)[None, :])


@torch.no_grad()
def cem_plan(encoder, predictor, probe, pos, vel, goal_xy, horizon=10,
             samples=128, iters=8, elite=16, seed=0):
    """Последовательность толчков. Стоимость — квадрат пикселей до центра зоны."""
    t0 = time.perf_counter()
    z0 = encoder(_state_tensor(pos, vel))
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


def execute(zone_index, seed=1, horizon=10, max_steps=20, samples=128, iters=8):
    """MPC по плоскому миру. Кадры — для страницы, не для сети."""
    encoder, predictor, probe = bundle()
    w = World(int(seed), world="flat")
    frames = [w.frame()[0]]
    ball = [w.pos.astype(float).copy()]
    w.step(4)
    frames.append(w.frame()[0])
    ball.append(w.pos.astype(float).copy())
    cx, cy = ZONE_CENTERS[zone_index]
    steps = []
    success = False
    for t in range(max_steps):
        seq, cost, dt = cem_plan(
            encoder, predictor, probe, w.pos, w.vel, (cx, cy),
            horizon=horizon, samples=samples, iters=iters, seed=seed)
        action = int(seq[0])
        w.step(action)
        frames.append(w.frame()[0])
        ball.append(w.pos.astype(float).copy())
        steps.append({
            "step": t + 1,
            "action": ACTION_NAMES[action],
            "cost": round(cost, 4),
            "plan_time": round(dt, 3),
            "pos": [round(float(v), 1) for v in w.pos],
        })
        if in_zone(w.pos, zone_index):
            success = True
            break
    miss = float(np.hypot(w.pos[0] - cx, w.pos[1] - cy))
    return {
        "zone": zone_index,
        "zone_name": ZONES[zone_index]["name"],
        "seed": int(seed),
        "success": success,
        "steps_used": len(steps),
        "miss_px": round(miss, 2),
        "steps": steps,
        "final_pos": [round(float(v), 1) for v in w.pos],
        "frames": frames,
        "ball": ball,
    }

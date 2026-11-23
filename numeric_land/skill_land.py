"""Навык посадки над площадкой: короткий поиск, сетка 162, линия 114.

Горизонт 3, все 729 последовательностей. Терминал на земле, vy = 6.
Модель решения и правило с наведением действие здесь не выбирают.
"""
import numpy as np
import torch

from numeric_land.data import PAIRS
from numeric_land.expert_hover import HOVER_CAP, NAV_TOL
from numeric_land.field import FUEL0, GROUND, SCALE_VEL, step_field
from numeric_land.goal import target_state
from numeric_land.models import LATENT, LOSS_WEIGHTS, action_pairs
from numeric_land.pad import outcome_at
from numeric_land.relenc import encode_rel, scale_rel
from numeric_land.skills import N_WORLD, world_of

TAG = "numeric-land-skill-land"
LINE = 114
GRID_N = 162
HORIZON = 3
N_ACT = 9
N_SEQ = 729
GOAL_VY = 6.0
GOAL_HEIGHT0 = 0.0
GOAL_FUEL_TRAIN = 10.0
EXPERT_FRAC = 1.0
GS = (2.0, 8.0, 15.0)
PADS = (256.0, 200.0)
DXS = (-float(NAV_TOL), 0.0, float(NAV_TOL))
HEIGHTS = (32.0, 64.0, 128.0)
VYS = (0.0, 6.0, 12.0)


def locks() -> None:
    if TAG != "numeric-land-skill-land":
        raise RuntimeError(TAG)
    if HORIZON != 3 or N_ACT != 9 or N_SEQ != 729 or N_ACT ** HORIZON != N_SEQ:
        raise RuntimeError((HORIZON, N_ACT, N_SEQ))
    if len(LOSS_WEIGHTS) != HORIZON or LOSS_WEIGHTS != (1.0, 0.5, 0.25):
        raise RuntimeError(LOSS_WEIGHTS)
    if GOAL_VY != 6.0 or GOAL_HEIGHT0 != 0.0 or GOAL_FUEL_TRAIN != 10.0:
        raise RuntimeError((GOAL_VY, GOAL_HEIGHT0, GOAL_FUEL_TRAIN))
    if LINE != 114 or GRID_N != 162 or EXPERT_FRAC != 1.0:
        raise RuntimeError((LINE, GRID_N, EXPERT_FRAC))
    if abs(float(NAV_TOL) - 6.0) > 0.0 or float(FUEL0) != 10.0 or float(GROUND) != 512.0:
        raise RuntimeError((NAV_TOL, FUEL0, GROUND))
    if float(SCALE_VEL) != 120.0 or int(LATENT) != 32 or int(N_WORLD) != 4:
        raise RuntimeError((SCALE_VEL, LATENT, N_WORLD))
    if int(HOVER_CAP) != 900 or len(PAIRS) != N_ACT:
        raise RuntimeError((HOVER_CAP, len(PAIRS)))
    if GS != (2.0, 8.0, 15.0) or PADS != (256.0, 200.0):
        raise RuntimeError((GS, PADS))
    if DXS != (-6.0, 0.0, 6.0) or HEIGHTS != (32.0, 64.0, 128.0) or VYS != (0.0, 6.0, 12.0):
        raise RuntimeError((DXS, HEIGHTS, VYS))
    if 3 * 2 * 3 * 3 * 3 != GRID_N:
        raise RuntimeError(GRID_N)


def line_ok(soft: int) -> bool:
    """114 из 162 проходит, 113 нет. Сравнение целых, без округления доли."""
    return int(soft) >= int(LINE)


def terminal_state(pad_x, fuel):
    """Точка на площадке, на земле, vx = 0, vy = 6. Высота окна сюда не входит."""
    raw = target_state(float(pad_x), float(GOAL_VY), float(GOAL_HEIGHT0), fuel0=float(fuel))
    if raw.shape != (1, 5):
        raise RuntimeError(raw.shape)
    return raw


def train_height_and_vy(batch: int):
    """Высота и vy для skill_loss. Ноль и 6, не высота начала окна и не vy пула."""
    n = int(batch)
    height0 = np.zeros(n, np.float64)
    vys = np.full(n, GOAL_VY, np.float64)
    return height0, vys


def grid():
    """Все сочетания тяжести, центра, dx, высоты и vy. Ровно 162, не на земле."""
    locks()
    rows = []
    seen = set()
    for g in GS:
        for pad in PADS:
            for height in HEIGHTS:
                for dx in DXS:
                    for vy in VYS:
                        key = (float(g), float(pad), float(dx), float(height), float(vy))
                        if key in seen:
                            raise RuntimeError(key)
                        seen.add(key)
                        y = float(GROUND) - float(height)
                        if not y < float(GROUND):
                            raise RuntimeError(y)
                        rows.append({
                            "g": float(g),
                            "pad_x": float(pad),
                            "dx": float(dx),
                            "height": float(height),
                            "vy": float(vy),
                            "x": float(pad) + float(dx),
                            "y": y,
                            "vx": 0.0,
                            "fuel": float(FUEL0),
                        })
    if len(rows) != GRID_N:
        raise RuntimeError(len(rows))
    return tuple(rows)


def pair_table():
    index = torch.arange(N_SEQ)
    columns = []
    base = N_ACT ** (HORIZON - 1)
    for _ in range(HORIZON):
        columns.append(torch.div(index, base, rounding_mode="floor") % N_ACT)
        base //= N_ACT
    actions = torch.stack(columns, dim=1)
    like = torch.zeros(1, dtype=torch.float32)
    pairs = action_pairs(actions, like)
    if actions.shape != (N_SEQ, HORIZON) or pairs.shape != (N_SEQ, HORIZON, 2):
        raise RuntimeError((actions.shape, pairs.shape))
    if int(actions[0, 0]) != 0 or int(actions[81, 0]) != 1 or int(actions[728, 0]) != 8:
        raise RuntimeError("порядок действий разъехался с индексом iy*3+ix")
    steps = [pairs[:, k].contiguous() for k in range(HORIZON)]
    return actions, steps


def world_batch(g, pad_x):
    raw = world_of(
        np.asarray([float(g)], np.float64),
        np.asarray([float(pad_x)], np.float64),
        np.asarray([float(GOAL_VY)], np.float64),
    )
    if tuple(raw.shape) != (1, 4):
        raise RuntimeError(raw.shape)
    if not np.allclose(raw[0, 3], float(GOAL_VY) / float(SCALE_VEL)):
        raise RuntimeError(raw[0])
    tensor = torch.from_numpy(np.ascontiguousarray(raw)).expand(N_SEQ, -1).contiguous()
    if tensor.shape != (N_SEQ, 4):
        raise RuntimeError(tensor.shape)
    return tensor


def goal_z(ema, pad_x, fuel, cache):
    key = float(fuel)
    hit = cache.get(key)
    if hit is not None:
        return hit
    raw = terminal_state(pad_x, key)
    scaled = scale_rel(raw, np.asarray([float(pad_x)], np.float64))
    if scaled.shape != (1, 7):
        raise RuntimeError(scaled.shape)
    latent = ema(torch.from_numpy(np.ascontiguousarray(scaled)))
    if latent.shape != (1, LATENT):
        raise RuntimeError(latent.shape)
    cache[key] = latent
    return latent


def first_action(predictor, z, z_goal, world_b, step_pairs, actions):
    """Три шага. При равной стоимости argmin берёт меньший индекс."""
    if z.shape != (1, LATENT) or z_goal.shape != (1, LATENT):
        raise RuntimeError((tuple(z.shape), tuple(z_goal.shape)))
    if world_b.shape != (N_SEQ, int(N_WORLD)) or len(step_pairs) != HORIZON:
        raise RuntimeError((tuple(world_b.shape), len(step_pairs)))
    current = z.expand(N_SEQ, -1)
    cost = None
    for k, weight in enumerate(LOSS_WEIGHTS):
        current = predictor(current, step_pairs[k], world_b)
        step = (current - z_goal).abs().mean(dim=-1)
        cost = step.mul(float(weight)) if cost is None else cost.add(step, alpha=float(weight))
    index = int(torch.argmin(cost).item())
    return int(actions[index, 0].item())


@torch.no_grad()
def fly_one(predictor, encoder, ema, actions, step_pairs, world_b, row):
    """Один старт сетки. no_grad, не inference_mode: encode_rel зовёт numpy."""
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    fuel = float(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    grounded = False
    steps = 0
    cache = {}
    pads = np.asarray([pad_x], np.float64)
    while steps < int(HOVER_CAP) and not grounded:
        state = np.asarray([[x, y, vx, vy, fuel]], np.float64)
        z = torch.from_numpy(np.ascontiguousarray(encode_rel(encoder, state, pads)))
        action = first_action(
            predictor, z, goal_z(ema, pad_x, fuel, cache), world_b, step_pairs, actions)
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, int(action), g)
        steps += 1
    result = outcome_at(x, y, vy, grounded, pad_x)
    if result not in ("мягкая", "удар", "не сел"):
        raise RuntimeError(result)
    return result, int(steps)


@torch.no_grad()
def fly_grid(name, predictor, encoder, ema):
    actions, step_pairs = pair_table()
    predictor.eval()
    encoder.eval()
    ema.eval()
    buckets = {}
    order = []
    worlds = {}
    soft = crash = miss = 0
    current = None
    for row in grid():
        key = (row["g"], row["pad_x"], row["height"])
        if key != current:
            if current is not None:
                if buckets[current]["n"] != 9:
                    raise RuntimeError((current, buckets[current]["n"]))
                _emit(name, current, buckets[current])
            current = key
            if key not in buckets:
                order.append(key)
                buckets[key] = {"soft": 0, "crash": 0, "miss": 0, "n": 0, "steps": 0}
        world_key = (row["g"], row["pad_x"])
        if world_key not in worlds:
            worlds[world_key] = world_batch(row["g"], row["pad_x"])
        result, n_steps = fly_one(
            predictor, encoder, ema, actions, step_pairs, worlds[world_key], row)
        cell = buckets[key]
        cell["n"] += 1
        cell["steps"] += int(n_steps)
        if result == "мягкая":
            cell["soft"] += 1
            soft += 1
        elif result == "удар":
            cell["crash"] += 1
            crash += 1
        else:
            cell["miss"] += 1
            miss += 1
    if current is not None:
        if buckets[current]["n"] != 9:
            raise RuntimeError((current, buckets[current]["n"]))
        _emit(name, current, buckets[current])
    cells = []
    for key in order:
        g, pad, height = key
        cell = buckets[key]
        if cell["n"] != 9 or cell["soft"] + cell["crash"] + cell["miss"] != 9:
            raise RuntimeError((key, cell))
        cells.append({
            "g": float(g),
            "pad_x": float(pad),
            "height": float(height),
            "n": 9,
            "soft": int(cell["soft"]),
            "crash": int(cell["crash"]),
            "miss": int(cell["miss"]),
            "steps_mean": round(float(cell["steps"]) / 9.0, 3),
        })
    if soft + crash + miss != GRID_N or len(cells) != 18:
        raise RuntimeError((soft, crash, miss, len(cells)))
    return {
        "name": name,
        "soft": int(soft),
        "crash": int(crash),
        "miss": int(miss),
        "of": int(GRID_N),
        "cells": cells,
    }


def _emit(name, key, cell) -> None:
    g, pad, height = key
    print(
        f"{name} g={g:g} pad={pad:g} h={height:g} "
        f"soft={cell['soft']} crash={cell['crash']} miss={cell['miss']} "
        f"of={cell['n']}",
        flush=True,
    )

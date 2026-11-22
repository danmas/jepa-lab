"""Навыки наведения и посадки на замороженном семиканальном кодировщике.

Предиктор новый. Терминальная цель считается через ``scale_rel``: ``goal_loss``
делит состояние на пять каналов и для этого кодировщика не подходит.
Действие замкнутого контура навык не выбирает.
"""
import numpy as np
import torch
import torch.nn.functional as F

from numeric_land.goal import target_state, world_vector
from numeric_land.models import LATENT, action_pairs, collapse_penalty
from numeric_land.relenc import PROBE_RATIO, scale_rel
from numeric_land.train_goal import COLLAPSE_WEIGHT, LOSS_WEIGHTS, W_TERM

SKILL_RATIO = PROBE_RATIO
N_WORLD = 4
PHASES = ("approach", "land", "shared")


def skill_flag(skill_l1: float, shared_l1: float) -> bool:
    """Истина, только если L1 навыка seed 0 не выше 0.9 от общего и общий L1 больше нуля."""
    if float(shared_l1) <= 0.0:
        return False
    return float(skill_l1) <= SKILL_RATIO * float(shared_l1)


def _goal_batch(pads, vys, height0) -> np.ndarray:
    pads = np.asarray(pads, np.float64)
    vys = np.asarray(vys, np.float64)
    height0 = np.asarray(height0, np.float64)
    rows = []
    for i in range(len(pads)):
        raw = target_state(pads[i], vys[i], height0[i])
        rows.append(scale_rel(raw, np.asarray([pads[i]], np.float64))[0])
    return np.asarray(rows, np.float32)


def skill_loss(encoder, predictor, ema, states, actions, world, pads, vys, height0):
    """Тот же состав, что у ``goal_loss``: три шага L1, штраф коллапса, терминал 0.25.

    Кодировщик на входе уже семиканальный. Градиент в него не идёт.
    """
    if float(W_TERM) != 0.25:
        raise RuntimeError(W_TERM)
    b, t, dim = states.shape
    flat = encoder(states.reshape(b * t, dim)).detach()
    penalty = collapse_penalty(flat)
    z = flat.view(b, t, LATENT)[:, 0]
    pred_loss = states.new_zeros(())
    parts = {}
    world_step = world if predictor.n_world else None
    for k, weight in enumerate(LOSS_WEIGHTS):
        z = predictor(z, action_pairs(actions[:, k], z), world_step)
        with torch.no_grad():
            target = ema(states[:, k + 1])
        step = F.l1_loss(z, target)
        pred_loss = pred_loss + weight * step
        parts[f"l{k + 1}"] = float(step.detach())
    z_h = predictor(z, action_pairs(actions[:, -1], z), world_step)
    goal = torch.from_numpy(_goal_batch(pads, vys, height0))
    with torch.no_grad():
        z_goal = ema(goal)
    term = F.l1_loss(z_h, z_goal)
    total = pred_loss + COLLAPSE_WEIGHT * penalty + W_TERM * term
    parts["penalty"] = float(penalty.detach())
    parts["term"] = float(term.detach())
    return total, parts


def world_of(gs, pads, vys) -> np.ndarray:
    """Четвёрка (g/9, 1, pad_x/512, desired_vy/120). Флага режима в ней нет."""
    return world_vector(gs, pads, vys)

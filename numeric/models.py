"""Две маленькие сети: JEPA в латенте 32 и прямой предиктор состояния.

Ширина 64 у предиктора та же, что у скрытого слоя кодировщика: в спеке ширина
предиктора не названа отдельно. Активация SiLU, как у пиксельной лаборатории.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from numeric.data import ACTION_PAIR, N_STATE

LATENT = 32
HIDDEN = 64
LOSS_WEIGHTS = (1.0, 0.5, 0.25)
COLLAPSE_WEIGHT = 0.1
COLLAPSE_STD = 0.1
# Чтобы градиент std существовал, когда координата схлопнулась в константу.
_STD_EPS = 1e-8


def action_pairs(actions: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Индекс 0…8 -> (dx, dy) из {-1, 0, +1}. Обучаемого эмбеддинга нет."""
    table = torch.as_tensor(ACTION_PAIR, device=like.device, dtype=like.dtype)
    return table[actions.long()]


class NumericEncoder(nn.Module):
    """4 -> 64 -> 64 -> 32, выход на единичной сфере."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATE, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, LATENT),
        )

    def forward(self, state):
        return F.normalize(self.net(state), dim=-1)


class NumericPredictor(nn.Module):
    """Латент и (dx, dy) -> приращение латента, сумма снова на сфере."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT + 2, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, LATENT),
        )

    def forward(self, z, pair):
        delta = self.net(torch.cat([z, pair], dim=-1))
        return F.normalize(z + delta, dim=-1)


class DirectPredictor(nn.Module):
    """Масштабированное состояние и (dx, dy) -> приращение следующего состояния.

    Без сферы и без EMA.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATE + 2, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, N_STATE),
        )

    def forward(self, state, pair):
        return state + self.net(torch.cat([state, pair], dim=-1))


def collapse_penalty(z: torch.Tensor) -> torch.Tensor:
    """Среднее по координатам relu(0.1 - std батча)^2. z: (N, 32)."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + _STD_EPS)
    return torch.relu(COLLAPSE_STD - std).square().mean()


def jepa_loss(encoder, predictor, ema, states, actions):
    """Открытые три шага, веса 1, 0.5, 0.25, плюс штраф схлопывания.

    states: (B, 6, 4) уже масштабированы. actions: (B, 5) индексы.
    На вход шага k+1 идёт предсказание, не кодировка истинного s_k.
    Цель шага — ema(s_k) со stop-grad. Штраф считается по кодировкам
    истинных состояний окна, не по предсказаниям.
    """
    b, t, _ = states.shape
    flat = encoder(states.reshape(b * t, N_STATE))
    penalty = collapse_penalty(flat)
    z = flat.view(b, t, LATENT)[:, 0]
    pred_loss = states.new_zeros(())
    parts = {}
    for k, weight in enumerate(LOSS_WEIGHTS):
        z = predictor(z, action_pairs(actions[:, k], z))
        with torch.no_grad():
            target = ema(states[:, k + 1])
        step = F.l1_loss(z, target)
        pred_loss = pred_loss + weight * step
        parts[f"l{k + 1}"] = float(step.detach())
    total = pred_loss + COLLAPSE_WEIGHT * penalty
    parts["penalty"] = float(penalty.detach())
    return total, parts


def direct_loss(predictor, states, actions):
    """Тот же открытый трёхшаговый L1 в масштабированных числах. Штрафа сферы нет."""
    z = states[:, 0]
    total = states.new_zeros(())
    parts = {}
    for k, weight in enumerate(LOSS_WEIGHTS):
        z = predictor(z, action_pairs(actions[:, k], z))
        step = F.l1_loss(z, states[:, k + 1])
        total = total + weight * step
        parts[f"l{k + 1}"] = float(step.detach())
    return total, parts

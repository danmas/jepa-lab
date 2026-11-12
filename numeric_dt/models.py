"""JEPA: кодировщик состояния с dt или без него, предиктор приращения латента.

Предиктор получает латент и (dx, dy). dt отдельным входом не подаётся.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from numeric_dt.data import PAIRS

LATENT = 32
HIDDEN = 64
LOSS_WEIGHTS = (1.0, 0.5, 0.25)
COLLAPSE_WEIGHT = 0.1
COLLAPSE_STD = 0.1
_STD_EPS = 1e-8
N_IN = {"with-dt": 5, "no-dt": 4}


def action_pairs(actions: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    table = torch.as_tensor(PAIRS, device=like.device, dtype=like.dtype)
    return table[actions.long()]


class DtEncoder(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = int(n_in)
        self.net = nn.Sequential(
            nn.Linear(self.n_in, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, LATENT),
        )

    def forward(self, state):
        return F.normalize(self.net(state), dim=-1)


class DtPredictor(nn.Module):
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


def collapse_penalty(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + _STD_EPS)
    return torch.relu(COLLAPSE_STD - std).square().mean()


def jepa_loss(encoder, predictor, ema, states, actions):
    """Открытые три шага. states уже масштабированы: (B, 6, D)."""
    b, t, dim = states.shape
    flat = encoder(states.reshape(b * t, dim))
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

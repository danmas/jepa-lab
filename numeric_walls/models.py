"""Кодировщик только мяча. Стенки конкатенируются к видимому предиктору."""
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
N_WALL = {"vis": 4, "hid": 0}


def action_pairs(actions: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    table = torch.as_tensor(PAIRS, device=like.device, dtype=like.dtype)
    return table[actions.long()]


class BallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, LATENT),
        )

    def forward(self, state):
        return F.normalize(self.net(state), dim=-1)


class WallPredictor(nn.Module):
    def __init__(self, n_wall: int):
        super().__init__()
        self.n_wall = int(n_wall)
        self.net = nn.Sequential(
            nn.Linear(LATENT + 2 + self.n_wall, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, LATENT),
        )

    def forward(self, z, pair, walls=None):
        parts = [z, pair]
        if self.n_wall:
            parts.append(walls)
        delta = self.net(torch.cat(parts, dim=-1))
        return F.normalize(z + delta, dim=-1)


def predictor_input(z, pair, walls, variant: str) -> torch.Tensor:
    if variant == "vis":
        return torch.cat([z, pair, walls], dim=-1)
    if variant == "hid":
        return torch.cat([z, pair], dim=-1)
    raise ValueError(variant)


def collapse_penalty(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + _STD_EPS)
    return torch.relu(COLLAPSE_STD - std).square().mean()


def jepa_loss(encoder, predictor, ema, states, actions, walls):
    """states уже масштабированы: (B, 6, 5). walls: (B, 4) или None у скрытого."""
    b, t, dim = states.shape
    flat = encoder(states.reshape(b * t, dim))
    penalty = collapse_penalty(flat)
    z = flat.view(b, t, LATENT)[:, 0]
    pred_loss = states.new_zeros(())
    parts = {}
    wall_step = walls if predictor.n_wall else None
    for k, weight in enumerate(LOSS_WEIGHTS):
        z = predictor(z, action_pairs(actions[:, k], z), wall_step)
        with torch.no_grad():
            target = ema(states[:, k + 1])
        step = F.l1_loss(z, target)
        pred_loss = pred_loss + weight * step
        parts[f"l{k + 1}"] = float(step.detach())
    total = pred_loss + COLLAPSE_WEIGHT * penalty
    parts["penalty"] = float(penalty.detach())
    return total, parts

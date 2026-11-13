"""Кодировщик посадки. Тяжесть конкатенируется только к видимому предиктору."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from numeric_land.data import PAIRS

LATENT = 32
HIDDEN = 64
LOSS_WEIGHTS = (1.0, 0.5, 0.25)
COLLAPSE_WEIGHT = 0.1
COLLAPSE_STD = 0.1
_STD_EPS = 1e-8
N_WORLD = {"vis": 2, "hid": 0}
N_OUT = 5


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


class LandPredictor(nn.Module):
    def __init__(self, n_world: int):
        super().__init__()
        self.n_world = int(n_world)
        self.net = nn.Sequential(
            nn.Linear(LATENT + 2 + self.n_world, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, LATENT),
        )

    def forward(self, z, pair, world=None):
        parts = [z, pair]
        if self.n_world:
            parts.append(world)
        delta = self.net(torch.cat(parts, dim=-1))
        return F.normalize(z + delta, dim=-1)


def predictor_input(z, pair, world, variant: str) -> torch.Tensor:
    if variant == "vis":
        return torch.cat([z, pair, world], dim=-1)
    if variant == "hid":
        return torch.cat([z, pair], dim=-1)
    raise ValueError(variant)


class Readout(nn.Module):
    """Один латент → (x, y, vx, vy, fuel). Это не корневой Probe на четыре числа."""

    def __init__(self, latent: int = LATENT, n_out: int = N_OUT):
        super().__init__()
        self.fc = nn.Linear(int(latent), int(n_out))

    def forward(self, z):
        return self.fc(z)


def fit_readout(z: torch.Tensor, y: torch.Tensor, lam: float = 1e-2) -> Readout:
    """Ridge со свободным членом. y — пять физических чисел, не доли."""
    z = z.double()
    y = y.double()
    ones = torch.ones(z.shape[0], 1, dtype=torch.float64)
    za = torch.cat([z, ones], dim=1)
    eye = torch.eye(za.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(za.T @ za + lam * eye, za.T @ y)
    head = Readout(z.shape[1], y.shape[1])
    with torch.no_grad():
        head.fc.weight.copy_(w[:-1].T.float())
        head.fc.bias.copy_(w[-1].float())
    return head


def collapse_penalty(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + _STD_EPS)
    return torch.relu(COLLAPSE_STD - std).square().mean()


def jepa_loss(encoder, predictor, ema, states, actions, world, freeze_encoder=False):
    """states: (B, 4, 5) уже поделены. actions: (B, 3). world у скрытого — None."""
    b, t, dim = states.shape
    flat = encoder(states.reshape(b * t, dim))
    if freeze_encoder:
        flat = flat.detach()
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
    total = pred_loss + COLLAPSE_WEIGHT * penalty
    parts["penalty"] = float(penalty.detach())
    return total, parts

"""Модели: энкодер, декодер, предиктор (с действием), EMA-цель, probe, VICReg-штраф.

У JEPA-модели и у пиксельного бейзлайна ОДНА И ТА ЖЕ архитектура. Отличается только
то, что предсказывается:

  JEPA  -> предиктор угадывает эмбеддинг следующего кадра   (loss в латенте)
  PIXEL -> предиктор угадывает сам кадр следующего шага     (loss в пикселях)

Предиктор опционально принимает действие (`n_actions > 0`) — так модель становится
action-conditioned world model, по которому можно планировать.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 64
LATENT = 48


def default_action_table(n_actions: int = 9) -> torch.Tensor:
    """9 действий -> нормированные направления (dx, dy) ∈ {-1,0,1}².

    Индекс = (dy+1)*3 + (dx+1), как в data.ACTIONS. Геометрия вместо эмбеддинга:
    сеть сразу видит «куда толкнули», а не учит 9 абстрактных векторов с нуля.
    """
    idx = torch.arange(n_actions)
    dy = torch.div(idx, 3, rounding_mode="floor") - 1
    dx = idx % 3 - 1
    return torch.stack([dx, dy], dim=1).float()


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    """64x64 -> вектор латента (LATENT).

    unit_norm=True нормирует выход на единичную сферу (стиль BYOL/VICReg). Без этого
    латент «раздувается»: норма растёт до сотен, MSE выглядит большой, а относительная
    ошибка маленькой, и стоимость плана перестаёт быть интерпретируемой.
    """

    def __init__(self, latent: int = LATENT, unit_norm: bool = True):
        super().__init__()
        self.unit_norm = unit_norm
        self.body = nn.Sequential(
            conv_block(1, 32),    # 32x32
            conv_block(32, 64),   # 16x16
            conv_block(64, 96),   # 8x8
            conv_block(96, 96),   # 4x4
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(96 * 4 * 4, latent))

    def forward(self, x):
        z = self.head(self.body(x))
        return F.normalize(z, dim=1) if self.unit_norm else z


class Decoder(nn.Module):
    """Вектор латента -> кадр 64x64. Нужен для визуализации предсказаний."""

    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, 96 * 4 * 4)
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(96, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.body(self.fc(z).view(-1, 96, 4, 4))


class Predictor(nn.Module):
    """(z_{t-1}, z_t[, a]) -> z_{t+1}.

    n_actions=0 — версия без управления (совместимость с v0.2).
    unit_norm=True — выход тоже на единичной сфере, чтобы MSE был ограничен [0,4]
    и стоимость плана читалась как «косинусное расстояние».
    """

    def __init__(self, latent: int = LATENT, hidden: int = 512,
                 n_actions: int = 0, act_dim: int = 16, unit_norm: bool = True):
        super().__init__()
        self.n_actions = n_actions
        self.unit_norm = unit_norm
        self.net = nn.Sequential(
            nn.Linear(2 * latent + (2 if n_actions else 0), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent),
        )
        if n_actions:
            # Действие подаётся ГЕОМЕТРИЧЕСКИ (нормированный импульс), а не как обучаемый
            # эмбеддинг: так сеть сразу видит «куда и насколько толкнули» и не может
            # проигнорировать действие — оно входит и в контекст, и в аддитивный путь.
            self.register_buffer("act_vec", default_action_table(n_actions), persistent=False)
            self.act_net = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(),
                                         nn.Linear(hidden, latent))

    def forward(self, z_prev, z_cur, action=None):
        # предсказываем ПРИРАЩЕНИЕ к текущему латенту: «продолжение движения» достаётся
        # бесплатно, и сети остаётся выучить только сдвиг, который вносит действие
        if self.n_actions:
            if action is None:
                action = torch.full((z_prev.shape[0],), 4, dtype=torch.long, device=z_prev.device)
            a_vec = self.act_vec[action]                       # (B,2), значения ±IMPULSE/4
            delta = self.net(torch.cat([z_prev, z_cur, a_vec], dim=1)) + self.act_net(a_vec)
        else:
            delta = self.net(torch.cat([z_prev, z_cur], dim=1))
        z = z_cur + delta
        return F.normalize(z, dim=1) if self.unit_norm else z


class WorldModel(nn.Module):
    def __init__(self, latent: int = LATENT, n_actions: int = 0, unit_norm: bool = True):
        super().__init__()
        self.encoder = Encoder(latent, unit_norm=unit_norm)
        self.decoder = Decoder(latent)
        self.predictor = Predictor(latent, n_actions=n_actions, unit_norm=unit_norm)

    def params_count(self):
        return sum(p.numel() for p in self.parameters())


def load_model(path, threads: int = 8):
    """Загружает чекпоинт, восстанавливая флаги архитектуры из его args."""
    torch.set_num_threads(threads)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {}) or {}
    model = WorldModel(n_actions=ck.get("n_actions", a.get("n_actions", 0)),
                       unit_norm=a.get("unit_norm", False))
    model.load_state_dict(ck["model"])
    model.eval()
    ema = EMATarget(model.encoder)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    return model, ema, ck


class EMATarget(nn.Module):
    """Копия энкодера, обновляемая скользящим средним (BYOL/JEPA-приём).

    Нужна, чтобы цель предсказания не «убегала» вместе с энкодером.
    """

    def __init__(self, encoder: nn.Module, momentum: float = 0.99):
        super().__init__()
        self.module = copy.deepcopy(encoder)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, encoder: nn.Module):
        for tp, p in zip(self.module.parameters(), encoder.parameters()):
            tp.mul_(self.momentum).add_(p.detach(), alpha=1 - self.momentum)
        for tb, b in zip(self.module.buffers(), encoder.buffers()):
            tb.copy_(b)

    def forward(self, x):
        return self.module(x)


class Probe(nn.Module):
    """Пара латентов (z_{t-1}, z_t) -> физическое состояние (x, y, vx, vy).

    Пара, а не один латент: в одном кадре скорости физически нет — мяч симметричен,
    поэтому скорость читается только из двух последовательных латентов.
    Обучается ridge, не градиентом.
    """

    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc = nn.Linear(2 * latent, 4)

    @torch.no_grad()
    def forward(self, z_prev, z_cur):
        return self.fc(torch.cat([z_prev, z_cur], dim=1))


def fit_ridge(z: torch.Tensor, y: torch.Tensor, lam: float = 1.0) -> Probe:
    """Закрытая форма ridge-регрессии: W = (ZᵀZ + λI)⁻¹ ZᵀY, со свободным членом."""
    z = z.double()
    y = y.double()
    ones = torch.ones(z.shape[0], 1, dtype=torch.float64)
    za = torch.cat([z, ones], dim=1)
    a = za.T @ za + lam * torch.eye(za.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(a, za.T @ y)          # (latent+1, 4)
    probe = Probe(z.shape[1] // 2)
    with torch.no_grad():
        probe.fc.weight.copy_(w[:-1].T.float())
        probe.fc.bias.copy_(w[-1].float())
    return probe


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> list:
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return [round(float(1 - r / t), 4) for r, t in zip(ss_res, ss_tot)]


def variance_term(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg-штраф: требует, чтобы каждая размерность латента «дышала».

    Страховка от коллапса (когда энкодер выдаёт константу и loss = 0).
    """
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()

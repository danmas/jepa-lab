"""Синтетический мир: мяч под гравитацией + управление импульсами, кадры 64x64.

Физика: каждый кадр к скорости добавляется действие-импульс (9 вариантов:
-dx,-dy ∈ {-1,0,+1}), затем гравитация, затем упругие отскоки от стен/пола/потолка.
Модель должна по двум кадрам восстановить состояние (позиция + скорость),
научиться предсказывать, как его меняет действие, и планировать достижение цели.

Данные НЕ хранятся на диске: генерируются на лету по сиду.
"""
import numpy as np

SIZE = 64
RADIUS = 5.0
GRAVITY = 0.15          # слабая гравитация: движение задаёт действие, а не инерция
SIGMA = 2.8
IMPULSE = 4.0          # сила действия, px/кадр — доминирует над собственной скоростью,
                       # иначе предиктору выгодно игнорировать действие
VEL_DECAY = 0.6        # инерция затухает: скорость = 0.6·v + действие
ZONE_HALF = 8          # половина стороны целевой зоны (16x16)

# 9 действий: индекс = (dy+1)*3 + (dx+1)
ACTIONS = np.array([[(dx) * IMPULSE, (dy) * IMPULSE]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.float32)
N_ACTIONS = len(ACTIONS)
NOOP = 4               # индекс действия (0, 0)

ACTION_NAMES = ["влево-вверх", "вверх", "вправо-вверх",
                "влево", "на месте", "вправо",
                "влево-вниз", "вниз", "вправо-вниз"]

# 9 целевых зон 3x3
ZONE_CENTERS = [(14, 14), (32, 14), (50, 14),
                (14, 32), (32, 32), (50, 32),
                (14, 50), (32, 50), (50, 50)]

ZONES = [
    {"name": "верхний левый", "syn": ["левый верхний", "слева сверху", "верх-лево"], "center": (14, 14)},
    {"name": "верхний центр", "syn": ["верх", "сверху", "по центру сверху"], "center": (32, 14)},
    {"name": "верхний правый", "syn": ["правый верхний", "справа сверху", "верх-право"], "center": (50, 14)},
    {"name": "средний левый", "syn": ["слева", "левый центр"], "center": (14, 32)},
    {"name": "центр", "syn": ["посередине", "в середине", "центральная зона"], "center": (32, 32)},
    {"name": "средний правый", "syn": ["справа", "правый центр"], "center": (50, 32)},
    {"name": "нижний левый", "syn": ["левый нижний", "слева снизу", "низ-лево"], "center": (14, 50)},
    {"name": "нижний центр", "syn": ["низ", "снизу", "по центру снизу"], "center": (32, 50)},
    {"name": "нижний правый", "syn": ["правый нижний", "справа снизу", "низ-право"], "center": (50, 50)},
]


def simulate(batch: int, horizon: int, rng: np.random.Generator, actions=None,
             size: int = SIZE, r: float = RADIUS, g: float = GRAVITY) -> np.ndarray:
    """-> позиции мяча (B, T, 2) float32.

    actions: индексы действий (B, T-1) — действие i переводит кадр i в кадр i+1.
             None означает «без управления».
    """
    pos = rng.uniform(r + 2, size - 1 - r - 2, size=(batch, 2)).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, size=batch).astype(np.float32)
    spd = rng.uniform(0.3, 1.0, size=batch).astype(np.float32)
    vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], axis=1)

    out = np.empty((batch, horizon, 2), np.float32)
    lo, hi = r, size - 1 - r
    for t in range(horizon):
        if actions is not None and t > 0:
            vel = VEL_DECAY * vel + ACTIONS[actions[:, t - 1]]
        vel[:, 1] += g
        pos = pos + vel
        for ax in range(2):
            m = pos[:, ax] < lo
            pos[m, ax] = lo + (lo - pos[m, ax]); vel[m, ax] *= -1
            m = pos[:, ax] > hi
            pos[m, ax] = hi - (pos[m, ax] - hi); vel[m, ax] *= -1
        out[:, t] = pos
    return out


def render(positions: np.ndarray, size: int = SIZE, sigma: float = SIGMA) -> np.ndarray:
    """Позиции (...,2) -> кадры (...,size,size) в диапазоне [-1,1]."""
    flat = positions.reshape(-1, 2)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d2 = (xx[None] - flat[:, 0, None, None]) ** 2 + (yy[None] - flat[:, 1, None, None]) ** 2
    img = np.exp(-d2 / (2 * sigma ** 2))          # яркое пятно, мягкие края
    img = img * 2.0 - 1.0                          # [0,1] -> [-1,1]
    return img.reshape(*positions.shape[:-1], size, size)


def make_batch(batch: int, rng: np.random.Generator, horizon: int = 6):
    """-> (кадры (B,H,1,64,64), действия (B,H-1), позиции (B,H,2)).

    Действия случайные: модель учит динамику мира под случайную политику.
    """
    acts = rng.integers(0, N_ACTIONS, size=(batch, horizon - 1)).astype(np.int64)
    pos = simulate(batch, horizon, rng, acts)
    frames = render(pos)[:, :, None, :, :].astype(np.float32)
    return frames, acts, pos


def make_episode(length: int, seed: int):
    """Один эпизод без управления: -> (кадры (T,1,64,64), позиции (T,2))."""
    rng = np.random.default_rng(seed)
    pos = simulate(1, length, rng)[0]
    return render(pos)[:, None, :, :].astype(np.float32), pos


def zone_frame(zone_index: int, size: int = SIZE) -> np.ndarray:
    """Кадр-цель: мяч в центре зоны, форма (1,1,64,64).
    Маркер зоны рисуется только при визуализации."""
    cx, cy = ZONE_CENTERS[zone_index]
    return render(np.array([[cx, cy]], np.float32))[None].astype(np.float32)


def in_zone(pos, zone_index: int, half: int = ZONE_HALF) -> bool:
    cx, cy = ZONE_CENTERS[zone_index]
    return abs(float(pos[0]) - cx) <= half and abs(float(pos[1]) - cy) <= half


def find_zone(text: str):
    """Слова -> индекс зоны. Побеждает САМОЕ ДЛИННОЕ совпадение.

    Иначе «верхний правый» ловится синонимом «верх» у зоны «верхний центр».
    """
    t = text.lower().replace("ё", "е").strip()
    best, best_len = None, 0
    for i, z in enumerate(ZONES):
        for key in [z["name"], *z["syn"]]:
            if key in t and len(key) > best_len:
                best, best_len = i, len(key)
    return best


def step_state(pos, vel, action: int, size: int = SIZE, r: float = RADIUS,
               g: float = GRAVITY):
    """Один шаг физики из явного состояния -> (pos, vel). Действие — индекс 0..8."""
    vel = VEL_DECAY * np.array(vel, np.float32) + ACTIONS[action]
    vel[1] += g
    pos = np.array(pos, np.float32) + vel
    lo, hi = r, size - 1 - r
    for ax in range(2):
        if pos[ax] < lo:
            pos[ax] = lo + (lo - pos[ax]); vel[ax] *= -1
        if pos[ax] > hi:
            pos[ax] = hi - (pos[ax] - hi); vel[ax] *= -1
    return pos, vel


class World:
    """Пошаговый мир для исполнения плана: держит состояние и применяет действия."""

    def __init__(self, seed: int, size: int = SIZE):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.pos = self.rng.uniform(RADIUS + 2, size - 1 - RADIUS - 2, size=2).astype(np.float32)
        ang = self.rng.uniform(0, 2 * np.pi)
        spd = self.rng.uniform(0.3, 1.0)
        self.vel = np.array([np.cos(ang) * spd, np.sin(ang) * spd], np.float32)

    def frame(self) -> np.ndarray:
        return render(self.pos.reshape(1, 2))[0][None, :, :].astype(np.float32)

    def step(self, action: int) -> np.ndarray:
        self.pos, self.vel = step_state(self.pos, self.vel, action, self.size)
        return self.frame()

    def state(self):
        return self.pos.copy(), self.vel.copy()

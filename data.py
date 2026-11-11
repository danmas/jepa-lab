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

# Режимы мира. flat — нынешняя физика, остальные не смешиваются в одном обучении.
WORLDS = ("flat", "jitter", "noise", "balls", "floor")
NOISE_SIGMA = 0.1
SLIP = 0.25
BUMP_X = (20.0, 44.0)
BUMP_H = 4.0
BUMP_HALF = 8.0
FLAT_FLOOR = SIZE - 1 - RADIUS   # 58: нижний предел центра мяча на ровном полу

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


def make_batch(batch: int, rng: np.random.Generator, horizon: int = 6, world: str = "flat"):
    """-> (кадры (B,H,1,64,64), действия (B,H-1), позиции (B,H,2)).

    Действия случайные: модель учит динамику мира под случайную политику.
    В jitter в физику уходит подменённый толчок, а в лосс — выбранный:
    по выбранному действию будущее не одно. Позиции — всегда управляемый мяч.
    """
    if world not in WORLDS:
        raise ValueError(world)
    acts = rng.integers(0, N_ACTIONS, size=(batch, horizon - 1)).astype(np.int64)
    if world == "flat":
        pos = simulate(batch, horizon, rng, acts)
        frames = render(pos)[:, :, None, :, :].astype(np.float32)
        return frames, acts, pos
    if world == "jitter":
        executed = slip_actions(acts, rng)
        pos = simulate(batch, horizon, rng, executed)
        frames = render(pos)[:, :, None, :, :].astype(np.float32)
        return frames, acts, pos
    if world == "noise":
        pos = simulate(batch, horizon, rng, acts)
        frames = render(pos)[:, :, None, :, :].astype(np.float32)
        return add_obs_noise(frames, rng), acts, pos
    if world == "balls":
        pos, pos2 = simulate_balls(batch, horizon, rng, acts)
        frames = render_pair(pos, pos2)[:, :, None, :, :].astype(np.float32)
        return frames, acts, pos
    pos = simulate_floor(batch, horizon, rng, acts)
    frames = render(pos)[:, :, None, :, :].astype(np.float32)
    return paint_floor(frames), acts, pos


def make_episode(length: int, seed: int):
    """Один эпизод без управления: -> (кадры (T,1,64,64), позиции (T,2))."""
    rng = np.random.default_rng(seed)
    pos = simulate(1, length, rng)[0]
    return render(pos)[:, None, :, :].astype(np.float32), pos


def zone_frame(zone_index: int, size: int = SIZE, world: str = "flat",
               bump_h: float = BUMP_H, centers=None) -> np.ndarray:
    """Кадр-цель: мяч в центре зоны, форма (1,1,64,64).

    Второй мяч в цель не входит. В режиме floor на кадре есть тот же пол, что в шаге.
    Маркер зоны рисуется только при визуализации.
    """
    cx, cy = ZONE_CENTERS[zone_index]
    img = render(np.array([[cx, cy]], np.float32))[None].astype(np.float32)
    if world == "floor":
        img = paint_floor(img, height=bump_h, centers=centers)
    return img


def goal_frame(zone_index: int, world: str = "flat", sigma: float = NOISE_SIGMA,
               bump_h: float = BUMP_H, centers=None) -> np.ndarray:
    """Кадр цели, который видит модель. Шум цели фиксирован сидом 1000+зона."""
    if world == "noise":
        rng = np.random.default_rng(1000 + int(zone_index))
        return add_obs_noise(zone_frame(zone_index), rng, sigma)
    if world == "floor":
        return zone_frame(zone_index, world="floor", bump_h=bump_h, centers=centers)
    return zone_frame(zone_index)


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


def slip_actions(actions: np.ndarray, rng: np.random.Generator, slip: float = SLIP) -> np.ndarray:
    """С вероятностью slip заменить индекс толчка равномерным из девяти."""
    out = np.array(actions, copy=True)
    mask = rng.random(out.shape) < slip
    n = int(np.count_nonzero(mask))
    if n:
        out[mask] = rng.integers(0, N_ACTIONS, size=n)
    return out.astype(np.int64)


def add_obs_noise(frames: np.ndarray, rng: np.random.Generator, sigma: float = NOISE_SIGMA) -> np.ndarray:
    """Гауссов шум только на кадре. Физику не двигает."""
    noise = rng.normal(0.0, float(sigma), size=frames.shape).astype(np.float32)
    return np.clip(frames.astype(np.float32) + noise, -1.0, 1.0).astype(np.float32)


def bump_centers(n: int) -> tuple:
    """Центры холмов по числу. Два холма — обученная пара, не равномерная сетка."""
    n = int(n)
    if n == 0:
        return ()
    if n == 1:
        return (32.0,)
    if n == 2:
        return (float(BUMP_X[0]), float(BUMP_X[1]))
    if n not in (3, 4, 5):
        raise ValueError(f"число холмов {n} вне 0…5")
    return tuple(8.0 + i * (56.0 - 8.0) / (n - 1) for i in range(n))


def gravity_of(g_steps: int) -> float:
    """Шаг ползунка 0.05. Три шага — ровно обученная тяжесть, не 3*0.05 в двоичном виде."""
    if int(g_steps) == 3:
        return float(GRAVITY)
    return float(g_steps) * 0.05


def bump_height(x, height: float = BUMP_H, half: float = BUMP_HALF, centers=None) -> float:
    """Высота треугольных бугров в точке x. Ноль вне скатов. Наложение берёт большую."""
    best = 0.0
    xf = float(x)
    xs = BUMP_X if centers is None else centers
    for cx in xs:
        d = abs(xf - cx)
        if d < half:
            best = max(best, float(height) * (1.0 - d / half))
    return best


def bump_slope(x, height: float = BUMP_H, half: float = BUMP_HALF, centers=None) -> float:
    """dh/dx активного ската. На ровном полу и на самом гребне — 0."""
    best_h, best_s = 0.0, 0.0
    xf = float(x)
    xs = BUMP_X if centers is None else centers
    for cx in xs:
        dx = xf - cx
        d = abs(dx)
        if d < half:
            h = float(height) * (1.0 - d / half)
            slope = 0.0 if d < 1e-9 else ((height / half) if dx < 0 else -(height / half))
            if h >= best_h:
                best_h, best_s = h, float(slope)
    return best_s


def floor_limit(x, height: float = BUMP_H, centers=None) -> float:
    """Нижний предел центра мяча: больше y на картинке — ниже. Ровный пол = 58."""
    return float(FLAT_FLOOR) - bump_height(x, height, centers=centers)


def zone_floor_margin(height: float = BUMP_H, samples: int = 65):
    """Минимальный запас «пол − центр зоны» по отрезку x зоны, не только в точке центра.

    В центре нижней боковой зоны плечо бугра низкое, и высота 8 px там всё ещё
    оставляет запас. По всему отрезку [cx−8, cx+8] тот же бугор запас съедает.
    """
    worst, zone = float("inf"), None
    for i, (cx, cy) in enumerate(ZONE_CENTERS):
        xs = np.linspace(cx - ZONE_HALF, cx + ZONE_HALF, samples)
        margin = min(floor_limit(x, height) - cy for x in xs)
        if margin < worst:
            worst, zone = float(margin), i
    return worst, zone


def blocked_zone_names(height: float = BUMP_H, centers=None, samples: int = 65) -> list:
    """Зоны, у которых холм съедает запас по отрезку x ниже 2 px. Запуск не запрещает."""
    names = []
    for i, (cx, cy) in enumerate(ZONE_CENTERS):
        xs = np.linspace(cx - ZONE_HALF, cx + ZONE_HALF, samples)
        margin = min(floor_limit(x, height, centers=centers) - cy for x in xs)
        if margin < 2.0:
            names.append(ZONES[i]["name"])
    return names


def assert_zone_floor(height: float = BUMP_H) -> float:
    margin, zone = zone_floor_margin(height)
    if margin < 2.0:
        raise RuntimeError(
            f"запас зон {margin:.3f} px < 2 при высоте бугра {height} (зона {zone})")
    return margin


def _reflect(vel, x, height: float, centers=None):
    """Зеркало о внешнюю нормаль пола: n ∝ (−h', −1), восстановление 1, трения нет."""
    slope = bump_slope(x, height, centers=centers)
    normal = np.array([-slope, -1.0], np.float64)
    normal /= np.linalg.norm(normal)
    v = np.asarray(vel, np.float64)
    return v - 2.0 * float(np.dot(v, normal)) * normal


def integrate_floor(pos, vel, height: float = BUMP_H, size: int = SIZE, r: float = RADIUS,
                    centers=None):
    """Сдвиг на вектор скорости подшагами не длиннее 1 px. Сквозь бугор не проходит."""
    pos = np.array(pos, np.float64)
    vel = np.array(vel, np.float64)
    speed = float(np.hypot(vel[0], vel[1]))
    lo, hi = float(r), float(size - 1 - r)

    def limit_at(x):
        return floor_limit(x, height, centers)

    if speed < 1e-12:
        limit = limit_at(pos[0])
        if pos[1] > limit:
            pos[1] = limit
        return pos.astype(np.float32), vel.astype(np.float32)
    direction = vel / speed
    left = speed
    for _ in range(64):
        if left < 1e-8:
            break
        step_len = min(1.0, left)
        delta = direction * step_len
        end = pos + delta
        if end[1] > limit_at(end[0]) + 1e-8:
            a, b = 0.0, 1.0
            for _ in range(16):
                mid = 0.5 * (a + b)
                probe = pos + delta * mid
                if probe[1] > limit_at(probe[0]):
                    b = mid
                else:
                    a = mid
            if a < 1e-4:
                direction = _reflect(direction, pos[0], height, centers)
                nrm = float(np.linalg.norm(direction))
                if nrm < 1e-12:
                    break
                direction = direction / nrm
                pos[1] = limit_at(pos[0]) - 1e-3
                left = max(0.0, left - 1e-3)
            else:
                hit = pos + delta * a
                hit[1] = limit_at(hit[0])
                direction = _reflect(direction, hit[0], height, centers)
                nrm = float(np.linalg.norm(direction))
                direction = direction / max(nrm, 1e-12)
                pos = hit + direction * 1e-4
                left -= step_len * a
        else:
            pos = end
            left -= step_len
        if pos[0] < lo:
            pos[0] = lo + (lo - pos[0])
            direction[0] *= -1
        elif pos[0] > hi:
            pos[0] = hi - (pos[0] - hi)
            direction[0] *= -1
        if pos[1] < lo:
            pos[1] = lo + (lo - pos[1])
            direction[1] *= -1
    vel_out = direction * speed
    limit = limit_at(pos[0])
    if pos[1] > limit:
        pos[1] = limit
        if vel_out[1] > 0:
            vel_out = _reflect(vel_out, pos[0], height, centers)
    return pos.astype(np.float32), vel_out.astype(np.float32)


def step_floor(pos, vel, action: int, size: int = SIZE, r: float = RADIUS,
               g: float = GRAVITY, height: float = BUMP_H, centers=None):
    """Шаг режима floor: то же затухание и толчок, затем подшаги по буграм."""
    vel = VEL_DECAY * np.array(vel, np.float64) + np.asarray(ACTIONS[int(action)], np.float64)
    vel[1] += g
    return integrate_floor(pos, vel, height=height, size=size, r=r, centers=centers)


def step_other(pos, vel, size: int = SIZE, r: float = RADIUS):
    """Второй мяч: постоянная скорость, зеркальные стены, без тяжести и без толчка."""
    pos = np.array(pos, np.float32) + np.array(vel, np.float32)
    vel = np.array(vel, np.float32).copy()
    lo, hi = r, size - 1 - r
    for ax in range(2):
        if pos[ax] < lo:
            pos[ax] = lo + (lo - pos[ax])
            vel[ax] *= -1
        elif pos[ax] > hi:
            pos[ax] = hi - (pos[ax] - hi)
            vel[ax] *= -1
    return pos, vel


def _free_start(rng, size, r, height, centers=None):
    """Старт управляемого мяча над полом, в том же ящике, что и плоский мир."""
    for _ in range(1000):
        pos = rng.uniform(r + 2, size - 1 - r - 2, size=2).astype(np.float32)
        if float(pos[1]) <= floor_limit(pos[0], height, centers) - 0.05:
            return pos
    pos[1] = np.float32(floor_limit(pos[0], height, centers) - 0.05)
    return pos


def simulate_floor(batch: int, horizon: int, rng: np.random.Generator, actions=None,
                   size: int = SIZE, r: float = RADIUS, g: float = GRAVITY,
                   height: float = BUMP_H) -> np.ndarray:
    """Позиции управляемого мяча на буграх, (B, T, 2)."""
    pos = np.empty((batch, 2), np.float32)
    for i in range(batch):
        pos[i] = _free_start(rng, size, r, height)
    ang = rng.uniform(0, 2 * np.pi, size=batch).astype(np.float32)
    spd = rng.uniform(0.3, 1.0, size=batch).astype(np.float32)
    vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], axis=1).astype(np.float32)
    out = np.empty((batch, horizon, 2), np.float32)
    for t in range(horizon):
        for i in range(batch):
            action = NOOP if actions is None or t == 0 else int(actions[i, t - 1])
            # simulate без действий всё равно добавляет гравитацию каждый кадр.
            # Здесь действие NOOP на первом кадре совпадает с «ещё не толкнули»,
            # но затухание на нулевой скорости его не отличает от шага с гравитацией.
            if actions is None or t == 0:
                v = vel[i].astype(np.float64)
                v[1] += g
                pos[i], vel[i] = integrate_floor(pos[i], v, height=height, size=size, r=r)
            else:
                pos[i], vel[i] = step_floor(pos[i], vel[i], action, size=size, r=r, g=g, height=height)
        out[:, t] = pos
    return out


def simulate_balls(batch: int, horizon: int, rng: np.random.Generator, actions=None,
                   size: int = SIZE, r: float = RADIUS, g: float = GRAVITY):
    """Управляемый мяч как в плоском simulate, второй — после него, тем же генератором."""
    pos = rng.uniform(r + 2, size - 1 - r - 2, size=(batch, 2)).astype(np.float64)
    ang = rng.uniform(0, 2 * np.pi, size=batch).astype(np.float64)
    spd = rng.uniform(0.3, 1.0, size=batch).astype(np.float64)
    vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], axis=1)
    pos2 = np.empty((batch, 2), np.float64)
    for i in range(batch):
        placed = False
        for _ in range(1000):
            p = rng.uniform(r + 2, size - 1 - r - 2, size=2)
            if np.hypot(p[0] - pos[i, 0], p[1] - pos[i, 1]) >= 2 * r:
                pos2[i] = p
                placed = True
                break
        if not placed:
            raise RuntimeError("не удалось разместить второй мяч")
    ang2 = rng.uniform(0, 2 * np.pi, size=batch).astype(np.float64)
    spd2 = rng.uniform(1.5, 2.5, size=batch).astype(np.float64)
    vel2 = np.stack([np.cos(ang2) * spd2, np.sin(ang2) * spd2], axis=1)
    out = np.empty((batch, horizon, 2), np.float32)
    out2 = np.empty((batch, horizon, 2), np.float32)
    lo, hi = r, size - 1 - r
    for t in range(horizon):
        if actions is not None and t > 0:
            vel = VEL_DECAY * vel + ACTIONS[actions[:, t - 1]]
        vel[:, 1] += g
        pos = pos + vel
        pos2 = pos2 + vel2
        for body, body_vel in ((pos, vel), (pos2, vel2)):
            for ax in range(2):
                m = body[:, ax] < lo
                body[m, ax] = lo + (lo - body[m, ax])
                body_vel[m, ax] *= -1
                m = body[:, ax] > hi
                body[m, ax] = hi - (body[m, ax] - hi)
                body_vel[m, ax] *= -1
        out[:, t] = pos
        out2[:, t] = pos2
    return out, out2


def render_pair(pos: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """Два одинаковых пятна. Столкновений нет, яркости складываются и обрезаются."""
    a = (render(pos) + 1.0) * 0.5
    b = (render(pos2) + 1.0) * 0.5
    return (np.clip(a + b, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)


def paint_floor(frames: np.ndarray, height: float = BUMP_H, centers=None) -> np.ndarray:
    """Форма пола на кадре. Пятно мяча не затирается. От тяжести рисунок не зависит."""
    img = np.array(frames, dtype=np.float32, copy=True)
    flat = img.reshape(-1, SIZE, SIZE)
    surf = np.array([floor_limit(x, height, centers) for x in range(SIZE)], np.float64)
    solid = np.arange(SIZE, dtype=np.float64)[:, None] >= surf[None, :]
    for i in range(flat.shape[0]):
        bg = flat[i] < -0.85
        flat[i][solid & bg] = np.float32(-0.15)
    return img


def jitter_gate(radius_px: float, slip: float = SLIP) -> dict:
    """Порог 6 px объявлен заранее. Не больше 6 — это не проверка неоднозначности."""
    radius = float(radius_px)
    ok = radius > 6.0
    return {
        "slip": float(slip),
        "radius_px": radius,
        "threshold_px": 6.0,
        "train": ok,
        "label": None if ok else "не проверка неоднозначности",
    }


def measure_jitter_ambiguity(n_seeds: int = 200, slip: float = SLIP) -> dict:
    """Старт (32, 32), нулевая скорость, десять толчков вправо/влево, 200 сидов подмены.

    Радиус — 90-й перцентиль расстояния финальной точки до покоординатной медианы.
    """
    nominal = [5, 3] * 5
    finals = np.empty((n_seeds, 2), np.float64)
    for s in range(n_seeds):
        pos = np.array([32.0, 32.0], np.float64)
        vel = np.zeros(2, np.float64)
        rng = np.random.default_rng(s)
        for action in nominal:
            if float(rng.random()) < slip:
                action = int(rng.integers(0, N_ACTIONS))
            pos, vel = step_state(pos, vel, int(action))
            pos = np.asarray(pos, np.float64)
            vel = np.asarray(vel, np.float64)
        finals[s] = pos
    med = np.median(finals, axis=0)
    dist = np.hypot(finals[:, 0] - med[0], finals[:, 1] - med[1])
    radius = float(np.percentile(dist, 90))
    rec = jitter_gate(radius, slip)
    rec.update({
        "n_seeds": n_seeds,
        "start": [32.0, 32.0],
        "actions": nominal,
        "action_names": ["вправо", "влево"] * 5,
    })
    return rec


class World:
    """Пошаговый мир для исполнения плана: держит состояние и применяет действия.

    Плоский режим берёт те же три числа генератора, что и раньше: место, угол, скорость.
    Подмена толчка и шум кадра идут отдельными потоками и плоский старт не сдвигают.
    Второй мяч берёт числа уже после управляемого.
    """

    def __init__(self, seed: int, size: int = SIZE, world: str = "flat",
                 slip: float = SLIP, noise_sigma: float = NOISE_SIGMA,
                 bump_h: float = BUMP_H, centers=None, gravity: float = GRAVITY):
        if world not in WORLDS:
            raise ValueError(world)
        self.size = size
        self.world = world
        self.slip = float(slip)
        self.noise_sigma = float(noise_sigma)
        self.bump_h = float(bump_h)
        self.centers = None if centers is None else tuple(float(c) for c in centers)
        self.gravity = float(gravity)
        self.rng = np.random.default_rng(seed)
        if world == "floor":
            self.pos = _free_start(self.rng, size, RADIUS, self.bump_h, self.centers)
        else:
            self.pos = self.rng.uniform(RADIUS + 2, size - 1 - RADIUS - 2, size=2).astype(np.float32)
        ang = self.rng.uniform(0, 2 * np.pi)
        spd = self.rng.uniform(0.3, 1.0)
        self.vel = np.array([np.cos(ang) * spd, np.sin(ang) * spd], np.float32)
        self.slip_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 17]))
        self.noise_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 29]))
        self.pos2 = None
        self.vel2 = None
        self.other_speed = None
        if world == "balls":
            self._spawn_other()

    def _spawn_other(self):
        placed = False
        for _ in range(1000):
            p = self.rng.uniform(RADIUS + 2, self.size - 1 - RADIUS - 2, size=2).astype(np.float32)
            if np.hypot(p[0] - self.pos[0], p[1] - self.pos[1]) >= 2 * RADIUS:
                self.pos2 = p
                placed = True
                break
        if not placed:
            raise RuntimeError("не удалось разместить второй мяч")
        ang = float(self.rng.uniform(0, 2 * np.pi))
        spd = float(self.rng.uniform(1.5, 2.5))
        self.vel2 = np.array([np.cos(ang) * spd, np.sin(ang) * spd], np.float32)
        self.other_speed = spd

    def frame(self) -> np.ndarray:
        img = render(self.pos.reshape(1, 2))[0]
        if self.world == "balls":
            other = render(self.pos2.reshape(1, 2))[0]
            img = np.clip((img + 1.0) * 0.5 + (other + 1.0) * 0.5, 0.0, 1.0) * 2.0 - 1.0
        img = img[None, :, :].astype(np.float32)
        if self.world == "floor":
            img = paint_floor(img, height=self.bump_h, centers=self.centers)
        if self.world == "noise":
            img = add_obs_noise(img, self.noise_rng, self.noise_sigma)
        return img

    def step(self, action: int) -> np.ndarray:
        executed = int(action)
        if self.world == "jitter" and float(self.slip_rng.random()) < self.slip:
            executed = int(self.slip_rng.integers(0, N_ACTIONS))
        if self.world == "floor":
            self.pos, self.vel = step_floor(
                self.pos, self.vel, executed, self.size,
                g=self.gravity, height=self.bump_h, centers=self.centers)
        else:
            self.pos, self.vel = step_state(self.pos, self.vel, executed, self.size)
        if self.world == "balls":
            self.pos2, self.vel2 = step_other(self.pos2, self.vel2, self.size)
        return self.frame()

    def state(self):
        return self.pos.copy(), self.vel.copy()

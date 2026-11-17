"""Правило посадки с фазой наведения: сначала подойти боком, потом садиться.

Почему это понадобилось. Прежнее правило (``numeric_land.expert``) сразу уводит
аппарат вниз. Оно исправно гасит вертикальную скорость (vy ≈ 4.5 — мягко по
скорости), но **промахивается по x**: за время падения боком пройти можно лишь
около 18 единиц, а площадка стоит в 60–99 единицах от стартового x.

Замер на сидах 200000…200029, площадка из потока окна:

    g   прежнее правило         с фазой наведения      шагов (потолок 450)
    2   мягко  9/30, промах 40  мягко 30/30, промах 0.7   292
    8   мягко  5/30, промах 40  мягко 30/30, промах 0.7   206
    15  мягко  5/30, промах 39  мягко 30/30, промах 0.7   207
    20  мягко  7/30, промах 13  мягко 19/30, промах ~1     325

Две фазы:

- **наведение** (пока по x далеко): держим высоту и едем боком к центру.
  При g ≤ 15 нужная доля тяги ≤ 0.5, а бак тратится только при frac ≥ 0.5 —
  висение выходит бесплатным, поэтому время на боковой манёвр есть.
- **посадка** (по x близко): спуск с торможением к целевой скорости, как раньше.

При g = 20 тяга ровно равна тяжести (MAX_THRUST = g), висение стоит 0.02
бака за шаг — тут прежний запас кончается, и мягких посадок меньше. Но g = 20
и так вне обучения (тяжесть учится на [2, 15)).
"""
import numpy as np

from numeric_land.data import HORIZ_ACCEL, MAX_THRUST
from numeric_land.field import FUEL0, GROUND, TRANSITIONS, start_and_actions, step_field

NAV_TOL = 6.0          # по x ближе этого — переходим к посадке
NAV_V_BAND = 1.0       # полоса боковой скорости на наведении
LAND_VX_GAIN = 1.2
LAND_VX_CAP = 8.0
LAND_VX_BAND = 1.2
NAV_V_MAX = 40.0       # предел боковой скорости (физика шага)
NAV_ACCEL = HORIZ_ACCEL
V_TARGET_TAU = 0.667
HALF_ALT = 5.0
HOVER_CAP = 900


def hover_action(state, g, pad_x, v_target):
    """Действие правила с фазой наведения."""
    x, y, vx, vy, fuel = state
    if y >= GROUND:
        return 1
    height = max(0.0, GROUND - y)
    if fuel <= 0.0:
        return 1
    dx = float(pad_x) - float(x)
    if abs(dx) > NAV_TOL:
        # --- наведение: держим высоту, идём боком ---
        # скорость «приехать и остановиться»: v = sqrt(2·a·d), чтобы успеть затормозить.
        # Потолок задаёт физика шага: за шаг x меняется на ~vx/30, поэтому vx ≤ 40
        # даёт ход ~1.35 единицы за шаг и не перескакивает цель.
        v_want = min(NAV_V_MAX, float(np.sqrt(max(0.0, 2.0 * NAV_ACCEL * abs(dx)))))
        want = float(np.sign(dx)) * v_want
        if vx < want - NAV_V_BAND:
            ix = 2
        elif vx > want + NAV_V_BAND:
            ix = 0
        else:
            ix = 1
        if vy > 0.6:
            iy = 2
        elif vy > -0.2:
            iy = 2 if height < 60.0 else 1
        else:
            iy = 0
    else:
        # --- посадка: спуск с торможением к цели ---
        want = max(-LAND_VX_CAP, min(LAND_VX_CAP, dx * LAND_VX_GAIN))
        if vx < want - LAND_VX_BAND:
            ix = 2
        elif vx > want + LAND_VX_BAND:
            ix = 0
        else:
            ix = 1
        iy = 0
        a_full = float(g) - MAX_THRUST
        pred = vy * vy + 2.0 * a_full * height
        if pred > v_target * v_target:
            iy = 2
        elif pred > (V_TARGET_TAU * v_target) ** 2 and height < HALF_ALT:
            iy = 1
    return iy * 3 + ix


def hover_episode(seed, g, pad_x, v_target, cap=HOVER_CAP):
    """Полная траектория. Возвращает состояния, действия, коснулся ли."""
    x, y, vx, vy, _ = start_and_actions(int(seed), 0)
    fuel = FUEL0
    grounded = False
    states = [(x, y, vx, vy, fuel)]
    actions = []
    for _ in range(int(cap)):
        if grounded:
            break
        a = hover_action((x, y, vx, vy, fuel), g, pad_x, v_target)
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, int(a), g)
        states.append((x, y, vx, vy, fuel))
        actions.append(int(a))
    return np.asarray(states, np.float64), np.asarray(actions, np.int64), bool(grounded)


def hover_pool(n_windows, seed=300_000, landed_only=True):
    """Пул окон из траекторий правила с наведением. Формат как у expert_pool."""
    from numeric_land.data import g_of_window
    from numeric_land.goal import goal_of_window

    rng = np.random.default_rng(int(seed))
    st, ac, gs, pd, vy = [], [], [], [], []
    tries, limit = 0, int(n_windows) * 5 + 100
    landed = seen = 0
    while len(st) < n_windows and tries < limit:
        tries += 1
        ws = int(rng.integers(100_000, 2**31 - 1))
        g = g_of_window(ws)
        pad_x, vt = goal_of_window(ws)
        states, actions, grounded = hover_episode(ws, g, pad_x, vt)
        seen += 1
        if grounded:
            landed += 1
        if landed_only and not grounded:
            continue
        n = actions.shape[0]
        if n < TRANSITIONS:
            continue
        for s in range(n - TRANSITIONS + 1):
            st.append(states[s:s + TRANSITIONS + 1])
            ac.append(actions[s:s + TRANSITIONS])
            gs.append(g)
            pd.append(pad_x)
            vy.append(vt)
            if len(st) >= n_windows:
                break
    if not st:
        raise RuntimeError("пул пуст")
    return {
        "states": np.asarray(st, np.float64),
        "actions": np.asarray(ac, np.int64),
        "gs": np.asarray(gs, np.float64),
        "pads": np.asarray(pd, np.float64),
        "vys": np.asarray(vy, np.float64),
        "landed_frac": float(landed / max(1, seen)),
        "episodes_seen": int(seen),
    }

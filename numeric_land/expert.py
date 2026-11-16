"""Экспертные окна для world-модели: траектории правила посадки.

Правило то же, что кнопка «правило в коде» на демо: подрулить к центру площадки
и тормозить к земле. Здесь оно параметризовано целью (желаемой скоростью касания),
поэтому даёт траектории под каждый desired_vy.

Зачем: случайные окна (TRANSITIONS = 3 шага от старта высоко над площадкой) почти
не показывают динамику ВБЛИЗИ ЗЕМЛИ. Правило садится — его траектории добавляют
предиктору данные о том, как аппарат ведёт себя у площадки. Гипотеза: это улучшит
и мягкость, и точность по центру.
"""
import numpy as np

from numeric_land.data import MAX_THRUST, g_of_window
from numeric_land.field import FUEL0, GROUND, TRANSITIONS, start_and_actions, step_field
from numeric_land.goal import goal_of_window, make_batch as _make_random

V_TARGET_TAU = 0.667     # порог «половина тяги» как доля целевой скорости
HALF_ALT = 5.0
EXPERT_CAP = 450
AIM_GAIN = 1.5
AIM_CAP = 8.0
VX_BAND = 1.2
POOL_SEED0 = 200_000
POOL_DEFAULT = 20_000


def rule_action(state, g, pad_x, v_target):
    """Действие правила: подрулить к pad_x и затормозить к v_target у земли."""
    x, y, vx, vy, fuel = state
    if y >= GROUND:
        return 1
    height = max(0.0, GROUND - y)
    aim = max(-AIM_CAP, min(AIM_CAP, (float(pad_x) - float(x)) * AIM_GAIN))
    if vx < aim - VX_BAND:
        ix = 2
    elif vx > aim + VX_BAND:
        ix = 0
    else:
        ix = 1
    iy = 0
    if fuel > 0.0 and height > 0.0:
        pred = vy * vy + 2.0 * (float(g) - MAX_THRUST) * height
        if pred > v_target * v_target:
            iy = 2
        elif pred > (V_TARGET_TAU * v_target) ** 2 and height < HALF_ALT:
            iy = 1
    return iy * 3 + ix


def expert_episode(seed, g, pad_x, v_target):
    """Полная траектория правила. Возвращает состояния (T+1,5), действия (T,), коснулся ли."""
    x, y, vx, vy, _ = start_and_actions(int(seed), 0)
    fuel = FUEL0
    grounded = False
    states = [(x, y, vx, vy, fuel)]
    actions = []
    for _ in range(EXPERT_CAP):
        if grounded:
            break
        a = rule_action((x, y, vx, vy, fuel), g, pad_x, v_target)
        x, y, vx, vy, fuel, grounded = step_field(x, y, vx, vy, fuel, grounded, int(a), g)
        states.append((x, y, vx, vy, fuel))
        actions.append(int(a))
    return np.asarray(states, np.float64), np.asarray(actions, np.int64), bool(grounded)


def expert_pool(n_windows, seed=POOL_SEED0, landed_only=True):
    """Пул окон из траекторий правила. Каждое окно — (TRANSITIONS+1) состояний и TRANSITIONS действий.

    Из одного эпизода берутся ВСЕ скользящие окна, поэтому окон у земли заведомо больше:
    эпизод даёт много окон, и последние (у земли) в пуле встречаются часто.
    """
    rng = np.random.default_rng(int(seed))
    st, ac, gs, pd, vy = [], [], [], [], []
    tries = 0
    limit = int(n_windows) * 5 + 100
    landed = 0
    seen = 0
    while len(st) < n_windows and tries < limit:
        tries += 1
        ws = int(rng.integers(100_000, 2**31 - 1))
        g = g_of_window(ws)
        pad_x, vt = goal_of_window(ws)
        states, actions, grounded = expert_episode(ws, g, pad_x, vt)
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


def expert_window(window_seed, rng):
    """Одно окно из траектории правила для данного сида. Половина раз — последнее (у земли)."""
    g = g_of_window(int(window_seed))
    pad_x, vt = goal_of_window(int(window_seed))
    states, actions, _ = expert_episode(window_seed, g, pad_x, vt)
    n = actions.shape[0]
    if n < TRANSITIONS:
        return None
    starts = n - TRANSITIONS + 1
    s = starts - 1 if rng.random() < 0.5 else int(rng.integers(0, starts))
    return (states[s:s + TRANSITIONS + 1].astype(np.float64),
            actions[s:s + TRANSITIONS].astype(np.int64), g, pad_x, vt)


def make_batch_mixed(rng, batch=128, pool=None, expert_frac=0.5):
    """Смесь случайных окон и экспертных. Формат тот же, что у goal.make_batch."""
    n_expert = int(round(batch * float(expert_frac)))
    n_rand = batch - n_expert
    states = np.empty((batch, TRANSITIONS + 1, 5), np.float64)
    actions = np.empty((batch, TRANSITIONS), np.int64)
    gs = np.empty(batch, np.float64)
    pads = np.empty(batch, np.float64)
    vys = np.empty(batch, np.float64)
    if n_rand:
        s, a, g_, p_, v_ = _make_random(rng, n_rand, "range")
        states[:n_rand] = s
        actions[:n_rand] = a
        gs[:n_rand] = g_
        pads[:n_rand] = p_
        vys[:n_rand] = v_
    if n_expert:
        if pool is None:
            raise ValueError("нужен пул экспертных окон")
        total = pool["states"].shape[0]
        idx = rng.integers(0, total, size=n_expert)
        states[n_rand:] = pool["states"][idx]
        actions[n_rand:] = pool["actions"][idx]
        gs[n_rand:] = pool["gs"][idx]
        pads[n_rand:] = pool["pads"][idx]
        vys[n_rand:] = pool["vys"][idx]
    return states, actions, gs, pads, vys
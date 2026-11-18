"""Треки: что именно видит модель при обучении.

Кнопка «обучающие треки» на демо площадки. Четыре вида, все детерминированы сидом —
это ровно те числа, что уходят в обучение, а не приблизительная картинка.

- ``window`` — окно обучения: TRANSITIONS+1 состояний и TRANSITIONS случайных
  действий. Именно окнами (а не траекториями) кормится предиктор.
- ``flight`` — полный полёт-источник: те же 450 случайных действий, откуда окно
  вырезано. Нужен, чтобы видеть, какую долю падения составляет окно.
- ``expert`` — полная траектория старого правила посадки (см. numeric_land.expert).
- ``hover`` — траектория правила с наведением. Тяжесть и цель из сида те же,
  что у головы режима: учебные сиды 98000…98511.

Окно всегда начинается у земли: x ∈ [240, 272], y ∈ [464, 488] при GROUND = 512.
Из одного старта окно покрывает доли единицы высоты — поэтому по нему не видно
ни подвода, ни касания.
"""
import numpy as np

from numeric_land.data import DT, g_of_window
from numeric_land.expert import expert_episode
from numeric_land.expert_hover import NAV_TOL, hover_episode, hover_mode
from numeric_land.field import (
    EPISODE_CAP, FUEL0, GROUND, TRANSITIONS, rollout_states, start_and_actions,
    step_field,
)
from numeric_land.goal import goal_of_window
from numeric_land.pad import outcome_at, pad_of_window

KINDS = ("window", "flight", "expert", "hover")


def _pt(x, y, vy, fuel, grounded, action=None):
    p = {
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "vy": round(float(vy), 2),
        "fuel": round(float(fuel), 2),
        "grounded": bool(grounded),
    }
    if action is not None:
        p["action"] = int(action)
    return p


def _track(kind, seed, g, pad_x, desired_vy, states, actions, grounded):
    """Общая сборка ответа. states (T+1,5), actions (T,) — либо пустой массив."""
    states = np.asarray(states, np.float64)
    actions = np.asarray(actions, np.int64)
    touched = bool(grounded) or bool((states[:, 1] >= GROUND - 1e-9).any())
    points = []
    for i, s in enumerate(states):
        act = int(actions[i]) if i < len(actions) else None
        points.append(_pt(s[0], s[1], s[3], s[4], s[1] >= GROUND - 1e-9, act))
    h_from = float(GROUND - states[0][1])
    h_to = float(GROUND - states[-1][1])
    fin_x = float(states[-1][0])
    return {
        "kind": kind,
        "seed": int(seed),
        "g": round(float(g), 4),
        "pad_x": round(float(pad_x), 2),
        "desired_vy": None if desired_vy is None else round(float(desired_vy), 2),
        "n_states": int(states.shape[0]),
        "n_actions": int(actions.shape[0]),
        "duration_s": round(float(actions.shape[0]) * DT, 4),
        "height_from": round(h_from, 2),
        "height_to": round(h_to, 2),
        "height_covered": round(h_from - h_to, 2),
        "touched": touched,
        "result": outcome_at(fin_x, states[-1][1], states[-1][3], touched, pad_x),
        "points": points,
    }


def random_window(seed, g=None):
    """Окно как в обучении. Тяжесть берётся потоком окна, если не задана."""
    seed = int(seed)
    gi = float(g_of_window(seed)) if g is None else float(g)
    x, y, vx, vy, actions = start_and_actions(seed, TRANSITIONS)
    states = rollout_states(x, y, vx, vy, actions, gi)
    return _track("window", seed, gi, pad_of_window(seed), None, states, actions,
                  bool(states[:, 1].max() >= GROUND - 1e-9))


def random_flight(seed, g=None, cap=EPISODE_CAP):
    """Полный полёт на случайных действиях. Префикс действий тот же, что у окна."""
    seed = int(seed)
    gi = float(g_of_window(seed)) if g is None else float(g)
    x, y, vx, vy, actions = start_and_actions(seed, int(cap))
    # обрываем на касании: дальше шаг замирает и рисовать нечего
    rows = rollout_states(x, y, vx, vy, actions, gi)
    touch = np.flatnonzero(rows[:, 1] >= GROUND - 1e-9)
    if len(touch):
        k = int(touch[0])
        rows, actions = rows[:k + 1], actions[:k]
    return _track("flight", seed, gi, pad_of_window(seed), None, rows, actions,
                  bool(len(touch)))


def expert_flight(seed, g=None, pad_x=None, desired_vy=None):
    """Полная траектория правила посадки. Цель — поток окна, если не задана."""
    seed = int(seed)
    gi = float(g_of_window(seed)) if g is None else float(g)
    pad_i, vy_i = goal_of_window(seed)
    if pad_x is not None:
        pad_i = float(pad_x)
    if desired_vy is not None:
        vy_i = float(desired_vy)
    states, actions, grounded = expert_episode(seed, gi, pad_i, vy_i)
    return _track("expert", seed, gi, pad_i, vy_i, states, actions, grounded)


def expert_final_window(seed, g=None, pad_x=None, desired_vy=None):
    """Последнее окно траектории правила — то самое мягкое касание у земли."""
    seed = int(seed)
    gi = float(g_of_window(seed)) if g is None else float(g)
    pad_i, vy_i = goal_of_window(seed)
    if pad_x is not None:
        pad_i = float(pad_x)
    if desired_vy is not None:
        vy_i = float(desired_vy)
    states, actions, grounded = expert_episode(seed, gi, pad_i, vy_i)
    n = actions.shape[0]
    if n < TRANSITIONS:
        raise ValueError("у траектории правила меньше трёх шагов")
    s0 = n - TRANSITIONS
    return _track("expert-window", seed, gi, pad_i, vy_i,
                  states[s0:s0 + TRANSITIONS + 1], actions[s0:s0 + TRANSITIONS],
                  grounded)


def hover_flight(seed, g=None, pad_x=None, desired_vy=None):
    """Траектория правила с наведением. Цель и тяжесть — потоки окна, если не заданы.

    Это тот же эпизод, из которого голова режима берёт метки: посадка при
    ``|pad_x − x| ≤ NAV_TOL``. Кадр касания режима не имеет.
    """
    seed = int(seed)
    gi = float(g_of_window(seed)) if g is None else float(g)
    pad_i, vy_i = goal_of_window(seed)
    if pad_x is not None:
        pad_i = float(pad_x)
    if desired_vy is not None:
        vy_i = float(desired_vy)
    states, actions, grounded = hover_episode(seed, gi, pad_i, vy_i)
    track = _track("hover", seed, gi, pad_i, vy_i, states, actions, grounded)
    first = None
    for i, row in enumerate(states):
        state = tuple(float(v) for v in row)
        mode = hover_mode(state, gi, pad_i, vy_i)
        track["points"][i]["mode"] = mode
        if mode == "land" and first is None:
            first = i
    track["first_land_step"] = first
    track["nav_tol"] = float(NAV_TOL)
    return track


def build(kind, seed, g=None, pad_x=None, desired_vy=None):
    """Ответ для /api/track: главный трек плюс фон, если он помогает читать."""
    if kind not in KINDS:
        raise ValueError("вид: window, flight, expert или hover")
    if kind == "window":
        win = random_window(seed, g)
        return {"kind": kind, "seed": int(seed), "track": win,
                "context": random_flight(seed, g)}
    if kind == "flight":
        return {"kind": kind, "seed": int(seed),
                "track": random_flight(seed, g), "context": None}
    if kind == "hover":
        return {"kind": kind, "seed": int(seed),
                "track": hover_flight(seed, g, pad_x, desired_vy), "context": None}
    return {"kind": kind, "seed": int(seed),
            "track": expert_flight(seed, g, pad_x, desired_vy), "context": None}

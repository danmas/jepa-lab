"""Ориентир трубы с торможением: желаемая боковая скорость наведения с запасом.

Правило не правится и остаётся эталоном. Меняется только то, что труба берёт как цель:
на ветке наведения желаемая боковая скорость умножается на ``BRAKE_GAIN``.

    v_want = gain · min(NAV_V_MAX, sqrt(2 · NAV_ACCEL · |dx|))

Вертикальная ветка, полоса, ``NAV_TOL``, потолки посадки, ``v_target`` и горизонт трубы
не меняются. При ``gain = 1.0`` профиль обязан совпадать с ``roll_tube`` побитово —
это условие ``equiv_ok`` в прогоне ``numeric_land.skill_brake``.

Зачем. Исследование финальной фазы (1.4.34) показало: труба идёт наведением на пределе
профиля, перелетает площадку и уходит от центра на сотни пикселей, а при ``g = 15``
висение стоит бака. Замер коэффициента — ``_research_kappa.py``.

Обучения нет.
"""
import numpy as np

from numeric_land.data import HORIZ_ACCEL, MAX_THRUST
from numeric_land.expert_hover import (
    GROUND,
    HALF_ALT,
    NAV_ACCEL,
    LAND_VX_BAND,
    LAND_VX_CAP,
    LAND_VX_GAIN,
    NAV_TOL,
    NAV_V_BAND,
    NAV_V_MAX,
    V_TARGET_TAU,
    hover_action,
    hover_mode,
)
from numeric_land.field import step_field
from numeric_land.skill_tube import TUBE_TARGETS, _encode_now, choose_action, tube_latents

BRAKE_GAIN = 0.6
GAIN_EXACT = 1.0


def locks() -> None:
    """Константы профиля и правила, от которых зависит замер."""
    if float(BRAKE_GAIN) != 0.6:
        raise RuntimeError(BRAKE_GAIN)
    if float(NAV_V_MAX) != 40.0 or float(NAV_ACCEL) != 12.0 or float(MAX_THRUST) != 20.0:
        raise RuntimeError((NAV_V_MAX, NAV_ACCEL, MAX_THRUST))
    if float(NAV_ACCEL) != float(HORIZ_ACCEL):
        raise RuntimeError((NAV_ACCEL, HORIZ_ACCEL))
    if float(NAV_TOL) != 6.0 or float(NAV_V_BAND) != 1.0:
        raise RuntimeError((NAV_TOL, NAV_V_BAND))
    if float(LAND_VX_CAP) != 8.0 or float(LAND_VX_GAIN) != 1.2 or float(LAND_VX_BAND) != 1.2:
        raise RuntimeError((LAND_VX_CAP, LAND_VX_GAIN, LAND_VX_BAND))
    if int(TUBE_TARGETS) != 3 or float(GROUND) != 512.0:
        raise RuntimeError((TUBE_TARGETS, GROUND))


def _approach_action_brake(x, y, vx, vy, g, pad_x, v_target, gain):
    """Наведение с запасом по боковой скорости. Вертикаль как у правила."""
    del g, v_target
    height = max(0.0, GROUND - y)
    dx = float(pad_x) - float(x)
    v_want = float(gain) * min(NAV_V_MAX, float(np.sqrt(max(0.0, 2.0 * NAV_ACCEL * abs(dx)))))
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
    return iy * 3 + ix


def brake_action(state, g, pad_x, v_target, gain=BRAKE_GAIN) -> int:
    """Действие профиля с запасом. На земле и при пустом баке — 1, как у правила."""
    gain = float(gain)
    if gain <= 0.0 or gain > 1.0:
        raise RuntimeError(gain)
    natural = hover_mode(state, float(g), float(pad_x), float(v_target))
    if natural is None:
        return 1
    if natural == "land":
        return int(hover_action(state, float(g), float(pad_x), float(v_target), mode="land"))
    x, y, vx, vy, _fuel = (float(v) for v in state)
    return int(_approach_action_brake(x, y, vx, vy, float(g), float(pad_x), float(v_target), gain))


def roll_tube_brake(state, g, pad_x, gain=BRAKE_GAIN, v_target=6.0):
    """Три кадра ориентира. Касание раньше третьего шага повторяет кадр касания."""
    x, y, vx, vy, fuel = (float(v) for v in state)
    frames = []
    held = None
    calls = 0
    clipped = False
    for k in range(TUBE_TARGETS):
        if held is not None:
            frames.append(held)
            continue
        calls += 1
        action = int(brake_action((x, y, vx, vy, fuel), float(g), float(pad_x), float(v_target), gain))
        x, y, vx, vy, fuel, grounded = step_field(x, y, vx, vy, fuel, False, action, float(g))
        held_state = (float(x), float(y), float(vx), float(vy), float(fuel))
        frames.append(held_state)
        if grounded and k < TUBE_TARGETS - 1:
            held = held_state
            clipped = True
    if len(frames) != TUBE_TARGETS or calls < 1 or calls > TUBE_TARGETS:
        raise RuntimeError((len(frames), calls, clipped))
    return tuple(frames), int(calls), bool(clipped)


def prepare_brake(encoder, ema, row, gain=BRAKE_GAIN):
    """Латент текущего состояния и латенты ориентира с запасом. Состояние не меняется."""
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    fuel = float(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    before = (x, y, vx, vy, fuel)
    frames, calls, clipped = roll_tube_brake(before, g, pad_x, gain)
    if (x, y, vx, vy, fuel) != before:
        raise RuntimeError("профиль подменил состояние")
    z = _encode_now(encoder, x, y, vx, vy, fuel, pad_x)
    z_tube = tube_latents(ema, frames, pad_x)
    return z, z_tube, frames, calls, clipped


def brake_target_action(predictor, encoder, ema, actions, step_pairs, world_b, state, g, pad_x,
                        gain=BRAKE_GAIN):
    """argmin по 729 планам против ориентира с запасом."""
    x, y, vx, vy, fuel = (float(v) for v in state)
    row = {
        "x": x, "y": y, "vx": vx, "vy": vy, "fuel": fuel,
        "g": float(g), "pad_x": float(pad_x),
    }
    before = (x, y, vx, vy, fuel)
    z, z_tube, _frames, calls, clipped = prepare_brake(encoder, ema, row, gain)
    if (x, y, vx, vy, fuel) != before:
        raise RuntimeError("профиль подменил состояние")
    if calls < 1:
        raise RuntimeError(calls)
    action, _cost = choose_action(predictor, z, z_tube, world_b, step_pairs, actions)
    return int(action), bool(clipped)

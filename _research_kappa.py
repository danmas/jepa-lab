"""Подбор коэффициента запаса в цели трубы (вариант 2: торможение в ориентире).

Ориентир трубы строится не правилом как есть, а «осторожным профилем»: желаемая
боковая скорость на наведении умножается на gain < 1. Вертикальная ветка не меняется.

    python _research_kappa.py

Проверяет два свойства и мерит сетку:
  1) gain = 1.0 даёт ровно ту же трубу, что ``roll_tube`` (побитово, по кадрам);
  2) на провальных клетках перелёт и пустые баки падают, на знакомых — не растут.
"""
import json
import os

import numpy as np
import torch

from numeric_land.data import HORIZ_ACCEL, MAX_THRUST, action_parts
from numeric_land.expert_hover import (
    HALF_ALT, HOVER_CAP, LAND_VX_BAND, LAND_VX_CAP, LAND_VX_GAIN, NAV_TOL, NAV_V_BAND,
    NAV_V_MAX, V_TARGET_TAU, hover_action,
)
from numeric_land.field import FUEL0, GROUND, start_and_actions, step_field
from numeric_land.relenc import OUT, write_json
from numeric_land.skill_assemble import classify, load_bundle, take_stamps
from numeric_land.skill_land import pair_table, world_batch
from numeric_land.skill_tube import GOAL_VY, TUBE_TARGETS, plan_costs, roll_tube, tube_latents, _encode_now

GAINS = (1.0, 0.85, 0.7, 0.55)
CELLS = (("15/406", 15.0, 406.0), ("20/356", 20.0, 356.0), ("20/406", 20.0, 406.0),
         ("15/256", 15.0, 256.0), ("2/406", 2.0, 406.0))
SEEDS = range(210_000, 210_008)
SOFT = 8.0


def brake_action(state, g, pad_x, v_target, gain):
    """Правило с множителем запаса на желаемую боковую скорость наведения."""
    x, y, vx, vy, fuel = (float(v) for v in state)
    if y >= GROUND or fuel <= 0.0:
        return 1
    dx = float(pad_x) - x
    if abs(dx) > NAV_TOL:
        height = max(0.0, GROUND - y)
        v_want = float(gain) * min(NAV_V_MAX, float(np.sqrt(max(0.0, 2.0 * HORIZ_ACCEL * abs(dx)))))
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
    want = max(-LAND_VX_CAP, min(LAND_VX_CAP, dx * LAND_VX_GAIN))
    if vx < want - LAND_VX_BAND:
        ix = 2
    elif vx > want + LAND_VX_BAND:
        ix = 0
    else:
        ix = 1
    iy = 0
    a_full = float(g) - MAX_THRUST
    pred = vy * vy + 2.0 * a_full * max(0.0, GROUND - y)
    if pred > float(v_target) ** 2:
        iy = 2
    elif pred > (V_TARGET_TAU * float(v_target)) ** 2 and (GROUND - y) < HALF_ALT:
        iy = 1
    return iy * 3 + ix


def roll_tube_gain(state, g, pad_x, gain):
    x, y, vx, vy, fuel = (float(v) for v in state)
    frames = []
    held = None
    for k in range(TUBE_TARGETS):
        if held is not None:
            frames.append(held)
            continue
        action = int(brake_action((x, y, vx, vy, fuel), float(g), float(pad_x), GOAL_VY, gain))
        x, y, vx, vy, fuel, grounded = step_field(x, y, vx, vy, fuel, False, action, float(g))
        held_state = (float(x), float(y), float(vx), float(vy), float(fuel))
        frames.append(held_state)
        if grounded and k < TUBE_TARGETS - 1:
            held = held_state
    return tuple(frames)


def prepare_gain(encoder, ema, row, gain):
    x = float(row["x"]); y = float(row["y"]); vx = float(row["vx"])
    vy = float(row["vy"]); fuel = float(row["fuel"]); pad_x = float(row["pad_x"])
    frames = roll_tube_gain((x, y, vx, vy, fuel), float(row["g"]), pad_x, gain)
    z = _encode_now(encoder, x, y, vx, vy, fuel, pad_x)
    z_tube = tube_latents(ema, frames, pad_x)
    return z, z_tube, frames


def fly(seed, g, pad_x, gain, bundle, actions, step_pairs):
    x, y, vx, vy, _ = start_and_actions(int(seed), 0)
    fuel = float(FUEL0)
    grounded = False
    steps = 0
    rows = []
    world_b = world_batch(g, pad_x)
    while steps < int(HOVER_CAP) and not grounded:
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        mode, _prob = classify(bundle["early"], bundle["encoder"], state, g, pad_x)
        if mode is None:
            action = 1
        else:
            row = {"x": state[0], "y": state[1], "vx": state[2], "vy": state[3],
                   "fuel": state[4], "g": g, "pad_x": pad_x}
            z, z_tube, _frames = prepare_gain(bundle["encoder"], bundle["ema"], row, gain)
            cost = plan_costs(bundle["pad"] if mode == "land" else bundle["approach"],
                              z, z_tube, world_b, step_pairs)
            action = int(actions[int(torch.argmin(cost).item()), 0].item())
        rows.append({"step": steps, "h": float(GROUND) - float(y), "vy": float(vy),
                     "vx": float(vx), "dx": float(x) - float(pad_x), "mode": mode,
                     "frac": float(action_parts(int(action), 10.0)[2])})
        x, y, vx, vy, fuel, grounded = step_field(
            state[0], state[1], state[2], state[3], state[4], False, int(action), g)
        steps += 1
        if float(fuel) <= 0.0 and not grounded:
            break
    soft = bool(grounded and abs(float(x) - float(pad_x)) <= 8.0 and float(vy) < SOFT)
    entered = False
    peak = 0.0
    for row in rows:
        if not entered and abs(row["dx"]) <= NAV_TOL:
            entered = True
        if entered:
            peak = max(peak, abs(row["dx"]))
    ap = [abs(row["vx"]) for row in rows if row["mode"] == "approach"]
    return {"seed": int(seed), "soft": soft, "steps": int(steps),
            "dry": bool(float(fuel) <= 0.0 and not grounded),
            "drift": peak, "vx_approach": float(np.mean(ap)) if ap else None,
            "burn": float(np.mean([1.0 if row["frac"] >= 0.5 and row["mode"] else 0.0 for row in rows]))}


def check_equivalence():
    """gain = 1.0 обязан совпасть с roll_tube и с hover_action на живых состояниях."""
    states = []
    for seed in range(210_000, 210_004):
        for g, pad in ((15.0, 406.0), (8.0, 406.0), (15.0, 256.0)):
            x, y, vx, vy, _ = start_and_actions(seed, 0)
            fuel = float(FUEL0)
            grounded = False
            for _ in range(60):
                if grounded:
                    break
                states.append(((float(x), float(y), float(vx), float(vy), float(fuel)), g, pad))
                a = int(hover_action((x, y, vx, vy, fuel), g, pad, GOAL_VY))
                x, y, vx, vy, fuel, grounded = step_field(x, y, vx, vy, fuel, False, a, g)
    bad_action = 0
    bad_frames = 0
    for state, g, pad in states:
        if int(brake_action(state, g, pad, GOAL_VY, 1.0)) != int(hover_action(state, g, pad, GOAL_VY)):
            bad_action += 1
        if roll_tube_gain(state, g, pad, 1.0) != tuple(roll_tube(state, g, pad)[0]):
            bad_frames += 1
    return {"states": len(states), "bad_action": int(bad_action), "bad_frames": int(bad_frames)}


def main():
    torch.set_num_threads(8)
    before = take_stamps()
    equiv = check_equivalence()
    print(json.dumps({"equivalence": equiv}, ensure_ascii=False), flush=True)
    if equiv["bad_action"] or equiv["bad_frames"]:
        raise SystemExit("gain = 1.0 не совпал с правилом")
    bundle = load_bundle()
    actions, step_pairs = pair_table()
    report = []
    for gain in GAINS:
        for name, g, pad in CELLS:
            eps = [fly(seed, g, pad, gain, bundle, actions, step_pairs) for seed in SEEDS]
            drift = [e["drift"] for e in eps]
            vx = [e["vx_approach"] for e in eps if e["vx_approach"] is not None]
            stats = {"gain": gain, "cell": name, "g": g, "pad_x": pad, "n": len(eps),
                     "soft": sum(1 for e in eps if e["soft"]),
                     "dry": sum(1 for e in eps if e["dry"]),
                     "steps_median": float(np.median([e["steps"] for e in eps])),
                     "drift_median": round(float(np.median(drift)), 1),
                     "drift_max": round(float(max(drift)), 1),
                     "vx_approach": round(float(np.mean(vx)), 2) if vx else None,
                     "burn": round(float(np.mean([e["burn"] for e in eps])), 3)}
            report.append(stats)
            print(json.dumps(stats, ensure_ascii=False), flush=True)
    if take_stamps() != before:
        raise SystemExit("замер сдвинул штамп чекпоинта")
    path = os.path.join(OUT, "research_kappa.json")
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}, второй замер не затирает")
    write_json(path, {"gains": list(GAINS), "cells": [[n, g, p] for n, g, p in CELLS],
                      "seeds": [SEEDS[0], SEEDS[-1]], "equivalence": equiv,
                      "report": report, "heads_trained": False, "stamps_unchanged": True})
    print(path, flush=True)


if __name__ == "__main__":
    main()

"""Исследование финальной фазы при g = 15: почему труба не садится на дальний центр.

Замер, а не изменение. Три клетки по 30 эпизодов: провал (g=15, центр 406), две нормы
(g=15, центр 256 и g=8, центр 406). Для каждой клетки считается труба и правило.

    python _research_phase15.py

Пишет out/research_phase15.json и печатает сводку. Обучения нет, чекпоинты только читаются.
"""
import json
import os

import numpy as np
import torch

from numeric_land.data import MAX_THRUST, action_parts
from numeric_land.expert_hover import HOVER_CAP, hover_action
from numeric_land.field import FUEL0, GROUND, start_and_actions, step_field
from numeric_land.relenc import OUT, write_json
from numeric_land.skill_assemble import classify, load_bundle, take_stamps
from numeric_land.skill_land import pair_table, world_batch
from numeric_land.skill_tube import plan_costs, prepare

CELLS = (("15/406 провал", 15.0, 406.0), ("15/256 норма", 15.0, 256.0), ("8/406 норма", 8.0, 406.0))
SEEDS = range(210_000, 210_030)
SOFT = 8.0
NAV_TOL = 6.0


def hover_fuel_per_step(g: float) -> float:
    """Топливо на висение за шаг: средняя тяга равна тяжести, цена 0.02 за полную тягу."""
    return 0.001 * float(g)


def fly(seed, g, pad_x, use_tube, bundle, actions, step_pairs):
    x, y, vx, vy, _ = start_and_actions(int(seed), 0)
    fuel = float(FUEL0)
    grounded = False
    steps = 0
    rows = []
    world_b = world_batch(g, pad_x)
    while steps < int(HOVER_CAP) and not grounded:
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        mode, prob = classify(bundle["early"], bundle["encoder"], state, g, pad_x)
        a_rule = 1 if mode is None else int(hover_action(state, g, pad_x, 6.0, mode=mode))
        a_tube = a_rule
        cost_min = cost_rule = spread = tgt_vy = tgt_h = None
        if mode is not None and use_tube:
            row = {"x": state[0], "y": state[1], "vx": state[2], "vy": state[3],
                   "fuel": state[4], "g": g, "pad_x": pad_x}
            z, z_tube, frames, _calls, _clip = prepare(bundle["encoder"], bundle["ema"], row)
            cost = plan_costs(bundle["pad"] if mode == "land" else bundle["approach"],
                              z, z_tube, world_b, step_pairs)
            arr = cost.detach().numpy()
            index = int(np.argmin(arr))
            a_tube = int(actions[index, 0].item())
            cost_min = float(arr[index])
            first = actions[:, 0].numpy()
            mask = first == a_rule
            cost_rule = float(arr[mask].min()) if bool(mask.any()) else None
            spread = float(np.percentile(arr, 95) - np.percentile(arr, 5))
            tgt_vy = float(frames[0][3])
            tgt_h = float(GROUND - frames[0][1])
        action = a_tube if (use_tube and mode is not None) else a_rule
        phys_tube = step_field(state[0], state[1], state[2], state[3], state[4], False, int(a_tube), g)
        phys_rule = step_field(state[0], state[1], state[2], state[3], state[4], False, int(a_rule), g)
        rows.append({
            "step": steps, "h": float(GROUND) - float(y), "vy": float(vy), "vx": float(vx),
            "dx": float(x) - float(pad_x), "fuel": float(fuel), "mode": mode,
            "frac": float(action_parts(int(action), 10.0)[2]),
            "a_tube": int(a_tube), "a_rule": int(a_rule),
            "tgt_vy": tgt_vy, "tgt_h": tgt_h, "cost_min": cost_min, "cost_rule": cost_rule,
            "spread": spread,
            "step_vy_tube": float(phys_tube[3]), "step_vy_rule": float(phys_rule[3]),
        })
        x, y, vx, vy, fuel, grounded = step_field(
            state[0], state[1], state[2], state[3], state[4], False, int(action), g)
        steps += 1
        if float(fuel) <= 0.0 and not grounded:
            break
    vy_touch = float(vy)
    soft = bool(grounded and abs(float(x) - float(pad_x)) <= 8.0 and vy_touch < SOFT)
    return {"seed": int(seed), "steps": int(steps), "soft": soft, "grounded": bool(grounded),
            "fuel_left": float(fuel), "dry": bool(float(fuel) <= 0.0 and not grounded),
            "vy_touch": vy_touch, "dx_touch": abs(float(x) - float(pad_x)), "rows": rows}


def digest(name, g, pad_x, tube, rule):
    out = {"cell": name, "g": float(g), "pad_x": float(pad_x), "n": len(tube)}
    out["tube_soft"] = int(sum(1 for e in tube if e["soft"]))
    out["rule_soft"] = int(sum(1 for e in rule if e["soft"]))
    out["tube_dry"] = int(sum(1 for e in tube if e["dry"]))
    out["tube_steps_median"] = float(np.median([e["steps"] for e in tube]))
    out["rule_steps_median"] = float(np.median([e["steps"] for e in rule]))
    reach = []
    drift = []
    vx_ap = []
    first_over = []
    burn = []
    gap = []
    target_over = 0
    steps_n = 0
    dvy_land = []
    for ep in tube:
        entered = False
        peak = 0.0
        for row in ep["rows"]:
            steps_n += 1
            if not entered and abs(row["dx"]) <= NAV_TOL:
                entered = True
                reach.append(row["step"])
            if entered:
                peak = max(peak, abs(row["dx"]))
            if row["mode"] == "approach":
                vx_ap.append(abs(row["vx"]))
            if row["frac"] >= 0.5 and row["mode"] is not None:
                burn.append(1.0)
            if row["mode"] == "land" and row["tgt_vy"] is not None and row["tgt_vy"] > SOFT:
                target_over += 1
            if row["mode"] == "land":
                dvy_land.append(row["step_vy_tube"] - row["step_vy_rule"])
            if row["cost_rule"] is not None and row["spread"]:
                gap.append((row["cost_rule"] - row["cost_min"]) / row["spread"])
            if not first_over and row["vy"] > SOFT:
                first_over.append({"step": row["step"], "h": round(row["h"], 3),
                                   "vy": round(row["vy"], 3), "dx": round(row["dx"], 3),
                                   "a": row["a_tube"], "tgt_vy": None if row["tgt_vy"] is None
                                   else round(row["tgt_vy"], 3)})
        drift.append(peak)
    out["reach_median_step"] = float(np.median(reach)) if reach else None
    out["share_reached_corridor"] = round(len(reach) / len(tube), 3)
    out["drift_after_reach_median_px"] = round(float(np.median(drift)), 1) if drift else None
    out["drift_after_reach_max_px"] = round(float(max(drift)), 1) if drift else None
    out["vx_mean_approach_tube"] = round(float(np.mean(vx_ap)), 3) if vx_ap else None
    rule_vx = [abs(row["vx"]) for ep in rule for row in ep["rows"] if row["mode"] == "approach"]
    out["vx_mean_approach_rule"] = round(float(np.mean(rule_vx)), 3) if rule_vx else None
    out["share_burn_steps"] = round(len(burn) / max(1, steps_n), 4)
    out["share_target_above_soft_in_land"] = round(target_over / max(1, len(dvy_land)), 4)
    out["dvy_land_mean"] = round(float(np.mean(dvy_land)), 4) if dvy_land else None
    out["cost_gap_norm_mean"] = round(float(np.mean(gap)), 4) if gap else None
    out["first_vy_over_soft"] = first_over[:3]
    out["hover_fuel_per_step"] = round(hover_fuel_per_step(g), 4)
    out["hover_step_budget"] = int(round(float(FUEL0) / hover_fuel_per_step(g)))
    return out


def main():
    torch.set_num_threads(8)
    before = take_stamps()
    bundle = load_bundle()
    actions, step_pairs = pair_table()
    report = []
    for name, g, pad_x in CELLS:
        tube = [fly(seed, g, pad_x, True, bundle, actions, step_pairs) for seed in SEEDS]
        rule = [fly(seed, g, pad_x, False, bundle, actions, step_pairs) for seed in SEEDS]
        stats = digest(name, g, pad_x, tube, rule)
        report.append(stats)
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    if take_stamps() != before:
        raise SystemExit("замер сдвинул штамп чекпоинта")
    path = os.path.join(OUT, "research_phase15.json")
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}, второй замер не затирает")
    write_json(path, {"cells": CELLS, "seeds": [SEEDS[0], SEEDS[-1]], "report": report,
                      "heads_trained": False, "stamps_unchanged": True})
    print(path, flush=True)


if __name__ == "__main__":
    main()

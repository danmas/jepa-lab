"""Контроль до прогона ``numeric-land-skill-brake``. Ничего не летает и не обучает.

Проверяет то, на чём держится замер:
  1) профиль с запасом: ``v_want`` умножается на ``0.6``, вертикаль и потолки посадки не тронуты;
  2) ``gain = 1.0`` совпадает с ``roll_tube`` по кадрам и с ``hover_action`` по действию;
  3) сетка: ``600`` основных эпизодов и ``180`` в блоке повтора, сиды не пересекаются;
  4) записи этапа 4 и разделения: ``407``, ``509``, ``557``, ``150`` по клеткам;
  5) арифметика восьми условий и допуск клетки ``3``;
  6) защита отчёта: без тега, с чужим тегом, поверх файла;
  7) порог головы ``0.5``, вход ``33``, действие ``1`` на земле и при пустом баке;
  8) чтение: пять чекпоинтов только на чтение, ``s1``/``s2`` в контур не попадают.

    python _check_numeric_land_skill_brake.py
"""
import json
import os
import sys

import numpy as np
import torch

from numeric_land.data import HORIZ_ACCEL, MAX_THRUST
from numeric_land.expert_hover import NAV_TOL, NAV_V_BAND, NAV_V_MAX, hover_action
from numeric_land.field import FUEL0, GROUND
from numeric_land.mode import THRESHOLD, V_TARGET_LOOP, n_in_of
from numeric_land.relenc import OUT
from numeric_land.skill_assemble import GRID_G, GRID_N, GRID_PADS, N_SEEDS, SEED0, mode_of, take_stamps
from numeric_land.skill_brake import (
    CELL_TOL, CONTROL_RULE, CONTROL_TUBE, EQUIV_STATES, EXPECTED_REPEAT, EXPECTED_SPLIT, LINE,
    KNOWN_LINE, PROBLEM_CELLS, PROBLEM_LINE, REFERENCE_NAME, REPEAT_LINE, TAG, cell_ok, conditions,
    json_path, known_ok, line_ok, locks, problem_ok, refuse_report, refuse_tag, repeat_ok, stray_pt,
)
from numeric_land.skill_guard import REP_G, REP_N, REP_PADS, REP_SEED0
from numeric_land.skill_tube import GOAL_VY, roll_tube
from numeric_land.target_brake import BRAKE_GAIN, brake_action, roll_tube_brake

FAILED = []


def check(name, ok, detail=""):
    if ok:
        print(f"  ok {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def states_720():
    from numeric_land.field import start_and_actions, step_field

    states = []
    for seed in range(SEED0, SEED0 + 4):
        for g, pad in ((15.0, 406.0), (8.0, 406.0), (15.0, 256.0)):
            x, y, vx, vy, _acts = start_and_actions(int(seed), 0)
            fuel = float(FUEL0)
            grounded = False
            for _step in range(60):
                if grounded:
                    break
                states.append(((float(x), float(y), float(vx), float(vy), float(fuel)), g, pad))
                action = int(hover_action((x, y, vx, vy, fuel), g, pad, GOAL_VY))
                x, y, vx, vy, fuel, grounded = step_field(
                    x, y, vx, vy, fuel, False, action, g)
    return states


def synthetic_reference():
    """Синтетика с настоящей арифметикой: сумма трубы `557`, известных `180`, правила `407`."""
    known = {(g, pad) for g in (2.0, 8.0, 15.0) for pad in (256.0, 200.0)}
    reference = {"rule_early": {}, "skill_tube": {}, "totals": {}}
    rest = 13
    rich = 7
    for g in GRID_G:
        for pad in GRID_PADS:
            key = (g, pad)
            if key in known:
                tube = 30
            elif rest > 0:
                tube = 27
                rest -= 1
            else:
                tube = 26
            rule = 21 if rich > 0 else 20
            rich -= 1 if rich > 0 else 0
            reference["rule_early"][key] = {"soft": rule}
            reference["skill_tube"][key] = {"soft": tube}
    reference["totals"] = {"rule_early": CONTROL_RULE, "skill_tube": CONTROL_TUBE}
    if sum(cell["soft"] for cell in reference["skill_tube"].values()) != 557:
        raise SystemExit("синтетика трубы не даёт 557")
    if sum(cell["soft"] for cell in reference["rule_early"].values()) != 407:
        raise SystemExit("синтетика правила не даёт 407")
    return reference


def synth_column(name, reference, key, soft_map=None, dry_cell=None, cells_soft=None):
    cells = []
    for (g, pad), row in reference[key].items():
        soft = int(cells_soft[(g, pad)]) if cells_soft and (g, pad) in cells_soft else int(row["soft"])
        fuel_out = 1 if dry_cell == (g, pad) else 0
        cells.append({"g": float(g), "pad_x": float(pad), "soft": soft,
                      "crash": int(N_SEEDS) - soft - fuel_out, "miss": 0,
                      "fuel_out": fuel_out, "of": int(N_SEEDS), "steps_mean": 100.0,
                      "drift_median_px": 5.0, "vx_approach_mean": 10.0, "clipped": 0})
    return {"name": name, "kind": "brake", "soft": int(sum(c["soft"] for c in cells)),
            "crash": int(sum(c["crash"] for c in cells)), "miss": 0,
            "fuel_out": int(sum(c["fuel_out"] for c in cells)), "of": int(GRID_N),
            "tube_clipped": 0, "cells": cells, "by_g": [], "by_pad": []}


def main():
    torch.set_num_threads(8)
    print(f"контроль {TAG}")
    locks()
    print(f"  ok locks gain={BRAKE_GAIN} line={LINE} known={KNOWN_LINE} tol={CELL_TOL}")

    # 1.1 профиль: запас по боковой скорости, вертикаль и посадка не тронуты
    check("gain_kappa", float(BRAKE_GAIN) == 0.6, f"{BRAKE_GAIN}")
    check("nav_constants", (float(NAV_V_MAX), float(HORIZ_ACCEL), float(NAV_V_BAND), float(NAV_TOL))
          == (40.0, 12.0, 1.0, 6.0), f"{NAV_V_MAX}/{HORIZ_ACCEL}/{NAV_V_BAND}/{NAV_TOL}")
    check("max_thrust", float(MAX_THRUST) == 20.0, f"{MAX_THRUST}")
    state_far = (256.0, 56.0, 0.0, 0.0, 10.0)              # pad 406, dx = +150, vx = 0
    want_exact = min(NAV_V_MAX, float(np.sqrt(2.0 * HORIZ_ACCEL * 150.0)))
    fast = (256.0, 56.0, 0.6 * want_exact, 0.0, 10.0)      # скорость ровно на профиле с запасом
    full = (256.0, 56.0, want_exact, 0.0, 10.0)            # скорость на полном профиле
    slow = (256.0, 56.0, 0.0, 0.0, 10.0)
    check("gain_holds_at_brake_speed", int(brake_action(fast, 8.0, 406.0, GOAL_VY)) % 3 == 1,
          f"a={brake_action(fast, 8.0, 406.0, GOAL_VY)} при vx={0.6 * want_exact:.2f}")
    check("gain_brakes_at_full_speed", int(brake_action(full, 8.0, 406.0, GOAL_VY)) % 3 == 0,
          f"a={brake_action(full, 8.0, 406.0, GOAL_VY)} при vx={want_exact:.2f} — правило бы разгоняло")
    check("gain_accelerates_from_zero", int(brake_action(slow, 8.0, 406.0, GOAL_VY)) % 3 == 2,
          f"a={brake_action(slow, 8.0, 406.0, GOAL_VY)}")
    check("gain_zero_is_rule_on_far_state",
          int(brake_action(state_far, 8.0, 406.0, GOAL_VY, 1.0)) == int(hover_action(state_far, 8.0, 406.0, GOAL_VY)),
          "gain=1 совпал")
    land_states = [(300.0 + 0.5 * k, 40.0 + 0.7 * k, 3.0 - 0.1 * k, 4.0 + 0.5 * k, 10.0)
                   for k in range(40)]
    bad_land = sum(1 for state in land_states
                   if int(brake_action(state, 15.0, 302.0, GOAL_VY)) != int(hover_action(state, 15.0, 302.0, GOAL_VY)))
    check("land_branch_is_rule", bad_land == 0, f"расхождений {bad_land}")
    check("ground_and_dry", int(brake_action((256.0, GROUND, 0.0, 3.0, 10.0), 15.0, 406.0, GOAL_VY)) == 1
          and int(brake_action((256.0, 56.0, 0.0, 0.0, 0.0), 15.0, 406.0, GOAL_VY)) == 1, "действие 1")

    # 1.2 эквивалентность
    states = states_720()
    check("equiv_states", len(states) == int(EQUIV_STATES), f"{len(states)}")
    bad_action = sum(1 for state, g, pad in states
                     if int(brake_action(state, g, pad, GOAL_VY, 1.0)) != int(hover_action(state, g, pad, GOAL_VY)))
    bad_frames = sum(1 for state, g, pad in states
                     if roll_tube_brake(state, g, pad, 1.0, GOAL_VY)[0] != tuple(roll_tube(state, g, pad)[0]))
    check("equiv_action", bad_action == 0, f"расхождений {bad_action}")
    check("equiv_frames", bad_frames == 0, f"расхождений {bad_frames}")
    frames, calls, clipped = roll_tube_brake((256.0, 56.0, 0.0, 6.0, 10.0), 8.0, 406.0)
    check("tube_targets", len(frames) == 3 and 1 <= calls <= 3, f"кадров {len(frames)} вызовов {calls} обрезка {clipped}")

    # 1.3 сетка
    main_keys = {(g, pad) for g in GRID_G for pad in GRID_PADS}
    rep_keys = {(g, pad) for g in REP_G for pad in REP_PADS}
    check("grid_main", len(main_keys) == 20 and len(GRID_G) * len(GRID_PADS) * N_SEEDS == GRID_N == 600,
          f"{len(main_keys)} клеток {GRID_N} эпизодов")
    check("grid_repeat", len(rep_keys) == 6 and len(REP_G) * len(REP_PADS) * N_SEEDS == REP_N == 180,
          f"{len(rep_keys)} клеток {REP_N} эпизодов")
    check("seeds_main", (SEED0, SEED0 + N_SEEDS - 1) == (210000, 210029), f"{SEED0}")
    check("seeds_repeat", (REP_SEED0, REP_SEED0 + N_SEEDS - 1) == (220000, 220029), f"{REP_SEED0}")
    check("seeds_disjoint",
          not (set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & set(range(SEED0, SEED0 + N_SEEDS)))
          and not (set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & set(range(99000, 99030))), "пересечений нет")

    # 1.4 записи
    path = os.path.join(OUT, REFERENCE_NAME)
    check("reference_exists", os.path.isfile(path), path)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        columns = {column["name"]: column for column in data["columns"]}
        check("reference_rule", int(columns["rule_early"]["soft"]) == int(CONTROL_RULE),
              f"{columns['rule_early']['soft']}")
        check("reference_tube", int(columns["skill_tube"]["soft"]) == int(CONTROL_TUBE),
              f"{columns['skill_tube']['soft']}")
        check("reference_cells", len(columns["rule_early"]["cells"]) == 20
              and len(columns["skill_tube"]["cells"]) == 20, "по 20 клеток")
    split_path = os.path.join(OUT, "numeric_land_skill_split.json")
    if os.path.isfile(split_path):
        with open(split_path, encoding="utf-8") as fh:
            split = json.load(fh)
        check("split_record", int(split["columns"][2]["soft"]) == int(EXPECTED_SPLIT)
              and int(split["repeat_columns"][2]["soft"]) == int(EXPECTED_REPEAT),
              f"{split['columns'][2]['soft']}/{split['repeat_columns'][2]['soft']}")
    else:
        check("split_record", False, "нет числовой записи разделения")

    # 1.5 арифметика условий
    check("line", line_ok(LINE) and not line_ok(LINE - 1), f"{LINE}/{LINE - 1}")
    check("known", known_ok(KNOWN_LINE) and not known_ok(KNOWN_LINE - 1), f"{KNOWN_LINE}/{KNOWN_LINE - 1}")
    check("problem", problem_ok(PROBLEM_LINE) and not problem_ok(PROBLEM_LINE - 1),
          f"{PROBLEM_LINE}/{PROBLEM_LINE - 1}")
    check("repeat", repeat_ok(REPEAT_LINE) and not repeat_ok(REPEAT_LINE - 1), f"{REPEAT_LINE}/{REPEAT_LINE - 1}")
    check("cell_tol", cell_ok(14, 17) and not cell_ok(13, 17), "14 при 17 проходит, 13 нет")
    reference = synthetic_reference()
    equiv_ok = {"states": int(EQUIV_STATES), "bad_action": 0, "bad_frames": 0, "ok": True}
    rule_col = synth_column("rule_early", reference, "rule_early")
    base_brake = synth_column("tube_brake", reference, "skill_tube")
    rep_rule = {"name": "rule_early", "soft": 150, "cells": [], "of": int(REP_N)}
    rep_ok = {"name": "tube_brake", "soft": 150, "cells": [], "of": int(REP_N)}
    numbers = conditions(base_brake, rule_col, rep_ok, rep_rule, reference, equiv_ok)
    check("conditions_base", numbers["control_ok"] and numbers["no_regression"] and numbers["dry_ok"]
          and numbers["repeat_ok"] and numbers["equiv_ok"], f"мягких {base_brake['soft']}")
    drop = {(PROBLEM_CELLS[0]): int(reference["skill_tube"][PROBLEM_CELLS[0]]["soft"])}
    soft_map = dict(drop)
    soft_map[PROBLEM_CELLS[0]] = PROBLEM_LINE - 1
    weak = synth_column("tube_brake", reference, "skill_tube", cells_soft=soft_map)
    numbers_problem = conditions(weak, rule_col, rep_ok, rep_rule, reference, equiv_ok)
    check("problem_caught", not numbers_problem["problem_ok"], f"{numbers_problem['problem_cells']}")
    dry_map = {(15.0, 406.0): int(reference["skill_tube"][(15.0, 406.0)]["soft"])}
    dry_col = synth_column("tube_brake", reference, "skill_tube", dry_cell=(15.0, 406.0))
    numbers_dry = conditions(dry_col, rule_col, rep_ok, rep_rule, reference, equiv_ok)
    check("dry_caught", not numbers_dry["dry_ok"], f"сухих {numbers_dry['heavy_fuel_out']}")
    del dry_map
    reg_map = {(8.0, 200.0): int(reference["skill_tube"][(8.0, 200.0)]["soft"]) - int(CELL_TOL) - 1}
    reg_col = synth_column("tube_brake", reference, "skill_tube", cells_soft=reg_map)
    numbers_reg = conditions(reg_col, rule_col, rep_ok, rep_rule, reference, equiv_ok)
    check("regression_caught", not numbers_reg["no_regression"], f"{numbers_reg['regression_cells']}")
    numbers_rep = conditions(base_brake, rule_col, {"name": "tube_brake", "soft": REPEAT_LINE - 1,
                                                    "cells": [], "of": int(REP_N)}, rep_rule,
                             reference, equiv_ok)
    check("repeat_caught", not numbers_rep["repeat_ok"], f"{numbers_rep['repeat_soft']}")
    numbers_line = conditions(weak, rule_col, rep_ok, rep_rule, reference, equiv_ok)
    check("line_counts_weak_column", int(weak["soft"]) < int(LINE) or numbers_line["line_ok"],
          f"сумма {weak['soft']}")
    broken = {"states": int(EQUIV_STATES), "bad_action": 1, "bad_frames": 0, "ok": False}
    numbers_equiv = conditions(base_brake, rule_col, rep_ok, rep_rule, reference, broken)
    check("equiv_caught", not numbers_equiv["equiv_ok"] and not numbers_equiv["pilot_pass"], "зачёт не зачтён")

    # 1.6 защита отчёта
    refused = 0
    for tag in ("", "numeric-land-skill-split", "numeric-land-skill-brake-2"):
        try:
            refuse_tag(tag)
        except SystemExit:
            refused += 1
    check("refuse_tag", refused == 3, f"{refused} из 3")
    try:
        refuse_tag(TAG)
        tag_ok = True
    except SystemExit:
        tag_ok = False
    check("accept_tag", tag_ok, TAG)
    target = os.path.join(OUT, "_check_brake_hold.json")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("{}")
    try:
        refuse_report(target)
        report_refused = False
    except SystemExit:
        report_refused = True
    os.remove(target)
    check("refuse_report", report_refused, "поверх файла не пишет")
    check("report_absent", not os.path.isfile(json_path()), json_path())
    check("stray_pt", not stray_pt(), "landd_brake_*.pt нет")

    # 1.7 голова и режим
    check("threshold", float(THRESHOLD) == 0.5 and mode_of(0.5) == "land" and mode_of(0.499) == "approach",
          "0.5 → land, 0.499 → approach")
    check("head_input", int(n_in_of("latent")) == 33, f"{n_in_of('latent')}")
    check("v_target", float(V_TARGET_LOOP) == 6.0 and float(GOAL_VY) == 6.0, f"{V_TARGET_LOOP}/{GOAL_VY}")
    check("nav_tol_again", float(NAV_TOL) == 6.0, f"{NAV_TOL}")

    # 1.8 чтение
    before = take_stamps()
    names = sorted(before)
    check("stamps_read", len(names) >= 4 and all(before[name] for name in names if before[name]),
          f"{len(names)} файлов")
    from numeric_land.skill_assemble import EARLY_PATH, approach_path, latent_path, pad_path

    s0 = [os.path.basename(EARLY_PATH), os.path.basename(latent_path()),
          os.path.basename(pad_path(0)), os.path.basename(approach_path())]
    check("s0_paths", s0 == ["landd_early_s0.pt", "landd_latent_s0.pt", "landk_pad_s0.pt",
                             "landk_approach_s0.pt"], ",".join(s0))
    after = take_stamps()
    check("stamps_stable", before == after, "чтение не сдвинуло штампы")

    if FAILED:
        print("ПРОВАЛ:", ", ".join(FAILED))
        return 1
    print(f"ok brake gain={BRAKE_GAIN} line={LINE} known={KNOWN_LINE} tol={CELL_TOL} "
          f"problem={PROBLEM_LINE} repeat={REPEAT_LINE} equiv_states={EQUIV_STATES} "
          f"grid={GRID_N} repeat_n={REP_N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

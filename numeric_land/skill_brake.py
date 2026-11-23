"""Труба с торможением в цели: ориентир строится профилем с запасом ``BRAKE_GAIN = 0.6``.

Сетка — та же, что в этапе 4 (600 эпизодов, сиды 210000…210029), чтобы прямо сравнить
с записями 509 и 407. Блок повтора — свежие сиды 220000…220029, тяжести 15/20,
центры 106/356/406 (180 эпизодов), как в разделении по тяжести.

Столбцы: ``rule_early`` (контроль, правило) и ``tube_brake`` (зачёт, труба с запасом).

    python -m numeric_land.skill_brake --tag numeric-land-skill-brake
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from numeric_land.expert_hover import NAV_TOL, hover_action
from numeric_land.field import FUEL0, GROUND, VY_SOFT, start_and_actions, step_field
from numeric_land.mode import V_TARGET_LOOP
from numeric_land.pad import outcome_at
from numeric_land.relenc import OUT, ROOT, write_json
from numeric_land.skill_assemble import (
    GRID_G, GRID_N, GRID_PADS, KNOWN_G, KNOWN_N, KNOWN_PADS, N_SEEDS, SEED0,
    RESULT_CODE, classify, load_bundle, predictor_of, rule_action, take_stamps,
)
from numeric_land.skill_guard import REP_G, REP_N, REP_PADS, REP_SEED0, episodes
from numeric_land.skill_land import pair_table, world_batch
from numeric_land.skill_tube import GOAL_VY, roll_tube
from numeric_land.target_brake import BRAKE_GAIN, brake_action, brake_target_action, roll_tube_brake
from numeric_land.train_goal import THREADS

TAG = "numeric-land-skill-brake"
JSON_NAME = "numeric_land_skill_brake.json"
LINE = 557
KNOWN_LINE = 179
CELL_TOL = 3
PROBLEM_CELLS = ((15.0, 406.0), (20.0, 356.0))
PROBLEM_LINE = 24
REPEAT_LINE = 150
CONTROL_RULE = 407
CONTROL_TUBE = 509
EXPECTED_SPLIT = 557
EXPECTED_REPEAT = 150
EQUIV_STATES = 720
REFERENCE_NAME = "numeric_land_skill_assemble.json"


def locks() -> None:
    """Замыкание констант: сетка, линия, коэффициент, правило."""
    if TAG != "numeric-land-skill-brake":
        raise RuntimeError(TAG)
    if GRID_G != (2.0, 8.0, 15.0, 20.0) or GRID_PADS != (106.0, 200.0, 256.0, 356.0, 406.0):
        raise RuntimeError((GRID_G, GRID_PADS))
    if SEED0 != 210_000 or N_SEEDS != 30 or GRID_N != 600:
        raise RuntimeError((SEED0, N_SEEDS, GRID_N))
    if REP_SEED0 != 220_000 or REP_N != 180 or REP_G != (15.0, 20.0):
        raise RuntimeError((REP_SEED0, REP_N, REP_G))
    if REP_PADS != (106.0, 356.0, 406.0):
        raise RuntimeError(REP_PADS)
    if float(BRAKE_GAIN) != 0.6:
        raise RuntimeError(BRAKE_GAIN)
    if int(LINE) != 557 or int(KNOWN_LINE) != 179 or int(CELL_TOL) != 3:
        raise RuntimeError((LINE, KNOWN_LINE, CELL_TOL))
    if tuple(float(v) for pair in PROBLEM_CELLS for v in pair) != (15.0, 406.0, 20.0, 356.0):
        raise RuntimeError(PROBLEM_CELLS)
    if int(PROBLEM_LINE) != 24 or int(REPEAT_LINE) != 150:
        raise RuntimeError((PROBLEM_LINE, REPEAT_LINE))
    if int(CONTROL_RULE) != 407 or int(CONTROL_TUBE) != 509:
        raise RuntimeError((CONTROL_RULE, CONTROL_TUBE))
    if int(EXPECTED_SPLIT) != 557 or int(EXPECTED_REPEAT) != 150:
        raise RuntimeError((EXPECTED_SPLIT, EXPECTED_REPEAT))
    if int(EQUIV_STATES) != 720:
        raise RuntimeError(EQUIV_STATES)
    if float(NAV_TOL) != 6.0 or float(VY_SOFT) != 8.0 or float(V_TARGET_LOOP) != 6.0:
        raise RuntimeError((NAV_TOL, VY_SOFT, V_TARGET_LOOP))
    if float(GOAL_VY) != 6.0:
        raise RuntimeError(GOAL_VY)
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise RuntimeError((FUEL0, GROUND, THREADS))
    if set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & set(range(SEED0, SEED0 + N_SEEDS)):
        raise RuntimeError("сиды блока повтора совпали с основной сеткой")
    if set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & set(range(99000, 99030)):
        raise RuntimeError("сиды блока повтора совпали с этапом 3")
    if not known_ok(KNOWN_LINE) or known_ok(KNOWN_LINE - 1):
        raise RuntimeError("линия знакомой части сдвинута")
    if not line_ok(LINE) or line_ok(LINE - 1):
        raise RuntimeError("линия 557 сдвинута")
    if not problem_ok(PROBLEM_LINE) or problem_ok(PROBLEM_LINE - 1):
        raise RuntimeError("линия провальных клеток сдвинута")
    if not repeat_ok(REPEAT_LINE) or repeat_ok(REPEAT_LINE - 1):
        raise RuntimeError("линия повтора сдвинута")
    if not cell_ok(14, 17) or cell_ok(13, 17):
        raise RuntimeError("допуск клетки 3 сдвинут")


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit("нужен тег numeric-land-skill-brake. Другие теги запрещены.")


def json_path() -> str:
    return os.path.join(OUT, JSON_NAME)


def refuse_report(path: str) -> None:
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}, второй прогон отчёт не затирает")


def stray_pt():
    return glob.glob(os.path.join(OUT, "landd_brake_*.pt"))


def known_ok(soft) -> bool:
    return int(soft) >= int(KNOWN_LINE)


def line_ok(soft) -> bool:
    return int(soft) >= int(LINE)


def problem_ok(soft) -> bool:
    return int(soft) >= int(PROBLEM_LINE)


def repeat_ok(soft) -> bool:
    return int(soft) >= int(REPEAT_LINE)


def cell_ok(soft, reference) -> bool:
    return int(soft) >= int(reference) - int(CELL_TOL)


def equivalence() -> dict:
    """gain = 1.0 обязан совпасть с roll_tube по кадрам и с hover_action по действию."""
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
    if len(states) != int(EQUIV_STATES):
        raise RuntimeError(len(states))
    bad_action = 0
    bad_frames = 0
    for state, g, pad in states:
        if int(brake_action(state, g, pad, GOAL_VY, 1.0)) != int(hover_action(state, g, pad, GOAL_VY)):
            bad_action += 1
        if roll_tube_brake(state, g, pad, 1.0, GOAL_VY)[0] != tuple(roll_tube(state, g, pad)[0]):
            bad_frames += 1
    return {"states": int(len(states)), "bad_action": int(bad_action),
            "bad_frames": int(bad_frames),
            "ok": bool(bad_action == 0 and bad_frames == 0)}


def reference_cells():
    """Запись этапа 4: столбцы по клеткам для контроля и допуска."""
    path = os.path.join(OUT, REFERENCE_NAME)
    if not os.path.isfile(path):
        raise SystemExit(f"нет {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    columns = {column["name"]: column for column in data["columns"]}
    out = {}
    for name in ("rule_early", "skill_tube"):
        cells = {}
        for cell in columns[name]["cells"]:
            cells[(float(cell["g"]), float(cell["pad_x"]))] = cell
        if len(cells) != 20:
            raise RuntimeError((name, len(cells)))
        out[name] = cells
    out["totals"] = {"rule_early": int(columns["rule_early"]["soft"]),
                     "skill_tube": int(columns["skill_tube"]["soft"])}
    return out


def _cell(bucket) -> dict:
    if int(bucket["of"]) != int(N_SEEDS):
        raise RuntimeError(bucket["of"])
    if int(bucket["soft"]) + int(bucket["crash"]) + int(bucket["miss"]) != int(N_SEEDS):
        raise RuntimeError(bucket)
    return {
        "g": float(bucket["g"]),
        "pad_x": float(bucket["pad_x"]),
        "soft": int(bucket["soft"]),
        "crash": int(bucket["crash"]),
        "miss": int(bucket["miss"]),
        "fuel_out": int(bucket["fuel_out"]),
        "of": int(N_SEEDS),
        "steps_mean": round(float(bucket["steps"]) / float(N_SEEDS), 3),
        "drift_median_px": None if not bucket["drift"] else round(
            float(np.median(np.asarray(bucket["drift"], np.float64))), 2),
        "vx_approach_mean": None if not bucket["vx_n"] else round(
            float(bucket["vx_sum"]) / float(bucket["vx_n"]), 3),
        "clipped": int(bucket["clipped"]),
    }


def _group(cells, key):
    order = []
    buckets = {}
    for cell in cells:
        mark = cell[key]
        if mark not in buckets:
            order.append(mark)
            buckets[mark] = {"soft": 0, "crash": 0, "miss": 0, "fuel_out": 0, "of": 0,
                             "steps": 0.0, "key": mark}
        bucket = buckets[mark]
        for field in ("soft", "crash", "miss", "fuel_out", "of"):
            bucket[field] += int(cell[field])
        bucket["steps"] += float(cell["steps_mean"])
    sizes = {}
    for cell in cells:
        sizes[cell[key]] = sizes.get(cell[key], 0) + 1
    return [{"key": buckets[mark]["key"],
             "soft": buckets[mark]["soft"],
             "crash": buckets[mark]["crash"],
             "miss": buckets[mark]["miss"],
             "fuel_out": buckets[mark]["fuel_out"],
             "of": buckets[mark]["of"],
             "steps_mean": round(buckets[mark]["steps"] / sizes[mark], 3)}
            for mark in order]


@torch.no_grad()
def fly_episode(kind, head, encoder, ema, predictors, actions, step_pairs, world_b, row):
    x, y, vx, vy, _acts = start_and_actions(int(row["seed"]), 0)
    if (float(x), float(y), float(vx), float(vy)) != (
        float(row["x"]), float(row["y"]), float(row["vx"]), float(row["vy"]),
    ):
        raise RuntimeError("старт разошёлся со start_and_actions")
    fuel = float(FUEL0)
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    if kind not in ("rule", "brake"):
        raise RuntimeError(kind)
    grounded = False
    steps = 0
    fuel_out = False
    clipped_n = 0
    entered = False
    drift = 0.0
    vx_sum = 0.0
    vx_n = 0
    while steps < 900 and not grounded:
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        mode, _prob = classify(head, encoder, state, g, pad_x)
        clipped = False
        if mode is None:
            action = 1
        elif kind == "rule":
            action = rule_action(state, g, pad_x, mode)
        else:
            action, clipped = brake_target_action(
                predictor_of(mode, predictors), encoder, ema, actions, step_pairs, world_b,
                state, g, pad_x)
        if clipped:
            clipped_n += 1
        if mode == "approach":
            vx_sum += abs(float(vx))
            vx_n += 1
        if not entered and abs(float(x) - pad_x) <= NAV_TOL:
            entered = True
        if entered:
            drift = max(drift, abs(float(x) - pad_x))
        x, y, vx, vy, fuel, grounded = step_field(
            float(x), float(y), float(vx), float(vy), float(fuel), False, int(action), float(g))
        steps += 1
        if float(fuel) <= 0.0 and not grounded:
            fuel_out = True
    result = outcome_at(x, y, vy, grounded, pad_x)
    if result not in RESULT_CODE:
        raise RuntimeError(result)
    return {
        "seed": int(row["seed"]),
        "g": float(g),
        "pad_x": float(pad_x),
        "result": result,
        "steps": int(steps),
        "fuel_out": bool(fuel_out),
        "clipped": int(clipped_n),
        "drift": float(drift),
        "vx_sum": float(vx_sum),
        "vx_n": int(vx_n),
    }


def fly_column(name, kind, rows, n, head, encoder, ema, predictors, actions, step_pairs):
    if len(rows) != int(n):
        raise RuntimeError((len(rows), n))
    buckets = {}
    order = []
    worlds = {}
    tube_clipped = 0
    for row in rows:
        key = (float(row["g"]), float(row["pad_x"]))
        if key not in buckets:
            order.append(key)
            buckets[key] = {"g": key[0], "pad_x": key[1], "soft": 0, "crash": 0, "miss": 0,
                            "fuel_out": 0, "of": 0, "steps": 0.0, "drift": [], "vx_sum": 0.0,
                            "vx_n": 0, "clipped": 0}
        if key not in worlds:
            worlds[key] = world_batch(key[0], key[1])
        out = fly_episode(kind, head, encoder, ema, predictors, actions, step_pairs, worlds[key], row)
        tube_clipped += int(out["clipped"])
        bucket = buckets[key]
        bucket["of"] += 1
        bucket["steps"] += float(out["steps"])
        bucket["drift"].append(float(out["drift"]))
        bucket["vx_sum"] += float(out["vx_sum"])
        bucket["vx_n"] += int(out["vx_n"])
        bucket["clipped"] += int(out["clipped"])
        if out["result"] == "мягкая":
            bucket["soft"] += 1
        elif out["result"] == "удар":
            bucket["crash"] += 1
        else:
            bucket["miss"] += 1
        if out["fuel_out"]:
            bucket["fuel_out"] += 1
    cells = [_cell(buckets[key]) for key in order]
    soft = sum(cell["soft"] for cell in cells)
    return {
        "name": name,
        "kind": kind,
        "soft": int(soft),
        "crash": int(sum(cell["crash"] for cell in cells)),
        "miss": int(sum(cell["miss"] for cell in cells)),
        "fuel_out": int(sum(cell["fuel_out"] for cell in cells)),
        "of": int(n),
        "tube_clipped": int(tube_clipped),
        "cells": cells,
        "by_g": _group(cells, "g"),
        "by_pad": _group(cells, "pad_x"),
    }


def _column(columns, name):
    for column in columns:
        if column["name"] == name:
            return column
    raise RuntimeError(name)


def conditions(brake, rule, rep_brake, rep_rule, reference, equiv) -> dict:
    """Восемь условий спеки."""
    cell_rule = {(cell["g"], cell["pad_x"]): cell for cell in rule["cells"]}
    cell_brake = {(cell["g"], cell["pad_x"]): cell for cell in brake["cells"]}
    if set(cell_rule) != set(cell_brake):
        raise RuntimeError("клетки столбцов не совпали")
    control_rows = 0
    for key, cell in cell_rule.items():
        if int(cell["soft"]) != int(reference["rule_early"][key]["soft"]):
            control_rows += 1
    control = bool(int(rule["soft"]) == int(CONTROL_RULE) and control_rows == 0)
    known_soft = sum(int(cell_brake[(g, pad)]["soft"]) for g in KNOWN_G for pad in KNOWN_PADS)
    known = known_ok(known_soft)
    losses = []
    for key, cell in cell_brake.items():
        if not cell_ok(int(cell["soft"]), int(reference["skill_tube"][key]["soft"])):
            losses.append({"g": key[0], "pad_x": key[1], "soft": int(cell["soft"]),
                           "reference": int(reference["skill_tube"][key]["soft"])})
    no_regression = bool(int(brake["soft"]) >= int(CONTROL_TUBE) and not losses)
    line = line_ok(int(brake["soft"]))
    problems = {f"{g:g}/{pad:g}": int(cell_brake[(g, pad)]["soft"]) for g, pad in PROBLEM_CELLS}
    problem = bool(all(problem_ok(value) for value in problems.values()))
    heavy = [cell for cell in brake["cells"] if float(cell["g"]) >= 15.0]
    heavy_dry = int(sum(cell["fuel_out"] for cell in heavy))
    dry = bool(heavy_dry == 0)
    repeat = bool(repeat_ok(int(rep_brake["soft"])) and int(rep_brake["soft"]) >= int(rep_rule["soft"]))
    eq = bool(equiv["ok"])
    pilot_pass = bool(control and eq and known and no_regression and line and problem and dry and repeat)
    return {
        "control_ok": control,
        "control_rows_diff": int(control_rows),
        "equiv_ok": eq,
        "known_ok": known,
        "known_soft": int(known_soft),
        "known_of": int(KNOWN_N),
        "no_regression": no_regression,
        "regression_cells": losses,
        "line_ok": line,
        "problem_ok": problem,
        "problem_cells": problems,
        "dry_ok": dry,
        "heavy_fuel_out": int(heavy_dry),
        "repeat_ok": repeat,
        "repeat_soft": int(rep_brake["soft"]),
        "repeat_rule": int(rep_rule["soft"]),
        "pilot_pass": pilot_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    refuse_tag(args.tag)
    locks()
    torch.set_num_threads(THREADS)
    report = json_path()
    refuse_report(report)
    if stray_pt():
        raise SystemExit("уже есть landd_brake_*.pt, новый чекпоинт писать нельзя")
    print(f"BRAKE_START grid={GRID_N} repeat={REP_N} gain={BRAKE_GAIN}", flush=True)
    hover_path = os.path.join(ROOT, "numeric_land", "expert_hover.py")
    hover_bytes = open(hover_path, "rb").read()
    tube_path = os.path.join(ROOT, "numeric_land", "skill_tube.py")
    tube_bytes = open(tube_path, "rb").read()
    before = take_stamps()
    bundle = load_bundle()
    if take_stamps() != before:
        raise RuntimeError("чтение сдвинуло штамп")
    print("BRAKE_LOADED", flush=True)
    predictors = {"land": bundle["pad"], "approach": bundle["approach"]}
    encoder = bundle["encoder"]
    ema = bundle["ema"]
    early = bundle["early"]
    actions, step_pairs = pair_table()
    reference = reference_cells()
    equiv = equivalence()
    print(f"BRAKE_EQUIV states={equiv['states']} bad_action={equiv['bad_action']} "
          f"bad_frames={equiv['bad_frames']} ok={int(equiv['ok'])}", flush=True)
    if not equiv["ok"]:
        raise RuntimeError("gain = 1.0 не совпал с roll_tube")

    rows_main = episodes(SEED0, GRID_G, GRID_PADS)
    rows_rep = episodes(REP_SEED0, REP_G, REP_PADS)
    if len(rows_main) != GRID_N or len(rows_rep) != REP_N:
        raise RuntimeError((len(rows_main), len(rows_rep)))

    main_cols = []
    main_cols.append(fly_column("rule_early", "rule", rows_main, GRID_N, early, encoder, ema,
                                predictors, actions, step_pairs))
    print(f"BRAKE_COLUMN rule_early={_column(main_cols, 'rule_early')['soft']}", flush=True)
    main_cols.append(fly_column("tube_brake", "brake", rows_main, GRID_N, early, encoder, ema,
                                predictors, actions, step_pairs))
    brake = _column(main_cols, "tube_brake")
    rule = _column(main_cols, "rule_early")
    print(f"BRAKE_COLUMN tube_brake={brake['soft']} rule_early={rule['soft']}", flush=True)

    rep_cols = []
    rep_cols.append(fly_column("rule_early", "rule", rows_rep, REP_N, early, encoder, ema,
                               predictors, actions, step_pairs))
    rep_cols.append(fly_column("tube_brake", "brake", rows_rep, REP_N, early, encoder, ema,
                               predictors, actions, step_pairs))
    rep_brake = _column(rep_cols, "tube_brake")
    rep_rule = _column(rep_cols, "rule_early")
    print(f"BRAKE_REPEAT tube_brake={rep_brake['soft']}/{REP_N} "
          f"rule_early={rep_rule['soft']}/{REP_N}", flush=True)

    if take_stamps() != before:
        raise RuntimeError("полёт сдвинул штамп")
    if open(hover_path, "rb").read() != hover_bytes:
        raise RuntimeError("правило изменилось")
    if open(tube_path, "rb").read() != tube_bytes:
        raise RuntimeError("skill_tube изменился")
    if stray_pt():
        raise RuntimeError("появился landd_brake_*.pt")

    numbers = conditions(brake, rule, rep_brake, rep_rule, reference, equiv)
    payload = {
        "tag": TAG,
        "grid_n": GRID_N,
        "grid_g": [float(g) for g in GRID_G],
        "grid_pads": [float(pad) for pad in GRID_PADS],
        "cell_n": int(N_SEEDS),
        "repeat_n": int(REP_N),
        "repeat_g": [float(g) for g in REP_G],
        "repeat_pads": [float(pad) for pad in REP_PADS],
        "repeat_seeds": [int(REP_SEED0), int(REP_SEED0 + N_SEEDS - 1)],
        "brake_gain": float(BRAKE_GAIN),
        "line": int(LINE),
        "known_line": int(KNOWN_LINE),
        "cell_tol": int(CELL_TOL),
        "problem_line": int(PROBLEM_LINE),
        "problem_cells": [[float(g), float(pad)] for g, pad in PROBLEM_CELLS],
        "repeat_line": int(REPEAT_LINE),
        "known_n": int(KNOWN_N),
        "known_g": [float(g) for g in KNOWN_G],
        "known_pads": [float(pad) for pad in KNOWN_PADS],
        "control_rule": int(CONTROL_RULE),
        "control_tube": int(CONTROL_TUBE),
        "expected_split": int(EXPECTED_SPLIT),
        "expected_repeat": int(EXPECTED_REPEAT),
        "equivalence": equiv,
        "control_ok": bool(numbers["control_ok"]),
        "control_rows_diff": int(numbers["control_rows_diff"]),
        "equiv_ok": bool(numbers["equiv_ok"]),
        "known": {"soft": int(numbers["known_soft"]), "of": int(numbers["known_of"]),
                  "line": int(KNOWN_LINE), "ok": bool(numbers["known_ok"])},
        "known_ok": bool(numbers["known_ok"]),
        "no_regression": bool(numbers["no_regression"]),
        "regression_cells": numbers["regression_cells"],
        "line_ok": bool(numbers["line_ok"]),
        "problem_ok": bool(numbers["problem_ok"]),
        "problem_cells_soft": numbers["problem_cells"],
        "dry_ok": bool(numbers["dry_ok"]),
        "heavy_fuel_out": int(numbers["heavy_fuel_out"]),
        "repeat_ok": bool(numbers["repeat_ok"]),
        "repeat_soft": int(numbers["repeat_soft"]),
        "repeat_rule": int(numbers["repeat_rule"]),
        "pilot_pass": bool(numbers["pilot_pass"]),
        "threshold": 0.5,
        "horizon": 3,
        "tube_targets": 3,
        "n_sequences": 729,
        "cem": False,
        "rule_action_in_scored": False,
        "heads_trained": False,
        "stamps_unchanged": True,
        "v_target": float(V_TARGET_LOOP),
        "seeds": [int(SEED0), int(SEED0 + N_SEEDS - 1)],
        "step_origin": 0,
        "height_gt_128_is_gate": False,
        "heads": {"early": "landd_early_s0.pt"},
        "skills": {"land": "landk_pad_s0.pt", "approach": "landk_approach_s0.pt"},
        "prior_records": {"stage4_tube": int(CONTROL_TUBE), "stage4_rule": int(CONTROL_RULE),
                          "split_soft": int(EXPECTED_SPLIT), "split_repeat": int(EXPECTED_REPEAT)},
        "columns": main_cols,
        "repeat_columns": rep_cols,
    }
    if take_stamps() != before:
        raise RuntimeError("перед записью штамп сдвинулся")
    write_json(report, payload)
    if take_stamps() != before:
        raise RuntimeError("запись отчёта задела чекпоинт")
    print(
        f"BRAKE_DONE brake={brake['soft']} rule={rule['soft']} "
        f"known={numbers['known_soft']}/180 known_ok={int(numbers['known_ok'])} "
        f"no_regression={int(numbers['no_regression'])} line_ok={int(numbers['line_ok'])} "
        f"problem_ok={int(numbers['problem_ok'])} dry_ok={int(numbers['dry_ok'])} "
        f"repeat={numbers['repeat_soft']}/{REP_N} repeat_ok={int(numbers['repeat_ok'])} "
        f"control_ok={int(numbers['control_ok'])} equiv_ok={int(numbers['equiv_ok'])} "
        f"pilot_pass={int(numbers['pilot_pass'])}",
        flush=True,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

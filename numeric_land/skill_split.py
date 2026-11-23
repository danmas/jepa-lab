"""Разделение по тяжести: при g ≥ 15 эпизод ведёт правило, при g ≤ 8 — труба.

Контроллер выбирается один раз на эпизод по тяжести, обратной передачи нет. Обучения нет.

Основная сетка — та же, что в этапе 4 (600 эпизодов, сиды 210000…210029), чтобы прямо
сравнить с записями 509 и 407. Блок повтора — свежие сиды 220000…220029, тяжести 15/20,
центры 106/356/406 (180 эпизодов).

    python -m numeric_land.skill_split --tag numeric-land-skill-split
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from numeric_land.expert_hover import NAV_TOL
from numeric_land.field import FUEL0, GROUND, VY_SOFT, start_and_actions, step_field
from numeric_land.mode import V_TARGET_LOOP
from numeric_land.pad import outcome_at
from numeric_land.relenc import OUT, ROOT, write_json
from numeric_land.skill_assemble import (
    GRID_G, GRID_N, GRID_PADS, KNOWN_G, KNOWN_N, KNOWN_PADS, N_SEEDS, SEED0,
    STAGE1_HEIGHT, load_bundle, take_stamps,
)
from numeric_land.skill_assemble import (
    RESULT_CODE, SwitchLog, classify, predictor_of, rule_action, switch_fields, tube_action,
)
from numeric_land.skill_guard import REP_G, REP_N, REP_PADS, REP_SEED0, episodes
from numeric_land.train_goal import THREADS

TAG = "numeric-land-skill-split"
JSON_NAME = "numeric_land_skill_split.json"
SPLIT_G = 15.0
LINE = 550
KNOWN_LINE = 179
CELL_TOL = 3
CONTROL_TUBE = 509
CONTROL_RULE = 407
ORACLE_EXPECT = 560
EXPECTED_SPLIT = 557
EXPECTED_REPEAT = 150


def locks() -> None:
    if TAG != "numeric-land-skill-split":
        raise RuntimeError(TAG)
    if GRID_G != (2.0, 8.0, 15.0, 20.0) or GRID_PADS != (106.0, 200.0, 256.0, 356.0, 406.0):
        raise RuntimeError((GRID_G, GRID_PADS))
    if SEED0 != 210_000 or N_SEEDS != 30 or GRID_N != 600:
        raise RuntimeError((SEED0, N_SEEDS, GRID_N))
    if REP_SEED0 != 220_000 or REP_N != 180 or REP_G != (15.0, 20.0):
        raise RuntimeError((REP_SEED0, REP_N, REP_G))
    if REP_PADS != (106.0, 356.0, 406.0):
        raise RuntimeError(REP_PADS)
    if float(SPLIT_G) != 15.0:
        raise RuntimeError(SPLIT_G)
    if int(LINE) != 550 or int(KNOWN_LINE) != 179 or int(CELL_TOL) != 3:
        raise RuntimeError((LINE, KNOWN_LINE, CELL_TOL))
    if int(CONTROL_TUBE) != 509 or int(CONTROL_RULE) != 407:
        raise RuntimeError((CONTROL_TUBE, CONTROL_RULE))
    if int(ORACLE_EXPECT) != 560 or int(EXPECTED_SPLIT) != 557 or int(EXPECTED_REPEAT) != 150:
        raise RuntimeError((ORACLE_EXPECT, EXPECTED_SPLIT, EXPECTED_REPEAT))
    if float(NAV_TOL) != 6.0 or float(VY_SOFT) != 8.0 or float(V_TARGET_LOOP) != 6.0:
        raise RuntimeError((NAV_TOL, VY_SOFT, V_TARGET_LOOP))
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise RuntimeError((FUEL0, GROUND, THREADS))
    if float(STAGE1_HEIGHT) != 128.0:
        raise RuntimeError(STAGE1_HEIGHT)
    if set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & set(range(SEED0, SEED0 + N_SEEDS)):
        raise RuntimeError("сиды блока повтора совпали с основной сеткой")
    if set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & set(range(99000, 99030)):
        raise RuntimeError("сиды блока повтора совпали с этапом 3")
    if controller_of(20.0) != "rule" or controller_of(15.0) != "rule":
        raise RuntimeError("тяжёлый эпизод не отдан правилу")
    if controller_of(8.0) != "tube" or controller_of(2.0) != "tube":
        raise RuntimeError("лёгкий эпизод не отдан трубе")
    if not known_ok(179) or known_ok(178):
        raise RuntimeError("линия знакомой части сдвинута")
    if not gain_ok(550) or gain_ok(549):
        raise RuntimeError("линия 550 сдвинута")
    if not cell_ok(14, 17) or cell_ok(13, 17):
        raise RuntimeError("допуск клетки 3 сдвинут")


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit("нужен тег numeric-land-skill-split. Другие теги запрещены.")


def json_path() -> str:
    return os.path.join(OUT, JSON_NAME)


def refuse_report(path: str) -> None:
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}, второй прогон отчёт не затирает")


def stray_pt():
    return glob.glob(os.path.join(OUT, "landd_split_*.pt"))


def _cell(bucket) -> dict:
    """Ячейка 30 эпизодов. Внутри — шаги и число эпизодов под правилом."""
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
        "rule_steps_mean": round(float(bucket["rule"]) / float(N_SEEDS), 3),
        "rule_episodes": int(bucket["rule_eps"]),
    }


def _group(cells, key):
    order = []
    buckets = {}
    for cell in cells:
        mark = cell[key]
        if mark not in buckets:
            order.append(mark)
            buckets[mark] = {"soft": 0, "crash": 0, "miss": 0, "fuel_out": 0, "of": 0, "rule_eps": 0}
        bucket = buckets[mark]
        for name in ("soft", "crash", "miss", "fuel_out", "of"):
            bucket[name] += int(cell[name])
        bucket["rule_eps"] += int(cell["rule_episodes"])
    rows = []
    for mark in order:
        row = dict(buckets[mark])
        row[key] = mark
        rows.append(row)
    return rows


def controller_of(g) -> str:
    """Один раз на эпизод: тяжёлый — правило, лёгкий — труба."""
    return "rule" if float(g) >= float(SPLIT_G) else "tube"


def known_ok(soft: int) -> bool:
    return int(soft) >= int(KNOWN_LINE)


def gain_ok(soft: int) -> bool:
    return int(soft) >= int(LINE)


def cell_ok(soft: int, other_soft: int) -> bool:
    return int(soft) >= int(other_soft) - int(CELL_TOL)


@torch.no_grad()
def fly_episode(kind, head, encoder, ema, predictors, actions, step_pairs, world_b, row, track):
    x, y, vx, vy, _acts = start_and_actions(int(row["seed"]), 0)
    if (float(x), float(y), float(vx), float(vy)) != (
        float(row["x"]), float(row["y"]), float(row["vx"]), float(row["vy"]),
    ):
        raise RuntimeError("старт разошёлся со start_and_actions")
    fuel = float(FUEL0)
    if float(row["fuel"]) != fuel:
        raise RuntimeError(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    chosen = controller_of(g) if kind == "split" else kind
    if chosen not in ("rule", "tube"):
        raise RuntimeError(chosen)
    grounded = False
    steps = 0
    fuel_out = False
    clipped_n = 0
    rule_steps = 0
    tube_steps = 0
    log = SwitchLog() if track else None
    while steps < 900 and not grounded:
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        mode, _prob = classify(head, encoder, state, g, pad_x)
        clipped = False
        if mode is None:
            action = 1
        elif chosen == "rule":
            action = rule_action(state, g, pad_x, mode)
            rule_steps += 1
        else:
            action, clipped = tube_action(
                predictor_of(mode, predictors), encoder, ema, actions, step_pairs, world_b,
                state, g, pad_x)
            tube_steps += 1
        if clipped:
            clipped_n += 1
        if log is not None and mode is not None:
            log.observe(mode, steps, float(GROUND) - float(y), abs(float(x) - pad_x))
        x, y, vx, vy, fuel, grounded = step_field(
            float(x), float(y), float(vx), float(vy), float(fuel), False, int(action), float(g))
        steps += 1
        if float(fuel) <= 0.0 and not grounded:
            fuel_out = True
    result = outcome_at(x, y, vy, grounded, pad_x)
    if result not in RESULT_CODE:
        raise RuntimeError(result)
    out = {
        "seed": int(row["seed"]),
        "g": float(g),
        "pad_x": float(pad_x),
        "result": result,
        "steps": int(steps),
        "fuel_out": bool(fuel_out),
        "tube_clipped": int(clipped_n),
        "controller": chosen,
        "rule_steps": int(rule_steps),
        "tube_steps": int(tube_steps),
        "rule_share": float(rule_steps) / float(max(1, rule_steps + tube_steps)),
    }
    if log is not None:
        out.update(switch_fields(log))
    return out


def summarize(rows, n: int) -> dict:
    if len(rows) != int(n):
        raise RuntimeError((len(rows), n))
    steps, heights, dxs = [], [], []
    n_no = n_open = n_hi = 0
    n_rule = 0
    rule_steps = 0
    for row in rows:
        bucket = row["bucket"]
        if bucket == "switch":
            height = float(row["switch_height"])
            steps.append(float(row["switch_step"]))
            heights.append(height)
            dxs.append(float(row["switch_abs_dx"]))
            if height > float(STAGE1_HEIGHT):
                n_hi += 1
        elif bucket == "no_switch":
            n_no += 1
        elif bucket == "opened_in_land":
            n_open += 1
        else:
            raise RuntimeError(bucket)
        if row["controller"] == "rule":
            n_rule += 1
        rule_steps += int(row["rule_steps"])
    n_switch = len(steps)
    if n_switch + n_no + n_open != int(n):
        raise RuntimeError((n_switch, n_no, n_open))

    def med(values):
        if not values:
            return None
        value = float(np.median(np.asarray(values, np.float64)))
        if not np.isfinite(value):
            raise RuntimeError(value)
        return value

    return {
        "n": int(n),
        "n_switch": int(n_switch),
        "n_no_switch": int(n_no),
        "n_opened_in_land": int(n_open),
        "median_step": med(steps),
        "median_height": med(heights),
        "median_abs_dx": med(dxs),
        "share_no_switch": float(n_no) / float(n),
        "n_height_gt_128": int(n_hi),
        "share_height_gt_128": None if n_switch == 0 else float(n_hi) / float(n_switch),
        "height_gt_128_is_gate": False,
        "n_rule_episodes": int(n_rule),
        "share_rule_episodes": float(n_rule) / float(n),
        "rule_steps": int(rule_steps),
    }


@torch.no_grad()
def fly_column(name, kind, rows, n_expected, head, encoder, ema, predictors, actions, step_pairs, track):
    from numeric_land.skill_land import world_batch
    cells = []
    failures = []
    tracked = []
    current = None
    bucket = None
    clipped_n = 0
    worlds = {}
    t_cell = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        key = (float(row["g"]), float(row["pad_x"]))
        if key not in worlds:
            worlds[key] = world_batch(row["g"], row["pad_x"])
        if key != current:
            if bucket is not None:
                cell = _cell(bucket)
                cells.append(cell)
                print(
                    f"SPLIT_CELL column={name} g={cell['g']:g} pad={cell['pad_x']:g} "
                    f"soft={cell['soft']} crash={cell['crash']} miss={cell['miss']} "
                    f"fuel_out={cell['fuel_out']} sec={time.perf_counter() - t_cell:.1f}",
                    flush=True,
                )
            current = key
            bucket = {
                "g": key[0], "pad_x": key[1],
                "soft": 0, "crash": 0, "miss": 0, "fuel_out": 0, "of": 0, "steps": 0,
                "rule": 0, "rule_eps": 0,
            }
            t_cell = time.perf_counter()
        ep = fly_episode(
            kind, head, encoder, ema, predictors, actions, step_pairs, worlds[key], row, track)
        code = RESULT_CODE[ep["result"]]
        bucket[code] += 1
        bucket["of"] += 1
        bucket["steps"] += int(ep["steps"])
        bucket["rule"] += int(ep["rule_steps"])
        bucket["rule_eps"] += 1 if ep["controller"] == "rule" else 0
        if ep["fuel_out"]:
            bucket["fuel_out"] += 1
        clipped_n += int(ep["tube_clipped"])
        if ep["result"] != "мягкая":
            failures.append({
                "seed": int(ep["seed"]), "g": float(ep["g"]), "pad_x": float(ep["pad_x"]),
                "result": ep["result"], "steps": int(ep["steps"]), "fuel_out": bool(ep["fuel_out"]),
            })
        if track:
            tracked.append(dict(ep))
        print(
            f"SPLIT_EP column={name} i={index} n={n_expected} steps={ep['steps']} "
            f"{ep['controller']} {code}",
            flush=True,
        )
    if bucket is None:
        raise RuntimeError("пустая сетка")
    cell = _cell(bucket)
    cells.append(cell)
    print(
        f"SPLIT_CELL column={name} g={cell['g']:g} pad={cell['pad_x']:g} "
        f"soft={cell['soft']} crash={cell['crash']} miss={cell['miss']} "
        f"fuel_out={cell['fuel_out']} sec={time.perf_counter() - t_cell:.1f}",
        flush=True,
    )
    if len(cells) != len({(item["g"], item["pad_x"]) for item in cells}):
        raise RuntimeError("клетка посчитана дважды")
    soft = int(sum(item["soft"] for item in cells))
    crash = int(sum(item["crash"] for item in cells))
    miss = int(sum(item["miss"] for item in cells))
    fuel_c = int(sum(item["fuel_out"] for item in cells))
    rule_eps = int(sum(item["rule_episodes"] for item in cells))
    if soft + crash + miss != int(n_expected):
        raise RuntimeError((soft, crash, miss, n_expected))
    column = {
        "name": name,
        "kind": kind,
        "scored": bool(name == "skill_split"),
        "soft": soft,
        "crash": crash,
        "miss": miss,
        "fuel_out": fuel_c,
        "of": int(n_expected),
        "tube_clipped": int(clipped_n),
        "cells": cells,
        "by_g": _group(cells, "g"),
        "by_pad": _group(cells, "pad_x"),
        "failures": failures,
        "n_rule_episodes": int(rule_eps),
        "share_rule_episodes": float(rule_eps) / float(len(cells)),
    }
    if track:
        column["episodes"] = tracked
        column["switches"] = summarize(tracked, n_expected)
    return column


def _column(columns, name):
    found = [column for column in columns if column["name"] == name]
    if len(found) != 1:
        raise RuntimeError(name)
    return found[0]


def marks(cells) -> dict:
    return {(float(cell["g"]), float(cell["pad_x"])): cell for cell in cells}


def conditions(split, tube, rule, rep_split, rep_tube, rep_rule) -> dict:
    """Шесть условий: знакомое, клетки, сумма, тождество, повтор."""
    g_marks, t_marks, r_marks = marks(split["cells"]), marks(tube["cells"]), marks(rule["cells"])
    if set(g_marks) != set(t_marks) or set(g_marks) != set(r_marks):
        raise RuntimeError("клетки столбцов не совпали")
    known_soft = 0
    known_of = 0
    cells_pass = True
    no_regress = int(split["soft"]) >= int(CONTROL_TUBE)
    identity = True
    oracle = 0
    for key, cell in g_marks.items():
        g = float(key[0])
        rule_soft = int(r_marks[key]["soft"])
        tube_soft = int(t_marks[key]["soft"])
        cell["rule_soft"] = rule_soft
        cell["tube_soft"] = tube_soft
        cell["controller"] = controller_of(g)
        cell["vs_rule"] = int(cell["soft"]) - rule_soft
        cell["vs_tube"] = int(cell["soft"]) - tube_soft
        cell["cell_pass"] = bool(cell_ok(cell["soft"], rule_soft))
        if not cell["cell_pass"]:
            cells_pass = False
        if int(cell["soft"]) < tube_soft - int(CELL_TOL):
            no_regress = False
        expected = rule_soft if controller_of(g) == "rule" else tube_soft
        if int(cell["soft"]) != expected:
            identity = False
        if float(key[0]) in KNOWN_G and float(key[1]) in KNOWN_PADS:
            known_soft += int(cell["soft"])
            known_of += int(cell["of"])
        oracle += max(tube_soft, rule_soft)
    if known_of != KNOWN_N:
        raise RuntimeError((known_of, KNOWN_N))
    if oracle != int(ORACLE_EXPECT):
        raise RuntimeError(f"потолок {oracle}, ждали {ORACLE_EXPECT}")
    if int(split["soft"]) != int(EXPECTED_SPLIT):
        identity = False
    rep_g = marks(rep_split["cells"])
    rep_t = marks(rep_tube["cells"])
    rep_r = marks(rep_rule["cells"])
    if set(rep_g) != set(rep_r) or set(rep_g) != set(rep_t):
        raise RuntimeError("клетки блока повтора не совпали")
    repeat = int(rep_split["soft"]) == int(EXPECTED_REPEAT)
    for key, cell in rep_g.items():
        rule_soft = int(rep_r[key]["soft"])
        tube_soft = int(rep_t[key]["soft"])
        cell["rule_soft"] = rule_soft
        cell["tube_soft"] = tube_soft
        cell["controller"] = controller_of(float(key[0]))
        cell["vs_rule"] = int(cell["soft"]) - rule_soft
        cell["vs_tube"] = int(cell["soft"]) - tube_soft
        cell["cell_pass"] = bool(cell_ok(cell["soft"], rule_soft))
        if int(cell["soft"]) != rule_soft:
            repeat = False
    return {
        "known_soft": int(known_soft),
        "known_of": int(known_of),
        "known_ok": bool(known_ok(known_soft)),
        "cells_ok": bool(cells_pass),
        "no_regression": bool(no_regress),
        "gain_ok": bool(gain_ok(split["soft"])),
        "repeat_ok": bool(repeat),
        "identity_ok": bool(identity),
        "oracle": int(oracle),
        "pilot_pass": bool(
            known_ok(known_soft) and cells_pass and no_regress and gain_ok(split["soft"])
            and repeat and identity),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    refuse_tag(args.tag)
    locks()
    torch.set_num_threads(THREADS)
    report = json_path()
    refuse_report(report)
    if stray_pt():
        raise SystemExit("уже есть landd_split_*.pt, новый чекпоинт писать нельзя")
    print("SPLIT_START grid=600 repeat=180", flush=True)
    hover_path = os.path.join(ROOT, "numeric_land", "expert_hover.py")
    hover_bytes = open(hover_path, "rb").read()
    before = take_stamps()
    bundle = load_bundle()
    if take_stamps() != before:
        raise RuntimeError("чтение сдвинуло штамп")
    print("SPLIT_LOADED", flush=True)
    predictors = {"land": bundle["pad"], "approach": bundle["approach"]}
    encoder = bundle["encoder"]
    ema = bundle["ema"]
    early = bundle["early"]
    from numeric_land.skill_land import pair_table
    actions, step_pairs = pair_table()

    rows_main = episodes(SEED0, GRID_G, GRID_PADS)
    rows_rep = episodes(REP_SEED0, REP_G, REP_PADS)
    if len(rows_main) != GRID_N or len(rows_rep) != REP_N:
        raise RuntimeError((len(rows_main), len(rows_rep)))

    main_cols = []
    main_cols.append(fly_column(
        "rule_early", "rule", rows_main, GRID_N, early, encoder, ema, predictors,
        actions, step_pairs, False))
    main_cols.append(fly_column(
        "skill_tube", "tube", rows_main, GRID_N, early, encoder, ema, predictors,
        actions, step_pairs, False))
    main_cols.append(fly_column(
        "skill_split", "split", rows_main, GRID_N, early, encoder, ema, predictors,
        actions, step_pairs, True))
    rule = _column(main_cols, "rule_early")
    tube = _column(main_cols, "skill_tube")
    split = _column(main_cols, "skill_split")
    control = bool(int(tube["soft"]) == int(CONTROL_TUBE) and int(rule["soft"]) == int(CONTROL_RULE))
    print(
        f"SPLIT_CONTROL tube={tube['soft']} rule={rule['soft']} split={split['soft']} "
        f"ok={int(control)}",
        flush=True,
    )

    rep_cols = []
    rep_cols.append(fly_column(
        "rule_early", "rule", rows_rep, REP_N, early, encoder, ema, predictors,
        actions, step_pairs, False))
    rep_cols.append(fly_column(
        "skill_tube", "tube", rows_rep, REP_N, early, encoder, ema, predictors,
        actions, step_pairs, False))
    rep_cols.append(fly_column(
        "skill_split", "split", rows_rep, REP_N, early, encoder, ema, predictors,
        actions, step_pairs, False))
    rep_rule = _column(rep_cols, "rule_early")
    rep_tube = _column(rep_cols, "skill_tube")
    rep_split = _column(rep_cols, "skill_split")

    if take_stamps() != before:
        raise RuntimeError("полёт сдвинул штамп")
    if open(hover_path, "rb").read() != hover_bytes:
        raise RuntimeError("правило изменилось")
    if stray_pt():
        raise RuntimeError("появился landd_split_*.pt")

    numbers = conditions(split, tube, rule, rep_split, rep_tube, rep_rule)
    if not control and numbers["pilot_pass"]:
        raise RuntimeError("контроль не воспроизвёлся, а зачёт закрыт")
    switches = split["switches"]
    if switches["height_gt_128_is_gate"] or switches["n"] != GRID_N:
        raise RuntimeError("сводка переключений сломана")
    payload = {
        "tag": TAG,
        "grid_n": GRID_N,
        "grid_g": [float(g) for g in GRID_G],
        "grid_pads": [float(pad) for pad in GRID_PADS],
        "cell_n": int(N_SEEDS),
        "split_g": float(SPLIT_G),
        "line": int(LINE),
        "known_line": int(KNOWN_LINE),
        "cell_tol": int(CELL_TOL),
        "known_n": int(KNOWN_N),
        "known_g": [float(g) for g in KNOWN_G],
        "known_pads": [float(pad) for pad in KNOWN_PADS],
        "repeat_n": int(REP_N),
        "repeat_g": [float(g) for g in REP_G],
        "repeat_pads": [float(pad) for pad in REP_PADS],
        "repeat_seeds": [int(REP_SEED0), int(REP_SEED0 + N_SEEDS - 1)],
        "control_tube": int(CONTROL_TUBE),
        "control_rule": int(CONTROL_RULE),
        "control_ok": bool(control),
        "oracle": int(numbers["oracle"]),
        "expected_split": int(EXPECTED_SPLIT),
        "expected_repeat": int(EXPECTED_REPEAT),
        "known": {"soft": int(numbers["known_soft"]), "of": int(numbers["known_of"]),
                  "line": int(KNOWN_LINE), "ok": bool(numbers["known_ok"])},
        "known_ok": bool(numbers["known_ok"]),
        "cells_ok": bool(numbers["cells_ok"]),
        "no_regression": bool(numbers["no_regression"]),
        "gain_ok": bool(numbers["gain_ok"]),
        "repeat_ok": bool(numbers["repeat_ok"]),
        "identity_ok": bool(numbers["identity_ok"]),
        "pilot_pass": bool(numbers["pilot_pass"]),
        "threshold": 0.5,
        "horizon": 3,
        "n_sequences": 729,
        "cem": False,
        "rule_action_in_scored": False,
        "heads_trained": False,
        "stamps_unchanged": True,
        "tube_clipped": int(split["tube_clipped"]),
        "v_target": float(V_TARGET_LOOP),
        "seeds": [int(SEED0), int(SEED0 + N_SEEDS - 1)],
        "step_origin": 0,
        "tube_targets": 3,
        "height_gt_128_is_gate": False,
        "heads": {"early": "landd_early_s0.pt"},
        "skills": {"land": "landk_pad_s0.pt", "approach": "landk_approach_s0.pt"},
        "prior_records": {
            "stage4_tube": int(CONTROL_TUBE),
            "stage4_rule": int(CONTROL_RULE),
            "stage4_terminal": 20,
            "stage4_oracle": int(ORACLE_EXPECT),
            "guard_soft": 508,
        },
        "switches": switches,
        "columns": main_cols,
        "repeat_columns": rep_cols,
    }
    if take_stamps() != before:
        raise RuntimeError("перед записью штамп сдвинулся")
    write_json(report, payload)
    if take_stamps() != before:
        raise RuntimeError("запись отчёта задела чекпоинт")
    print(
        f"SPLIT_DONE split={split['soft']} tube={tube['soft']} rule={rule['soft']} "
        f"oracle={numbers['oracle']} known={numbers['known_soft']}/180 "
        f"known_ok={int(numbers['known_ok'])} cells_ok={int(numbers['cells_ok'])} "
        f"no_regression={int(numbers['no_regression'])} gain_ok={int(numbers['gain_ok'])} "
        f"repeat={rep_split['soft']}/{REP_N} repeat_ok={int(numbers['repeat_ok'])} "
        f"identity_ok={int(numbers['identity_ok'])} pilot_pass={int(numbers['pilot_pass'])}",
        flush=True,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

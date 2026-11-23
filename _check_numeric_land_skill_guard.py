"""Контроль этапа «страховка» до прогона. python _check_numeric_land_skill_guard.py"""
import inspect
import json
import os
import sys
import tempfile

import torch

from numeric_land.decision import n_in_of
from numeric_land.expert_hover import NAV_TOL
from numeric_land.field import FUEL0, GROUND, VY_SOFT
from numeric_land.mode import V_TARGET_LOOP
from numeric_land.relenc import OUT
from numeric_land.skill_assemble import GRID_N, KNOWN_G, KNOWN_PADS, N_SEEDS, classify, mode_of
from numeric_land.skill_guard import (
    CELL_TOL, CONTROL_RULE, CONTROL_TUBE, GUARD_DX, GUARD_G, LINE, ORACLE_EXPECT,
    REP_G, REP_N, REP_PADS, REP_SEED0, TAG, cell_ok, conditions, gain_ok, known_same,
    locks, main_episodes, outside_state, repeat_episodes, refuse_report, refuse_tag,
    stray_pt, take_stamps,
)
from numeric_land.train_goal import THREADS

ASSEMBLE_REPORT = os.path.join(OUT, "numeric_land_skill_assemble.json")


def _expect_exit(fn):
    try:
        fn()
    except SystemExit as exc:
        if exc.code in (0, None):
            raise SystemExit("отказ вышел с нулевым кодом")
        return
    raise SystemExit("ожидался отказ")


def _cells(pairs):
    """Клетки из списка (g, pad, soft)."""
    return [{"g": float(g), "pad_x": float(pad), "soft": int(soft), "crash": 0, "miss": 0,
             "fuel_out": 0, "of": int(N_SEEDS), "steps_mean": 100.0,
             "guard_steps_mean": 0.0, "gate_fired": 0} for g, pad, soft in pairs]


def _from_stage4():
    """Двадцать клеток этапа 4: (g, pad, tube, rule)."""
    data = json.load(open(ASSEMBLE_REPORT, encoding="utf-8"))
    columns = {column["name"]: column for column in data["columns"]}
    tube = {(cell["g"], cell["pad_x"]): int(cell["soft"]) for cell in columns["skill_tube"]["cells"]}
    rule = {(cell["g"], cell["pad_x"]): int(cell["soft"]) for cell in columns["rule_early"]["cells"]}
    keys = sorted(tube)
    if len(keys) != 20 or int(data["grid_n"]) != 600:
        raise SystemExit((len(keys), data["grid_n"]))
    return keys, tube, rule


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    hover_path = os.path.join(root, "numeric_land", "expert_hover.py")
    hover_bytes = open(hover_path, "rb").read()
    if not hover_bytes:
        raise SystemExit("нет правила")
    if not os.path.isfile(ASSEMBLE_REPORT):
        raise SystemExit("нет отчёта этапа 4, потолок 560 взять неоткуда")

    if TAG != "numeric-land-skill-guard":
        raise SystemExit(TAG)
    if float(GUARD_G) != 15.0 or float(GUARD_DX) != 100.0:
        raise SystemExit((GUARD_G, GUARD_DX))
    if int(LINE) != 540 or int(ORACLE_EXPECT) != 560 or int(CELL_TOL) != 3:
        raise SystemExit((LINE, ORACLE_EXPECT, CELL_TOL))
    if int(CONTROL_TUBE) != 509 or int(CONTROL_RULE) != 407:
        raise SystemExit((CONTROL_TUBE, CONTROL_RULE))
    if float(NAV_TOL) != 6.0 or float(VY_SOFT) != 8.0 or float(V_TARGET_LOOP) != 6.0:
        raise SystemExit("допуск, порог касания или v_target сдвинуты")
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise SystemExit("бак, земля или потоки сдвинуты")
    if n_in_of("latent") != 33 or mode_of(0.5) != "land" or mode_of(0.499) != "approach":
        raise SystemExit("вход головы или порог 0.5 сдвинуты")
    if not gain_ok(540) or gain_ok(539):
        raise SystemExit("линия 540 сдвинута")
    if not cell_ok(27, 30) or cell_ok(26, 30):
        raise SystemExit("допуск клетки 3 сдвинут")
    locks()

    # Ворота: смотрят на состояние, не на клетку.
    if not outside_state(256.0, 20.0, 406.0) or not outside_state(240.0, 15.0, 356.0):
        raise SystemExit("ворота не видят дальний подвод")
    if not outside_state(300.0, 15.0, 200.0) or not outside_state(100.0, 15.0, 200.0):
        raise SystemExit("ворота не срабатывают по |x − pad| при большой тяжести")
    if outside_state(150.0, 15.0, 200.0) or outside_state(150.0, 14.9, 200.0):
        raise SystemExit("ворота срабатывают ближе 100 или ниже 15")
    if outside_state(256.0, 8.0, 406.0) or outside_state(256.0, 2.0, 106.0):
        raise SystemExit("ворота срабатывают при малой тяжести")
    if outside_state(256.0, 20.0, 256.0):
        raise SystemExit("ворота срабатывают у центральной площадки")

    rows = main_episodes()
    if len(rows) != 600:
        raise SystemExit(len(rows))
    rep = repeat_episodes()
    if len(rep) != REP_N:
        raise SystemExit(len(rep))
    if {(row["g"], row["pad_x"]) for row in rep} != {(g, pad) for g in REP_G for pad in REP_PADS}:
        raise SystemExit("клетки блока повтора не те")
    if set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & {int(row["seed"]) for row in rows}:
        raise SystemExit("сиды блока повтора совпали с основной сеткой")
    if len({(row["seed"], row["g"], row["pad_x"]) for row in rows}) != 600:
        raise SystemExit("в основной сетке повтор")
    known_rows = [row for row in rows
                  if row["g"] in KNOWN_G and row["pad_x"] in KNOWN_PADS]
    if len(known_rows) != 180:
        raise SystemExit(len(known_rows))
    if not all(row["outside_at_start"] is False for row in known_rows):
        raise SystemExit("ворота срабатывают на старте знакомой клетки")
    if any(row["outside_at_start"] != outside_state(row["x"], row["g"], row["pad_x"]) for row in rows):
        raise SystemExit("признак старта не совпал с воротами")

    # Условия считаются на настоящих числах этапа 4.
    keys, tube_soft, rule_soft = _from_stage4()
    tube_cells = _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys])
    rule_cells = _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys])
    rep_keys = [(g, pad) for g in REP_G for pad in REP_PADS]
    rep_tube_cells = _cells([(g, pad, 20) for g, pad in rep_keys])
    rep_rule_cells = _cells([(g, pad, 18) for g, pad in rep_keys])
    rep_guard_cells = _cells([(g, pad, 25) for g, pad in rep_keys])

    pure = conditions(
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": tube_cells, "soft": CONTROL_TUBE},
        {"cells": rule_cells, "soft": CONTROL_RULE},
        {"cells": rep_guard_cells, "soft": 150},
        {"cells": rep_tube_cells, "soft": 120},
        {"cells": rep_rule_cells, "soft": 108},
    )
    if pure["known_same"] is not True or pure["cells_ok"] or pure["gain_ok"]:
        raise SystemExit(("чистая труба не должна закрывать линию", pure))
    if pure["no_regression"] is not True or pure["repeat_ok"] is not True:
        raise SystemExit(pure)
    if pure["oracle"] != 560 or pure["gap_to_oracle"] != 51:
        raise SystemExit((pure["oracle"], pure["gap_to_oracle"]))

    best = {key: max(tube_soft[key], rule_soft[key]) for key in keys}
    oracle_soft = sum(best.values())
    guarded = conditions(
        {"cells": _cells([(g, pad, best[(g, pad)]) for g, pad in keys]), "soft": oracle_soft},
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_RULE},
        {"cells": _cells([(g, pad, 25) for g, pad in rep_keys]), "soft": 150},
        {"cells": _cells([(g, pad, 20) for g, pad in rep_keys]), "soft": 120},
        {"cells": _cells([(g, pad, 18) for g, pad in rep_keys]), "soft": 108},
    )
    if not guarded["cells_ok"] or not guarded["gain_ok"] or not guarded["pilot_pass"]:
        raise SystemExit(("потолок по клеткам должен закрывать линию", guarded))
    if guarded["oracle"] != 560 or oracle_soft != 560:
        raise SystemExit((guarded["oracle"], oracle_soft))

    bad_repeat = conditions(
        {"cells": _cells([(g, pad, best[(g, pad)]) for g, pad in keys]), "soft": oracle_soft},
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_RULE},
        {"cells": _cells([(g, pad, 14) for g, pad in rep_keys]), "soft": 84},
        {"cells": _cells([(g, pad, 20) for g, pad in rep_keys]), "soft": 120},
        {"cells": _cells([(g, pad, 18) for g, pad in rep_keys]), "soft": 108},
    )
    if bad_repeat["repeat_ok"]:
        raise SystemExit("блок повтора не ловит клетку ниже правила минус 3")
    late = [(g, pad, best[(g, pad)]) for g, pad in keys]
    late[0] = (late[0][0], late[0][1], 0)
    broken = conditions(
        {"cells": _cells(late), "soft": oracle_soft - 30},
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_RULE},
        {"cells": _cells([(g, pad, 25) for g, pad in rep_keys]), "soft": 150},
        {"cells": _cells([(g, pad, 20) for g, pad in rep_keys]), "soft": 120},
        {"cells": _cells([(g, pad, 18) for g, pad in rep_keys]), "soft": 108},
    )
    if broken["gain_ok"] or broken["pilot_pass"] or broken["gap_to_oracle"] != 30:
        raise SystemExit(("просадка на 30 эпизодов не поймана", broken))

    known_keys = [(g, pad) for g in KNOWN_G for pad in KNOWN_PADS]
    drifted = [(g, pad, 29 if index == 0 else 30) for index, (g, pad) in enumerate(known_keys)]
    steady = [(g, pad, 30) for g, pad in known_keys]
    if known_same(_cells(drifted), _cells(steady)) is not False:
        raise SystemExit("сверка знакомой части не видит расхождение")
    ok_cells = _cells([(g, pad, 30) for g in KNOWN_G for pad in KNOWN_PADS])
    if known_same(ok_cells, ok_cells) is not True:
        raise SystemExit("сверка знакомой части не принимает равные счета")
    try:
        known_same(_cells([(g, pad, 30) for g in KNOWN_G for pad in KNOWN_PADS]), _cells([]))
    except RuntimeError:
        pass
    else:
        raise SystemExit("сверка знакомой части не требует всех шести клеток")

    ground = (256.0, float(GROUND), 0.0, 0.0, 10.0)
    empty = (256.0, float(GROUND) - 40.0, 0.0, 0.0, 0.0)
    for state in (ground, empty):
        mode, prob = classify(None, None, state, 8.0, 256.0)
        if mode is not None or prob is not None:
            raise SystemExit("на земле или при пустом баке режим не пустой")

    module = sys.modules["numeric_land.skill_guard"]
    src = inspect.getsource(module)
    for banned in ("torch.save", "AdamW", "backward", "cem_plan", "inference_mode"):
        if banned in src:
            raise SystemExit(f"модуль содержит {banned}")
    flight = inspect.getsource(module.fly_episode)
    if flight.count("outside(state, g, pad_x)") != 1:
        raise SystemExit("ворота считаются не по состоянию")
    if 'kind == "guard"' not in flight or 'kind == "tube"' not in flight or 'kind == "rule"' not in flight:
        raise SystemExit("в шаге нет трёх источников действия")
    if "if mode is None:" not in flight or "action = 1" not in flight:
        raise SystemExit("на земле или при пустом баке действие не 1")
    if "hover_action" in inspect.getsource(module.tube_action) if hasattr(module, "tube_action") else False:
        raise SystemExit("навык берёт действие у правила")
    main_src = inspect.getsource(module.main)
    if main_src.count("fly_column(") != 6:
        raise SystemExit("прогон зовёт не шесть столбцов")
    if not (0 <= main_src.find("refuse_tag") < main_src.find("refuse_report") < main_src.find('"rule_early"')):
        raise SystemExit("отказ стоит не до полёта")
    if not (main_src.find("main_episodes") < main_src.find("repeat_episodes")):
        raise SystemExit("сначала считается блок повтора")
    for flag in (
        '"guard_g": float(GUARD_G)', '"guard_dx": float(GUARD_DX)', '"line": int(LINE)',
        '"cell_tol": int(CELL_TOL)', '"control_ok": bool(control)', '"oracle"',
        '"gap_to_oracle"', '"known_same": bool(numbers["known_same"])',
        '"cells_ok": bool(numbers["cells_ok"])', '"no_regression": bool(numbers["no_regression"])',
        '"gain_ok": bool(numbers["gain_ok"])', '"repeat_ok": bool(numbers["repeat_ok"])',
        '"pilot_pass": bool(numbers["pilot_pass"])', '"heads_trained": False',
        '"stamps_unchanged": True', '"repeat_columns": rep_cols',
    ):
        if flag not in main_src:
            raise SystemExit(f"в отчёте нет {flag}")
    for banned in ("landd_early_s1", "landk_pad_s1", "landk_approach_s1", "landk_approach_s2"):
        if banned in main_src:
            raise SystemExit(f"{banned} попал в прогон")

    _expect_exit(lambda: refuse_tag("numeric-land-skill-assemble"))
    fd, temp_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        _expect_exit(lambda: refuse_report(temp_json))
    finally:
        os.remove(temp_json)
    if os.path.isfile(module.json_path()):
        raise SystemExit("отчёт уже есть, контроль не летит поверх него")

    # Ворота не срабатывают в знакомых клетках на настоящем полёте.
    before = take_stamps()
    torch.set_num_threads(THREADS)
    bundle = None
    from numeric_land.skill_guard import fly_episode
    from numeric_land.skill_land import pair_table, world_batch
    bundle = sys.modules["numeric_land.skill_assemble"].load_bundle()
    actions, step_pairs = pair_table()
    preds = {"land": bundle["pad"], "approach": bundle["approach"]}
    sample = [row for row in known_rows if float(row["g"]) == 15.0][0]
    ep = fly_episode(
        "guard", bundle["early"], bundle["encoder"], bundle["ema"], preds,
        actions, step_pairs, world_batch(sample["g"], sample["pad_x"]), sample, True)
    if int(ep["guard_steps"]) != 0 or ep["gate_fired"] is not False:
        raise SystemExit(("ворота сработали в знакомой клетке", ep["g"], ep["pad_x"], ep["guard_steps"]))
    if ep["result"] != "мягкая":
        raise SystemExit(("знакомая клетка перестала садиться", ep["result"]))
    outside_row = [row for row in rows if row["g"] == 20.0 and row["pad_x"] == 406.0][0]
    ep_out = fly_episode(
        "tube", bundle["early"], bundle["encoder"], bundle["ema"], preds,
        actions, step_pairs, world_batch(outside_row["g"], outside_row["pad_x"]), outside_row, False)
    if ep_out["gate_fired"]:
        raise SystemExit("чистая труба включила ворота")
    if take_stamps() != before:
        raise SystemExit("контроль сдвинул штамп")
    if open(hover_path, "rb").read() != hover_bytes:
        raise SystemExit("правило изменилось")
    if stray_pt():
        raise SystemExit("после контроля появился landd_guard_*.pt")
    print(
        "ok gate 15/100 known cells clean oracle 560 line 540 cell tol 3 "
        "repeat block 220000..220029 six cells mode 0.5 land 0.499 approach",
        flush=True,
    )


if __name__ == "__main__":
    main()

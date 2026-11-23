"""Контроль разделения по тяжести до прогона. python _check_numeric_land_skill_split.py"""
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
from numeric_land.skill_assemble import (
    GRID_N, KNOWN_G, KNOWN_PADS, N_SEEDS, classify, mode_of,
)
from numeric_land.skill_guard import REP_G, REP_N, REP_PADS, REP_SEED0
from numeric_land.skill_split import (
    CELL_TOL, CONTROL_RULE, CONTROL_TUBE, EXPECTED_REPEAT, EXPECTED_SPLIT, KNOWN_LINE,
    LINE, ORACLE_EXPECT, SPLIT_G, TAG, cell_ok, conditions, controller_of, gain_ok,
    known_ok, locks, refuse_report, refuse_tag, stray_pt, take_stamps,
)
from numeric_land.skill_guard import episodes
from numeric_land.train_goal import THREADS

ASSEMBLE_REPORT = os.path.join(OUT, "numeric_land_skill_assemble.json")
GUARD_REPORT = os.path.join(OUT, "numeric_land_skill_guard.json")


def _expect_exit(fn):
    try:
        fn()
    except SystemExit as exc:
        if exc.code in (0, None):
            raise SystemExit("отказ вышел с нулевым кодом")
        return
    raise SystemExit("ожидался отказ")


def _cells(pairs):
    return [{"g": float(g), "pad_x": float(pad), "soft": int(soft), "crash": 0, "miss": 0,
             "fuel_out": 0, "of": int(N_SEEDS), "steps_mean": 100.0,
             "rule_steps_mean": 0.0, "rule_episodes": 0} for g, pad, soft in pairs]


def _stage4():
    data = json.load(open(ASSEMBLE_REPORT, encoding="utf-8"))
    columns = {column["name"]: column for column in data["columns"]}
    tube = {(cell["g"], cell["pad_x"]): int(cell["soft"]) for cell in columns["skill_tube"]["cells"]}
    rule = {(cell["g"], cell["pad_x"]): int(cell["soft"]) for cell in columns["rule_early"]["cells"]}
    if len(tube) != 20 or int(data["grid_n"]) != 600:
        raise SystemExit((len(tube), data["grid_n"]))
    return sorted(tube), tube, rule


def _guard_repeat():
    data = json.load(open(GUARD_REPORT, encoding="utf-8"))
    columns = {column["name"]: column for column in data["repeat_columns"]}
    rule = int(columns["rule_early"]["soft"])
    tube = int(columns["skill_tube"]["soft"])
    if int(columns["rule_early"]["of"]) != REP_N:
        raise SystemExit(columns["rule_early"]["of"])
    return rule, tube


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    hover_path = os.path.join(root, "numeric_land", "expert_hover.py")
    hover_bytes = open(hover_path, "rb").read()
    if not hover_bytes:
        raise SystemExit("нет правила")
    for path in (ASSEMBLE_REPORT, GUARD_REPORT):
        if not os.path.isfile(path):
            raise SystemExit(f"нет {path}")

    if TAG != "numeric-land-skill-split":
        raise SystemExit(TAG)
    if float(SPLIT_G) != 15.0:
        raise SystemExit(SPLIT_G)
    if int(LINE) != 550 or int(KNOWN_LINE) != 179 or int(CELL_TOL) != 3:
        raise SystemExit((LINE, KNOWN_LINE, CELL_TOL))
    if int(CONTROL_TUBE) != 509 or int(CONTROL_RULE) != 407:
        raise SystemExit((CONTROL_TUBE, CONTROL_RULE))
    if int(ORACLE_EXPECT) != 560 or int(EXPECTED_SPLIT) != 557 or int(EXPECTED_REPEAT) != 150:
        raise SystemExit((ORACLE_EXPECT, EXPECTED_SPLIT, EXPECTED_REPEAT))
    if float(NAV_TOL) != 6.0 or float(VY_SOFT) != 8.0 or float(V_TARGET_LOOP) != 6.0:
        raise SystemExit("допуск, порог касания или v_target сдвинуты")
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise SystemExit("бак, земля или потоки сдвинуты")
    if n_in_of("latent") != 33 or mode_of(0.5) != "land" or mode_of(0.499) != "approach":
        raise SystemExit("вход головы или порог 0.5 сдвинуты")
    if controller_of(15.0) != "rule" or controller_of(20.0) != "rule":
        raise SystemExit("тяжёлый эпизод не отдан правилу")
    if controller_of(8.0) != "tube" or controller_of(2.0) != "tube" or controller_of(14.9) != "tube":
        raise SystemExit("лёгкий эпизод не отдан трубе")
    if not known_ok(179) or known_ok(178):
        raise SystemExit("линия знакомой части сдвинута")
    if not gain_ok(550) or gain_ok(549):
        raise SystemExit("линия 550 сдвинута")
    if not cell_ok(14, 17) or cell_ok(13, 17) or not cell_ok(27, 30) or cell_ok(26, 30):
        raise SystemExit("допуск клетки 3 сдвинут")
    locks()

    rows = episodes(210_000, (2.0, 8.0, 15.0, 20.0), (106.0, 200.0, 256.0, 356.0, 406.0))
    if len(rows) != GRID_N:
        raise SystemExit(len(rows))
    rep = episodes(REP_SEED0, REP_G, REP_PADS)
    if len(rep) != REP_N:
        raise SystemExit(len(rep))
    if {(row["g"], row["pad_x"]) for row in rep} != {(g, pad) for g in REP_G for pad in REP_PADS}:
        raise SystemExit("клетки блока повтора не те")
    if set(range(REP_SEED0, REP_SEED0 + N_SEEDS)) & {int(row["seed"]) for row in rows}:
        raise SystemExit("сиды блока повтора совпали с основной сеткой")
    heavy = [row for row in rows if row["g"] >= SPLIT_G]
    light = [row for row in rows if row["g"] < SPLIT_G]
    if len(heavy) != 300 or len(light) != 300:
        raise SystemExit((len(heavy), len(light)))
    if any(row["g"] < SPLIT_G for row in rep):
        raise SystemExit("в блоке повтора есть лёгкая клетка")

    keys, tube_soft, rule_soft = _stage4()
    expected = sum(rule_soft[key] if controller_of(float(key[0])) == "rule" else tube_soft[key]
                   for key in keys)
    if expected != EXPECTED_SPLIT:
        raise SystemExit(f"тождество даёт {expected}, ждали {EXPECTED_SPLIT}")
    rep_rule, rep_tube = _guard_repeat()
    if rep_rule != EXPECTED_REPEAT:
        raise SystemExit(f"блок повтора у правила {rep_rule}, ждали {EXPECTED_REPEAT}")

    rep_keys = [(g, pad) for g in REP_G for pad in REP_PADS]
    rep_split_cells = _cells([(g, pad, rep_rule) for g, pad in rep_keys])
    rep_rule_cells = _cells([(g, pad, rep_rule) for g, pad in rep_keys])
    rep_tube_cells = _cells([(g, pad, rep_tube) for g, pad in rep_keys])

    good = conditions(
        {"cells": _cells([(g, pad, rule_soft[(g, pad)] if controller_of(g) == "rule"
                          else tube_soft[(g, pad)]) for g, pad in keys]), "soft": expected},
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_RULE},
        {"cells": _cells([(g, pad, rep_rule) for g, pad in rep_keys]), "soft": rep_rule},
        {"cells": _cells([(g, pad, rep_tube) for g, pad in rep_keys]), "soft": rep_tube},
        {"cells": rep_rule_cells, "soft": rep_rule},
    )
    if good["oracle"] != ORACLE_EXPECT:
        raise SystemExit(good["oracle"])
    if not good["identity_ok"] or not good["cells_ok"] or not good["gain_ok"]:
        raise SystemExit(("тождество по клеткам не закрывает линию", good))
    if not good["repeat_ok"] or not good["pilot_pass"]:
        raise SystemExit(good)
    if good["known_soft"] != 180 or not good["known_ok"]:
        raise SystemExit((good["known_soft"], good["known_ok"]))

    wrong = [(g, pad, rule_soft[(g, pad)] if controller_of(g) == "rule" else tube_soft[(g, pad)])
             for g, pad in keys]
    wrong[0] = (wrong[0][0], wrong[0][1], 12)
    broken_identity = conditions(
        {"cells": _cells(wrong), "soft": expected - (tube_soft[keys[0]] - 12)},
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_RULE},
        {"cells": _cells([(g, pad, rep_rule) for g, pad in rep_keys]), "soft": rep_rule},
        {"cells": _cells([(g, pad, rep_tube) for g, pad in rep_keys]), "soft": rep_tube},
        {"cells": rep_rule_cells, "soft": rep_rule},
    )
    if broken_identity["identity_ok"] or broken_identity["pilot_pass"]:
        raise SystemExit("подмена клетки не поймана")

    weak = conditions(
        {"cells": _cells([(g, pad, max(rule_soft[(g, pad)], tube_soft[(g, pad)]) - 4)
                          for g, pad in keys]), "soft": expected - 60},
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_RULE},
        {"cells": rep_split_cells, "soft": rep_rule},
        {"cells": rep_tube_cells, "soft": rep_tube},
        {"cells": rep_rule_cells, "soft": rep_rule},
    )
    if weak["gain_ok"] or weak["identity_ok"] or weak["pilot_pass"]:
        raise SystemExit(("просадка 60 эпизодов не поймана", weak))

    bad_repeat = conditions(
        {"cells": _cells([(g, pad, rule_soft[(g, pad)] if controller_of(g) == "rule"
                          else tube_soft[(g, pad)]) for g, pad in keys]), "soft": expected},
        {"cells": _cells([(g, pad, tube_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_TUBE},
        {"cells": _cells([(g, pad, rule_soft[(g, pad)]) for g, pad in keys]), "soft": CONTROL_RULE},
        {"cells": _cells([(g, pad, rep_tube) for g, pad in rep_keys]), "soft": rep_tube},
        {"cells": _cells([(g, pad, rep_tube) for g, pad in rep_keys]), "soft": rep_tube},
        {"cells": rep_rule_cells, "soft": rep_rule},
    )
    if bad_repeat["repeat_ok"]:
        raise SystemExit("блок повтора на трубе зачлось")

    ground = (256.0, float(GROUND), 0.0, 0.0, 10.0)
    empty = (256.0, float(GROUND) - 40.0, 0.0, 0.0, 0.0)
    for state in (ground, empty):
        mode, prob = classify(None, None, state, 8.0, 256.0)
        if mode is not None or prob is not None:
            raise SystemExit("на земле или при пустом баке режим не пустой")

    module = sys.modules["numeric_land.skill_split"]
    src = inspect.getsource(module)
    for banned in ("torch.save", "AdamW", "backward", "cem_plan", "inference_mode"):
        if banned in src:
            raise SystemExit(f"модуль содержит {banned}")
    flight = inspect.getsource(module.fly_episode)
    if "controller_of(g) if kind == \"split\"" not in flight:
        raise SystemExit("контроллер выбирается не по тяжести эпизода")
    if flight.count("controller_of(g)") != 1:
        raise SystemExit("контроллер выбирается повторно внутри эпизода")
    if "if mode is None:" not in flight or "action = 1" not in flight:
        raise SystemExit("на земле или при пустом баке действие не 1")
    main_src = inspect.getsource(module.main)
    if main_src.count("fly_column(") != 6:
        raise SystemExit("прогон зовёт не шесть столбцов")
    if not (0 <= main_src.find("refuse_tag") < main_src.find("refuse_report") < main_src.find('"rule_early"')):
        raise SystemExit("отказ стоит не до полёта")
    for flag in (
        '"split_g": float(SPLIT_G)', '"line": int(LINE)', '"known_line": int(KNOWN_LINE)',
        '"cell_tol": int(CELL_TOL)', '"control_ok": bool(control)', '"oracle"',
        '"expected_split": int(EXPECTED_SPLIT)', '"expected_repeat": int(EXPECTED_REPEAT)',
        '"known_ok": bool(numbers["known_ok"])', '"cells_ok": bool(numbers["cells_ok"])',
        '"no_regression": bool(numbers["no_regression"])', '"gain_ok": bool(numbers["gain_ok"])',
        '"repeat_ok": bool(numbers["repeat_ok"])', '"identity_ok": bool(numbers["identity_ok"])',
        '"pilot_pass": bool(numbers["pilot_pass"])', '"heads_trained": False',
        '"stamps_unchanged": True', '"repeat_columns": rep_cols',
    ):
        if flag not in main_src:
            raise SystemExit(f"в отчёте нет {flag}")
    for banned in ("landd_early_s1", "landk_pad_s1", "landk_approach_s1", "landk_approach_s2"):
        if banned in main_src:
            raise SystemExit(f"{banned} попал в прогон")

    _expect_exit(lambda: refuse_tag("numeric-land-skill-guard"))
    fd, temp_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        _expect_exit(lambda: refuse_report(temp_json))
    finally:
        os.remove(temp_json)
    if os.path.isfile(module.json_path()):
        raise SystemExit("отчёт уже есть, контроль не летит поверх него")

    # Настоящий полёт: лёгкий эпизод целиком на трубе, тяжёлый — целиком на правиле.
    before = take_stamps()
    torch.set_num_threads(THREADS)
    bundle = sys.modules["numeric_land.skill_assemble"].load_bundle()
    from numeric_land.skill_land import pair_table, world_batch
    from numeric_land.skill_split import fly_episode
    actions, step_pairs = pair_table()
    preds = {"land": bundle["pad"], "approach": bundle["approach"]}
    light_row = [row for row in rows if row["g"] == 8.0 and row["pad_x"] == 406.0][0]
    ep_light = fly_episode(
        "split", bundle["early"], bundle["encoder"], bundle["ema"], preds,
        actions, step_pairs, world_batch(light_row["g"], light_row["pad_x"]), light_row, False)
    if ep_light["controller"] != "tube" or int(ep_light["rule_steps"]) != 0:
        raise SystemExit(("лёгкий эпизод ушёл правилу", ep_light["controller"], ep_light["rule_steps"]))
    heavy_row = [row for row in rows if row["g"] == 15.0 and row["pad_x"] == 406.0][0]
    ep_heavy = fly_episode(
        "split", bundle["early"], bundle["encoder"], bundle["ema"], preds,
        actions, step_pairs, world_batch(heavy_row["g"], heavy_row["pad_x"]), heavy_row, True)
    if ep_heavy["controller"] != "rule" or int(ep_heavy["tube_steps"]) != 0:
        raise SystemExit(("тяжёлый эпизод ушёл трубе", ep_heavy["controller"], ep_heavy["tube_steps"]))
    if ep_heavy["result"] != "мягкая":
        raise SystemExit(("правило не село в клетке 15/406", ep_heavy["result"]))
    if take_stamps() != before:
        raise SystemExit("контроль сдвинул штамп")
    if open(hover_path, "rb").read() != hover_bytes:
        raise SystemExit("правило изменилось")
    if stray_pt():
        raise SystemExit("после контроля появился landd_split_*.pt")
    print(
        "ok split 15 heavy rule light tube identity 557 repeat 150 line 550 known 179 "
        "cell tol 3 mode 0.5 land 0.499 approach",
        flush=True,
    )


if __name__ == "__main__":
    main()

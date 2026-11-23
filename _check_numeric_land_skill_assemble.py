"""Контроль этапа 4 до прогона. python _check_numeric_land_skill_assemble.py"""
import inspect
import os
import sys
import tempfile

import numpy as np
import torch

from numeric_land.decision import n_in_of
from numeric_land.expert_hover import HOVER_CAP, NAV_TOL
from numeric_land.field import FUEL0, GROUND, VY_SOFT, start_and_actions, step_field
from numeric_land.mode import EXTRA_G, EXTRA_PAD, LOOP_N, LOOP_SEED0, V_TARGET_LOOP
from numeric_land.pad import OPEN_PADS
from numeric_land.relenc import encode_rel
from numeric_land.rollout_decision_threshold import _decide
from numeric_land.skill_assemble import (
    CELL_TOL, GAIN_MIN, GRID_G, GRID_N, GRID_PADS, KNOWN_G, KNOWN_LINE, KNOWN_N,
    KNOWN_PADS, N_SEEDS, RATIO_MIN, RULE_CELL_MIN, SEED0, TAG,
    SwitchLog, aggregate_ok, approach_path, by_domain, cell_ok, cells_block,
    classify, closed_skill_step, episodes, known_block, known_ok, load_bundle,
    mode_of, predictor_of, ratio_need, refuse_report, refuse_tag, rule_action,
    rule_from_head, stray_pt, summarize_switches, take_stamps, terminal_action,
    tube_action, tube_gain_ok, verify,
)
from numeric_land.skill_land import N_SEQ, first_action, goal_z, pair_table, world_batch
from numeric_land.skill_tube import choose_action, prepare
from numeric_land.train_goal import THREADS


def _expect_exit(fn):
    try:
        fn()
    except SystemExit as exc:
        if exc.code in (0, None):
            raise SystemExit("отказ вышел с нулевым кодом")
        return
    raise SystemExit("ожидался отказ")


def _switch_rows():
    """600 строк: три перехода, один старт в land, остальные без land."""
    rows = [
        {"bucket": "switch", "switch_step": 10, "switch_height": 40.0, "switch_abs_dx": 6.0},
        {"bucket": "switch", "switch_step": 30, "switch_height": 129.0, "switch_abs_dx": 2.0},
        {"bucket": "switch", "switch_step": 12, "switch_height": 128.0, "switch_abs_dx": 4.0},
        {"bucket": "opened_in_land", "switch_step": None, "switch_height": None, "switch_abs_dx": None},
    ]
    while len(rows) < GRID_N:
        rows.append({"bucket": "no_switch", "switch_step": None, "switch_height": None,
                     "switch_abs_dx": None})
    return rows


def _cells(soft_first):
    """Двадцать клеток по 30 эпизодов: первая клетка со своим счётом, остальные 30 из 30."""
    cells = []
    for index, g in enumerate(GRID_G):
        for jndex, pad in enumerate(GRID_PADS):
            first = index == 0 and jndex == 0
            soft = int(soft_first) if first else 30
            cells.append({
                "g": float(g), "pad_x": float(pad), "soft": int(soft), "crash": int(N_SEEDS - soft),
                "miss": 0, "fuel_out": 0, "of": int(N_SEEDS), "steps_mean": 100.0,
            })
    return cells


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    hover_path = os.path.join(root, "numeric_land", "expert_hover.py")
    demo_paths = [os.path.join(root, name) for name in ("demo_ui.py", "_land_demo.py", "_pad_demo.py")]
    hover_bytes = open(hover_path, "rb").read()
    if not hover_bytes:
        raise SystemExit("нет правила")
    for path in demo_paths:
        if not os.path.isfile(path):
            raise SystemExit(f"нет {path}")

    if TAG != "numeric-land-skill-assemble":
        raise SystemExit(TAG)
    if GRID_G != (2.0, 8.0, 15.0, 20.0) or GRID_PADS != (106.0, 200.0, 256.0, 356.0, 406.0):
        raise SystemExit((GRID_G, GRID_PADS))
    if SEED0 != 210_000 or N_SEEDS != 30 or GRID_N != 600:
        raise SystemExit((SEED0, N_SEEDS, GRID_N))
    if len(GRID_G) * len(GRID_PADS) * N_SEEDS != GRID_N:
        raise SystemExit(GRID_N)
    if max(GRID_G) > float(EXTRA_G) or max(GRID_PADS) != float(EXTRA_PAD):
        raise SystemExit((EXTRA_G, EXTRA_PAD))
    if set(GRID_PADS) != set(OPEN_PADS):
        raise SystemExit(OPEN_PADS)
    if not set(GRID_G) <= set(OPEN_PADS) | set(GRID_G):
        raise SystemExit(GRID_G)
    if set(range(SEED0, SEED0 + N_SEEDS)) & set(range(LOOP_SEED0, LOOP_SEED0 + LOOP_N)):
        raise SystemExit("сиды совпали с этапом 3")
    if KNOWN_G != (2.0, 8.0, 15.0) or KNOWN_PADS != (256.0, 200.0):
        raise SystemExit((KNOWN_G, KNOWN_PADS))
    if KNOWN_N != 180 or len(KNOWN_G) * len(KNOWN_PADS) * N_SEEDS != KNOWN_N:
        raise SystemExit(KNOWN_N)
    if KNOWN_LINE != 171 or KNOWN_N - KNOWN_LINE != 9:
        raise SystemExit((KNOWN_LINE, KNOWN_N - KNOWN_LINE))
    if abs(RATIO_MIN - 0.85) > 0.0 or GAIN_MIN != 18 or CELL_TOL != 3 or RULE_CELL_MIN != 3:
        raise SystemExit((RATIO_MIN, GAIN_MIN, CELL_TOL, RULE_CELL_MIN))
    if abs(float(NAV_TOL) - 6.0) > 0.0 or float(VY_SOFT) != 8.0 or int(HOVER_CAP) != 900:
        raise SystemExit("допуск, порог касания или потолок сдвинуты")
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or float(V_TARGET_LOOP) != 6.0:
        raise SystemExit("бак, земля или v_target сдвинуты")
    if THREADS != 8 or N_SEQ != 729 or n_in_of("latent") != 33:
        raise SystemExit("потоки, планы или ширина входа сдвинуты")
    if mode_of(0.5) != "land" or mode_of(0.499) != "approach":
        raise SystemExit("порог 0.5 разошёлся")

    if known_ok(170) or not known_ok(171) or not known_ok(180):
        raise SystemExit("линия знакомой части сдвинута")
    if ratio_need(400) != 340 or ratio_need(0) != 0:
        raise SystemExit(ratio_need(400))
    if aggregate_ok(339, 400) or not aggregate_ok(340, 400):
        raise SystemExit("отношение 85 % сломано")
    if cell_ok(21, 24) is not True or cell_ok(20, 24) is not False:
        raise SystemExit("клеточный допуск 3 сломан")
    if cell_ok(0, 2) is not None or cell_ok(30, 0) is not None:
        raise SystemExit("клетка вне домена правила не помечена")
    if tube_gain_ok(17) or not tube_gain_ok(18):
        raise SystemExit("разрыв 18 сдвинут")

    rows = episodes()
    if len(rows) != 600:
        raise SystemExit(len(rows))
    seen = set()
    per_seed = {}
    for row in rows:
        key = (int(row["seed"]), float(row["g"]), float(row["pad_x"]))
        if key in seen:
            raise SystemExit(f"повтор {key}")
        seen.add(key)
        per_seed[int(row["seed"])] = per_seed.get(int(row["seed"]), 0) + 1
        x, y, vx, vy, _acts = start_and_actions(int(row["seed"]), 0)
        if (float(row["x"]), float(row["y"]), float(row["vx"]), float(row["vy"])) != (
            float(x), float(y), float(vx), float(vy),
        ):
            raise SystemExit("старт не из start_and_actions")
        if float(row["fuel"]) != 10.0:
            raise SystemExit("бак не 10")
    if set(per_seed) != set(range(SEED0, SEED0 + N_SEEDS)):
        raise SystemExit(sorted(per_seed)[:3])
    if any(count != len(GRID_G) * len(GRID_PADS) for count in per_seed.values()):
        raise SystemExit("сид встретился не двадцать раз")
    cells_seen = {(float(row["g"]), float(row["pad_x"])) for row in rows}
    if len(cells_seen) != 20:
        raise SystemExit(len(cells_seen))

    # Блоки отчёта на синтетических столбцах.
    tube = {"cells": _cells(21), "soft": 21 + 30 * 19, "of": GRID_N}
    rule = {"cells": _cells(24), "soft": 24 + 30 * 19, "of": GRID_N}
    terminal = {"cells": _cells(0), "soft": 0, "of": GRID_N}
    known = known_block(tube)
    if known["of"] != 180 or known["line"] != KNOWN_LINE or known["ok"] is not True:
        raise SystemExit(known)
    blocked = verify(tube, rule, terminal)
    if blocked["cells_ok"] is not True or blocked["n_rule_out_of_domain"] != 0:
        raise SystemExit(blocked)
    if blocked["tube_gain"] != 21 + 30 * 19 or not blocked["tube_gain_ok"]:
        raise SystemExit(blocked["tube_gain"])
    if not blocked["aggregate_ok"] or blocked["pilot_pass"] is not True:
        raise SystemExit(blocked)
    narrow = verify({"cells": _cells(20), "soft": 20 + 30 * 19, "of": GRID_N}, rule, terminal)
    if narrow["cells_ok"] or narrow["pilot_pass"]:
        raise SystemExit("клетка ниже правила минус 3 зачлась")
    weak = verify({"cells": _cells(30), "soft": GRID_N, "of": GRID_N},
                  {"cells": _cells(1), "soft": 1 + 30 * 19, "of": GRID_N}, terminal)
    if weak["n_rule_out_of_domain"] != 1 or weak["cells_ok"] is not True:
        raise SystemExit(weak)
    edge = verify({"cells": _cells(30), "soft": GRID_N, "of": GRID_N},
                  {"cells": _cells(30), "soft": GRID_N, "of": GRID_N},
                  {"cells": _cells(30), "soft": GRID_N, "of": GRID_N})
    if edge["tube_gain_ok"] or edge["pilot_pass"]:
        raise SystemExit("нулевой разрыв зачёлся")
    marked = cells_block({"cells": _cells(30)}, {"cells": _cells(1)})
    if marked[0]["rule_soft"] != 1 or marked[0]["rule_out_of_domain"] is not True:
        raise SystemExit(marked[0])
    if marked[0]["cell_pass"] is not None:
        raise SystemExit("клетка вне домена получила вердикт")
    domains = by_domain({"cells": _cells(30)}, {"cells": _cells(30)})
    known_row = [row for row in domains if row["domain"] == "known"][0]
    edge_row = [row for row in domains if row["domain"] == "edge"][0]
    if known_row["of"] != 180 or edge_row["of"] != 420:
        raise SystemExit(domains)

    never = SwitchLog()
    if never.bucket() != "no_switch":
        raise SystemExit("пустой эпизод не «без переключения»")
    opened = SwitchLog()
    opened.observe("land", 0, 200.0, 1.0)
    if opened.bucket() != "opened_in_land" or opened.switch is not None:
        raise SystemExit("старт в land попал в медиану")
    edge_log = SwitchLog()
    edge_log.observe("approach", 3, 80.0, 40.0)
    edge_log.observe("land", 9, 50.0, 5.5)
    if edge_log.switch["step"] != 9 or edge_log.switch["abs_dx"] != 5.5:
        raise SystemExit(edge_log.switch)
    summary = summarize_switches(_switch_rows())
    if summary["n"] != GRID_N or summary["n_switch"] != 3 or summary["n_no_switch"] != 596:
        raise SystemExit((summary["n"], summary["n_switch"], summary["n_no_switch"]))
    if summary["n_height_gt_128"] != 1 or summary["height_gt_128_is_gate"]:
        raise SystemExit("доля выше 128 посчитана как гейт или неверно")

    params = list(inspect.signature(step_field).parameters)
    if "pad_x" in params or "mode" in params:
        raise SystemExit(f"физика видит лишнее: {params}")

    module = sys.modules["numeric_land.skill_assemble"]
    src = inspect.getsource(module)
    for banned in (
        "torch.save", "AdamW", "backward", "cem_plan", "inference_mode",
        "def plan_costs", "def roll_tube", "def goal_z", "def first_action",
    ):
        if banned in src:
            raise SystemExit(f"модуль содержит {banned}")
    if "choose_action" not in inspect.getsource(tube_action) or "prepare" not in inspect.getsource(tube_action):
        raise SystemExit("поиск по трубе не зовёт готовую стоимость")
    if "hover_action" in inspect.getsource(tube_action) or "hover_action" in inspect.getsource(terminal_action):
        raise SystemExit("навык берёт действие у правила")
    if "prepare" in inspect.getsource(terminal_action) or "plan_costs" in inspect.getsource(terminal_action):
        raise SystemExit("абляция пользуется трубой")
    if "first_action" not in inspect.getsource(terminal_action):
        raise SystemExit("абляция без терминала на земле")
    if "goal_z" not in inspect.getsource(terminal_action):
        raise SystemExit("абляция без цели на земле")
    if "hover_action" not in inspect.getsource(rule_action) or "mode=mode" not in inspect.getsource(rule_action):
        raise SystemExit("столбец правила не вызывает названную ветку")
    if "mode_of" not in inspect.getsource(classify):
        raise SystemExit("голова считает порог отдельно от mode_of")
    if "predictor_of" not in inspect.getsource(closed_skill_step):
        raise SystemExit("режим не выбирает предиктор")
    if "hover_action" in inspect.getsource(closed_skill_step):
        raise SystemExit("шаг контура сам вызывает правило")
    main_src = inspect.getsource(module.main)
    bundle_src = inspect.getsource(module.load_bundle)
    for banned in (
        "landd_early_s1", "landk_pad_s1", "landk_approach_s1", "landk_approach_s2",
        "pad_path(1)", "approach_file(1)", "approach_file(2)",
    ):
        if banned in main_src or banned in bundle_src:
            raise SystemExit(f"{banned} попал в прогон")
    if main_src.count("fly_column(") != 4:
        raise SystemExit("прогон зовёт не четыре столбца")
    early_at = main_src.find('"rule_early"')
    latent_at = main_src.find('"rule_latent"')
    tube_at = main_src.find('"skill_tube"')
    term_at = main_src.find('"skill_terminal"')
    write_at = main_src.find("write_json")
    if not (0 <= main_src.find("refuse_tag") < main_src.find("refuse_report") < early_at):
        raise SystemExit("отказ стоит не до полёта")
    if not (early_at < latent_at < tube_at < term_at < write_at):
        raise SystemExit("столбцы или запись отчёта стоят не в том порядке")
    for flag in (
        '"grid_n": GRID_N', '"known_n": int(KNOWN_N)', '"known_line": int(KNOWN_LINE)',
        '"ratio_min": RATIO_MIN', '"gain_min": int(GAIN_MIN)', '"cell_tol": int(CELL_TOL)',
        '"threshold": float(THRESHOLD)', '"horizon": HORIZON', '"n_sequences": N_SEQ',
        '"cem": False', '"rule_action_in_scored": False', '"heads_trained": False',
        '"stamps_unchanged": True', '"known_ok": bool(numbers["known"]["ok"])',
        '"aggregate_ok": bool(numbers["aggregate_ok"])', '"cells_ok": bool(numbers["cells_ok"])',
        '"tube_gain": int(numbers["tube_gain"])', '"tube_gain_ok": bool(numbers["tube_gain_ok"])',
        '"pilot_pass": bool(numbers["pilot_pass"])', '"tube_clipped": int(tube["tube_clipped"])',
        '"height_gt_128_is_gate": False', '"by_domain": domains',
    ):
        if flag not in main_src:
            raise SystemExit(f"в отчёте нет {flag}")

    _expect_exit(lambda: refuse_tag("numeric-land-skill-decision"))
    fd, temp_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        _expect_exit(lambda: refuse_report(temp_json))
    finally:
        os.remove(temp_json)
    if os.path.isfile(module.json_path()):
        raise SystemExit("отчёт уже есть, контроль не летит поверх него")

    torch.set_num_threads(THREADS)
    before = take_stamps()
    hover_before = open(hover_path, "rb").read()
    bundle = load_bundle()
    if take_stamps() != before:
        raise SystemExit("чтение сдвинуло штамп")
    if predictor_of("land", {"land": bundle["pad"], "approach": bundle["approach"]}) is not bundle["pad"]:
        raise SystemExit("land смотрит не в landk_pad_s0")
    if predictor_of("approach", {"land": bundle["pad"], "approach": bundle["approach"]}) is not bundle["approach"]:
        raise SystemExit("approach смотрит не в landk_approach_s0")
    if os.path.basename(approach_path()) != "landk_approach_s0.pt":
        raise SystemExit(approach_path())

    class _Boom:
        def __call__(self, *args, **kwargs):
            raise SystemExit("поиск вызван без режима")

    boom = {"land": _Boom(), "approach": _Boom()}
    ground = (256.0, float(GROUND), 0.0, 0.0, 10.0)
    empty = (256.0, float(GROUND) - 40.0, 0.0, 0.0, 0.0)
    for state in (ground, empty):
        _nxt, action, _clipped, mode, _prob = closed_skill_step(
            "tube", None, None, None, boom, None, None, None, state, 2.0, 256.0, None)
        if action != 1 or mode is not None:
            raise SystemExit("на земле или при пустом баке действие не 1")
        if rule_action(state, 2.0, 256.0, "land") != 1:
            raise SystemExit("ветка правила на земле не отдала 1")

    sample = rows[0]
    state = (float(sample["x"]), float(sample["y"]), float(sample["vx"]), float(sample["vy"]), 10.0)
    g = float(sample["g"])
    pad_x = float(sample["pad_x"])
    actions, step_pairs = pair_table()
    world_b = world_batch(g, pad_x)
    preds = {"land": bundle["pad"], "approach": bundle["approach"]}
    saved = module.hover_action

    def _boom_hover(*args, **kwargs):
        del args, kwargs
        raise SystemExit("зачётный шаг вызвал hover_action")

    module.hover_action = _boom_hover
    try:
        nxt, action, _clipped, mode, _prob = closed_skill_step(
            "tube", bundle["early"], bundle["encoder"], bundle["ema"], preds,
            actions, step_pairs, world_b, state, g, pad_x, {})
    finally:
        module.hover_action = saved
    if mode not in ("approach", "land"):
        raise SystemExit("первый старт не дал режим")
    z, z_tube, _frames, calls, _was = prepare(
        bundle["encoder"], bundle["ema"], {
            "x": state[0], "y": state[1], "vx": state[2], "vy": state[3], "fuel": state[4],
            "g": g, "pad_x": pad_x,
        })
    if calls < 1:
        raise SystemExit("труба не вызвала цель")
    expected, _cost = choose_action(
        predictor_of(mode, preds), z, z_tube, world_b, step_pairs, actions)
    if int(action) != int(expected):
        raise SystemExit("в эпизод ушло не действие argmin")
    manual = step_field(state[0], state[1], state[2], state[3], state[4], False, int(action), g)
    if nxt != manual:
        raise SystemExit("следующее состояние не step_field выбранного действия")

    cache = {}
    term_nxt, term_action, _term_clip, term_mode, _term_prob = closed_skill_step(
        "terminal", bundle["early"], bundle["encoder"], bundle["ema"], preds,
        actions, step_pairs, world_b, state, g, pad_x, cache)
    again = {}
    direct = terminal_action(
        predictor_of(term_mode, preds), bundle["encoder"], bundle["ema"],
        actions, step_pairs, world_b, state, g, pad_x, again)
    if int(term_action) != int(direct):
        raise SystemExit("абляция ушла не в first_action")
    raw = np.asarray([state], np.float64)
    pads = np.asarray([pad_x], np.float64)
    goal_action = first_action(
        predictor_of(term_mode, preds),
        torch.from_numpy(np.ascontiguousarray(encode_rel(bundle["encoder"], raw, pads))),
        goal_z(bundle["ema"], pad_x, state[4], {}),
        world_b, step_pairs, actions)
    if int(goal_action) != int(term_action):
        raise SystemExit("терминал на земле не совпал с goal_z")
    if term_nxt != step_field(state[0], state[1], state[2], state[3], state[4], False, int(term_action), g):
        raise SystemExit("шаг абляции не step_field")

    for head in (bundle["early"], bundle["latent"]):
        decide = _decide(head, bundle["encoder"], g, pad_x, 0.5)
        x, y, vx, vy, fuel = state
        grounded = False
        for _ in range(8):
            if grounded:
                break
            current = (float(x), float(y), float(vx), float(vy), float(fuel))
            mine, named = rule_from_head(head, bundle["encoder"], current, g, pad_x)
            ref = int(decide(current))
            if int(mine) != ref:
                raise SystemExit(f"правило разошлось: {mine} vs {ref}")
            if named is not None and mine != rule_action(current, g, pad_x, named):
                raise SystemExit("названный режим не дошёл до hover_action")
            x, y, vx, vy, fuel, grounded = step_field(x, y, vx, vy, fuel, False, int(mine), g)

    if take_stamps() != before:
        raise SystemExit("контроль сдвинул штамп")
    if open(hover_path, "rb").read() != hover_before:
        raise SystemExit("правило изменилось")
    if stray_pt():
        raise SystemExit("после контроля появился landd_assemble_*.pt")
    print(
        "ok grid 600 cells 20 known 171/180 ratio 400 -> 340 cell tol 3 "
        "gain 18 yes 17 no out_of_domain skipped mode 0.5 land 0.499 approach",
        flush=True,
    )


if __name__ == "__main__":
    main()

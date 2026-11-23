"""Контроль этапа 3 до прогона. python _check_numeric_land_skill_decision.py"""
import inspect
import os
import sys
import tempfile

import numpy as np
import torch

from numeric_land.decision import n_in_of
from numeric_land.expert_hover import HOVER_CAP, NAV_TOL
from numeric_land.field import FUEL0, GROUND, VY_SOFT, start_and_actions, step_field
from numeric_land.mode import V_TARGET_LOOP
from numeric_land.relenc import encode_rel
from numeric_land.rollout_decision_threshold import _decide
from numeric_land.skill_decision import (
    CONTROL_EARLY, CONTROL_LATENT, GAIN_MIN, GRID_N, LINE, RATIO_MIN, TAG,
    SwitchLog, approach_path, classify, closed_skill_step, control_ok, episodes,
    line_closed, load_bundle, mode_of, predictor_of, ratio_need, refuse_report,
    refuse_tag, rule_action, rule_from_head, stray_pt, summarize_switches,
    take_stamps, terminal_action, tube_action, tube_gain_ok, verdict,
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
    """180 строк: два перехода, один старт в land, остальные без land."""
    rows = []
    rows.append({
        "bucket": "switch", "switch_step": 10, "switch_height": 40.0, "switch_abs_dx": 6.0,
    })
    rows.append({
        "bucket": "switch", "switch_step": 30, "switch_height": 129.0, "switch_abs_dx": 2.0,
    })
    rows.append({
        "bucket": "switch", "switch_step": 12, "switch_height": 128.0, "switch_abs_dx": 4.0,
    })
    rows.append({
        "bucket": "opened_in_land", "switch_step": None, "switch_height": None, "switch_abs_dx": None,
    })
    while len(rows) < GRID_N:
        rows.append({
            "bucket": "no_switch", "switch_step": None, "switch_height": None, "switch_abs_dx": None,
        })
    return rows


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

    if TAG != "numeric-land-skill-decision":
        raise SystemExit(TAG)
    if LINE != 100 or GRID_N != 180 or GAIN_MIN != 18 or abs(RATIO_MIN - 0.85) > 0.0:
        raise SystemExit("линия, сетка или разрыв сдвинуты")
    if CONTROL_EARLY != 120 or CONTROL_LATENT != 108:
        raise SystemExit("контроль сдвинут")
    if abs(float(NAV_TOL) - 6.0) > 0.0 or float(VY_SOFT) != 8.0 or int(HOVER_CAP) != 900:
        raise SystemExit("допуск, порог касания или потолок сдвинуты")
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or float(V_TARGET_LOOP) != 6.0:
        raise SystemExit("бак, земля или v_target сдвинуты")
    if THREADS != 8 or N_SEQ != 729 or n_in_of("latent") != 33:
        raise SystemExit("потоки, планы или ширина входа сдвинуты")
    if mode_of(0.5) != "land" or mode_of(0.499) != "approach":
        raise SystemExit("порог 0.5 разошёлся")
    if ratio_need(120) != 102:
        raise SystemExit(f"ceil(0.85·120) = {ratio_need(120)}, ждали 102")
    if line_closed(99, 120) or line_closed(100, 120) or line_closed(101, 120):
        raise SystemExit("99, 100 или 101 проходят при правиле 120")
    if not line_closed(102, 120) or not line_closed(103, 120):
        raise SystemExit("102 или 103 не проходят при правиле 120")
    if control_ok(119, 108) or control_ok(121, 108) or control_ok(120, 107) or control_ok(120, 109):
        raise SystemExit("контроль принял не 120 и 108")
    if not control_ok(120, 108):
        raise SystemExit("контроль не принял запись")
    if tube_gain_ok(17) or not tube_gain_ok(18):
        raise SystemExit("разрыв 18 сдвинут")
    passed = verdict(103, 120, 108, 80)
    if not passed["pilot_pass"] or passed["tube_gain"] != 23 or passed["ratio_need"] != 102:
        raise SystemExit("зачёт 103 при правиле 120 не закрылся")
    soft_only = verdict(100, 120, 108, 0)
    if soft_only["line_ok"] or soft_only["pilot_pass"] or soft_only["control_ok"] is not True:
        raise SystemExit("100 при правиле 120 закрыл линию")
    short = verdict(180, 119, 108, 0)
    if short["control_ok"] or short["pilot_pass"]:
        raise SystemExit("правило 119 зачлось")
    gap = verdict(103, 120, 108, 86)
    if gap["tube_gain"] != 17 or gap["tube_gain_ok"] or gap["pilot_pass"]:
        raise SystemExit("разрыв 17 зачёлся")
    other = verdict(140, 121, 108, 100)
    if other["control_ok"] or other["pilot_pass"]:
        raise SystemExit("правило 121 зачлось")

    rows = episodes()
    if len(rows) != 180:
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
    if len(seen) != 180 or set(per_seed) != set(range(99000, 99030)):
        raise SystemExit("сетка не 99000…99029 × 2/8/15 × 256/200")
    if any(count != 6 for count in per_seed.values()):
        raise SystemExit("сид встретился не шесть раз")

    never = SwitchLog()
    if never.bucket() != "no_switch":
        raise SystemExit("пустой эпизод не «без переключения»")
    opened = SwitchLog()
    opened.observe("land", 0, 200.0, 1.0)
    if opened.bucket() != "opened_in_land" or opened.switch is not None:
        raise SystemExit("старт в land попал в медиану")
    edge = SwitchLog()
    edge.observe("approach", 3, 80.0, 40.0)
    edge.observe("land", 9, 50.0, 5.5)
    if edge.bucket() != "switch" or edge.switch["step"] != 9 or edge.switch["height"] != 50.0:
        raise SystemExit("переход approach → land записан не в момент land")
    if edge.switch["abs_dx"] != 5.5:
        raise SystemExit("|dx| перехода не тот")
    summary = summarize_switches(_switch_rows())
    if summary["n_switch"] != 3 or summary["n_no_switch"] != 176 or summary["n_opened_in_land"] != 1:
        raise SystemExit(summary)
    if summary["n_height_gt_128"] != 1 or summary["height_gt_128_is_gate"]:
        raise SystemExit("доля выше 128 посчитана как гейт или неверно")
    if summary["median_step"] != 12.0 or abs(summary["median_height"] - 128.0) > 0.0:
        raise SystemExit((summary["median_step"], summary["median_height"]))
    if abs(summary["median_abs_dx"] - 4.0) > 0.0:
        raise SystemExit(summary["median_abs_dx"])
    if summary["share_height_gt_128"] != 1.0 / 3.0:
        raise SystemExit(summary["share_height_gt_128"])

    params = list(inspect.signature(step_field).parameters)
    if "pad_x" in params or "mode" in params:
        raise SystemExit(f"физика видит лишнее: {params}")

    module = sys.modules["numeric_land.skill_decision"]
    src = inspect.getsource(module)
    if "новый вес" in src:
        raise SystemExit("в модуле есть запрещённая формулировка")
    head, _body = src.split("def refuse_tag", 1)
    if "refuse_tag" in head:
        raise SystemExit("чужой refuse_tag попал в импорт")
    for banned in (
        "torch.save", "AdamW", "backward", "cem_plan", "inference_mode",
        "def plan_costs", "def roll_tube", "def goal_z", "def first_action",
        "in_transfer_zone",
    ):
        if banned in src:
            raise SystemExit(f"модуль содержит {banned}")
    if "choose_action" not in inspect.getsource(tube_action) or "prepare" not in inspect.getsource(tube_action):
        raise SystemExit("поиск по трубе не зовёт готовую стоимость")
    if "hover_action" in inspect.getsource(tube_action) or "hover_action" in inspect.getsource(terminal_action):
        raise SystemExit("навык берёт действие у правила")
    if "plan_costs" in inspect.getsource(terminal_action) or "prepare" in inspect.getsource(terminal_action):
        raise SystemExit("абляция пользуется трубой")
    if "first_action" not in inspect.getsource(terminal_action) or "goal_z" not in inspect.getsource(terminal_action):
        raise SystemExit("абляция без терминала на земле")
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
        '"grid_n": GRID_N', '"line": LINE', '"ratio_min": RATIO_MIN',
        '"threshold": float(THRESHOLD)', '"horizon": HORIZON', '"n_sequences": N_SEQ',
        '"cem": False', '"rule_action_in_scored": False', '"heads_trained": False',
        '"stamps_unchanged": True', '"tube_gain": numbers["tube_gain"]',
        '"tube_gain_ok": numbers["tube_gain_ok"]', '"control_ok": numbers["control_ok"]',
        '"pilot_pass": numbers["pilot_pass"]', '"tube_clipped": int(tube["tube_clipped"])',
        '"height_gt_128_is_gate": False',
    ):
        if flag not in main_src:
            raise SystemExit(f"в отчёте нет {flag}")

    _expect_exit(lambda: refuse_tag("numeric-land-decision"))
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
    bundle = load_bundle()
    if take_stamps() != before:
        raise SystemExit("чтение сдвинуло штамп")
    if predictor_of("land", {"land": bundle["pad"], "approach": bundle["approach"]}) is not bundle["pad"]:
        raise SystemExit("land смотрит не в landk_pad_s0")
    if predictor_of("approach", {"land": bundle["pad"], "approach": bundle["approach"]}) is not bundle["approach"]:
        raise SystemExit("approach смотрит не в landk_approach_s0")

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
    import numeric_land.skill_decision as decision
    saved = decision.hover_action

    def _boom_hover(*args, **kwargs):
        del args, kwargs
        raise SystemExit("зачётный шаг вызвал hover_action")

    decision.hover_action = _boom_hover
    try:
        nxt, action, _clipped, mode, _prob = closed_skill_step(
            "tube", bundle["early"], bundle["encoder"], bundle["ema"], preds,
            actions, step_pairs, world_b, state, g, pad_x, {})
    finally:
        decision.hover_action = saved
    if mode not in ("approach", "land"):
        raise SystemExit("старт 99000 не дал режим")
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

    if os.path.basename(approach_path()) != "landk_approach_s0.pt":
        raise SystemExit(approach_path())
    if take_stamps() != before:
        raise SystemExit("контроль сдвинул штамп")
    if open(hover_path, "rb").read() != hover_bytes:
        raise SystemExit("правило изменилось")
    if stray_pt():
        raise SystemExit("после контроля появился landd_skill_*.pt")
    print(
        "ok grid 180 line 99 no 100 no 102 yes ratio 120 -> 102 "
        "threshold 0.5 land 0.499 approach gain 18 yes 17 no",
        flush=True,
    )


if __name__ == "__main__":
    main()

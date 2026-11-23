"""Контроль наведения до прогона. python _check_numeric_land_skill_approach.py"""
import inspect
import os
import sys
import tempfile

import torch

from numeric_land.expert_hover import HOVER_CAP, NAV_TOL
from numeric_land.field import FUEL0, GROUND, VY_SOFT, start_and_actions, step_field
from numeric_land.pad import pad_of_window
from numeric_land.skill_approach import (
    ENCODER_TAG, EXCLUDED_N, GRID_N, GS, HEIGHT_MIN, HOLD, LINE, N_SEEDS, SCORED_N,
    SEED0, STAGE1_HEIGHT, TAG, VY_HI, VY_LO, after_step, approach_file, excluded_start,
    field_rows, fly_rule, fly_skill, in_transfer_zone, line_ok, refuse_report, refuse_tag,
    split_grid, stray_pt, take_stamps,
)
from numeric_land.skill_land import N_SEQ, pair_table, world_batch
from numeric_land.skill_tube import (
    TUBE_TARGETS, baseline_path, choose_action, closed_step, load_frozen, load_predictor,
    plan_costs, prepare, roll_tube, tube_latents,
)
from numeric_land.train_goal import THREADS


def _clip_state():
    """Ниже 32 px, правило касается на первом шаге трубы."""
    for height in (0.4, 0.8, 1.0, 1.5, 2.0, 3.0):
        for vy in (24.0, 40.0, 60.0, 90.0):
            for g in (15.0, 8.0, 2.0):
                state = (256.0, float(GROUND) - height, 0.0, float(vy), float(FUEL0))
                frames, calls, clipped = roll_tube(state, g, 256.0)
                if not clipped or calls != 1:
                    continue
                if frames[0] != frames[1] or frames[1] != frames[2]:
                    raise SystemExit("касание не повторило последний состояние трубы")
                if float(frames[0][1]) < float(GROUND):
                    raise SystemExit("касание не дошло до земли")
                return state, g, frames
    raise SystemExit("не нашли касание правила на первом шаге трубы")


def _expect_exit(fn):
    try:
        fn()
    except SystemExit as exc:
        if exc.code in (0, None):
            raise SystemExit("отказ вышел с нулевым кодом")
        return
    raise SystemExit("ожидался отказ")


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

    if TAG != "numeric-land-skill-approach":
        raise SystemExit(TAG)
    if LINE != 65 or GRID_N != 90 or SCORED_N != 81 or EXCLUDED_N != 9:
        raise SystemExit("линия или сетка сдвинуты")
    if line_ok(64) or not line_ok(65):
        raise SystemExit("64 проходит или 65 не проходит")
    if HOLD != 3 or VY_LO != -4.0 or VY_HI != 12.0 or HEIGHT_MIN != 24.0:
        raise SystemExit("зона или удержание сдвинуты")
    if abs(float(NAV_TOL) - 6.0) > 0.0 or float(VY_SOFT) != 8.0 or int(HOVER_CAP) != 900:
        raise SystemExit("допуск, порог или потолок сдвинуты")
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise SystemExit("бак, земля или потоки сдвинуты")
    if N_SEQ != 729 or TUBE_TARGETS != 3 or SEED0 != 200000 or N_SEEDS != 30:
        raise SystemExit("поиск или сиды сдвинуты")
    if GS != (2.0, 8.0, 15.0):
        raise SystemExit(GS)

    params = list(inspect.signature(step_field).parameters)
    if "pad_x" in params or "mode" in params:
        raise SystemExit(f"физика видит лишнее: {params}")

    module = sys.modules["numeric_land.skill_approach"]
    src = inspect.getsource(module)
    if "новый вес" in src:
        raise SystemExit("в модуле есть запрещённая формулировка")
    head, _body = src.split("def refuse_tag", 1)
    if "refuse_tag" in head:
        raise SystemExit("чужой refuse_tag попал в импорт")
    for banned in (
        "terminal_state", "target_state", "goal_z", "cem_plan", "landd_",
        "outcome_at", "torch.save", "AdamW", "backward", "def plan_costs", "def roll_tube",
    ):
        if banned in src:
            raise SystemExit(f"модуль наведения содержит {banned}")
    if "closed_step" not in src:
        raise SystemExit("контур не зовёт трубу")
    if "hover_action" in inspect.getsource(fly_skill):
        raise SystemExit("навык берёт действие у правила")
    if "hover_action" not in inspect.getsource(fly_rule):
        raise SystemExit("справочное правило не вызвано")
    if "step_field" not in inspect.getsource(fly_rule):
        raise SystemExit("правило не шагает физикой")
    main_src = inspect.getsource(module.main)
    if main_src.count("fly_scored(") != 2 or main_src.count("fly_rule_rows(") != 1:
        raise SystemExit("прогон зовёт не те столбцы")
    if main_src.count("load_predictor(") != 2:
        raise SystemExit("читается не два предиктора")
    if "approach_file(1)" in main_src or "approach_file(2)" in main_src:
        raise SystemExit("s1 или s2 попали в прогон")
    if "landk_approach_s1" in main_src or "landk_approach_s2" in main_src:
        raise SystemExit("s1 или s2 названы в прогоне")
    for flag in (
        '"hover_action_action": False', '"tube_targets": TUBE_TARGETS',
        '"cem": False', '"decision_loaded": False',
        '"approach_s1_flown": False', '"approach_s2_flown": False',
        '"height_le_128_is_gate": False', '"passed": bool(approach_row["line_ok"])',
    ):
        if flag not in main_src:
            raise SystemExit(f"в отчёте нет {flag}")
    if not (main_src.find("refuse_tag") < main_src.find("refuse_report")
            < main_src.find("fly_scored") < main_src.find("write_json")):
        raise SystemExit("отказ или запись отчёта стоят не на месте")
    cost_src = inspect.getsource(plan_costs)
    if "terminal_state" in cost_src or "goal_z" in cost_src:
        raise SystemExit("в стоимости трубы есть терминал на земле")
    tube_src = inspect.getsource(sys.modules["numeric_land.skill_tube"])
    if "hover_action" not in inspect.getsource(roll_tube):
        raise SystemExit("труба не вызывает правило")
    if "def fly_one" not in tube_src:
        raise SystemExit("модуль трубы сломан")

    _expect_exit(lambda: refuse_tag("numeric-land-decision"))
    fd, temp_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        _expect_exit(lambda: refuse_report(temp_json))
    finally:
        os.remove(temp_json)

    pad = 300.0
    y24 = float(GROUND) - 24.0
    y239 = float(GROUND) - 23.9
    if not in_transfer_zone(pad + 6.0, y24, 0.0, 10.0, False, pad):
        raise SystemExit("|dx| = 6 не вошло в зону")
    if in_transfer_zone(pad + 6.1, y24, 0.0, 10.0, False, pad):
        raise SystemExit("|dx| = 6.1 вошло в зону")
    if not in_transfer_zone(pad - 6.0, y24, 0.0, 10.0, False, pad):
        raise SystemExit("dx = -6 не вошло в зону")
    if in_transfer_zone(pad - 6.1, y24, 0.0, 10.0, False, pad):
        raise SystemExit("dx = -6.1 вошло в зону")
    if not in_transfer_zone(pad, y24, 12.0, 10.0, False, pad):
        raise SystemExit("vy = 12 не вошло в зону")
    if in_transfer_zone(pad, y24, 12.5, 10.0, False, pad):
        raise SystemExit("vy = 12.5 вошло в зону")
    if not in_transfer_zone(pad, y24, -4.0, 10.0, False, pad):
        raise SystemExit("vy = -4 не вошло в зону")
    if in_transfer_zone(pad, y24, -4.1, 10.0, False, pad):
        raise SystemExit("vy = -4.1 вошло в зону")
    if not in_transfer_zone(pad, y24, 0.0, 10.0, False, pad):
        raise SystemExit("высота 24 не вошла в зону")
    if in_transfer_zone(pad, y239, 0.0, 10.0, False, pad):
        raise SystemExit("высота 23.9 вошла в зону")
    if in_transfer_zone(pad, y24, 0.0, 0.0, False, pad):
        raise SystemExit("пустой бак вошёл в зону")
    if in_transfer_zone(pad, y24, 0.0, 10.0, True, pad):
        raise SystemExit("касание вошло в зону")
    if in_transfer_zone(pad, float(GROUND), 0.0, 10.0, False, pad):
        raise SystemExit("земля вошла в зону")

    hold, outcome, fuel_out = 0, None, False
    for _ in range(2):
        hold, outcome, fuel_out = after_step(hold, pad, y24, 0.0, 10.0, False, pad)
        if outcome is not None or fuel_out or hold > 2:
            raise SystemExit("два шага в зоне уже успех")
    if hold != 2:
        raise SystemExit("удержание не дошло до 2")
    hold, outcome, fuel_out = after_step(hold, pad, y24, 0.0, 10.0, False, pad)
    if outcome != "передача" or hold != 3 or fuel_out:
        raise SystemExit("три шага в зоне не успех")

    hold = 0
    for _ in range(2):
        hold, outcome, fuel_out = after_step(hold, pad, y24, 0.0, 10.0, False, pad)
    hold, outcome, fuel_out = after_step(hold, pad + 20.0, y24, 0.0, 10.0, False, pad)
    if outcome is not None or hold != 0:
        raise SystemExit("выход после двух шагов засчитан успехом")

    hold = 0
    for _ in range(3):
        hold, outcome, fuel_out = after_step(hold, pad, y24, 12.5, 10.0, False, pad)
        if outcome is not None:
            raise SystemExit("vy = 12.5 удержалось как зона")
    if hold != 0:
        raise SystemExit("vy = 12.5 копит удержание")

    _hold, outcome, fuel_out = after_step(2, pad, y24, 0.0, 0.0, False, pad)
    if outcome != "промах" or not fuel_out:
        raise SystemExit("пустой бак на третьем шаге засчитан передачей")
    _hold, outcome, fuel_out = after_step(2, pad, float(GROUND), 0.0, 10.0, True, pad)
    if outcome != "удар" or fuel_out:
        raise SystemExit("касание без трёх шагов не удар")

    if not excluded_start({"x": pad + 6.0, "pad_x": pad}):
        raise SystemExit("старт с |dx| = 6 не исключён")
    if excluded_start({"x": pad + 6.1, "pad_x": pad}):
        raise SystemExit("старт с |dx| = 6.1 исключён")

    rows = field_rows()
    if len(rows) != 90:
        raise SystemExit("эпизодов не 90")
    seen = set()
    pads = set()
    for row in rows:
        key = (int(row["seed"]), float(row["g"]))
        if key in seen:
            raise SystemExit(f"пара повторилась {key}")
        seen.add(key)
        if int(row["seed"]) < 200000 or int(row["seed"]) > 200029:
            raise SystemExit("сид не из поля")
        if float(row["g"]) not in (2.0, 8.0, 15.0):
            raise SystemExit("чужая тяжесть")
        pad_x = float(pad_of_window(int(row["seed"])))
        if float(row["pad_x"]) != pad_x:
            raise SystemExit("площадка не из окна")
        if pad_x == 256.0 or pad_x == 200.0:
            raise SystemExit(f"площадка жёстко {pad_x}")
        if not (156.0 <= pad_x < 356.0):
            raise SystemExit("площадка вне окна обучения")
        pads.add(pad_x)
        x, y, vx, vy, actions = start_and_actions(int(row["seed"]), 0)
        if (float(row["x"]), float(row["y"]), float(row["vx"]), float(row["vy"])) != (x, y, vx, vy):
            raise SystemExit("старт не из start_and_actions")
        if len(actions) != 0 or float(row["fuel"]) != 10.0:
            raise SystemExit("бак или лишние действия старта")
        if abs((float(row["x"]) - pad_x) - float(row["dx"])) > 1e-9:
            raise SystemExit("dx разошёлся со стартом")
    if len(seen) != 90 or len(pads) < 2:
        raise SystemExit("сетка не 90 разных пар или площадка одна")
    scored, excluded = split_grid(rows)
    if len(scored) != 81 or len(excluded) != 9:
        raise SystemExit("зачёт не 81 или исключённых не 9")
    if any(abs(float(row["dx"])) > 6.0 for row in excluded):
        raise SystemExit("исключён старт дальше допуска")
    if any(abs(float(row["dx"])) <= 6.0 for row in scored):
        raise SystemExit("в зачёте остался старт над площадкой")
    seeds = sorted({int(row["seed"]) for row in excluded})
    if len(seeds) != 3:
        raise SystemExit(f"исключённые сиды не три: {seeds}")

    state, g_clip, frames = _clip_state()
    tube_clipped = 0
    if True:
        _frames, calls, clipped = roll_tube(state, g_clip, 256.0)
        if not clipped or calls != 1 or _frames != frames:
            raise SystemExit("обрезка трубы разошлась")
        tube_clipped += 1
    if tube_clipped != 1:
        raise SystemExit("tube_clipped не увеличился")

    torch.set_num_threads(THREADS)
    stamps = take_stamps()
    for name in ("landk_approach_s0.pt", "landk_approach_s1.pt", "landk_approach_s2.pt",
                 "landk_land_s0.pt", "landr_scratch_s0.pt"):
        if stamps.get(name) is None:
            raise SystemExit(f"нет штампа {name}")
    source, encoder, ema = load_frozen()
    predictor = load_predictor(approach_file(0), "approach", 0, ENCODER_TAG, source)
    load_predictor(baseline_path(), "land", 0, ENCODER_TAG, source)
    if take_stamps() != stamps:
        raise SystemExit("чтение чекпоинта сдвинуло штамп")
    if stray_pt():
        raise SystemExit("появился landk_approach_tube_*.pt")
    if os.path.basename(approach_file(0)) != "landk_approach_s0.pt":
        raise SystemExit("читается не s0")

    z_touch = tube_latents(ema, frames, 256.0)
    if not (torch.equal(z_touch[0], z_touch[1]) and torch.equal(z_touch[1], z_touch[2])):
        raise SystemExit("три латента касания не совпали")
    actions, step_pairs = pair_table()
    clip_row = {
        "x": float(state[0]), "y": float(state[1]), "vx": float(state[2]),
        "vy": float(state[3]), "fuel": float(state[4]),
        "g": float(g_clip), "pad_x": 256.0,
    }
    world_clip = world_batch(g_clip, 256.0)
    _nxt, _action, was_clipped, n_calls, _frames = closed_step(
        predictor, encoder, ema, actions, step_pairs, world_clip, clip_row)
    if not was_clipped or n_calls != 1:
        raise SystemExit("шаг контура не увидел обрезку трубы")

    sample = dict(scored[0])
    world_b = world_batch(sample["g"], sample["pad_x"])
    import numeric_land.skill_tube as tube
    saved_hover = tube.hover_action
    echoed = []
    try:
        for forced in (7, 3):
            def _forced(state, g, pad_x, v_target, mode=None, forced=forced):
                del state, g, pad_x, v_target, mode
                return forced

            tube.hover_action = _forced
            current = dict(sample)
            before = (
                float(current["x"]), float(current["y"]), float(current["vx"]),
                float(current["vy"]), float(current["fuel"]),
            )
            nxt, action, _clipped, calls, frames = closed_step(
                predictor, encoder, ema, actions, step_pairs, world_b, current)
            after = (
                float(current["x"]), float(current["y"]), float(current["vx"]),
                float(current["vy"]), float(current["fuel"]),
            )
            if after != before:
                raise SystemExit("труба записалась в эпизод")
            if calls != 3:
                raise SystemExit(f"на высоком старте труба вызвала правило {calls} раз")
            z, z_tube, _frames, _calls, _was = prepare(encoder, ema, current)
            expected, _cost = choose_action(
                predictor, z, z_tube, world_b, step_pairs, actions)
            if int(action) != int(expected):
                raise SystemExit("в эпизод ушло не действие argmin")
            manual = step_field(before[0], before[1], before[2], before[3], before[4],
                                False, int(action), float(current["g"]))
            if nxt != manual:
                raise SystemExit("следующее состояние не step_field выбранного действия")
            tube_state = (float(frames[0][0]), float(frames[0][1]), float(frames[0][2]),
                          float(frames[0][3]), float(frames[0][4]))
            manual_state = (float(manual[0]), float(manual[1]), float(manual[2]),
                            float(manual[3]), float(manual[4]))
            if int(action) != int(forced) and tube_state == manual_state:
                raise SystemExit("состояние трубы подставлено в эпизод")
            echoed.append((forced, int(action)))
    finally:
        tube.hover_action = saved_hover
    if not echoed:
        raise SystemExit("шаг контура не проверен")

    if open(hover_path, "rb").read() != hover_bytes:
        raise SystemExit("правило изменилось")
    if stray_pt():
        raise SystemExit("после контроля появился чекпоинт трубы")
    print(
        f"ok сетка {GRID_N} зачёт {SCORED_N} исключено {EXCLUDED_N} "
        f"сиды {seeds[0]} {seeds[1]} {seeds[2]} линия 64 нет 65 да",
        flush=True,
    )


if __name__ == "__main__":
    main()

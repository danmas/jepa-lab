"""Контроль трубы до прогона. python _check_numeric_land_skill_tube.py"""
import inspect
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from numeric_land.expert_hover import HOVER_CAP, NAV_TOL, hover_action
from numeric_land.field import FUEL0, GROUND, VY_SOFT, step_field
from numeric_land.models import LOSS_WEIGHTS
from numeric_land.pad import outcome_at
from numeric_land.skill_land import (
    DXS, GRID_N, HEIGHTS, HORIZON, LINE, N_SEQ, VYS, grid, line_ok, pair_table,
    terminal_state,
)
from numeric_land.skill_tube import (
    SIGN_SPEC, SPREAD_MIN, SPREAD_SPEC, TAG, TUBE_TARGETS, choose_action, closed_step,
    cost_spread, load_frozen, load_predictor, pad_path, plan_costs, prepare, refuse_report,
    refuse_tag, roll_tube, row_of, scratch_path, side_mins, sign_ok, take_stamps,
    tube_files, tube_latents, world_batch,
)
from numeric_land.skill_tube import baseline_path
from numeric_land.train_goal import THREADS


def _stamp(path):
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _run(root, args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        args, cwd=root, timeout=90, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env)


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
                    raise SystemExit("касание не повторило последний латент-состояние")
                if float(frames[0][1]) < float(GROUND):
                    raise SystemExit("касание ушло ниже земли или не дошло")
                return state, g, frames
    raise SystemExit("не нашли касание правила на первом шаге ниже 32 px")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    hover_path = os.path.join(root, "numeric_land", "expert_hover.py")
    demo_paths = [os.path.join(root, name) for name in ("demo_ui.py", "_land_demo.py", "_pad_demo.py")]
    hover_bytes = open(hover_path, "rb").read()
    demo_before = {path: _stamp(path) for path in demo_paths}
    if None in demo_before.values() or not hover_bytes:
        raise SystemExit("нет правила или демо-страницы")
    ckpt_before = {
        scratch_path(): _stamp(scratch_path()),
        pad_path(0): _stamp(pad_path(0)),
        pad_path(1): _stamp(pad_path(1)),
        baseline_path(): _stamp(baseline_path()),
    }
    if any(value is None for value in ckpt_before.values()):
        raise SystemExit("нет чекпоинта трубы")

    if TAG != "numeric-land-skill-tube":
        raise SystemExit(TAG)
    if HORIZON != 3 or N_SEQ != 729 or TUBE_TARGETS != 3 or 9 ** 3 != N_SEQ:
        raise SystemExit("горизонт не 3 или последовательностей не 729")
    if LINE != 114 or GRID_N != 162:
        raise SystemExit("линия или сетка сдвинуты")
    if line_ok(113) or not line_ok(114):
        raise SystemExit("113 проходит или 114 не проходит")
    if LOSS_WEIGHTS != (1.0, 0.5, 0.25):
        raise SystemExit(LOSS_WEIGHTS)
    if abs(float(NAV_TOL) - 6.0) > 0.0 or float(VY_SOFT) != 8.0 or int(HOVER_CAP) != 900:
        raise SystemExit("допуск, порог или потолок сдвинуты")
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise SystemExit("бак, земля или потоки сдвинуты")
    if abs(float(SPREAD_MIN) - 0.00140) > 0.0 or len(SPREAD_SPEC) != 6:
        raise SystemExit("порог разброса сдвинут")

    params = list(inspect.signature(step_field).parameters)
    if "pad_x" in params or "mode" in params:
        raise SystemExit(f"физика видит лишнее: {params}")
    if outcome_at(256.0, 512.0, 8.0, True, 256.0) != "удар":
        raise SystemExit("vy = 8 не удар")
    if outcome_at(256.0, 512.0, 7.9, True, 256.0) != "мягкая":
        raise SystemExit("vy < 8 не мягкая")

    rows = grid()
    if len(rows) != 162 or len({(r["g"], r["pad_x"], r["dx"], r["height"], r["vy"]) for r in rows}) != 162:
        raise SystemExit("сочетаний не 162")
    dxs = {r["dx"] for r in rows}
    if dxs != {-6.0, 0.0, 6.0} or DXS != (-6.0, 0.0, 6.0):
        raise SystemExit("|dx| сетки не ±6")
    if any(abs(abs(dx) - 6.1) < 1e-9 for dx in dxs):
        raise SystemExit("|dx| = 6.1 вошло в сетку")
    if any(r["g"] == 20.0 or r["pad_x"] == 406.0 for r in rows):
        raise SystemExit("в сетке есть тяжесть 20 или центр 406")
    if HEIGHTS != (32.0, 64.0, 128.0) or VYS != (0.0, 6.0, 12.0):
        raise SystemExit("высота или vy сетки сдвинуты")
    if any(r["y"] >= 512.0 or r["fuel"] != 10.0 or r["vx"] != 0.0 for r in rows):
        raise SystemExit("старт на земле или не с нулевой vx")

    tube_src = inspect.getsource(sys.modules["numeric_land.skill_tube"])
    for banned in (
        "terminal_state", "target_state", "goal_z", "cem_plan", "landd_",
        "landk_approach", "start_and_actions", "torch.save", "AdamW", "backward",
    ):
        if banned in tube_src:
            raise SystemExit(f"модуль трубы содержит {banned}")
    if "hover_action" not in tube_src:
        raise SystemExit("труба не вызывает правило")
    land_src = inspect.getsource(sys.modules["numeric_land.skill_land"])
    if "hover_action" in land_src:
        raise SystemExit("старый модуль посадки стал вызывать правило")
    body = inspect.getsource(sys.modules["numeric_land.skill_tube"].main)
    if not (body.find("refuse_tag") < body.find("refuse_report") < body.find("fly_grid") < body.find("write_json")):
        raise SystemExit("отказ или запись отчёта стоят не на месте")
    if body.count("fly_grid(") != 2:
        raise SystemExit("прогон зовёт не два предиктора")
    if "landk_pad_s1" in body.split("fly_grid", 1)[-1] and "landk_pad_s1" in body[body.find("fly_grid"):]:
        if 'fly_grid("landk_pad_s1"' in body or "fly_grid('landk_pad_s1'" in body:
            raise SystemExit("seed 1 летит")
    for flag in ('"hover_action_action": False', '"tube_targets": TUBE_TARGETS',
                 '"cem": False', '"seed1_flown": False', '"decision_loaded": False'):
        if flag not in body:
            raise SystemExit(f"в отчёте нет {flag}")

    actions, steps = pair_table()
    if int(actions[0, 0]) != 0 or int(actions[81, 0]) != 1 or int(actions[728, 0]) != 8:
        raise SystemExit("индекс последовательности")
    if tuple(int(actions[i, 0].item()) for i in (0, 3 * 81, 6 * 81)) != (0, 3, 6):
        raise SystemExit("влево — не индексы 0, 3, 6")
    if tuple(int(actions[i, 0].item()) for i in (2 * 81, 5 * 81, 8 * 81)) != (2, 5, 8):
        raise SystemExit("вправо — не индексы 2, 5, 8")

    state, g_clip, frames = _clip_state()
    if float(GROUND) - float(state[1]) >= 32.0:
        raise SystemExit("контрольное касание не ниже 32 px")
    direct = int(hover_action(state, g_clip, 256.0, 6.0))
    seen = {"n": 0, "action": None}

    def _wrapped(s, g, pad_x, v_target, mode=None):
        seen["n"] += 1
        seen["action"] = int(hover_action(s, g, pad_x, v_target, mode=mode))
        return seen["action"]

    import numeric_land.skill_tube as tube
    saved_hover = tube.hover_action
    tube.hover_action = _wrapped
    try:
        wrapped_frames, calls, clipped = roll_tube(state, g_clip, 256.0)
    finally:
        tube.hover_action = saved_hover
    if seen["n"] != 1 or calls != 1 or not clipped:
        raise SystemExit(f"обрезка вызвала правило не один раз: {seen['n']} {calls}")
    if wrapped_frames != frames:
        raise SystemExit("обрезка с подсчётом разошлась с трубой")
    if seen["action"] != direct:
        raise SystemExit("касание подменено не тем действием, которое вернуло правило")
    if seen["action"] == 1 and direct != 1:
        raise SystemExit("касание подменено свободным падением")
    tube_clipped = 0
    if clipped:
        tube_clipped += 1
    if tube_clipped != 1:
        raise SystemExit("tube_clipped не увеличился")

    torch.set_num_threads(THREADS)
    stamps = take_stamps()
    source, encoder, ema = load_frozen()
    predictor = load_predictor(pad_path(0), "pad", 0, "numeric-land-skill-land", source)
    load_predictor(pad_path(1), "pad", 1, "numeric-land-skill-land", source)
    load_predictor(baseline_path(), "land", 0, "numeric-land-decision", source)
    if take_stamps() != stamps:
        raise SystemExit("чтение чекпоинта сдвинуло штамп")
    if tube_files():
        raise SystemExit("появился landk_tube_*.pt")

    z_touch = tube_latents(ema, frames, 256.0)
    if not (torch.equal(z_touch[0], z_touch[1]) and torch.equal(z_touch[1], z_touch[2])):
        raise SystemExit("три латента касания не совпали")

    def _costs(spec):
        row = row_of(*spec)
        z, z_tube, tube_frames, n_calls, was_clipped = prepare(encoder, ema, row)
        world = world_batch(row["g"], row["pad_x"])
        cost = plan_costs(predictor, z, z_tube, world, steps)
        return row, z, z_tube, tube_frames, n_calls, was_clipped, cost

    for spec in SIGN_SPEC:
        row, _z, z_tube, tube_frames, n_calls, was_clipped, cost = _costs(spec)
        if n_calls != 3 or was_clipped or z_tube.shape != (3, 32):
            raise SystemExit(f"труба знака не из трёх вызовов: {spec} {n_calls}")
        if tube_frames[0] == tube_frames[1] == tube_frames[2]:
            raise SystemExit("на высоте 32 труба схлопнулась в одну точку")
        if not sign_ok(cost, actions, row["dx"]):
            left, right = side_mins(cost, actions)
            raise SystemExit(f"знак dx={row['dx']} неверен: влево {left:.6f} вправо {right:.6f}")
        action, again = choose_action(predictor, _z, z_tube, world_batch(row["g"], row["pad_x"]), steps, actions)
        if int(torch.argmin(again).item()) != int(torch.argmin(cost).item()):
            raise SystemExit("повторный argmin разошёлся")
        if action != int(actions[int(torch.argmin(cost).item()), 0].item()):
            raise SystemExit("действие взято не из argmin")
        ground = terminal_state(row["pad_x"], row["fuel"])
        from numeric_land.relenc import scale_rel
        scaled = scale_rel(ground, np.asarray([row["pad_x"]], np.float64))
        z_ground = ema(torch.from_numpy(np.ascontiguousarray(scaled))).expand(3, -1)
        cost_ground = plan_costs(
            predictor, _z, z_ground.contiguous(), world_batch(row["g"], row["pad_x"]), steps)
        if torch.allclose(cost, cost_ground):
            raise SystemExit("стоимость трубы совпала с терминалом на земле")

    sign_grid = []
    for g in (2.0, 8.0, 15.0):
        for height in (32.0, 64.0, 128.0):
            for dx in (6.0, -6.0):
                sign_grid.append((g, 256.0, dx, height, 6.0))
    if len(sign_grid) != 18:
        raise SystemExit(len(sign_grid))
    for spec in sign_grid:
        row, _z, _zt, _fr, n_calls, was_clipped, cost = _costs(spec)
        if n_calls != 3 or was_clipped:
            raise SystemExit(f"контрольный знак обрезал трубу: {spec}")
        if not sign_ok(cost, actions, row["dx"]):
            left, right = side_mins(cost, actions)
            raise SystemExit(
                f"знак g={row['g']:g} h={row['height']:g} dx={row['dx']:g} "
                f"влево {left:.6f} вправо {right:.6f}")

    spreads = []
    for spec in SPREAD_SPEC:
        _row, _z, _zt, _fr, n_calls, was_clipped, cost = _costs(spec)
        if n_calls != 3 or was_clipped:
            raise SystemExit(f"разброс обрезал трубу: {spec}")
        gap = cost_spread(cost)
        spreads.append((spec, gap))
        print(
            f"разброс g={spec[0]:g} h={spec[3]:g} dx={spec[2]:g} {gap:.5f}",
            flush=True,
        )
    short = [item for item in spreads if round(item[1], 5) < round(float(SPREAD_MIN), 5)]
    if short:
        text = " ".join(f"g={spec[0]:g}/h={spec[3]:g}:{gap:.5f}" for spec, gap in short)
        raise SystemExit(f"разброс ниже {SPREAD_MIN:.5f}: {text}")

    probe_row = row_of(15.0, 256.0, 6.0, 32.0, 6.0)
    saved_choose = tube.choose_action
    forced = {"action": 8}

    def _force(*args, **kwargs):
        _action, cost = saved_choose(*args, **kwargs)
        return int(forced["action"]), cost

    tube.choose_action = _force
    try:
        nxt, action, _clipped, n_calls, frames_step = closed_step(
            predictor, encoder, ema, actions, steps,
            world_batch(probe_row["g"], probe_row["pad_x"]), probe_row)
    finally:
        tube.choose_action = saved_choose
    expect = step_field(
        probe_row["x"], probe_row["y"], probe_row["vx"], probe_row["vy"], probe_row["fuel"],
        False, 8, probe_row["g"])
    if action != 8 or nxt != expect:
        raise SystemExit("физика шагнула не выбранным действием")
    if n_calls != 3:
        raise SystemExit("шаг контура построил трубу не тремя вызовами")
    if (probe_row["x"], probe_row["y"]) != (262.0, 480.0):
        raise SystemExit("труба переписала старт эпизода")
    if frames_step[0][0] == nxt[0] and frames_step[0][1] == nxt[1]:
        raise SystemExit("состояние трубы совпало с шагом эпизода")

    handle = tempfile.NamedTemporaryFile(prefix="tube-report-", suffix=".json", delete=False)
    handle.write(b'{"tag":"keep"}')
    handle.close()
    try:
        before_report = open(handle.name, "rb").read()
        try:
            refuse_report(handle.name)
        except SystemExit:
            pass
        else:
            raise SystemExit("второй прогон не отказал")
        if open(handle.name, "rb").read() != before_report:
            raise SystemExit("второй прогон переписал отчёт")
    finally:
        os.remove(handle.name)

    proc = _run(root, [sys.executable, "-m", "numeric_land.skill_tube", "--tag", "numeric-land-decision"])
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 or "numeric-land-skill-tube" not in text:
        raise SystemExit(f"чужой тег не отсечён\n{text}")
    if "полёт" in text:
        raise SystemExit(f"чужой тег дошёл до полёта\n{text}")

    try:
        refuse_tag("numeric-land-skill-tube")
    except SystemExit:
        raise SystemExit("свой тег отклонён")

    if open(hover_path, "rb").read() != hover_bytes:
        raise SystemExit("проверка изменила expert_hover.py")
    for path, stamp in demo_before.items():
        if _stamp(path) != stamp:
            raise SystemExit(f"проверка изменила {os.path.basename(path)}")
    for path, stamp in ckpt_before.items():
        if _stamp(path) != stamp:
            raise SystemExit(f"проверка изменила {os.path.basename(path)}")
    if take_stamps() != stamps:
        raise SystemExit("штамп после контроля разошёлся")
    print(
        "ok знак 18/18 разброс "
        + " ".join(f"{gap:.5f}" for _spec, gap in spreads),
        flush=True,
    )


if __name__ == "__main__":
    main()

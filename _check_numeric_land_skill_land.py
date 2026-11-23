"""Контроль навыка посадки над площадкой. python _check_numeric_land_skill_land.py"""
import inspect
import os
import subprocess
import sys

import numpy as np
import torch

from numeric_land.expert_hover import NAV_TOL, hover_mode
from numeric_land.field import FUEL0, GROUND, VY_SOFT, step_field
from numeric_land.pad import outcome_at
from numeric_land.relenc import accepted, bucket_of
from numeric_land.skill_land import (
    DXS, EXPERT_FRAC, GOAL_FUEL_TRAIN, GOAL_HEIGHT0, GOAL_VY, GRID_N, HEIGHTS, HORIZON,
    LINE, N_SEQ, TAG, VYS, first_action, grid, line_ok, pair_table, terminal_state,
    train_height_and_vy, world_batch,
)
from numeric_land import train_skill_land as TR
from numeric_land.train_goal import BATCH, COLLAPSE_WEIGHT, LOSS_WEIGHTS, W_TERM


def _stamp(path):
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _run(root, args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        args, cwd=root, timeout=60, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    hover_path = os.path.join(root, "numeric_land", "expert_hover.py")
    pad_path = os.path.join(root, "out", "landk_pad_s0.pt")
    hover_before = _stamp(hover_path)
    pad_before = _stamp(pad_path)
    if hover_before is None:
        raise SystemExit("нет expert_hover.py")

    if TAG != "numeric-land-skill-land" or TR.TAG != TAG:
        raise SystemExit("тег навыка не numeric-land-skill-land")
    if TAG == "numeric-land-decision" or TR.ENCODER_TAG == TAG:
        raise SystemExit("тег совпал с numeric-land-decision")
    if HORIZON != 3 or N_SEQ != 729 or 9 ** 3 != N_SEQ:
        raise SystemExit("горизонт не 3 или последовательностей не 729")
    if LINE != 114 or GRID_N != 162:
        raise SystemExit("линия или сетка сдвинуты")
    if not line_ok(114) or line_ok(113):
        raise SystemExit("113 или 114 разошлись с линией")
    if GOAL_VY != 6.0 or GOAL_HEIGHT0 != 0.0 or GOAL_FUEL_TRAIN != 10.0:
        raise SystemExit("терминал сдвинут")
    if EXPERT_FRAC != 1.0 or int(round(BATCH * EXPERT_FRAC)) != int(BATCH):
        raise SystemExit("доля эксперта не 1")
    if W_TERM != 0.25 or COLLAPSE_WEIGHT != 0.1 or LOSS_WEIGHTS != (1.0, 0.5, 0.25):
        raise SystemExit("лосс сдвинут")
    if abs(float(NAV_TOL) - 6.0) > 0.0 or float(VY_SOFT) != 8.0:
        raise SystemExit("допуск или порог скорости сдвинуты")
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0:
        raise SystemExit("бак или земля сдвинуты")

    params = list(inspect.signature(step_field).parameters)
    if "pad_x" in params or "mode" in params or "pad" in params:
        raise SystemExit(f"физика видит лишнее: {params}")
    try:
        step_field(256, 480, 0, 0, 10, False, 1, 8, pad_x=200, mode="land")
    except TypeError:
        pass
    else:
        raise SystemExit("шаг принял режим или pad_x")
    first = step_field(256, 480, 0, 0, 10, False, 1, 8)
    second = step_field(256, 480, 0, 0, 10, False, 1, 8)
    if first != second:
        raise SystemExit("шаг из (256, 480) не повторился")

    if outcome_at(256, 512, 8, True, 256) != "удар":
        raise SystemExit("vy = 8 внутри площадки не удар")
    if outcome_at(248, 512, 8, True, 256) != "удар":
        raise SystemExit("vy = 8 на краю площадки не удар")
    if outcome_at(256, 512, 7.9, True, 256) != "мягкая":
        raise SystemExit("vy < 8 внутри площадки не мягкая")

    rows = grid()
    if len(rows) != 162 or len({(r["g"], r["pad_x"], r["dx"], r["height"], r["vy"]) for r in rows}) != 162:
        raise SystemExit("сочетаний не 162")
    dxs = {r["dx"] for r in rows}
    if 6.0 not in dxs or -6.0 not in dxs or DXS != (-6.0, 0.0, 6.0):
        raise SystemExit("|dx| = 6 не входит в сетку")
    if any(abs(abs(dx) - 6.1) < 1e-9 for dx in dxs):
        raise SystemExit("|dx| = 6.1 вошло в сетку")
    if any(r["g"] == 20.0 or r["pad_x"] == 406.0 for r in rows):
        raise SystemExit("в сетке есть тяжесть 20 или центр 406")
    if 128.0 not in {r["height"] for r in rows} or HEIGHTS != (32.0, 64.0, 128.0):
        raise SystemExit("высота 128 выпала")
    if VYS != (0.0, 6.0, 12.0):
        raise SystemExit("vy сетки сдвинуты")
    example = None
    for row in rows:
        if (row["g"], row["pad_x"], row["dx"], row["height"], row["vy"]) == (
                2.0, 200.0, -6.0, 128.0, 12.0):
            example = row
    if example is None:
        raise SystemExit("пример старта выпал")
    if example["x"] != 194.0 or example["y"] != 384.0 or example["vx"] != 0.0:
        raise SystemExit(example)
    if example["vy"] != 12.0 or example["fuel"] != 10.0:
        raise SystemExit(example)
    if any(r["y"] >= 512.0 or r["vx"] != 0.0 or r["fuel"] != 10.0 for r in rows):
        raise SystemExit("старт на земле или не с нулевой vx")
    if line_ok(113) or not line_ok(114):
        raise SystemExit("линия после сетки разъехалась")
    if 128.0 not in {r["height"] for r in rows}:
        raise SystemExit("после 113 высота 128 удалена")

    term = terminal_state(200.0, 10.0)
    if float(term[0, 0]) != 200.0 or float(term[0, 1]) != 512.0 or float(term[0, 2]) != 0.0:
        raise SystemExit(term)
    if float(term[0, 3]) != 6.0 or float(term[0, 4]) != 10.0:
        raise SystemExit(term)
    height0, vys = train_height_and_vy(4)
    if not np.all(height0 == 0.0) or not np.all(vys == 6.0):
        raise SystemExit("терминал обучения смотрит на окно")
    loop = terminal_state(200.0, 7.5)
    if float(loop[0, 1]) != 512.0 or float(loop[0, 3]) != 6.0 or float(loop[0, 4]) != 7.5:
        raise SystemExit("бак контура не текущий")

    actions, steps = pair_table()
    if int(actions[0, 0]) != 0 or int(actions[81, 0]) != 1 or int(actions[728, 0]) != 8:
        raise SystemExit("индекс последовательности")
    if len(steps) != 3:
        raise SystemExit("в поиске не три шага")

    class _Flat:
        def __call__(self, z, pair, world=None):
            return z

    z = torch.zeros(1, 32)
    z_goal = torch.ones(1, 32)
    action = first_action(_Flat(), z, z_goal, world_batch(2.0, 256.0), steps, actions)
    if action != 0:
        raise SystemExit("при равной стоимости победил не меньший индекс")

    land = (250.0, 400.0, 0.0, 0.0, 10.0)
    edge = (250.0, 400.0, 0.0, 0.0, 10.0)
    outside = (249.9, 400.0, 0.0, 0.0, 10.0)
    if hover_mode(land, 8.0, 256.0, 6.0) != "land":
        raise SystemExit("|dx| = 6 не посадка")
    if hover_mode(edge, 8.0, 256.0, 6.0) != "land":
        raise SystemExit("край допуска отброшен фазой")
    if hover_mode(outside, 8.0, 256.0, 6.0) != "approach":
        raise SystemExit("|dx| = 6.1 не наведение")
    if bucket_of(("land", "land", "land", "land")) != "land" or not accepted("land", "land"):
        raise SystemExit("чистое land не входит в пул")
    if accepted("approach", "land") or accepted("mixed", "land") or accepted(None, "land"):
        raise SystemExit("стык, наведение или пустой режим вошли в пул посадки")
    if bucket_of(("approach", "land", "land", "land")) != "mixed":
        raise SystemExit("стык фаз не смешанный")

    src = inspect.getsource(sys.modules["numeric_land.skill_land"])
    for banned in ("hover_action", "cem_plan", "landd_", "landk_approach", "start_and_actions"):
        if banned in src:
            raise SystemExit(f"модуль полёта содержит {banned}")
    body = inspect.getsource(TR.main)
    i_refuse = body.find("refuse_tag(")
    i_threads = body.find("set_num_threads")
    i_scan = body.find("scan_range")
    if i_refuse < 0 or i_threads < 0 or i_scan < 0:
        raise SystemExit("в main нет отказа, потоков или пула")
    if not (i_refuse < i_threads and i_refuse < i_scan):
        raise SystemExit("отказ тега стоит после пула или потоков")

    proc = _run(root, [
        sys.executable, "-m", "numeric_land.train_skill_land",
        "--tag", "numeric-land-decision",
    ])
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        raise SystemExit(f"чужой тег принят\n{text}")
    if "numeric-land-skill-land" not in text:
        raise SystemExit(f"отказ не назвал нужный тег\n{text}")
    if "сбор" in text or "ms/step" in text or "AdamW" in text:
        raise SystemExit(f"чужой тег дошёл до пула или оптимизатора\n{text}")
    if _stamp(pad_path) != pad_before:
        raise SystemExit("чужой тег задел landk_pad_s0.pt")
    if _stamp(hover_path) != hover_before:
        raise SystemExit("чужой тег задел expert_hover.py")
    if _stamp(hover_path) != hover_before:
        raise SystemExit("проверка задела expert_hover.py")
    print("ok", flush=True)


if __name__ == "__main__":
    main()

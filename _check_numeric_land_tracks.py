"""Контроль модуля треков. python _check_numeric_land_tracks.py"""
import json
import os
import subprocess
import sys

import numpy as np

from numeric_land.field import GROUND, TRANSITIONS
from numeric_land.tracks import (
    KINDS, build, expert_flight, random_flight, random_window,
)


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # 1. окно обучения — ровно TRANSITIONS+1 состояний, старт у самой земли
    for seed in (100000, 100001, 100002, 200000):
        w = random_window(seed)
        if w["n_states"] != TRANSITIONS + 1 or w["n_actions"] != TRANSITIONS:
            raise SystemExit(f"окно не то: {w['n_states']}, {w['n_actions']}")
        h0 = w["height_from"]
        if not (0.0 < h0 < GROUND):
            raise SystemExit(f"окно не над землёй: высота {h0}")
        # окно короткое: покрывает доли единицы высоты
        if abs(w["height_covered"]) > 5.0:
            raise SystemExit(f"окно покрыло слишком много: {w['height_covered']}")
        if w["touched"]:
            raise SystemExit("окно не должно касаться земли (это 3 шага от старта)")
        if w["duration_s"] > TRANSITIONS / 30.0 + 1e-9:
            raise SystemExit(w["duration_s"])

    # 2. полёт-источник: те же общие состояния, что у окна (префикс)
    w = random_window(100000)
    f = random_flight(100000)
    if f["n_actions"] <= w["n_actions"]:
        raise SystemExit("полёт не длиннее окна")
    for i in range(w["n_states"]):
        if abs(w["points"][i]["x"] - f["points"][i]["x"]) > 1e-6:
            raise SystemExit(f"окно не префикс полёта, точка {i}")
        if abs(w["points"][i]["y"] - f["points"][i]["y"]) > 1e-6:
            raise SystemExit(f"окно не префикс полёта по y, точка {i}")

    # 3. эксперт: касается земли и покрывает всю высоту
    soft = 0
    for seed in range(200000, 200000 + 12):
        e = expert_flight(seed)
        if not e["touched"]:
            raise SystemExit(f"эксперт не коснулся, сид {seed}")
        if e["height_to"] > 1e-6:
            raise SystemExit(f"эксперт не дошёл до земли: {e['height_to']}")
        if e["height_covered"] < 20.0:
            raise SystemExit("эксперт покрыл мало высоты")
        if e["result"] == "мягкая":
            soft += 1
    if soft < 3:
        raise SystemExit(f"эксперт почти не садится мягко: {soft}/12")

    # 4. build отдаёт контекст только для окна; всё сериализуется
    b = build("window", 100000)
    if b["track"]["kind"] != "window" or b["context"] is None:
        raise SystemExit("build(window) без контекста")
    if build("flight", 100000)["context"] is not None:
        raise SystemExit("build(flight) с лишним контекстом")
    if build("expert", 100000)["context"] is not None:
        raise SystemExit("build(expert) с лишним контекстом")
    for kind in KINDS:
        json.dumps(build(kind, 100000), ensure_ascii=False)
    try:
        build("nonsense", 1)
    except ValueError:
        pass
    else:
        raise SystemExit("неизвестный вид принят")

    # 5. прежние контроли живы
    for chk in ("_check_numeric_land_field.py", "_check_numeric_land_pad.py",
                "_check_numeric_land_goal.py", "_check_numeric_land_expert.py"):
        r = subprocess.run([sys.executable, os.path.join(root, chk)], cwd=root)
        if r.returncode != 0:
            raise SystemExit(f"{chk} не прошёл")
    print("numeric land tracks: ок")


if __name__ == "__main__":
    main()

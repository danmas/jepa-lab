"""Навык наведения до зоны передачи. Обучения нет.

Сетка — поле: сиды 200000…200029, тяжести 2/8/15, площадка из окна.
Девять стартов с |x − pad_x| ≤ 6 в зачёт не входят. Успех — зона три шага подряд.
Стоимость — та же труба, что у посадки. Посадка здесь не измеряется.

    python -m numeric_land.skill_approach --tag numeric-land-skill-approach
"""
import argparse
import glob
import os

import numpy as np
import torch

from numeric_land.expert_hover import HOVER_CAP, NAV_TOL, hover_action
from numeric_land.field import FUEL0, GROUND, start_and_actions, step_field
from numeric_land.models import LOSS_WEIGHTS
from numeric_land.pad import pad_of_window
from numeric_land.relenc import OUT, ROOT, file_stamp, write_json
from numeric_land.relenc import TAG as ENCODER_TAG
from numeric_land.skill_land import HORIZON, N_SEQ, pair_table, world_batch
from numeric_land.skill_tube import (
    GOAL_VY, SOURCE_NAME, TUBE_TARGETS, baseline_path, closed_step, load_frozen,
    load_predictor, scratch_path,
)
from numeric_land.train_goal import THREADS
from numeric_land.train_skills import ckpt_path

TAG = "numeric-land-skill-approach"
LINE = 65
GRID_N = 90
SCORED_N = 81
EXCLUDED_N = 9
HOLD = 3
SEED0 = 200000
N_SEEDS = 30
GS = (2.0, 8.0, 15.0)
VY_LO = -4.0
VY_HI = 12.0
HEIGHT_MIN = 24.0
STAGE1_HEIGHT = 128.0
STAGE1_VYS = (0.0, 6.0, 12.0)
JSON_NAME = "numeric_land_skill_approach.json"
APPROACH_NAME = "landk_approach_s0.pt"
LAND_NAME = "landk_land_s0.pt"
DEMO_NAMES = ("demo_ui.py", "_land_demo.py", "_pad_demo.py")


def locks() -> None:
    if TAG != "numeric-land-skill-approach" or ENCODER_TAG == TAG:
        raise RuntimeError(TAG)
    if HORIZON != 3 or N_SEQ != 729 or TUBE_TARGETS != 3 or GOAL_VY != 6.0:
        raise RuntimeError((HORIZON, N_SEQ, TUBE_TARGETS, GOAL_VY))
    if LOSS_WEIGHTS != (1.0, 0.5, 0.25) or len(LOSS_WEIGHTS) != TUBE_TARGETS:
        raise RuntimeError(LOSS_WEIGHTS)
    if LINE != 65 or GRID_N != 90 or SCORED_N != 81 or EXCLUDED_N != 9:
        raise RuntimeError((LINE, GRID_N, SCORED_N, EXCLUDED_N))
    if HOLD != 3 or SEED0 != 200000 or N_SEEDS != 30 or GS != (2.0, 8.0, 15.0):
        raise RuntimeError((HOLD, SEED0, N_SEEDS, GS))
    if VY_LO != -4.0 or VY_HI != 12.0 or HEIGHT_MIN != 24.0:
        raise RuntimeError((VY_LO, VY_HI, HEIGHT_MIN))
    if abs(float(NAV_TOL) - 6.0) > 0.0 or int(HOVER_CAP) != 900:
        raise RuntimeError((NAV_TOL, HOVER_CAP))
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise RuntimeError((FUEL0, GROUND, THREADS))
    if STAGE1_VYS != (0.0, 6.0, 12.0) or STAGE1_HEIGHT != 128.0:
        raise RuntimeError((STAGE1_VYS, STAGE1_HEIGHT))
    if GRID_N != N_SEEDS * len(GS) or SCORED_N + EXCLUDED_N != GRID_N:
        raise RuntimeError((GRID_N, SCORED_N, EXCLUDED_N))


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "нужен тег numeric-land-skill-approach. Другие теги запрещены.")


def json_path() -> str:
    return os.path.join(OUT, JSON_NAME)


def refuse_report(path: str) -> None:
    """Второй прогон готовый отчёт не открывает на запись."""
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}, второй прогон отчёт не затирает")


def approach_file(seed: int) -> str:
    seed = int(seed)
    if seed not in (0, 1, 2):
        raise ValueError(seed)
    path = ckpt_path("approach", seed)
    expect = f"landk_approach_s{seed}.pt"
    if os.path.basename(path) != expect:
        raise RuntimeError(path)
    return path


def line_ok(success: int) -> bool:
    """65 из 81 проходит, 64 нет. Сравнение целых, без округления доли."""
    return int(success) >= int(LINE)


def in_transfer_zone(x, y, vy, fuel, grounded, pad_x) -> bool:
    """Все условия сразу. |dx| = 6 входит, 6.1 нет. vy = 12 входит, 12.5 нет."""
    if bool(grounded) or float(y) >= float(GROUND):
        return False
    if float(fuel) <= 0.0:
        return False
    if abs(float(x) - float(pad_x)) > float(NAV_TOL):
        return False
    if float(vy) < float(VY_LO) or float(vy) > float(VY_HI):
        return False
    height = float(GROUND) - float(y)
    if height < float(HEIGHT_MIN):
        return False
    return True


def excluded_start(row) -> bool:
    """Старт уже над площадкой: домен этапа 1, в линию 65 не входит."""
    return abs(float(row["x"]) - float(row["pad_x"])) <= float(NAV_TOL)


def after_step(hold, x, y, vy, fuel, grounded, pad_x):
    """Шаг после физики. Касание — удар, даже если до этого в зоне было два шага.

    Пустой бак без касания — промах. Два шага в зоне успехом не считаются.
    """
    if bool(grounded) or float(y) >= float(GROUND):
        return 0, "удар", False
    if in_transfer_zone(x, y, vy, fuel, grounded, pad_x):
        hold = int(hold) + 1
        if hold >= int(HOLD):
            return hold, "передача", False
        return hold, None, False
    if float(fuel) <= 0.0:
        return 0, "промах", True
    return 0, None, False


def field_rows():
    """90 эпизодов поля. Площадка из окна, старт start_and_actions(seed, 0)."""
    locks()
    rows = []
    seen = set()
    for seed in range(SEED0, SEED0 + N_SEEDS):
        pad_x = float(pad_of_window(seed))
        x, y, vx, vy, actions = start_and_actions(int(seed), 0)
        if len(actions) != 0:
            raise RuntimeError(len(actions))
        if not float(y) < float(GROUND):
            raise RuntimeError(y)
        for g in GS:
            key = (int(seed), float(g))
            if key in seen:
                raise RuntimeError(key)
            seen.add(key)
            rows.append({
                "seed": int(seed),
                "g": float(g),
                "pad_x": pad_x,
                "x": float(x),
                "y": float(y),
                "vx": float(vx),
                "vy": float(vy),
                "fuel": float(FUEL0),
                "dx": float(x) - pad_x,
            })
    if len(rows) != GRID_N or len(seen) != GRID_N:
        raise RuntimeError(len(rows))
    return tuple(rows)


def split_grid(rows):
    """Один и тот же отбор для навыка, базы и правила."""
    scored = []
    excluded = []
    for row in rows:
        if excluded_start(row):
            excluded.append(row)
        else:
            scored.append(row)
    if len(rows) != GRID_N or len(excluded) != EXCLUDED_N or len(scored) != SCORED_N:
        raise RuntimeError((len(rows), len(excluded), len(scored)))
    per_g = {float(g): 0 for g in GS}
    for row in scored:
        per_g[float(row["g"])] += 1
    expect = SCORED_N // len(GS)
    if any(per_g[float(g)] != expect for g in GS):
        raise RuntimeError(per_g)
    return tuple(scored), tuple(excluded)


def _result(outcome, fuel_out, steps, y, vy, clipped_n):
    height = None
    vy_out = None
    if outcome == "передача":
        height = float(GROUND) - float(y)
        vy_out = float(vy)
        if fuel_out:
            raise RuntimeError("передача с пустым баком")
    return {
        "outcome": outcome,
        "fuel_out": bool(fuel_out),
        "steps": int(steps),
        "height": height,
        "vy": vy_out,
        "tube_clipped": int(clipped_n),
    }


@torch.no_grad()
def fly_skill(predictor, encoder, ema, actions, step_pairs, world_b, row):
    """Эпизод до передачи, удара, пустого бака или потолка. no_grad: encode зовёт numpy."""
    if excluded_start(row):
        raise RuntimeError("в зачёт попал старт над площадкой")
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    fuel = float(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    hold = 0
    steps = 0
    clipped_n = 0
    while steps < int(HOVER_CAP):
        current = {
            "x": x, "y": y, "vx": vx, "vy": vy, "fuel": fuel,
            "g": g, "pad_x": pad_x,
        }
        nxt, _action, clipped, calls, _frames = closed_step(
            predictor, encoder, ema, actions, step_pairs, world_b, current)
        if calls < 1:
            raise RuntimeError(calls)
        if clipped:
            clipped_n += 1
        x, y, vx, vy, fuel, grounded = nxt
        steps += 1
        hold, outcome, fuel_out = after_step(hold, x, y, vy, fuel, grounded, pad_x)
        if outcome is not None:
            return _result(outcome, fuel_out, steps, y, vy, clipped_n)
    return _result("промах", False, steps, y, vy, clipped_n)


@torch.no_grad()
def fly_rule(row):
    """Справочный потолок: само правило до той же зоны и того же удержания."""
    if excluded_start(row):
        raise RuntimeError("в зачёт попал старт над площадкой")
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    fuel = float(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    hold = 0
    steps = 0
    while steps < int(HOVER_CAP):
        action = int(hover_action((x, y, vx, vy, fuel), g, pad_x, float(GOAL_VY)))
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, False, action, g)
        steps += 1
        hold, outcome, fuel_out = after_step(hold, x, y, vy, fuel, grounded, pad_x)
        if outcome is not None:
            return _result(outcome, fuel_out, steps, y, vy, 0)
    return _result("промах", False, steps, y, vy, 0)


def _record(bucket, result) -> None:
    bucket["n"] += 1
    bucket["tube_clipped"] += int(result["tube_clipped"])
    outcome = result["outcome"]
    if outcome == "передача":
        bucket["success"] += 1
        bucket["steps"].append(int(result["steps"]))
        bucket["height"].append(float(result["height"]))
        bucket["vy"].append(float(result["vy"]))
    elif outcome == "удар":
        bucket["crash"] += 1
    elif outcome == "промах":
        bucket["miss"] += 1
        if result["fuel_out"]:
            bucket["fuel_out"] += 1
    else:
        raise RuntimeError(outcome)


def _empty():
    return {
        "success": 0, "crash": 0, "miss": 0, "fuel_out": 0, "n": 0,
        "steps": [], "height": [], "vy": [], "tube_clipped": 0,
    }


def _median(values):
    if not values:
        return None
    return float(round(float(np.median(np.asarray(values, np.float64))), 4))


def _share(count, total):
    if int(total) <= 0:
        return None
    return float(round(float(count) / float(total), 4))


def _vy_in_stage1(vy) -> bool:
    return any(abs(float(vy) - mark) <= 1e-4 for mark in STAGE1_VYS)


def _pack(name, bucket, per_g, on_line):
    success = int(bucket["success"])
    n_height = sum(1 for value in bucket["height"] if float(value) <= float(STAGE1_HEIGHT))
    n_vy = sum(1 for value in bucket["vy"] if _vy_in_stage1(value))
    if success + int(bucket["crash"]) + int(bucket["miss"]) != SCORED_N:
        raise RuntimeError((name, bucket["success"], bucket["crash"], bucket["miss"]))
    if int(bucket["fuel_out"]) > int(bucket["miss"]) or int(bucket["n"]) != SCORED_N:
        raise RuntimeError((name, bucket["fuel_out"], bucket["n"]))
    by_g = []
    for g in GS:
        cell = per_g[float(g)]
        if int(cell["success"]) + int(cell["crash"]) + int(cell["miss"]) != int(cell["n"]):
            raise RuntimeError((g, cell))
        by_g.append({
            "g": float(g),
            "n": int(cell["n"]),
            "success": int(cell["success"]),
            "crash": int(cell["crash"]),
            "miss": int(cell["miss"]),
            "fuel_out": int(cell["fuel_out"]),
            "steps_median": _median(cell["steps"]),
            "height_median": _median(cell["height"]),
            "vy_median": _median(cell["vy"]),
        })
    row = {
        "name": name,
        "scored": bool(on_line),
        "success": success,
        "crash": int(bucket["crash"]),
        "miss": int(bucket["miss"]),
        "fuel_out": int(bucket["fuel_out"]),
        "of": int(SCORED_N),
        "tube_clipped": int(bucket["tube_clipped"]),
        "steps_median": _median(bucket["steps"]),
        "height_median": _median(bucket["height"]),
        "vy_median": _median(bucket["vy"]),
        "height_le_128": int(n_height),
        "height_le_128_share": _share(n_height, success),
        "vy_in_0_6_12": int(n_vy),
        "vy_in_0_6_12_share": _share(n_vy, success),
        "by_g": by_g,
    }
    if on_line:
        row["line_ok"] = bool(line_ok(success))
    elif "line_ok" in row:
        raise RuntimeError(name)
    return row


def _fly_rows(name, rows, step_fn, on_line):
    total = _empty()
    per_g = {float(g): _empty() for g in GS}
    n = len(rows)
    if n != SCORED_N:
        raise RuntimeError(n)
    for i, row in enumerate(rows):
        result = step_fn(row)
        _record(total, result)
        _record(per_g[float(row["g"])], result)
        kind = {"передача": "handoff", "удар": "crash", "промах": "miss"}[result["outcome"]]
        print(
            f"APPROACH_EP name={name} i={i + 1} n={n} seed={int(row['seed'])} "
            f"g={float(row['g']):g} steps={int(result['steps'])} {kind} "
            f"success={total['success']} crash={total['crash']} miss={total['miss']} "
            f"fuel_out={total['fuel_out']}",
            flush=True,
        )
    return _pack(name, total, per_g, on_line)


def fly_scored(name, predictor, encoder, ema, rows):
    actions, step_pairs = pair_table()
    predictor.eval()
    encoder.eval()
    ema.eval()
    worlds = {}

    def one(row):
        key = (float(row["g"]), float(row["pad_x"]))
        if key not in worlds:
            worlds[key] = world_batch(row["g"], row["pad_x"])
        return fly_skill(
            predictor, encoder, ema, actions, step_pairs, worlds[key], row)

    return _fly_rows(name, rows, one, name == APPROACH_NAME.replace(".pt", ""))


def fly_rule_rows(rows):
    return _fly_rows("rule_reach", rows, fly_rule, False)


def excluded_payload(excluded):
    out = []
    for row in excluded:
        out.append({
            "seed": int(row["seed"]),
            "g": float(row["g"]),
            "dx": float(round(float(row["dx"]), 6)),
        })
    return out


def stray_pt():
    return glob.glob(os.path.join(OUT, "landk_approach_tube_*.pt"))


def watched_paths():
    paths = {
        APPROACH_NAME: approach_file(0),
        "landk_approach_s1.pt": approach_file(1),
        "landk_approach_s2.pt": approach_file(2),
        LAND_NAME: baseline_path(),
        SOURCE_NAME: scratch_path(),
        "expert_hover.py": os.path.join(ROOT, "numeric_land", "expert_hover.py"),
    }
    for name in DEMO_NAMES:
        paths[name] = os.path.join(ROOT, name)
    return paths


def take_stamps():
    stamps = {}
    for name, path in watched_paths().items():
        stamps[name] = file_stamp(path)
        if stamps[name] is None:
            raise SystemExit(f"нет {name}")
    return stamps


def _finish_row_names(approach_row, land_row, rule_row) -> None:
    if approach_row["name"] != "landk_approach_s0":
        raise RuntimeError(approach_row["name"])
    if land_row["name"] != "landk_land_s0" or rule_row["name"] != "rule_reach":
        raise RuntimeError((land_row["name"], rule_row["name"]))
    if "line_ok" not in approach_row or "line_ok" in land_row or "line_ok" in rule_row:
        raise RuntimeError("линия проставлена не тому столбцу")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    refuse_tag(args.tag)
    locks()
    torch.set_num_threads(THREADS)
    report = json_path()
    refuse_report(report)
    if stray_pt():
        raise SystemExit("уже есть landk_approach_tube_*.pt, чекпоинт трубы писать нельзя")
    rows = field_rows()
    scored, excluded = split_grid(rows)
    before = take_stamps()
    source, encoder, ema = load_frozen()
    approach = load_predictor(approach_file(0), "approach", 0, ENCODER_TAG, source)
    land = load_predictor(baseline_path(), "land", 0, ENCODER_TAG, source)
    if take_stamps() != before:
        raise RuntimeError("чтение сдвинуло штамп")
    print(f"полёт landk_approach_s0, труба, зачёт {SCORED_N}", flush=True)
    approach_row = fly_scored("landk_approach_s0", approach, encoder, ema, scored)
    print(f"полёт landk_land_s0, труба, зачёт {SCORED_N}", flush=True)
    land_row = fly_scored("landk_land_s0", land, encoder, ema, scored)
    print(f"полёт rule_reach, зачёт {SCORED_N}", flush=True)
    rule_row = fly_rule_rows(scored)
    if take_stamps() != before:
        raise RuntimeError("полёт сдвинул штамп")
    if stray_pt():
        raise RuntimeError("появился landk_approach_tube_*.pt")
    _finish_row_names(approach_row, land_row, rule_row)
    if bool(approach_row["line_ok"]) != bool(line_ok(approach_row["success"])):
        raise RuntimeError("линия разошлась")
    payload = {
        "tag": TAG,
        "grid_n": GRID_N,
        "scored_n": SCORED_N,
        "excluded": excluded_payload(excluded),
        "line": LINE,
        "horizon": HORIZON,
        "n_sequences": N_SEQ,
        "cem": False,
        "zone": {
            "dx_max": float(NAV_TOL),
            "vy_min": float(VY_LO),
            "vy_max": float(VY_HI),
            "height_min": float(HEIGHT_MIN),
            "fuel_gt": 0.0,
            "touch": False,
            "hold": HOLD,
        },
        "hover_action_action": False,
        "tube_targets": TUBE_TARGETS,
        "tube_clipped": int(approach_row["tube_clipped"] + land_row["tube_clipped"]),
        "stamps_unchanged": True,
        "encoder": SOURCE_NAME,
        "decision_loaded": False,
        "approach_s1_flown": False,
        "approach_s2_flown": False,
        "height_le_128_is_gate": False,
        "stage1_vy": [0.0, 6.0, 12.0],
        "stage1_height_max": float(STAGE1_HEIGHT),
        "passed": bool(approach_row["line_ok"]),
        "rows": [approach_row, land_row, rule_row],
    }
    if payload["passed"] != bool(approach_row["line_ok"]):
        raise RuntimeError("вердикт смотрит не на столбец наведения")
    write_json(report, payload)
    print(
        f"APPROACH_DONE success={approach_row['success']} of={SCORED_N} line={LINE} "
        f"passed={int(payload['passed'])} land={land_row['success']} "
        f"rule={rule_row['success']} tube_clipped={payload['tube_clipped']}",
        flush=True,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

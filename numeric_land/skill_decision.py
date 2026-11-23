"""Голова называет режим, действие берёт навык. Обучения нет.

Четыре столбца на сетке 180. Зачёт — skill_tube: голова landd_early_s0,
посадка landk_pad_s0, наведение landk_approach_s0, стоимость по трубе.
Контроль — rule_early и rule_latent, действие из ветки правила.
Абляция — skill_terminal: те же голова и навыки, терминал на земле.

    python -m numeric_land.skill_decision --tag numeric-land-skill-decision
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from numeric_land.decision import THRESHOLD, DecisionHead, features_of, n_in_of
from numeric_land.expert_hover import HOVER_CAP, NAV_TOL, hover_action, hover_mode
from numeric_land.field import FUEL0, GROUND, VY_SOFT, start_and_actions, step_field
from numeric_land.mode import LOOP_N, LOOP_SEED0, PASS_G, PASS_PADS, V_TARGET_LOOP
from numeric_land.models import LOSS_WEIGHTS
from numeric_land.pad import outcome_at
from numeric_land.relenc import OUT, ROOT, encode_rel, file_stamp, write_json
from numeric_land.relenc import TAG as ENCODER_TAG
from numeric_land.rollout_decision import _load_head
from numeric_land.rollout_decision_skills import EARLY_PATH, EARLY_TAG, _same_source
from numeric_land.rollout_decision_threshold import _decide
from numeric_land.skill_land import (
    GOAL_VY, HORIZON, N_SEQ, first_action, goal_z, pair_table, world_batch,
)
from numeric_land.skill_tube import (
    SOURCE_NAME, TAG as TUBE_TAG, TUBE_TARGETS, choose_action, load_frozen, load_predictor,
    pad_path, prepare,
)
from numeric_land.skill_tube import LAND_TAG
from numeric_land.train_decision import ckpt_path
from numeric_land.train_goal import THREADS

TAG = "numeric-land-skill-decision"
LINE = 100
RATIO_MIN = 0.85
GAIN_MIN = 18
GRID_N = 180
CONTROL_EARLY = 120
CONTROL_LATENT = 108
STAGE1_HEIGHT = 128.0
JSON_NAME = "numeric_land_skill_decision.json"
DEMO_NAMES = ("demo_ui.py", "_land_demo.py", "_pad_demo.py")
OPTIONAL_NAMES = (
    "landd_early_s1.pt",
    "landk_pad_s1.pt",
    "landk_approach_s1.pt",
    "landk_approach_s2.pt",
)
RESULT_CODE = {"мягкая": "soft", "удар": "crash", "не сел": "miss"}


def locks() -> None:
    if TAG != "numeric-land-skill-decision":
        raise RuntimeError(TAG)
    if TAG in (ENCODER_TAG, LAND_TAG, EARLY_TAG, TUBE_TAG, "numeric-land-skill-approach"):
        raise RuntimeError(TAG)
    if HORIZON != 3 or N_SEQ != 729 or TUBE_TARGETS != 3 or GOAL_VY != 6.0:
        raise RuntimeError((HORIZON, N_SEQ, TUBE_TARGETS, GOAL_VY))
    if LOSS_WEIGHTS != (1.0, 0.5, 0.25) or len(LOSS_WEIGHTS) != HORIZON:
        raise RuntimeError(LOSS_WEIGHTS)
    if LINE != 100 or abs(RATIO_MIN - 0.85) > 0.0 or GAIN_MIN != 18 or GRID_N != 180:
        raise RuntimeError((LINE, RATIO_MIN, GAIN_MIN, GRID_N))
    if CONTROL_EARLY != 120 or CONTROL_LATENT != 108:
        raise RuntimeError((CONTROL_EARLY, CONTROL_LATENT))
    if LOOP_SEED0 != 99000 or LOOP_N != 30:
        raise RuntimeError((LOOP_SEED0, LOOP_N))
    if PASS_G != (2.0, 8.0, 15.0) or PASS_PADS != (256.0, 200.0):
        raise RuntimeError((PASS_G, PASS_PADS))
    if LOOP_N * len(PASS_G) * len(PASS_PADS) != GRID_N:
        raise RuntimeError(GRID_N)
    if abs(float(THRESHOLD) - 0.5) > 0.0 or float(V_TARGET_LOOP) != 6.0:
        raise RuntimeError((THRESHOLD, V_TARGET_LOOP))
    if n_in_of("latent") != 33:
        raise RuntimeError(n_in_of("latent"))
    if abs(float(NAV_TOL) - 6.0) > 0.0 or float(VY_SOFT) != 8.0 or int(HOVER_CAP) != 900:
        raise RuntimeError((NAV_TOL, VY_SOFT, HOVER_CAP))
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0 or THREADS != 8:
        raise RuntimeError((FUEL0, GROUND, THREADS))
    if STAGE1_HEIGHT != 128.0 or os.path.basename(EARLY_PATH) != "landd_early_s0.pt":
        raise RuntimeError((STAGE1_HEIGHT, EARLY_PATH))
    if SOURCE_NAME != "landr_scratch_s0.pt":
        raise RuntimeError(SOURCE_NAME)
    if mode_of(0.5) != "land" or mode_of(0.499) != "approach":
        raise RuntimeError("порог головы сдвинут")
    if ratio_need(120) != 102:
        raise RuntimeError(ratio_need(120))
    if line_closed(99, 120) or line_closed(100, 120) or line_closed(101, 120):
        raise RuntimeError("линия пропускает 99, 100 или 101 при правиле 120")
    if not line_closed(102, 120) or not line_closed(103, 120):
        raise RuntimeError("линия не берёт 102 или 103 при правиле 120")
    if control_ok(119, 108) or control_ok(121, 108) or control_ok(120, 107) or control_ok(120, 109):
        raise RuntimeError("контроль принял не запись")
    if not control_ok(120, 108):
        raise RuntimeError("контроль не принял 120 и 108")
    if tube_gain_ok(17) or not tube_gain_ok(18):
        raise RuntimeError("разрыв 18 сдвинут")


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "нужен тег numeric-land-skill-decision. Другие теги запрещены.")


def json_path() -> str:
    return os.path.join(OUT, JSON_NAME)


def refuse_report(path: str) -> None:
    """Второй прогон готовый отчёт не открывает на запись."""
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}, второй прогон отчёт не затирает")


def stray_pt():
    return glob.glob(os.path.join(OUT, "landd_skill_*.pt"))


def mode_of(prob: float) -> str:
    """0.5 — land, 0.499 — approach. Порог из головы, не из этого прогона."""
    return "land" if float(prob) >= float(THRESHOLD) else "approach"


def ratio_need(rule_early: int) -> int:
    """ceil(0.85 · n) целыми: ceil(85·n / 100). Долю не округляем."""
    n = int(rule_early)
    if n < 0:
        raise RuntimeError(n)
    return (85 * n + 99) // 100


def line_closed(skill_soft: int, rule_early: int) -> bool:
    """100 и ceil(0.85 · rule_early). Оба условия, сравнение целых."""
    soft = int(skill_soft)
    if soft < int(LINE):
        return False
    return soft >= ratio_need(rule_early)


def control_ok(early_soft: int, latent_soft: int) -> bool:
    """Записи 120 и 108. Другое число — контур собран не так, зачёт не закрывается."""
    return int(early_soft) == CONTROL_EARLY and int(latent_soft) == CONTROL_LATENT


def tube_gain_ok(gain: int) -> bool:
    return int(gain) >= GAIN_MIN


def verdict(skill_soft: int, rule_early: int, rule_latent: int, skill_terminal: int) -> dict:
    gain = int(skill_soft) - int(skill_terminal)
    closed = line_closed(skill_soft, rule_early)
    ok = control_ok(rule_early, rule_latent)
    gain_ok = tube_gain_ok(gain)
    return {
        "line_ok": bool(closed),
        "ratio_need": int(ratio_need(rule_early)),
        "control_ok": bool(ok),
        "tube_gain": int(gain),
        "tube_gain_ok": bool(gain_ok),
        "pilot_pass": bool(ok and closed and gain_ok),
    }


def episodes():
    """Сид × тяжесть × центр, каждый один раз. Старт — start_and_actions, бак 10."""
    locks()
    rows = []
    seen = set()
    for g in PASS_G:
        for pad in PASS_PADS:
            for seed in range(LOOP_SEED0, LOOP_SEED0 + LOOP_N):
                key = (int(seed), float(g), float(pad))
                if key in seen:
                    raise RuntimeError(key)
                seen.add(key)
                x, y, vx, vy, _acts = start_and_actions(int(seed), 0)
                rows.append({
                    "seed": int(seed),
                    "g": float(g),
                    "pad_x": float(pad),
                    "x": float(x),
                    "y": float(y),
                    "vx": float(vx),
                    "vy": float(vy),
                    "fuel": float(FUEL0),
                })
    if len(rows) != GRID_N or len(seen) != GRID_N:
        raise RuntimeError((len(rows), len(seen)))
    return tuple(rows)


class SwitchLog:
    """Первый переход approach → land. Старт уже в land — отдельное ведро."""

    def __init__(self):
        self.prev = None
        self.named_land = False
        self.switch = None

    def observe(self, mode, step, height, abs_dx):
        if mode not in ("approach", "land"):
            raise RuntimeError(mode)
        if mode == "land":
            self.named_land = True
            if self.prev == "approach" and self.switch is None:
                self.switch = {
                    "step": int(step),
                    "height": float(height),
                    "abs_dx": float(abs_dx),
                }
        self.prev = mode

    def bucket(self) -> str:
        if self.switch is not None:
            return "switch"
        if not self.named_land:
            return "no_switch"
        return "opened_in_land"


def switch_fields(log: SwitchLog) -> dict:
    bucket = log.bucket()
    if bucket == "switch":
        return {
            "bucket": bucket,
            "switch_step": int(log.switch["step"]),
            "switch_height": float(log.switch["height"]),
            "switch_abs_dx": float(log.switch["abs_dx"]),
        }
    return {
        "bucket": bucket,
        "switch_step": None,
        "switch_height": None,
        "switch_abs_dx": None,
    }


def summarize_switches(rows) -> dict:
    if len(rows) != GRID_N:
        raise RuntimeError(len(rows))
    steps, heights, dxs = [], [], []
    n_no = n_open = n_hi = 0
    for row in rows:
        bucket = row["bucket"]
        if bucket == "switch":
            height = float(row["switch_height"])
            steps.append(float(row["switch_step"]))
            heights.append(height)
            dxs.append(float(row["switch_abs_dx"]))
            if height > STAGE1_HEIGHT:
                n_hi += 1
        elif bucket == "no_switch":
            n_no += 1
        elif bucket == "opened_in_land":
            n_open += 1
        else:
            raise RuntimeError(bucket)
    n_switch = len(steps)
    if n_switch + n_no + n_open != GRID_N:
        raise RuntimeError((n_switch, n_no, n_open))

    def med(values):
        if not values:
            return None
        value = float(np.median(np.asarray(values, np.float64)))
        if not np.isfinite(value):
            raise RuntimeError(value)
        return value

    share_hi = None if n_switch == 0 else float(n_hi) / float(n_switch)
    return {
        "n": int(GRID_N),
        "n_switch": int(n_switch),
        "n_no_switch": int(n_no),
        "n_opened_in_land": int(n_open),
        "median_step": med(steps),
        "median_height": med(heights),
        "median_abs_dx": med(dxs),
        "share_no_switch": float(n_no) / float(GRID_N),
        "n_height_gt_128": int(n_hi),
        "share_height_gt_128": share_hi,
        "height_gt_128_is_gate": False,
    }


@torch.no_grad()
def classify(head, encoder, state, g, pad_x):
    """Пустой режим — None. Иначе порог 0.5 по латенту и каналу площадки."""
    if hover_mode(state, float(g), float(pad_x), V_TARGET_LOOP) is None:
        return None, None
    pads = np.asarray([float(pad_x)], np.float64)
    gs = np.asarray([float(g)], np.float64)
    state_a = np.asarray(state, np.float64).reshape(1, 5)
    z = encode_rel(encoder, state_a, pads)
    feat = features_of("latent", pads, gs, z)
    if feat.shape != (1, 33):
        raise RuntimeError(feat.shape)
    logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
    prob = float(torch.sigmoid(logit).item())
    return mode_of(prob), prob


def rule_action(state, g, pad_x, mode):
    """На земле и при пустом баке действие 1. Иначе названная ветка правила."""
    if hover_mode(state, float(g), float(pad_x), V_TARGET_LOOP) is None:
        return 1
    if mode not in ("approach", "land"):
        raise RuntimeError(mode)
    return int(hover_action(state, float(g), float(pad_x), V_TARGET_LOOP, mode=mode))


def rule_from_head(head, encoder, state, g, pad_x):
    mode, _prob = classify(head, encoder, state, g, pad_x)
    if mode is None:
        return 1, None
    return rule_action(state, g, pad_x, mode), mode


def predictor_of(mode, predictors):
    if mode not in ("approach", "land"):
        raise RuntimeError(mode)
    predictor = predictors[mode]
    if predictor is None:
        raise RuntimeError(mode)
    return predictor


@torch.no_grad()
def tube_action(predictor, encoder, ema, actions, step_pairs, world_b, state, g, pad_x):
    """argmin по 729 планам. Цель — труба, не терминал на земле."""
    x, y, vx, vy, fuel = (float(v) for v in state)
    row = {
        "x": x, "y": y, "vx": vx, "vy": vy, "fuel": fuel,
        "g": float(g), "pad_x": float(pad_x),
    }
    before = (x, y, vx, vy, fuel)
    z, z_tube, _frames, calls, clipped = prepare(encoder, ema, row)
    if (x, y, vx, vy, fuel) != before:
        raise RuntimeError("труба подменила состояние")
    if calls < 1:
        raise RuntimeError(calls)
    action, _cost = choose_action(predictor, z, z_tube, world_b, step_pairs, actions)
    return int(action), bool(clipped)


@torch.no_grad()
def terminal_action(predictor, encoder, ema, actions, step_pairs, world_b, state, g, pad_x, cache):
    """Та же сетка планов. Цель — терминал на земле, vy = 6, бак текущий."""
    x, y, vx, vy, fuel = (float(v) for v in state)
    pads = np.asarray([float(pad_x)], np.float64)
    raw = np.asarray([[x, y, vx, vy, fuel]], np.float64)
    z_np = encode_rel(encoder, raw, pads)
    z = torch.from_numpy(np.ascontiguousarray(z_np))
    action = first_action(
        predictor, z, goal_z(ema, pad_x, fuel, cache), world_b, step_pairs, actions)
    return int(action)


@torch.no_grad()
def closed_skill_step(
    kind, head, encoder, ema, predictors, actions, step_pairs, world_b, state, g, pad_x, cache,
):
    """Один шаг. Физика — step_field выбранного действия."""
    mode, prob = classify(head, encoder, state, g, pad_x)
    clipped = False
    if mode is None:
        action = 1
    elif kind == "rule":
        action = rule_action(state, g, pad_x, mode)
    elif kind == "tube":
        action, clipped = tube_action(
            predictor_of(mode, predictors), encoder, ema, actions, step_pairs, world_b,
            state, g, pad_x)
    elif kind == "terminal":
        action = terminal_action(
            predictor_of(mode, predictors), encoder, ema, actions, step_pairs, world_b,
            state, g, pad_x, cache)
    else:
        raise RuntimeError(kind)
    x, y, vx, vy, fuel = (float(v) for v in state)
    nxt = step_field(x, y, vx, vy, fuel, False, int(action), float(g))
    return nxt, int(action), bool(clipped), mode, prob


def assert_rule_matches(head, encoder, g, pad_x) -> None:
    """Несколько шагов против прежнего контура. Расхождение — стоп до сетки."""
    decide = _decide(head, encoder, float(g), float(pad_x), float(THRESHOLD))
    x, y, vx, vy, _acts = start_and_actions(LOOP_SEED0, 0)
    fuel = float(FUEL0)
    grounded = False
    for _ in range(8):
        if grounded:
            break
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        action, _mode = rule_from_head(head, encoder, state, g, pad_x)
        ref = int(decide(state))
        if int(action) != ref:
            raise RuntimeError(f"правило разошлось с прежним контуром: {action} vs {ref}")
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, int(action), float(g))


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
    grounded = False
    steps = 0
    fuel_out = False
    clipped_n = 0
    log = SwitchLog() if track else None
    cache = {}
    while steps < int(HOVER_CAP) and not grounded:
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        nxt, _action, clipped, mode, _prob = closed_skill_step(
            kind, head, encoder, ema, predictors, actions, step_pairs, world_b,
            state, g, pad_x, cache)
        if clipped:
            clipped_n += 1
        if log is not None and mode is not None:
            height = float(GROUND) - float(y)
            log.observe(mode, steps, height, abs(float(x) - pad_x))
        x, y, vx, vy, fuel, grounded = nxt
        steps += 1
        if float(fuel) <= 0.0 and not grounded:
            fuel_out = True
    result = outcome_at(x, y, vy, grounded, pad_x)
    if result not in RESULT_CODE:
        raise RuntimeError(result)
    out = {
        "seed": int(row["seed"]),
        "g": g,
        "pad_x": pad_x,
        "result": result,
        "steps": int(steps),
        "fuel_out": bool(fuel_out),
        "tube_clipped": int(clipped_n),
    }
    if log is not None:
        out.update(switch_fields(log))
    return out


def _group(cells, key):
    order = []
    buckets = {}
    for cell in cells:
        mark = cell[key]
        if mark not in buckets:
            order.append(mark)
            buckets[mark] = {"soft": 0, "crash": 0, "miss": 0, "fuel_out": 0, "of": 0}
        bucket = buckets[mark]
        for name in ("soft", "crash", "miss", "fuel_out", "of"):
            bucket[name] += int(cell[name])
    rows = []
    for mark in order:
        row = dict(buckets[mark])
        row[key] = mark
        rows.append(row)
    return rows


def _finish_cell(bucket):
    if int(bucket["of"]) != int(LOOP_N):
        raise RuntimeError(bucket["of"])
    if int(bucket["soft"]) + int(bucket["crash"]) + int(bucket["miss"]) != int(LOOP_N):
        raise RuntimeError(bucket)
    return {
        "g": float(bucket["g"]),
        "pad_x": float(bucket["pad_x"]),
        "soft": int(bucket["soft"]),
        "crash": int(bucket["crash"]),
        "miss": int(bucket["miss"]),
        "fuel_out": int(bucket["fuel_out"]),
        "of": int(LOOP_N),
        "steps_mean": round(float(bucket["steps"]) / float(LOOP_N), 3),
    }


@torch.no_grad()
def fly_column(name, kind, head, encoder, ema, predictors, actions, step_pairs, track):
    rows = episodes()
    cells = []
    failures = []
    tracked = []
    current = None
    bucket = None
    clipped_n = 0
    t_cell = time.perf_counter()
    worlds = {}
    for index, row in enumerate(rows, start=1):
        key = (float(row["g"]), float(row["pad_x"]))
        if key not in worlds:
            worlds[key] = world_batch(row["g"], row["pad_x"])
        if key != current:
            if bucket is not None:
                cell = _finish_cell(bucket)
                cells.append(cell)
                elapsed = time.perf_counter() - t_cell
                print(
                    f"DECISION_CELL column={name} g={cell['g']:g} pad={cell['pad_x']:g} "
                    f"soft={cell['soft']} crash={cell['crash']} miss={cell['miss']} "
                    f"fuel_out={cell['fuel_out']} sec={elapsed:.1f}",
                    flush=True,
                )
            current = key
            bucket = {
                "g": key[0], "pad_x": key[1],
                "soft": 0, "crash": 0, "miss": 0, "fuel_out": 0, "of": 0, "steps": 0,
            }
            t_cell = time.perf_counter()
        ep = fly_episode(
            kind, head, encoder, ema, predictors, actions, step_pairs, worlds[key], row, track)
        code = RESULT_CODE[ep["result"]]
        bucket[code] += 1
        bucket["of"] += 1
        bucket["steps"] += int(ep["steps"])
        if ep["fuel_out"]:
            bucket["fuel_out"] += 1
        clipped_n += int(ep["tube_clipped"])
        if ep["result"] != "мягкая":
            failures.append({
                "seed": int(ep["seed"]),
                "g": float(ep["g"]),
                "pad_x": float(ep["pad_x"]),
                "result": ep["result"],
                "steps": int(ep["steps"]),
                "fuel_out": bool(ep["fuel_out"]),
            })
        if track:
            tracked.append({
                "seed": int(ep["seed"]),
                "g": float(ep["g"]),
                "pad_x": float(ep["pad_x"]),
                "result": ep["result"],
                "steps": int(ep["steps"]),
                "fuel_out": bool(ep["fuel_out"]),
                "tube_clipped": int(ep["tube_clipped"]),
                "bucket": ep["bucket"],
                "switch_step": ep["switch_step"],
                "switch_height": ep["switch_height"],
                "switch_abs_dx": ep["switch_abs_dx"],
            })
        print(
            f"DECISION_EP column={name} i={index} n={GRID_N} "
            f"steps={ep['steps']} {code}",
            flush=True,
        )
    if bucket is None:
        raise RuntimeError("пустая сетка")
    cell = _finish_cell(bucket)
    cells.append(cell)
    elapsed = time.perf_counter() - t_cell
    print(
        f"DECISION_CELL column={name} g={cell['g']:g} pad={cell['pad_x']:g} "
        f"soft={cell['soft']} crash={cell['crash']} miss={cell['miss']} "
        f"fuel_out={cell['fuel_out']} sec={elapsed:.1f}",
        flush=True,
    )
    if len(cells) != len(PASS_G) * len(PASS_PADS):
        raise RuntimeError(len(cells))
    soft = int(sum(item["soft"] for item in cells))
    crash = int(sum(item["crash"] for item in cells))
    miss = int(sum(item["miss"] for item in cells))
    fuel_n = int(sum(item["fuel_out"] for item in cells))
    if soft + crash + miss != GRID_N:
        raise RuntimeError((soft, crash, miss))
    if track and len(tracked) != GRID_N:
        raise RuntimeError(len(tracked))
    column = {
        "name": name,
        "kind": kind,
        "scored": bool(name == "skill_tube"),
        "soft": soft,
        "crash": crash,
        "miss": miss,
        "fuel_out": fuel_n,
        "of": int(GRID_N),
        "tube_clipped": int(clipped_n),
        "cells": cells,
        "by_g": _group(cells, "g"),
        "by_pad": _group(cells, "pad_x"),
        "failures": failures,
    }
    if track:
        column["episodes"] = tracked
        column["switches"] = summarize_switches(tracked)
    return column


def approach_path() -> str:
    path = os.path.join(OUT, "landk_approach_s0.pt")
    if os.path.basename(path) != "landk_approach_s0.pt":
        raise RuntimeError(path)
    return path


def latent_path() -> str:
    path = ckpt_path("latent", 0)
    if os.path.basename(path) != "landd_latent_s0.pt":
        raise RuntimeError(path)
    return path


def optional_paths():
    return {
        "landd_early_s1.pt": os.path.join(OUT, "landd_early_s1.pt"),
        "landk_pad_s1.pt": pad_path(1),
        "landk_approach_s1.pt": os.path.join(OUT, "landk_approach_s1.pt"),
        "landk_approach_s2.pt": os.path.join(OUT, "landk_approach_s2.pt"),
    }


def watched_paths():
    paths = {
        "landd_early_s0.pt": EARLY_PATH,
        "landd_latent_s0.pt": latent_path(),
        "landk_pad_s0.pt": pad_path(0),
        "landk_approach_s0.pt": approach_path(),
        SOURCE_NAME: os.path.join(OUT, SOURCE_NAME),
        "expert_hover.py": os.path.join(ROOT, "numeric_land", "expert_hover.py"),
    }
    paths.update(optional_paths())
    for name in DEMO_NAMES:
        paths[name] = os.path.join(ROOT, name)
    return paths


def take_stamps():
    stamps = {}
    for name, path in watched_paths().items():
        stamps[name] = file_stamp(path)
        if stamps[name] is None and name not in OPTIONAL_NAMES:
            raise SystemExit(f"нет {name}")
    return stamps


def _freeze_head(head):
    head.requires_grad_(False)
    head.eval()
    return head


def load_early_head(source):
    path = EARLY_PATH
    if os.path.basename(path) != "landd_early_s0.pt":
        raise RuntimeError(path)
    before = file_stamp(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if file_stamp(path) != before:
        raise RuntimeError("чтение landd_early_s0 изменило файл")
    if ck.get("tag") != EARLY_TAG or ck.get("kind") != "early" or int(ck.get("seed")) != 0:
        raise RuntimeError(path)
    if float(ck.get("threshold")) != float(THRESHOLD) or int(ck.get("n_in")) != n_in_of("latent"):
        raise RuntimeError(path)
    if int(ck.get("n_in")) != 33:
        raise RuntimeError(ck.get("n_in"))
    _same_source(ck, source, "landd_early_s0")
    head = DecisionHead(n_in_of("latent"))
    head.load_state_dict(ck["head"])
    return _freeze_head(head)


def load_bundle():
    """Чтение. s1 и s2 сюда не входят."""
    source, encoder, ema = load_frozen()
    encoder.eval()
    ema.eval()
    early = load_early_head(source)
    latent_ck, latent = _load_head("latent", 0)
    if os.path.basename(latent_path()) != "landd_latent_s0.pt":
        raise RuntimeError(latent_path())
    if int(latent_ck.get("n_in")) != 33 or float(latent_ck.get("threshold")) != float(THRESHOLD):
        raise RuntimeError(latent_ck.get("n_in"))
    _same_source(latent_ck, source, "landd_latent_s0")
    latent = _freeze_head(latent)
    pad = load_predictor(pad_path(0), "pad", 0, LAND_TAG, source)
    approach = load_predictor(approach_path(), "approach", 0, ENCODER_TAG, source)
    pad_sd = pad.state_dict()
    approach_sd = approach.state_dict()
    if pad_sd.keys() == approach_sd.keys() and all(
        torch.equal(pad_sd[key], approach_sd[key]) for key in pad_sd
    ):
        raise RuntimeError("предикторы посадки и наведения совпали")
    early_sd = early.state_dict()
    latent_sd = latent.state_dict()
    if early_sd.keys() == latent_sd.keys() and all(
        torch.equal(early_sd[key], latent_sd[key]) for key in early_sd
    ):
        raise RuntimeError("головы early и latent совпали")
    return {
        "source": source,
        "encoder": encoder,
        "ema": ema,
        "early": early,
        "latent": latent,
        "pad": pad,
        "approach": approach,
    }


def _column_soft(columns, name):
    found = [column for column in columns if column["name"] == name]
    if len(found) != 1:
        raise RuntimeError(name)
    return found[0]


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
        raise SystemExit("уже есть landd_skill_*.pt, новый чекпоинт писать нельзя")
    print("DECISION_START grid=180", flush=True)
    hover_path = os.path.join(ROOT, "numeric_land", "expert_hover.py")
    hover_bytes = open(hover_path, "rb").read()
    before = take_stamps()
    bundle = load_bundle()
    if take_stamps() != before:
        raise RuntimeError("чтение сдвинуло штамп")
    print("DECISION_LOADED", flush=True)
    predictors = {"land": bundle["pad"], "approach": bundle["approach"]}
    encoder = bundle["encoder"]
    ema = bundle["ema"]
    early = bundle["early"]
    latent = bundle["latent"]
    assert_rule_matches(early, encoder, float(PASS_G[0]), float(PASS_PADS[0]))
    assert_rule_matches(latent, encoder, float(PASS_G[0]), float(PASS_PADS[0]))
    actions, step_pairs = pair_table()
    columns = []
    columns.append(fly_column(
        "rule_early", "rule", early, encoder, ema, predictors, actions, step_pairs, False))
    columns.append(fly_column(
        "rule_latent", "rule", latent, encoder, ema, predictors, actions, step_pairs, False))
    rule_early = _column_soft(columns, "rule_early")
    rule_latent = _column_soft(columns, "rule_latent")
    ok = control_ok(rule_early["soft"], rule_latent["soft"])
    print(
        f"DECISION_CONTROL early={rule_early['soft']} latent={rule_latent['soft']} "
        f"ok={int(ok)}",
        flush=True,
    )
    columns.append(fly_column(
        "skill_tube", "tube", early, encoder, ema, predictors, actions, step_pairs, True))
    columns.append(fly_column(
        "skill_terminal", "terminal", early, encoder, ema, predictors, actions, step_pairs, False))
    if take_stamps() != before:
        raise RuntimeError("полёт сдвинул штамп")
    if open(hover_path, "rb").read() != hover_bytes:
        raise RuntimeError("правило изменилось")
    if stray_pt():
        raise RuntimeError("появился landd_skill_*.pt")
    tube = _column_soft(columns, "skill_tube")
    terminal = _column_soft(columns, "skill_terminal")
    if not tube["scored"] or any(column["scored"] for column in columns if column is not tube):
        raise RuntimeError("зачётный столбец не skill_tube")
    if rule_early["tube_clipped"] or rule_latent["tube_clipped"] or terminal["tube_clipped"]:
        raise RuntimeError("обрезка трубы не в том столбце")
    numbers = verdict(tube["soft"], rule_early["soft"], rule_latent["soft"], terminal["soft"])
    if numbers["pilot_pass"] and not numbers["control_ok"]:
        raise RuntimeError("зачёт закрыт без контроля")
    switches = tube["switches"]
    if switches["height_gt_128_is_gate"] or switches["n"] != GRID_N:
        raise RuntimeError("сводка переключений сломана")
    payload = {
        "tag": TAG,
        "grid_n": GRID_N,
        "line": LINE,
        "ratio_min": RATIO_MIN,
        "ratio_need": numbers["ratio_need"],
        "threshold": float(THRESHOLD),
        "horizon": HORIZON,
        "n_sequences": N_SEQ,
        "cem": False,
        "rule_action_in_scored": False,
        "heads_trained": False,
        "tube_gain": numbers["tube_gain"],
        "tube_gain_ok": numbers["tube_gain_ok"],
        "control_ok": numbers["control_ok"],
        "stamps_unchanged": True,
        "pilot_pass": numbers["pilot_pass"],
        "line_ok": numbers["line_ok"],
        "tube_clipped": int(tube["tube_clipped"]),
        "gain_min": GAIN_MIN,
        "control_early": CONTROL_EARLY,
        "control_latent": CONTROL_LATENT,
        "v_target": float(V_TARGET_LOOP),
        "seeds": [int(LOOP_SEED0), int(LOOP_SEED0 + LOOP_N - 1)],
        "step_origin": 0,
        "step_weights": [float(weight) for weight in LOSS_WEIGHTS],
        "tube_targets": int(TUBE_TARGETS),
        "height_gt_128_is_gate": False,
        "encoder": SOURCE_NAME,
        "heads": {"early": "landd_early_s0.pt", "latent": "landd_latent_s0.pt"},
        "skills": {"land": "landk_pad_s0.pt", "approach": "landk_approach_s0.pt"},
        "early_s1_flown": False,
        "pad_s1_flown": False,
        "approach_s1_flown": False,
        "approach_s2_flown": False,
        "prior_closed_loop_soft": {
            "early_rule": 120,
            "latent_rule": 108,
            "skill_terminal_landk_land_s0": 6,
        },
        "switches": switches,
        "columns": columns,
    }
    if take_stamps() != before:
        raise RuntimeError("перед записью штамп сдвинулся")
    write_json(report, payload)
    if take_stamps() != before:
        raise RuntimeError("запись отчёта задела чекпоинт")
    if open(hover_path, "rb").read() != hover_bytes:
        raise RuntimeError("правило изменилось после записи")
    print(
        f"DECISION_DONE skill_tube={tube['soft']} rule_early={rule_early['soft']} "
        f"rule_latent={rule_latent['soft']} skill_terminal={terminal['soft']} "
        f"gain={numbers['tube_gain']} control_ok={int(numbers['control_ok'])} "
        f"line_ok={int(numbers['line_ok'])} pilot_pass={int(numbers['pilot_pass'])} "
        f"tube_clipped={tube['tube_clipped']}",
        flush=True,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

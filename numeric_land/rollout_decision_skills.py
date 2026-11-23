"""Действие от навыка в режиме, который назвала уже обученная голова.

    python -m numeric_land.rollout_decision_skills --tag numeric-land-decision-skills

Обучения нет. Порог 0.5 не двигается. Длинного CEM нет.
Поиск открытый и полный: горизонт 3, все 729 последовательностей,
берётся первое действие лучшей. Коэффициенты шагов — 1, 0.5, 0.25.
Цель — латент EMA от состояния на площадке, на земле, vx = 0, vy = 6,
бак равен текущему. height0 = 0 зафиксирован до прогона.
"""
import argparse
import os
import time

import numpy as np
import torch

from numeric_land.data import PAIRS
from numeric_land.decision import (
    CELL_NOISE, THRESHOLD, DecisionHead, assert_start, cell_is_result, features_of, n_in_of,
)
from numeric_land.expert_hover import HOVER_CAP, hover_action, hover_mode
from numeric_land.field import FUEL0, start_and_actions, step_field
from numeric_land.goal import target_state
from numeric_land.mode import LOOP_N, LOOP_SEED0, PASS_G, PASS_PADS, V_TARGET_LOOP
from numeric_land.models import LATENT, LOSS_WEIGHTS, LandPredictor, action_pairs
from numeric_land.pad import outcome_at
from numeric_land.relenc import (
    OUT, TAG as ENCODER_TAG, encode_rel, file_stamp, scale_rel, tensors_equal, write_json,
)
from numeric_land.rollout_decision import (
    MODE_JSON, OUT_JSON as DECISION_JSON, _encoder_from, _load_head, _load_json,
)
from numeric_land.rollout_decision_threshold import _decide
from numeric_land.skills import N_WORLD, world_of
from numeric_land.train_decision import ckpt_path
from numeric_land.train_relenc import ckpt_path as relenc_ckpt
from numeric_land.train_skills import ckpt_path as skill_ckpt

TAG = "numeric-land-decision-skills"
EARLY_TAG = "numeric-land-decision-early"
HORIZON = 3
N_ACT = 9
N_SEQ = 729
SOFT_LINE = 15
PART_LINE = 3
GOAL_HEIGHT0 = 0.0
THREADS = 8
PUBLISHED_SOFT = {"latent": 108, "early": 120}
READING_USEFUL = "навыки полезны при неидеальном переключении"
READING_PARTIAL = "навыки полезны на части клеток"
READING_MIXED = "навыки двигают клетки в разные стороны"
READING_WEAK = "прирост слабый: дело в моменте переключения"
OUT_PATH = os.path.join(OUT, "numeric_land_decision_skills.json")
EARLY_JSON = os.path.join(OUT, "numeric_land_decision_early.json")
EARLY_PATH = os.path.join(OUT, "landd_early_s0.pt")
SOURCE_NAME = "landr_scratch_s0.pt"


def _refuse(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "нужен тег numeric-land-decision-skills. Другие теги запрещены.")


def _guard() -> None:
    if HORIZON != 3 or N_ACT != 9 or N_SEQ != 729 or N_ACT ** HORIZON != N_SEQ:
        raise RuntimeError((HORIZON, N_ACT, N_SEQ))
    if LOSS_WEIGHTS != (1.0, 0.5, 0.25):
        raise RuntimeError(LOSS_WEIGHTS)
    if CELL_NOISE != PART_LINE or SOFT_LINE != 15:
        raise RuntimeError((CELL_NOISE, SOFT_LINE))
    if abs(float(THRESHOLD) - 0.5) > 0.0 or float(GOAL_HEIGHT0) != 0.0:
        raise RuntimeError("порог или height0 сдвинулись до прогона")
    if len(PAIRS) != N_ACT or int(N_WORLD) != 4:
        raise RuntimeError((len(PAIRS), N_WORLD))
    names = {
        os.path.abspath(OUT_PATH),
        os.path.abspath(DECISION_JSON),
        os.path.abspath(EARLY_JSON),
        os.path.abspath(MODE_JSON),
    }
    if len(names) != 4 or OUT_PATH.endswith(".pt"):
        raise RuntimeError(OUT_PATH)
    if os.path.isfile(OUT_PATH):
        raise SystemExit(f"уже есть {OUT_PATH}, второй прогон отчёт не затирает")


def _paths():
    return {
        "encoder": relenc_ckpt("scratch", 0),
        "latent": ckpt_path("latent", 0),
        "early": EARLY_PATH,
        "approach": skill_ckpt("approach", 0),
        "land": skill_ckpt("land", 0),
        "decision_json": DECISION_JSON,
        "early_json": EARLY_JSON,
        "mode_json": MODE_JSON,
    }


def _stamps(paths):
    stamps = {key: file_stamp(path) for key, path in paths.items()}
    missing = [key for key, stamp in stamps.items() if stamp is None]
    if missing:
        raise SystemExit(f"нет файла для чтения: {missing}")
    if os.path.basename(paths["encoder"]) != SOURCE_NAME:
        raise RuntimeError(paths["encoder"])
    for key in ("latent", "early", "approach", "land"):
        if not os.path.basename(paths[key]).endswith(".pt"):
            raise RuntimeError(paths[key])
    return stamps


def _pair_table():
    index = torch.arange(N_SEQ)
    columns = []
    base = N_ACT ** (HORIZON - 1)
    for _ in range(HORIZON):
        columns.append(torch.div(index, base, rounding_mode="floor") % N_ACT)
        base //= N_ACT
    actions = torch.stack(columns, dim=1)
    like = torch.zeros(1, dtype=torch.float32)
    pairs = action_pairs(actions, like)
    if actions.shape != (N_SEQ, HORIZON) or pairs.shape != (N_SEQ, HORIZON, 2):
        raise RuntimeError((actions.shape, pairs.shape))
    if int(actions[0, 0]) != 0 or int(actions[81, 0]) != 1 or int(actions[728, 0]) != 8:
        raise RuntimeError("порядок действий разъехался с индексом iy*3+ix")
    steps = [pairs[:, k].contiguous() for k in range(HORIZON)]
    return actions, steps


def _reading(skill_soft: int, rule_soft: int, deltas) -> str:
    """Четыре формулировки зафиксированы до прогона и после чисел не меняются."""
    gain = int(skill_soft) - int(rule_soft)
    dropped = any(cell_is_result(delta) and int(delta) < 0 for delta in deltas)
    gained = any(cell_is_result(delta) and int(delta) > 0 for delta in deltas)
    if gain >= SOFT_LINE and not dropped:
        return READING_USEFUL
    if gained and not dropped and gain >= PART_LINE:
        return READING_PARTIAL
    if gained and dropped:
        return READING_MIXED
    return READING_WEAK


def _switch_stat(steps) -> dict:
    never = sum(step is None for step in steps)
    values = [int(step) for step in steps if step is not None]
    if not values:
        return {
            "mean": None, "median": None, "min": None,
            "n_switched": 0, "n_never": int(never), "n": int(len(steps)),
        }
    arr = np.asarray(values, np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "n_switched": int(len(values)),
        "n_never": int(never),
        "n": int(len(steps)),
    }


def _load_scratch():
    path = relenc_ckpt("scratch", 0)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != ENCODER_TAG or ck.get("kind") != "scratch" or int(ck.get("seed", -1)) != 0:
        raise RuntimeError(path)
    if float(ck.get("w_probe")) != 0.1 or int(ck.get("n_in")) != 7:
        raise RuntimeError(path)
    encoder, ema = _encoder_from(ck, "latent")
    encoder.eval()
    ema.eval()
    return ck, encoder, ema


def _same_source(blob, scratch, label: str) -> None:
    if not tensors_equal(blob["encoder"], scratch["encoder"]):
        raise RuntimeError(f"кодировщик {label} не совпал с {SOURCE_NAME}")
    if not tensors_equal(blob["ema"], scratch["ema"]):
        raise RuntimeError(f"EMA {label} не совпала с {SOURCE_NAME}")


def _load_early(scratch):
    ck = torch.load(EARLY_PATH, map_location="cpu", weights_only=False)
    if ck.get("tag") != EARLY_TAG or ck.get("kind") != "early" or int(ck.get("seed")) != 0:
        raise RuntimeError(EARLY_PATH)
    if float(ck.get("threshold")) != float(THRESHOLD) or int(ck.get("n_in")) != n_in_of("latent"):
        raise RuntimeError(EARLY_PATH)
    _same_source(ck, scratch, "landd_early_s0")
    head = DecisionHead(n_in_of("latent"))
    head.load_state_dict(ck["head"])
    head.eval()
    return head


def _load_predictor(phase: str, scratch):
    path = skill_ckpt(phase, 0)
    name = os.path.basename(path)
    if name != f"landk_{phase}_s0.pt":
        raise RuntimeError(name)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != ENCODER_TAG or ck.get("phase") != phase or int(ck.get("seed")) != 0:
        raise RuntimeError(path)
    if int(ck.get("n_world")) != N_WORLD:
        raise RuntimeError(ck.get("n_world"))
    _same_source(ck, scratch, name)
    predictor = LandPredictor(N_WORLD)
    predictor.load_state_dict(ck["predictor"])
    predictor.eval()
    if int(predictor.n_world) != 4:
        raise RuntimeError(predictor.n_world)
    return predictor


def _published_early_soft(payload) -> int:
    found = None
    for row in payload["rows"]:
        if int(row.get("seed", -1)) == 0 and "soft" in row:
            found = int(row["soft"])
    if found is None:
        raise RuntimeError("в отчёте штрафа нет полёта seed 0")
    return found


def _check_published(decision, early) -> None:
    if int(decision["new_soft"]) != PUBLISHED_SOFT["latent"]:
        raise SystemExit(f"в отчёте решения мягких {decision['new_soft']}, ждали 108")
    early_soft = _published_early_soft(early)
    if early_soft != PUBLISHED_SOFT["early"]:
        raise SystemExit(f"в отчёте штрафа мягких {early_soft}, ждали 120")
    if float(decision["threshold"]) != 0.5 or float(early["threshold"]) != 0.5:
        raise SystemExit("опубликованный порог уже не 0.5")
    if decision.get("skills_choose_action") is not False or early.get("skills_choose_action") is not False:
        raise SystemExit("старый отчёт уже отдавал действие навыку")


def _mode_and_z(head, encoder, state, g, pad_x, pads, gs):
    if hover_mode(state, g, pad_x, V_TARGET_LOOP) is None:
        return None, None
    state_a = np.asarray(state, np.float64).reshape(1, 5)
    z = encode_rel(encoder, state_a, pads)
    feat = features_of("latent", pads, gs, z)
    logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
    prob = float(torch.sigmoid(logit)[0].item())
    mode = "land" if prob >= float(THRESHOLD) else "approach"
    return mode, z


def _world_batch(g, pad_x):
    raw = world_of(
        np.asarray([float(g)], np.float64),
        np.asarray([float(pad_x)], np.float64),
        np.asarray([float(V_TARGET_LOOP)], np.float64),
    )
    if tuple(raw.shape) != (1, 4):
        raise RuntimeError(raw.shape)
    tensor = torch.from_numpy(np.ascontiguousarray(raw))
    return tensor.expand(N_SEQ, -1).contiguous()


def _goal_z(ema, pad_x, fuel, cache):
    key = float(fuel)
    hit = cache.get(key)
    if hit is not None:
        return hit
    raw = target_state(float(pad_x), float(V_TARGET_LOOP), GOAL_HEIGHT0, fuel0=key)
    scaled = scale_rel(raw, np.asarray([float(pad_x)], np.float64))
    if scaled.shape != (1, 7):
        raise RuntimeError(scaled.shape)
    latent = ema(torch.from_numpy(np.ascontiguousarray(scaled)))
    if latent.shape != (1, LATENT):
        raise RuntimeError(latent.shape)
    cache[key] = latent
    return latent


def _first_action(predictor, z, z_goal, world_b, step_pairs, actions):
    if z.shape != (1, LATENT) or z_goal.shape != (1, LATENT):
        raise RuntimeError((z.shape, z_goal.shape))
    current = z.expand(N_SEQ, -1)
    cost = None
    for k, weight in enumerate(LOSS_WEIGHTS):
        current = predictor(current, step_pairs[k], world_b)
        step = (current - z_goal).abs().mean(dim=-1)
        cost = step.mul(float(weight)) if cost is None else cost.add(step, alpha=float(weight))
    index = int(torch.argmin(cost).item())
    return int(actions[index, 0].item())


def _check_rule_agrees(head, encoder, g, pad_x) -> None:
    x, y, vx, vy, _ = start_and_actions(LOOP_SEED0, 0)
    state = (float(x), float(y), float(vx), float(vy), float(FUEL0))
    pads = np.asarray([pad_x], np.float64)
    gs = np.asarray([g], np.float64)
    mode, _z = _mode_and_z(head, encoder, state, g, pad_x, pads, gs)
    if mode is None:
        mine = 1
    else:
        mine = int(hover_action(state, g, pad_x, V_TARGET_LOOP, mode=mode))
    ref = int(_decide(head, encoder, g, pad_x, float(THRESHOLD))(state))
    if mine != ref:
        raise RuntimeError(f"правило разошлось с прежним контуром: {mine} vs {ref}")


@torch.no_grad()
def _action_gap(predictor, encoder, world_b, step_pairs) -> float:
    """Один старт. Ноль значит, что два разных первых действия дают один латент.

    Первые 81 последовательностей делят одно и то же первое действие, поэтому
    вторая точка — индекс 81, а не 1.
    """
    x, y, vx, vy, _ = start_and_actions(LOOP_SEED0, 0)
    state = np.asarray([[x, y, vx, vy, FUEL0]], np.float64)
    pads = np.asarray([float(PASS_PADS[0])], np.float64)
    z = torch.from_numpy(np.ascontiguousarray(encode_rel(encoder, state, pads)))
    left = predictor(z, step_pairs[0][0:1], world_b[0:1])
    right = predictor(z, step_pairs[0][81:82], world_b[0:1])
    return float((left - right).abs().mean().item())


@torch.no_grad()
def _episode(seed, g, pad_x, head, encoder, ema, predictors, actions, step_pairs, source, world_b):
    x, y, vx, vy, _ = start_and_actions(int(seed), 0)
    fuel = float(FUEL0)
    grounded = False
    steps = 0
    first_land = None
    fallback = 0
    pads = np.asarray([pad_x], np.float64)
    gs = np.asarray([g], np.float64)
    cache = {}
    while steps < HOVER_CAP and not grounded:
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        mode, z = _mode_and_z(head, encoder, state, g, pad_x, pads, gs)
        if mode == "land" and first_land is None:
            first_land = int(steps)
        if mode is None:
            action = 1
        elif source == "rule":
            action = int(hover_action(state, g, pad_x, V_TARGET_LOOP, mode=mode))
        else:
            predictor = predictors.get(mode)
            if predictor is None:
                action = int(hover_action(state, g, pad_x, V_TARGET_LOOP, mode=mode))
                fallback += 1
            else:
                z_now = torch.from_numpy(np.ascontiguousarray(z))
                action = _first_action(
                    predictor, z_now, _goal_z(ema, pad_x, fuel, cache),
                    world_b, step_pairs, actions)
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, int(action), float(g))
        steps += 1
    return outcome_at(x, y, vy, grounded, pad_x), first_land, int(fallback)


def _fly_cell(label, source, head, encoder, ema, predictors, actions, step_pairs, g, pad_x):
    world_b = _world_batch(g, pad_x) if source == "skill" else None
    soft = crash = miss = 0
    lands = []
    fallback = 0
    t0 = time.perf_counter()
    for seed in range(LOOP_SEED0, LOOP_SEED0 + LOOP_N):
        result, first_land, n_fb = _episode(
            seed, g, pad_x, head, encoder, ema, predictors, actions, step_pairs, source, world_b)
        fallback += n_fb
        lands.append(first_land)
        if result == "мягкая":
            soft += 1
        elif result == "удар":
            crash += 1
        elif result == "не сел":
            miss += 1
        else:
            raise RuntimeError(result)
    if soft + crash + miss != LOOP_N:
        raise RuntimeError((soft, crash, miss))
    stat = _switch_stat(lands)
    elapsed = time.perf_counter() - t0
    print(
        f"{label} {source} g={g:g} pad={pad_x:g} "
        f"soft={soft} crash={crash} miss={miss} "
        f"land_mean={stat['mean']} land_median={stat['median']} "
        f"n_never={stat['n_never']} fallback={fallback} {elapsed:.1f} с",
        flush=True,
    )
    return {
        "g": float(g),
        "pad_x": float(pad_x),
        "soft": int(soft),
        "crash": int(crash),
        "miss": int(miss),
        "of": int(LOOP_N),
        "fallback": int(fallback),
        "switch": stat,
        "seconds": round(float(elapsed), 3),
    }


def _totals(cells):
    soft = int(sum(cell["soft"] for cell in cells))
    crash = int(sum(cell["crash"] for cell in cells))
    miss = int(sum(cell["miss"] for cell in cells))
    fallback = int(sum(cell["fallback"] for cell in cells))
    if soft + crash + miss != 180:
        raise RuntimeError((soft, crash, miss))
    return soft, crash, miss, fallback


def _row(head_name, file_name, rule_cells, skill_cells):
    rule_soft, rule_crash, rule_miss, rule_fb = _totals(rule_cells)
    skill_soft, skill_crash, skill_miss, skill_fb = _totals(skill_cells)
    paired = []
    deltas = []
    for rule, skill in zip(rule_cells, skill_cells):
        if rule["g"] != skill["g"] or rule["pad_x"] != skill["pad_x"]:
            raise RuntimeError("клетки правила и навыка разошлись")
        delta = int(skill["soft"]) - int(rule["soft"])
        deltas.append(delta)
        paired.append({
            "g": rule["g"],
            "pad_x": rule["pad_x"],
            "rule_soft": rule["soft"],
            "skill_soft": skill["soft"],
            "delta_soft": delta,
            "rule_crash": rule["crash"],
            "skill_crash": skill["crash"],
            "rule_miss": rule["miss"],
            "skill_miss": skill["miss"],
            "counts_as_cell_result": bool(cell_is_result(delta)),
            "rule_switch": rule["switch"],
            "skill_switch": skill["switch"],
        })
    published = int(PUBLISHED_SOFT[head_name])
    reading = _reading(skill_soft, rule_soft, deltas)
    print(
        f"{head_name}: правило {rule_soft}/180  навыки {skill_soft}/180  "
        f"дельта {skill_soft - rule_soft}  опубликовано {published}  {reading}",
        flush=True,
    )
    return {
        "head": head_name,
        "file": file_name,
        "published_soft": published,
        "matches_published": bool(rule_soft == published),
        "rule": {
            "soft": rule_soft, "crash": rule_crash, "miss": rule_miss,
            "of": 180, "fallback": rule_fb, "cells": rule_cells,
        },
        "skill": {
            "soft": skill_soft, "crash": skill_crash, "miss": skill_miss,
            "of": 180, "fallback": skill_fb, "cells": skill_cells,
        },
        "cell_delta": paired,
        "soft_minus_rule": int(skill_soft - rule_soft),
        "soft_minus_published": int(skill_soft - published),
        "reading": reading,
    }


@torch.no_grad()
def _bench(head, encoder, ema, predictors, actions, step_pairs) -> None:
    """Тридцать поисков на старте. По ним посадку не судим."""
    g, pad = float(PASS_G[0]), float(PASS_PADS[0])
    world_b = _world_batch(g, pad)
    x, y, vx, vy, _ = start_and_actions(LOOP_SEED0, 0)
    state = (float(x), float(y), float(vx), float(vy), float(FUEL0))
    pads = np.asarray([pad], np.float64)
    gs = np.asarray([g], np.float64)
    t0 = time.perf_counter()
    for _ in range(30):
        mode, z = _mode_and_z(head, encoder, state, g, pad, pads, gs)
        if mode is None or z is None:
            raise RuntimeError("старт 99000 не дал режим")
        _first_action(
            predictors[mode], torch.from_numpy(np.ascontiguousarray(z)),
            _goal_z(ema, pad, FUEL0, {}), world_b, step_pairs, actions)
    elapsed = (time.perf_counter() - t0) * 1000.0 / 30.0
    print(f"поиск 3×729: {elapsed:.2f} мс/шаг  (в зачёт не идёт)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    _refuse(args.tag)
    torch.set_num_threads(THREADS)
    _guard()
    paths = _paths()
    before = _stamps(paths)
    decision = _load_json(DECISION_JSON)
    early = _load_json(EARLY_JSON)
    mode_json = _load_json(MODE_JSON)
    _check_published(decision, early)
    x0, y0, _vx, _vy, _acts = start_and_actions(LOOP_SEED0, 0)
    assert_start(x0, y0, mode_json["start_99000"])

    scratch, encoder, ema = _load_scratch()
    latent_ck, latent_head = _load_head("latent", 0)
    _same_source(latent_ck, scratch, "landd_latent_s0")
    early_head = _load_early(scratch)
    approach = _load_predictor("approach", scratch)
    land = _load_predictor("land", scratch)
    if tensors_equal(approach.state_dict(), land.state_dict()):
        raise RuntimeError("предикторы наведения и посадки совпали")
    heads = (("latent", "landd_latent_s0.pt", latent_head), ("early", "landd_early_s0.pt", early_head))
    for _name, _file, head in heads:
        _check_rule_agrees(head, encoder, float(PASS_G[0]), float(PASS_PADS[0]))
    actions, step_pairs = _pair_table()
    predictors = {"approach": approach, "land": land}
    gaps = {
        "approach": _action_gap(approach, encoder, _world_batch(PASS_G[0], PASS_PADS[0]), step_pairs),
        "land": _action_gap(land, encoder, _world_batch(PASS_G[0], PASS_PADS[0]), step_pairs),
    }
    if gaps["approach"] == 0.0 or gaps["land"] == 0.0:
        raise RuntimeError(f"предиктор не отличает два действия: {gaps}")
    print(f"зазор действия approach {gaps['approach']:.6f}  land {gaps['land']:.6f}", flush=True)
    _bench(latent_head, encoder, ema, predictors, actions, step_pairs)

    rows = []
    grid = [(float(g), float(pad)) for g in PASS_G for pad in PASS_PADS]
    if len(grid) * LOOP_N != 180:
        raise RuntimeError("сетка не 180")
    for head_name, file_name, head in heads:
        flown = {}
        for source in ("rule", "skill"):
            cells = []
            for g, pad in grid:
                cells.append(_fly_cell(
                    head_name, source, head, encoder, ema, predictors,
                    actions, step_pairs, g, pad))
            flown[source] = cells
        rows.append(_row(head_name, file_name, flown["rule"], flown["skill"]))

    after = _stamps(paths)
    if after != before:
        raise SystemExit("прогон изменил старый файл, отчёт не публикуется")
    payload = {
        "tag": TAG,
        "threshold": float(THRESHOLD),
        "horizon": HORIZON,
        "n_sequences": N_SEQ,
        "cem": False,
        "horizon_40": False,
        "trained": False,
        "goal_height0": GOAL_HEIGHT0,
        "goal_vy": float(V_TARGET_LOOP),
        "goal_fuel": "current",
        "goal_x": "pad_x",
        "goal_vx": 0.0,
        "cost": "weighted mean L1 of predicted latent to ema(scale_rel(target_state))",
        "step_weights": [float(weight) for weight in LOSS_WEIGHTS],
        "tie_break": "lowest sequence index",
        "step_origin": 0,
        "switch_is_teacher_shift": False,
        "teacher_timing_recomputed": False,
        "head_retrained": False,
        "encoder": SOURCE_NAME,
        "skills": {"approach": "landk_approach_s0.pt", "land": "landk_land_s0.pt"},
        "v_target": float(V_TARGET_LOOP),
        "seeds": [LOOP_SEED0, LOOP_SEED0 + LOOP_N - 1],
        "action_gap": gaps,
        "published_soft": dict(PUBLISHED_SOFT),
        "soft_line": SOFT_LINE,
        "cell_line": PART_LINE,
        "readings": {
            "useful": READING_USEFUL,
            "partial": READING_PARTIAL,
            "mixed": READING_MIXED,
            "weak": READING_WEAK,
        },
        "rows": rows,
    }
    if _stamps(paths) != before:
        raise SystemExit("перед записью старый файл изменился")
    write_json(OUT_PATH, payload)
    if _stamps(paths) != before:
        raise SystemExit("запись отчёта задела старый файл")
    print(os.path.abspath(OUT_PATH), flush=True)
    for row in rows:
        print(f"итог {row['head']}: {row['reading']}", flush=True)


if __name__ == "__main__":
    main()

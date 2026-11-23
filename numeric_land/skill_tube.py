"""Поиск посадки по трубе правильных состояний. Обучения нет.

Труба на каждом шаге контура: три вызова hover_action от текущего состояния,
затем scale_rel и EMA landr_scratch_s0. Действие берёт argmin по 729 планам.
Терминальной точки на земле в стоимости нет.

    python -m numeric_land.skill_tube --tag numeric-land-skill-tube
"""
import argparse
import glob
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.expert_hover import HOVER_CAP, hover_action
from numeric_land.field import FUEL0, GROUND, step_field
from numeric_land.models import LATENT, LOSS_WEIGHTS, LandPredictor
from numeric_land.pad import outcome_at
from numeric_land.relenc import (
    OUT, ROOT, RelEncoder, encode_rel, file_stamp, scale_rel, tensors_equal, write_json,
)
from numeric_land.relenc import TAG as ENCODER_TAG
from numeric_land.skill_land import (
    GRID_N, HORIZON, LINE, N_SEQ, TAG as LAND_TAG, grid, line_ok, pair_table, world_batch,
)
from numeric_land.skills import N_WORLD
from numeric_land.train_goal import EMA_MOMENTUM, THREADS
from numeric_land.train_relenc import ckpt_path as relenc_ckpt_path
from numeric_land.train_skills import ckpt_path as land_ckpt_path

TAG = "numeric-land-skill-tube"
TUBE_TARGETS = 3
SPREAD_MIN = 0.00140
GOAL_VY = 6.0
LEFT_FIRST = (0, 3, 6)
RIGHT_FIRST = (2, 5, 8)
SOURCE_NAME = "landr_scratch_s0.pt"
PAD_NAMES = ("landk_pad_s0.pt", "landk_pad_s1.pt")
BASELINE_NAME = "landk_land_s0.pt"
JSON_NAME = "numeric_land_skill_tube.json"
DEMO_NAMES = ("demo_ui.py", "_land_demo.py", "_pad_demo.py")
# Шесть состояний замера разброса: центр 256, vy0 = 6, dx = +6,
# тяжести 2/8/15 и высоты 32/128. Нижняя граница того замера — 0.00140.
SPREAD_SPEC = (
    (2.0, 256.0, 6.0, 32.0, 6.0),
    (2.0, 256.0, 6.0, 128.0, 6.0),
    (8.0, 256.0, 6.0, 32.0, 6.0),
    (8.0, 256.0, 6.0, 128.0, 6.0),
    (15.0, 256.0, 6.0, 32.0, 6.0),
    (15.0, 256.0, 6.0, 128.0, 6.0),
)
# Знак из спеки: g=15, h=32, vy0=6, центр 256, dx = ±6.
SIGN_SPEC = (
    (15.0, 256.0, 6.0, 32.0, 6.0),
    (15.0, 256.0, -6.0, 32.0, 6.0),
)


def locks() -> None:
    if TAG != "numeric-land-skill-tube" or ENCODER_TAG == TAG or LAND_TAG == TAG:
        raise RuntimeError(TAG)
    if HORIZON != 3 or N_SEQ != 729 or TUBE_TARGETS != 3:
        raise RuntimeError((HORIZON, N_SEQ, TUBE_TARGETS))
    if LOSS_WEIGHTS != (1.0, 0.5, 0.25) or len(LOSS_WEIGHTS) != TUBE_TARGETS:
        raise RuntimeError(LOSS_WEIGHTS)
    if LINE != 114 or GRID_N != 162 or GOAL_VY != 6.0:
        raise RuntimeError((LINE, GRID_N, GOAL_VY))
    if abs(SPREAD_MIN - 0.00140) > 0.0 or int(HOVER_CAP) != 900:
        raise RuntimeError((SPREAD_MIN, HOVER_CAP))
    if len(SPREAD_SPEC) != 6 or len(SIGN_SPEC) != 2:
        raise RuntimeError((len(SPREAD_SPEC), len(SIGN_SPEC)))
    if float(FUEL0) != 10.0 or float(GROUND) != 512.0:
        raise RuntimeError((FUEL0, GROUND))
    if int(LATENT) != 32 or int(N_WORLD) != 4 or THREADS != 8:
        raise RuntimeError((LATENT, N_WORLD, THREADS))


def refuse_tag(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "нужен тег numeric-land-skill-tube. Другие теги запрещены.")


def json_path() -> str:
    return os.path.join(OUT, JSON_NAME)


def refuse_report(path: str) -> None:
    """Второй прогон готовый отчёт не открывает на запись."""
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}, второй прогон отчёт не затирает")


def row_of(g, pad_x, dx, height, vy):
    """Старт как в сетке: x = центр + dx, y = 512 − высота, vx = 0, бак 10."""
    y = float(GROUND) - float(height)
    if not y < float(GROUND):
        raise RuntimeError(y)
    return {
        "g": float(g),
        "pad_x": float(pad_x),
        "dx": float(dx),
        "height": float(height),
        "vy": float(vy),
        "x": float(pad_x) + float(dx),
        "y": y,
        "vx": 0.0,
        "fuel": float(FUEL0),
    }


def roll_tube(state, g, pad_x):
    """Три состояния правила от текущего s0. Касание раньше третьего шага повторяет s_j.

    hover_action здесь только цель. Возвращённые координаты в эпизод не пишутся.
    """
    x, y, vx, vy, fuel = (float(v) for v in state)
    frames = []
    calls = 0
    held = None
    clipped = False
    for k in range(TUBE_TARGETS):
        if held is not None:
            frames.append(held)
            continue
        calls += 1
        action = int(hover_action((x, y, vx, vy, fuel), float(g), float(pad_x), GOAL_VY))
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, False, action, float(g))
        held_state = (float(x), float(y), float(vx), float(vy), float(fuel))
        frames.append(held_state)
        if grounded and k < TUBE_TARGETS - 1:
            held = held_state
            clipped = True
    if len(frames) != TUBE_TARGETS or calls < 1 or calls > TUBE_TARGETS:
        raise RuntimeError((len(frames), calls, clipped))
    if clipped and held is None:
        raise RuntimeError("обрезка без состояния касания")
    return tuple(frames), int(calls), bool(clipped)


def tube_latents(ema, frames, pad_x):
    """scale_rel и EMA. Онлайн-кодировщик в цель трубы не входит."""
    raw = np.asarray(frames, np.float64)
    if raw.shape != (TUBE_TARGETS, 5):
        raise RuntimeError(raw.shape)
    pads = np.full(TUBE_TARGETS, float(pad_x), np.float64)
    scaled = scale_rel(raw, pads)
    if scaled.shape != (TUBE_TARGETS, 7):
        raise RuntimeError(scaled.shape)
    latent = ema(torch.from_numpy(np.ascontiguousarray(scaled)))
    if latent.shape != (TUBE_TARGETS, LATENT):
        raise RuntimeError(latent.shape)
    return latent


def plan_costs(predictor, z, z_tube, world_b, step_pairs):
    """Σ w_k · mean|z_k − z_tube_k|. Одна точка на земле сюда не подмешивается."""
    if z.shape != (1, LATENT) or z_tube.shape != (TUBE_TARGETS, LATENT):
        raise RuntimeError((tuple(z.shape), tuple(z_tube.shape)))
    if world_b.shape != (N_SEQ, int(N_WORLD)) or len(step_pairs) != HORIZON:
        raise RuntimeError((tuple(world_b.shape), len(step_pairs)))
    current = z.expand(N_SEQ, -1)
    cost = None
    for k, weight in enumerate(LOSS_WEIGHTS):
        current = predictor(current, step_pairs[k], world_b)
        step = (current - z_tube[k]).abs().mean(dim=-1)
        cost = step.mul(float(weight)) if cost is None else cost.add(step, alpha=float(weight))
    if cost.shape != (N_SEQ,):
        raise RuntimeError(cost.shape)
    return cost


def choose_action(predictor, z, z_tube, world_b, step_pairs, actions):
    """argmin. При равной стоимости остаётся меньший индекс последовательности."""
    cost = plan_costs(predictor, z, z_tube, world_b, step_pairs)
    index = int(torch.argmin(cost).item())
    return int(actions[index, 0].item()), cost


def side_mins(cost, actions):
    first = actions[:, 0]
    left = None
    right = None
    for index, value in enumerate(cost.detach().tolist()):
        action = int(first[index].item())
        if action in LEFT_FIRST:
            left = value if left is None else min(left, value)
        elif action in RIGHT_FIRST:
            right = value if right is None else min(right, value)
    if left is None or right is None:
        raise RuntimeError((left, right))
    return float(left), float(right)


def sign_ok(cost, actions, dx) -> bool:
    """dx > 0 — аппарат правее центра, дешевле первое действие влево. dx < 0 — наоборот."""
    left, right = side_mins(cost, actions)
    if float(dx) > 0.0:
        return left < right
    if float(dx) < 0.0:
        return right < left
    raise RuntimeError(dx)


def cost_spread(cost) -> float:
    arr = cost.detach().float().numpy()
    return float(np.percentile(arr, 95) - np.percentile(arr, 5))


def _encode_now(encoder, x, y, vx, vy, fuel, pad_x):
    state = np.asarray([[x, y, vx, vy, fuel]], np.float64)
    pads = np.asarray([float(pad_x)], np.float64)
    latent = encode_rel(encoder, state, pads)
    tensor = torch.from_numpy(np.ascontiguousarray(latent))
    if tensor.shape != (1, LATENT):
        raise RuntimeError(tensor.shape)
    return tensor


def prepare(encoder, ema, row):
    """Труба от текущего состояния. Координаты row не меняются."""
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    fuel = float(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    before = (x, y, vx, vy, fuel)
    frames, calls, clipped = roll_tube(before, g, pad_x)
    if (x, y, vx, vy, fuel) != before:
        raise RuntimeError("труба подменила состояние")
    z = _encode_now(encoder, x, y, vx, vy, fuel, pad_x)
    z_tube = tube_latents(ema, frames, pad_x)
    return z, z_tube, frames, calls, clipped


@torch.no_grad()
def closed_step(predictor, encoder, ema, actions, step_pairs, world_b, row):
    """Один шаг контура: цель из трубы, физика — step_field выбранного действия."""
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    fuel = float(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    before = (x, y, vx, vy, fuel)
    z, z_tube, frames, calls, clipped = prepare(encoder, ema, row)
    if (float(row["x"]), float(row["y"]), float(row["vx"]), float(row["vy"]), float(row["fuel"])) != before:
        raise RuntimeError("труба записалась в эпизод")
    action, _cost = choose_action(predictor, z, z_tube, world_b, step_pairs, actions)
    nxt = step_field(x, y, vx, vy, fuel, False, int(action), g)
    return nxt, int(action), bool(clipped), int(calls), frames


@torch.no_grad()
def fly_one(predictor, encoder, ema, actions, step_pairs, world_b, row):
    """Эпизод до земли или 900. no_grad: encode_rel зовёт numpy."""
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    fuel = float(row["fuel"])
    g = float(row["g"])
    pad_x = float(row["pad_x"])
    grounded = False
    steps = 0
    clipped_n = 0
    while steps < int(HOVER_CAP) and not grounded:
        current = {
            "x": x, "y": y, "vx": vx, "vy": vy, "fuel": fuel,
            "g": g, "pad_x": pad_x,
        }
        nxt, action, clipped, calls, _frames = closed_step(
            predictor, encoder, ema, actions, step_pairs, world_b, current)
        if clipped:
            clipped_n += 1
        if calls < 1:
            raise RuntimeError(calls)
        x, y, vx, vy, fuel, grounded = nxt
        steps += 1
        del action
    result = outcome_at(x, y, vy, grounded, pad_x)
    if result not in ("мягкая", "удар", "не сел"):
        raise RuntimeError(result)
    return result, int(steps), int(clipped_n)


@torch.no_grad()
def fly_grid(name, predictor, encoder, ema):
    actions, step_pairs = pair_table()
    predictor.eval()
    encoder.eval()
    ema.eval()
    buckets = {}
    order = []
    worlds = {}
    soft = crash = miss = 0
    clipped_n = 0
    current = None
    for row in grid():
        key = (row["g"], row["pad_x"], row["height"])
        if key != current:
            if current is not None:
                if buckets[current]["n"] != 9:
                    raise RuntimeError((current, buckets[current]["n"]))
                _emit(name, current, buckets[current])
            current = key
            if key not in buckets:
                order.append(key)
                buckets[key] = {
                    "soft": 0, "crash": 0, "miss": 0, "n": 0, "steps": 0, "tube_clipped": 0,
                }
            print(f"{name} старт g={key[0]:g} pad={key[1]:g} h={key[2]:g}", flush=True)
        world_key = (row["g"], row["pad_x"])
        if world_key not in worlds:
            worlds[world_key] = world_batch(row["g"], row["pad_x"])
        result, n_steps, n_clip = fly_one(
            predictor, encoder, ema, actions, step_pairs, worlds[world_key], row)
        print(
            f"{name} эпизод {int(buckets[key]['n']) + 1}/9 steps={n_steps} {result}",
            flush=True,
        )
        cell = buckets[key]
        cell["n"] += 1
        cell["steps"] += int(n_steps)
        cell["tube_clipped"] += int(n_clip)
        clipped_n += int(n_clip)
        if result == "мягкая":
            cell["soft"] += 1
            soft += 1
        elif result == "удар":
            cell["crash"] += 1
            crash += 1
        else:
            cell["miss"] += 1
            miss += 1
    if current is not None:
        if buckets[current]["n"] != 9:
            raise RuntimeError((current, buckets[current]["n"]))
        _emit(name, current, buckets[current])
    cells = []
    for key in order:
        g, pad, height = key
        cell = buckets[key]
        if cell["n"] != 9 or cell["soft"] + cell["crash"] + cell["miss"] != 9:
            raise RuntimeError((key, cell))
        cells.append({
            "g": float(g),
            "pad_x": float(pad),
            "height": float(height),
            "n": 9,
            "soft": int(cell["soft"]),
            "crash": int(cell["crash"]),
            "miss": int(cell["miss"]),
            "steps_mean": round(float(cell["steps"]) / 9.0, 3),
            "tube_clipped": int(cell["tube_clipped"]),
        })
    if soft + crash + miss != GRID_N or len(cells) != 18:
        raise RuntimeError((soft, crash, miss, len(cells)))
    return {
        "name": name,
        "soft": int(soft),
        "crash": int(crash),
        "miss": int(miss),
        "of": int(GRID_N),
        "tube_clipped": int(clipped_n),
        "cells": cells,
    }


def _emit(name, key, cell) -> None:
    g, pad, height = key
    print(
        f"{name} g={g:g} pad={pad:g} h={height:g} "
        f"soft={cell['soft']} crash={cell['crash']} miss={cell['miss']} "
        f"of={cell['n']} tube_clipped={cell['tube_clipped']}",
        flush=True,
    )


def scratch_path() -> str:
    path = relenc_ckpt_path("scratch", 0)
    if os.path.basename(path) != SOURCE_NAME:
        raise RuntimeError(path)
    return path


def pad_path(seed: int) -> str:
    seed = int(seed)
    if seed not in (0, 1):
        raise ValueError(seed)
    path = os.path.join(OUT, f"landk_pad_s{seed}.pt")
    if os.path.basename(path) not in PAD_NAMES:
        raise RuntimeError(path)
    return path


def baseline_path() -> str:
    path = land_ckpt_path("land", 0)
    if os.path.basename(path) != BASELINE_NAME:
        raise RuntimeError(path)
    return path


def watched_paths():
    paths = {
        SOURCE_NAME: scratch_path(),
        PAD_NAMES[0]: pad_path(0),
        PAD_NAMES[1]: pad_path(1),
        BASELINE_NAME: baseline_path(),
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


def load_frozen():
    """Чтение. Штамп до и после load обязан совпасть."""
    path = scratch_path()
    before = file_stamp(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if file_stamp(path) != before:
        raise RuntimeError("чтение кодировщика изменило файл")
    if ck.get("kind") != "scratch" or int(ck.get("seed", -1)) != 0:
        raise RuntimeError(path)
    if ck.get("tag") != ENCODER_TAG:
        raise RuntimeError(ck.get("tag"))
    encoder = RelEncoder()
    encoder.load_state_dict(ck["encoder"])
    encoder.requires_grad_(False)
    encoder.eval()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    ema.load_state_dict(ck["ema"])
    for param in ema.parameters():
        param.requires_grad_(False)
    ema.eval()
    return ck, encoder, ema


def load_predictor(path, expect_phase, expect_seed, expect_tag, source):
    before = file_stamp(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if file_stamp(path) != before:
        raise RuntimeError(f"чтение {os.path.basename(path)} изменило файл")
    if ck.get("tag") != expect_tag:
        raise RuntimeError((path, ck.get("tag")))
    if ck.get("phase") != expect_phase or int(ck.get("seed", -1)) != int(expect_seed):
        raise RuntimeError((path, ck.get("phase"), ck.get("seed")))
    if int(ck.get("n_world", -1)) != int(N_WORLD):
        raise RuntimeError(ck.get("n_world"))
    if not tensors_equal(ck["encoder"], source["encoder"]):
        raise RuntimeError(f"кодировщик {os.path.basename(path)} не совпал с {SOURCE_NAME}")
    if not tensors_equal(ck["ema"], source["ema"]):
        raise RuntimeError(f"EMA {os.path.basename(path)} не совпала с {SOURCE_NAME}")
    predictor = LandPredictor(N_WORLD)
    predictor.load_state_dict(ck["predictor"])
    predictor.requires_grad_(False)
    predictor.eval()
    return predictor


def tube_files():
    return glob.glob(os.path.join(OUT, "landk_tube_*.pt"))


def _row(name, flown, scored):
    row = {
        "name": name,
        "scored": bool(scored),
        "soft": int(flown["soft"]),
        "crash": int(flown["crash"]),
        "miss": int(flown["miss"]),
        "of": int(GRID_N),
        "tube_clipped": int(flown["tube_clipped"]),
        "cells": flown["cells"],
    }
    if scored:
        row["line_ok"] = bool(line_ok(row["soft"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    refuse_tag(args.tag)
    locks()
    torch.set_num_threads(THREADS)
    report = json_path()
    refuse_report(report)
    if tube_files():
        raise SystemExit("уже есть landk_tube_*.pt, чекпоинт трубы писать нельзя")
    before = take_stamps()
    source, encoder, ema = load_frozen()
    pad = load_predictor(pad_path(0), "pad", 0, LAND_TAG, source)
    seed1 = load_predictor(pad_path(1), "pad", 1, LAND_TAG, source)
    base = load_predictor(baseline_path(), "land", 0, ENCODER_TAG, source)
    del seed1
    if take_stamps() != before:
        raise RuntimeError("чтение сдвинуло штамп")
    print(f"полёт landk_pad_s0, труба, {GRID_N} стартов", flush=True)
    pad_flown = fly_grid("landk_pad_s0", pad, encoder, ema)
    print(f"полёт landk_land_s0, труба, {GRID_N} стартов", flush=True)
    base_flown = fly_grid("landk_land_s0", base, encoder, ema)
    if take_stamps() != before:
        raise RuntimeError("полёт сдвинул штамп")
    if tube_files():
        raise RuntimeError("появился landk_tube_*.pt")
    pad_row = _row("landk_pad_s0", pad_flown, True)
    base_row = _row("landk_land_s0", base_flown, False)
    if "line_ok" in base_row:
        raise RuntimeError("база закрыла бы линию")
    if bool(pad_row["line_ok"]) != bool(line_ok(pad_row["soft"])):
        raise RuntimeError("линия разошлась")
    payload = {
        "tag": TAG,
        "horizon": HORIZON,
        "n_sequences": N_SEQ,
        "cem": False,
        "grid_n": GRID_N,
        "line": LINE,
        "hover_action_action": False,
        "tube_targets": TUBE_TARGETS,
        "tube_clipped": int(pad_row["tube_clipped"] + base_row["tube_clipped"]),
        "stamps_unchanged": True,
        "encoder": SOURCE_NAME,
        "seed1_flown": False,
        "decision_loaded": False,
        "approach_loaded": False,
        "passed": bool(pad_row["line_ok"]),
        "prior_landk_pad_s0_soft": 4,
        "rule_ceiling_soft": 156,
        "rows": [pad_row, base_row],
    }
    write_json(report, payload)
    print(
        f"итог landk_pad_s0 {pad_row['soft']}/{GRID_N} линия {LINE} "
        f"{'пройдена' if payload['passed'] else 'не пройдена'}; "
        f"landk_land_s0 {base_row['soft']}/{GRID_N} в линию не входит; "
        f"tube_clipped {payload['tube_clipped']}",
        flush=True,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

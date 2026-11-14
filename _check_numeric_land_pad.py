"""Контроль центра площадки без обучения. python _check_numeric_land_pad.py"""
import os
import subprocess
import sys

import numpy as np
import torch

from numeric_land.field import FUEL0, GROUND, start_and_actions, step_field
from numeric_land.models import N_WORLD, LandPredictor, action_pairs, predictor_input
from numeric_land.pad import (
    PAD_HALF, TRAIN_PAD_HI, TRAIN_PAD_LO, make_batch, outcome_at, pad_of_window,
    physics_step, world_vector,
)
from numeric_land.plan_pad import step_cost
from numeric_land.train_pad import OUT, refuse_tag


def close(got, exp, tol=1e-6):
    if abs(float(got) - float(exp)) > tol:
        raise SystemExit(f"{got} != {exp}")


def main():
    a = physics_step(256.0, 464.0, 0.0, 0.0, FUEL0, False, 1, 8.0)
    b = step_field(256.0, 464.0, 0.0, 0.0, FUEL0, False, 1, 8.0)
    if a != b:
        raise SystemExit("обёртка разошлась с интегратором")
    if physics_step(256.0, 464.0, 0.0, 0.0, FUEL0, False, 1, 8.0) != a:
        raise SystemExit("повторный шаг разошёлся")

    if outcome_at(250.0, GROUND, 7.9, True, 256.0) != "мягкая":
        raise SystemExit("старый центр не узнал площадку")
    if outcome_at(240.0, GROUND, 7.9, True, 256.0) != "удар":
        raise SystemExit("мимо отрезка")
    if outcome_at(250.0, GROUND, 8.0, True, 256.0) != "удар":
        raise SystemExit("порог включил 8")
    if outcome_at(300.0, GROUND, 1.0, True, 300.0) != "мягкая":
        raise SystemExit("свой центр")
    if outcome_at(290.0, GROUND, 1.0, True, 300.0) != "удар":
        raise SystemExit("полуширина")
    if outcome_at(256.0, 480.0, 0.0, False, 256.0) != "не сел":
        raise SystemExit("зависание")
    if PAD_HALF != 8.0:
        raise SystemExit("полуширина")
    lo = 256.0 - PAD_HALF
    hi = 256.0 + PAD_HALF
    if (lo, hi) != (248.0, 264.0):
        raise SystemExit("отрезок 256")

    first = pad_of_window(93000)
    if pad_of_window(93000) != first:
        raise SystemExit("центр окна не повторился")
    if not (TRAIN_PAD_LO <= first < TRAIN_PAD_HI):
        raise SystemExit("центр вне [156, 356)")
    for seed in range(20):
        pad = pad_of_window(seed)
        if not (TRAIN_PAD_LO <= pad < TRAIN_PAD_HI):
            raise SystemExit(pad)
    before = start_and_actions(93000, 4)
    pad_of_window(93000)
    after = start_and_actions(93000, 4)
    if before[0] != after[0] or before[4].tolist() != after[4].tolist():
        raise SystemExit("центр съел генератор старта")
    x2, y2, vx2, vy2, a2 = start_and_actions(93000, 4)
    x20, y20, vx20, vy20, a20 = start_and_actions(93000, 4)
    if (x2, y2, vx2, vy2, a2.tolist()) != (x20, y20, vx20, vy20, a20.tolist()):
        raise SystemExit("старт зависит от вызова")
    if not (240.0 <= x2 <= 272.0 and 464.0 <= y2 <= 488.0):
        raise SystemExit("старт уехал")

    world = world_vector(np.array([9.0]), np.array([256.0]))
    if world.shape != (1, 3):
        raise SystemExit(world.shape)
    if world.dtype != np.float32:
        raise SystemExit(world.dtype)
    close(world[0, 0], 1.0)
    close(world[0, 1], 1.0)
    close(world[0, 2], 0.5)
    world4 = world_vector(np.array([4.0]), np.array([512.0]))
    close(world4[0, 0], np.float32(4.0 / 9.0))
    close(world4[0, 2], 1.0)

    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.zeros(1, 32) + 0.2, dim=-1)
    pair = action_pairs(torch.tensor([8]), z)
    vis = predictor_input(z, pair, torch.from_numpy(world), "vis")
    hid = predictor_input(z, pair, torch.from_numpy(world), "hid")
    if vis.shape[-1] != 37 or hid.shape[-1] != 34:
        raise SystemExit((vis.shape, hid.shape))
    if any(abs(float(v) - 12.0) < 1e-6 for v in vis.reshape(-1)):
        raise SystemExit("во входе оказалось 12")
    if abs(float(vis[0, -1]) - 0.5) > 1e-6 or abs(float(vis[0, -2]) - 1.0) > 1e-6:
        raise SystemExit("тройка мира")
    if LandPredictor(3).n_world != 3 or LandPredictor(0).n_world != 0:
        raise SystemExit("ширина предиктора")
    if N_WORLD["vis"] != 2 or N_WORLD["hid"] != 0:
        raise SystemExit("поле 512 сменило ширину мира")

    states, _acts, gs, pads = make_batch(np.random.default_rng(0), 128, "range")
    if states.shape != (128, 4, 5) or pads.shape != (128,):
        raise SystemExit("батч")
    if pads.min() < TRAIN_PAD_LO or pads.max() >= TRAIN_PAD_HI:
        raise SystemExit("батч вышел из диапазона центра")
    if len(set(np.round(gs, 5))) < 2:
        raise SystemExit("батч тяжести не разнообразен")

    pred = torch.zeros(2, 5)
    pred[:, 0] = 400.0
    pred[:, 1] = GROUND
    pred[:, 4] = torch.tensor([0.0, 10.0])
    here = step_cost(pred, 400.0)
    there = step_cost(pred, 256.0)
    if float(here[0]) != 0.0 or float(here[1]) != 0.0:
        raise SystemExit(f"свой центр не нуль: {here}")
    if float(here[0]) != float(here[1]):
        raise SystemExit("топливо вошло в стоимость")
    close(float(there[0]), (144.0 / 512.0) ** 2)
    if float(there[0]) != float(there[1]):
        raise SystemExit("топливо вошло в чужой центр")

    try:
        refuse_tag("numeric-land-field")
    except SystemExit:
        pass
    else:
        raise SystemExit("тег поля 512 принят")

    root = os.path.dirname(os.path.abspath(__file__))
    landp = os.path.join(OUT, "landp_range_vis_s0.pt")
    landf = os.path.join(OUT, "landf_range_vis_s0.pt")
    before_p = os.stat(landp) if os.path.isfile(landp) else None
    before_f = os.stat(landf) if os.path.isfile(landf) else None
    proc = subprocess.run(
        [sys.executable, "-m", "numeric_land.train_pad", "--tag", "numeric-land-field"],
        cwd=root, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        raise SystemExit("чужой тег завершился нулём")
    after_p = os.stat(landp) if os.path.isfile(landp) else None
    created = before_p is None and after_p is not None
    touched = (
        before_p is not None and after_p is not None
        and (before_p.st_mtime_ns, before_p.st_size) != (after_p.st_mtime_ns, after_p.st_size)
    )
    if created or touched:
        raise SystemExit("чужой тег создал чекпоинт")
    proc_field = subprocess.run(
        [sys.executable, "-m", "numeric_land.train_field", "--tag", "numeric-land-pad"],
        cwd=root, capture_output=True, text=True,
    )
    if proc_field.returncode == 0:
        raise SystemExit("поле 512 приняло новый тег")
    after_f = os.stat(landf) if os.path.isfile(landf) else None
    if before_f is None or after_f is None:
        raise SystemExit("нет исходного кодировщика")
    if (before_f.st_mtime_ns, before_f.st_size) != (after_f.st_mtime_ns, after_f.st_size):
        raise SystemExit("поле 512 переписало свой чекпоинт")
    proc64 = subprocess.run(
        [sys.executable, "-m", "numeric_land.train", "--tag", "numeric-land-pad"],
        cwd=root, capture_output=True, text=True,
    )
    if proc64.returncode == 0:
        raise SystemExit("поле 64 приняло новый тег")

    run = subprocess.run([sys.executable, os.path.join(root, "_check_numeric_land_field.py")], cwd=root)
    if run.returncode != 0:
        raise SystemExit("_check_numeric_land_field.py не прошёл")
    print("numeric land pad: ок")


if __name__ == "__main__":
    main()

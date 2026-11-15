"""Контроль цели без обучения. python _check_numeric_land_goal.py"""
import os
import subprocess
import sys

import numpy as np
import torch

from numeric_land.field import FUEL0, GROUND, SCALE_VEL, SCALE_XY, start_and_actions, step_field
from numeric_land.goal import (
    DESIRED_VY, GOAL_DIM, N_WORLD, TRAIN_PAD_HI, TRAIN_PAD_LO, goal_of_window,
    make_batch, outcome_at, physics_step, target_state, world_vector,
)
from numeric_land.models import LandPredictor, predictor_input
from numeric_land.plan_goal import step_cost
from numeric_land.train_goal import OUT, refuse_tag


def close(got, exp, tol=1e-6):
    if abs(float(got) - float(exp)) > tol:
        raise SystemExit(f"{got} != {exp}")


def main():
    # 1. Физика от цели не зависит
    a = physics_step(256.0, 464.0, 0.0, 0.0, FUEL0, False, 1, 8.0)
    b = step_field(256.0, 464.0, 0.0, 0.0, FUEL0, False, 1, 8.0)
    if a != b:
        raise SystemExit("обёртка разошлась с интегратором")
    if physics_step(256.0, 464.0, 0.0, 0.0, FUEL0, False, 1, 8.0) != a:
        raise SystemExit("повторный шаг разошёлся")

    # 2. Успех зависит только от pad_x
    if outcome_at(202.0, GROUND, 7.9, True, 200.0) != "мягкая":
        raise SystemExit("свой центр не узнал площадку")
    if outcome_at(215.0, GROUND, 7.9, True, 200.0) != "удар":
        raise SystemExit("мимо отрезка")
    if outcome_at(202.0, GROUND, 8.0, True, 200.0) != "удар":
        raise SystemExit("порог включил 8")
    if outcome_at(202.0, 480.0, 0.0, False, 200.0) != "не сел":
        raise SystemExit("зависание")

    # 3. Цель окна воспроизводима и в диапазоне
    first = goal_of_window(97000)
    if goal_of_window(97000) != first:
        raise SystemExit("цель окна не повторилась")
    if not (TRAIN_PAD_LO <= first[0] < TRAIN_PAD_HI):
        raise SystemExit("центр вне [156, 356)")
    if first[1] not in DESIRED_VY:
        raise SystemExit("desired_vy вне набора")
    for seed in range(50):
        pad, vy = goal_of_window(seed)
        if not (TRAIN_PAD_LO <= pad < TRAIN_PAD_HI):
            raise SystemExit(pad)
        if vy not in DESIRED_VY:
            raise SystemExit(vy)

    # 4. Цель не съедает генератор старта
    before = start_and_actions(97000, 4)
    goal_of_window(97000)
    after = start_and_actions(97000, 4)
    if before[0] != after[0] or before[4].tolist() != after[4].tolist():
        raise SystemExit("цель съела генератор старта")
    x2, y2, vx2, vy2, a2 = start_and_actions(97000, 4)
    if not (240.0 <= x2 <= 272.0 and 464.0 <= y2 <= 488.0):
        raise SystemExit("старт уехал")

    # 5. Четвёрка мира
    world = world_vector(np.array([9.0]), np.array([256.0]), np.array([0.0]))
    if world.shape != (1, 4):
        raise SystemExit(world.shape)
    if world.dtype != np.float32:
        raise SystemExit(world.dtype)
    close(world[0, 0], 1.0)
    close(world[0, 1], 1.0)
    close(world[0, 2], 0.5)
    close(world[0, 3], 0.0)
    w2 = world_vector(np.array([4.0]), np.array([512.0]), np.array([90.0]))
    close(w2[0, 0], np.float32(4.0 / 9.0))
    close(w2[0, 2], 1.0)
    close(w2[0, 3], np.float32(90.0 / SCALE_VEL))

    # 6. Ширина предиктора vis/hid
    z = torch.nn.functional.normalize(torch.zeros(1, 32) + 0.2, dim=-1)
    pair = __import__("numeric_land.models", fromlist=["action_pairs"]).action_pairs(torch.tensor([8]), z)
    vis = predictor_input(z, pair, torch.from_numpy(world), "vis")
    hid = predictor_input(z, pair, torch.from_numpy(world), "hid")
    if vis.shape[-1] != 38 or hid.shape[-1] != 34:
        raise SystemExit((vis.shape, hid.shape))
    if LandPredictor(N_WORLD["vis"]).n_world != 4 or LandPredictor(N_WORLD["hid"]).n_world != 0:
        raise SystemExit("ширина предиктора")
    if GOAL_DIM != 2:
        raise SystemExit("goal_dim")

    # 7. Целевое состояние под терминальный лосс
    tgt = target_state(200.0, 6.0, 30.0)
    if tgt.shape != (1, 5):
        raise SystemExit(tgt.shape)
    close(tgt[0, 0], 200.0)
    close(tgt[0, 1], GROUND - 30.0)
    close(tgt[0, 3], 6.0)
    close(tgt[0, 4], FUEL0)

    # 8. Стоимость стоит в нуле над своим центром
    pred = torch.zeros(2, 5)
    pred[:, 0] = 400.0
    pred[:, 1] = GROUND
    pred[:, 4] = torch.tensor([0.0, 10.0])
    here = step_cost(pred, 400.0)
    there = step_cost(pred, 256.0)
    if float(here[0]) != 0.0:
        raise SystemExit(f"свой центр не нуль: {here}")
    close(float(there[0]), (144.0 / 512.0) ** 2)
    if float(here[0]) != float(here[1]):
        raise SystemExit("топливо вошло в стоимость")

    # 9. Батч
    states, _acts, gs, pads, vys = make_batch(np.random.default_rng(0), 128, "range")
    if states.shape != (128, 4, 5) or pads.shape != (128,) or vys.shape != (128,):
        raise SystemExit("батч")
    if pads.min() < TRAIN_PAD_LO or pads.max() >= TRAIN_PAD_HI:
        raise SystemExit("батч вышел из диапазона центра")
    if set(np.unique(vys)) - set(DESIRED_VY):
        raise SystemExit("батч вышел из набора desired_vy")
    if len(set(np.round(gs, 5))) < 2:
        raise SystemExit("батч тяжести не разнообразен")

    # 10. Чужие теги отклоняются
    for bad in ("numeric-land-pad", "numeric-land-field"):
        try:
            refuse_tag(bad)
        except SystemExit:
            pass
        else:
            raise SystemExit(f"тег {bad} принят")

    root = os.path.dirname(os.path.abspath(__file__))
    landg = os.path.join(OUT, "landg_range_vis_s0.pt")
    landp = os.path.join(OUT, "landp_range_vis_s0.pt")
    before_p = os.stat(landp) if os.path.isfile(landp) else None
    proc = subprocess.run(
        [sys.executable, "-m", "numeric_land.train_pad", "--tag", "numeric-land-goal"],
        cwd=root, capture_output=True, text=True)
    if proc.returncode == 0:
        raise SystemExit("площадка приняла тег цели")
    after_p = os.stat(landp) if os.path.isfile(landp) else None
    if before_p is not None and after_p is not None:
        if (before_p.st_mtime_ns, before_p.st_size) != (after_p.st_mtime_ns, after_p.st_size):
            raise SystemExit("площадка переписала свой чекпоинт")
    proc2 = subprocess.run(
        [sys.executable, "-m", "numeric_land.train_field", "--tag", "numeric-land-goal"],
        cwd=root, capture_output=True, text=True)
    if proc2.returncode == 0:
        raise SystemExit("поле 512 приняло тег цели")

    run = subprocess.run([sys.executable, os.path.join(root, "_check_numeric_land_pad.py")], cwd=root)
    if run.returncode != 0:
        raise SystemExit("_check_numeric_land_pad.py не прошёл")
    print("numeric land goal: ок")


if __name__ == "__main__":
    main()
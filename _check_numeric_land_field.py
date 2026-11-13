"""Контроль большого поля без обучения. python _check_numeric_land_field.py"""
import os
import subprocess
import sys

import numpy as np
import torch

from numeric_land.field import (
    FUEL0, GROUND, PAD_HI, PAD_LO,
    make_batch, outcome, scale_state, start_and_actions, step_field, world_vector,
)
from numeric_land.models import N_WORLD, LandPredictor, action_pairs, predictor_input
from numeric_land.plan_field import step_cost
from numeric_land.train_field import OUT, refuse_tag


def close(got, exp, tol=1e-9):
    if abs(float(got) - float(exp)) > tol:
        raise SystemExit(f"{got} != {exp}")


def main():
    x, y, vx, vy, fuel, hit = step_field(256.0, 256.0, 0.0, 0.0, FUEL0, False, 1, 9.0)
    close(vx, 0.0)
    close(vy, 0.3)
    close(y - 256.0, 0.01)
    close(fuel, 10.0)
    if hit:
        raise SystemExit("свободное падение названо касанием")

    slow = step_field(256.0, 256.0, 0.0, 0.0, FUEL0, False, 1, 2.0)
    fast = step_field(256.0, 256.0, 0.0, 0.0, FUEL0, False, 1, 15.0)
    close(fast[3] - slow[3], 13.0 / 30.0)
    close(fast[1] - slow[1], 13.0 / 900.0)

    touch = step_field(256.0, 25599.0 / 50.0, 0.0, 1.0, FUEL0, False, 1, 9.0)
    close(touch[0], 256.0)
    close(touch[1], GROUND)
    close(touch[3], 13.0 / 10.0)
    close(touch[2], 0.0)
    if not touch[5]:
        raise SystemExit("касание не отмечено")
    frozen = step_field(*touch[:5], True, 8, 9.0)
    close(frozen[1], GROUND)
    close(frozen[3], 13.0 / 10.0)
    close(frozen[4], touch[4])

    side = step_field(-1.0, 400.0, -1.0, 0.0, FUEL0, False, 1, 9.0)
    close(side[0], -31.0 / 30.0)
    close(side[2], -1.0)
    if side[1] == GROUND:
        raise SystemExit("открытый x назван землёй")

    x, y, vx, vy, fuel, hit = 256.0, 256.0, 0.0, 0.0, FUEL0, False
    for _ in range(300):
        x, y, vx, vy, fuel, hit = step_field(x, y, vx, vy, fuel, hit, 2, 2.0)
    if not (x > 512.0 and vx > 0.0 and not hit and abs(fuel - 10.0) < 1e-9):
        raise SystemExit(f"бок упёрся в номинальный край: {x}, {vx}, {hit}, {fuel}")

    x, y, vx, vy, fuel, hit = 256.0, 464.0, 0.0, 0.0, FUEL0, False
    for step in range(1, 121):
        x, y, vx, vy, fuel, hit = step_field(x, y, vx, vy, fuel, hit, 1, 2.0)
    if hit or y >= GROUND:
        raise SystemExit("побережье село за 120 шагов")
    landed = None
    for step in range(121, 450):
        x, y, vx, vy, fuel, hit = step_field(x, y, vx, vy, fuel, hit, 1, 2.0)
        if hit:
            landed = step
            break
    if landed is None or y != GROUND or abs(fuel - 10.0) > 1e-9:
        raise SystemExit(f"побережье не село до 450: {landed}, {y}, {fuel}")

    full = step_field(256.0, 400.0, 0.0, 0.0, FUEL0, False, 8, 9.0)
    close(full[2], 0.4)
    close(full[3], -11.0 / 30.0)
    close(full[4], 9.98)
    empty = step_field(256.0, 400.0, 0.0, 0.0, 0.0, False, 8, 9.0)
    close(empty[2], 0.0)
    close(empty[3], 0.3)
    close(empty[4], 0.0)

    sx, sy, svx, svy, _a = start_and_actions(80000, 15)
    tx, ty, tvx, tvy, tb = start_and_actions(80000, 15)
    if (sx, sy, svx, svy) != (tx, ty, tvx, tvy) or _a.tolist() != tb.tolist():
        raise SystemExit("лента 80000 разошлась")
    if not (240.0 <= sx <= 272.0 and 464.0 <= sy <= 488.0):
        raise SystemExit(f"старт вне окна: {sx}, {sy}")

    scaled = scale_state(np.array([[256.0, 512.0, 0.0, 0.0, 10.0]]))
    if scaled.shape != (1, 5) or not np.allclose(scaled[0], [0.5, 1.0, 0.0, 0.0, 1.0]):
        raise SystemExit(f"масштаб {scaled}")
    world = world_vector(np.array([9.0]))
    if world.shape != (1, 2) or not np.allclose(world[0], [1.0, 1.0]):
        raise SystemExit(f"мир {world}")
    world4 = world_vector(np.array([4.0]))
    close(world4[0, 0], np.float32(4.0 / 9.0))

    z = torch.zeros(1, 32)
    pair = action_pairs(torch.tensor([8]), z)
    vis = predictor_input(z, pair, torch.from_numpy(world), "vis")
    hid = predictor_input(z, pair, None, "hid")
    if vis.shape[-1] != 36 or hid.shape[-1] != 34:
        raise SystemExit((vis.shape, hid.shape))
    if any(abs(float(v) - 12.0) < 1e-6 for v in vis.reshape(-1)):
        raise SystemExit("во входе оказалось 12")
    if vis.shape[-1] != 32 + 2 + 2:
        raise SystemExit("ширина вошла отдельным каналом")

    states, _acts, gs = make_batch(np.random.default_rng(0), 128, "range")
    if states.shape != (128, 4, 5) or len(set(np.round(gs, 5))) < 2:
        raise SystemExit("батч диапазона не разнообразен")

    same = torch.zeros(2, 5)
    same[:, 0] = 256.0
    same[:, 1] = 512.0
    same[:, 4] = torch.tensor([0.0, 10.0])
    costs = step_cost(same)
    if float(costs[0]) != 0.0 or float(costs[1]) != 0.0:
        raise SystemExit(f"топливо вошло в стоимость: {costs}")

    soft = outcome(256.0, GROUND, 7.9, True)
    crash = outcome(200.0, GROUND, 1.0, True)
    miss = outcome(256.0, 480.0, 1.0, False)
    if (soft, crash, miss) != ("мягкая", "удар", "не сел"):
        raise SystemExit((soft, crash, miss))
    if not (PAD_LO == 248.0 and PAD_HI == 264.0):
        raise SystemExit("площадка сдвинулась")

    try:
        refuse_tag("numeric-land")
    except SystemExit:
        pass
    else:
        raise SystemExit("тег numeric-land принят")

    before = os.path.isfile(os.path.join(OUT, "landf_range_vis_s0.pt"))
    proc = subprocess.run(
        [sys.executable, "-m", "numeric_land.train_field", "--tag", "numeric-land"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        raise SystemExit("чужой тег завершился нулём")
    if os.path.isfile(os.path.join(OUT, "landf_range_vis_s0.pt")) != before:
        raise SystemExit("чужой тег создал чекпоинт")
    proc64 = subprocess.run(
        [sys.executable, "-m", "numeric_land.train", "--tag", "numeric-land-field"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True,
    )
    if proc64.returncode == 0:
        raise SystemExit("поле 64 приняло новый тег")

    if N_WORLD["vis"] != 2 or N_WORLD["hid"] != 0:
        raise SystemExit(N_WORLD)
    if LandPredictor(2).n_world != 2:
        raise SystemExit("предиктор")

    root = os.path.dirname(os.path.abspath(__file__))
    for script in ("_check_numeric_land.py", "_check_flat.py"):
        run = subprocess.run([sys.executable, os.path.join(root, script)], cwd=root)
        if run.returncode != 0:
            raise SystemExit(f"{script} не прошёл")
    print("numeric land field: ок")


if __name__ == "__main__":
    main()

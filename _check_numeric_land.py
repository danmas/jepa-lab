"""Контроль посадки без обучения. python _check_numeric_land.py"""
import os
import subprocess
import sys

import numpy as np
import torch

from numeric_land.data import (
    DT, GROUND, HORIZ_ACCEL, make_batch, outcome, scale_state, start_and_actions,
    step_lander, world_vector,
)
from numeric_land.models import LandPredictor, N_WORLD, action_pairs, predictor_input
from numeric_land.train import OUT, refuse_tag


def _close(got, exp, tol=1e-9):
    if abs(float(got) - float(exp)) > tol:
        raise SystemExit(f"{got} != {exp}")


def check_physics():
    x, y, vx, vy, fuel, hit = step_lander(32, 32, 0, 0, 1, False, 1, 9)
    _close(vx, 0.0)
    _close(vy, 0.3)
    _close(y - 32.0, 0.01)
    _close(fuel, 1.0)
    if hit:
        raise SystemExit("свободное падение названо касанием")
    slow = step_lander(32, 32, 0, 0, 1, False, 1, 2)
    fast = step_lander(32, 32, 0, 0, 1, False, 1, 15)
    _close(slow[3], 1.0 / 15.0)
    _close(fast[3], 0.5)
    _close(fast[3] - slow[3], 13.0 / 30.0)
    _close(fast[1] - slow[1], 13.0 / 900.0)
    touch = step_lander(32, 3199.0 / 50.0, 0, 1, 1, False, 1, 9)
    _close(touch[0], 32.0)
    _close(touch[1], 64.0)
    _close(touch[3], 1.3)
    _close(touch[2], 0.0)
    if not touch[5]:
        raise SystemExit("касание не отмечено")
    frozen = step_lander(*touch[:5], True, 8, 9)
    _close(frozen[1], 64.0)
    _close(frozen[3], 1.3)
    _close(frozen[4], touch[4])
    side = step_lander(0.01, 32, -1, 0, 1, False, 1, 9)
    _close(side[0], 0.0)
    _close(side[2], 0.0)
    if side[1] == GROUND:
        raise SystemExit("бок назван землёй")
    empty = step_lander(32, 32, 0, 0, 0, False, 8, 9)
    _close(empty[2], 0.0)
    _close(empty[3], 0.3)
    _close(empty[4], 0.0)
    full = step_lander(32, 32, 0, 0, 1, False, 8, 9)
    _close(full[2], 0.4)
    _close(full[3], -11.0 / 30.0)
    _close(full[4], 0.98)
    half = step_lander(32, 32, 0, 0, 1, False, 4, 9)
    _close(half[2], 0.0)
    _close(half[3], -1.0 / 30.0)
    _close(half[4], 0.99)


def check_windows():
    states, _actions, gs = make_batch(np.random.default_rng(0), 128, "range")
    if len(set(np.round(gs, 6))) < 2:
        raise SystemExit("батч схлопнул тяжести")
    if np.any(gs < 2.0) or np.any(gs >= 15.0):
        raise SystemExit("g вне [2, 15)")
    fixed, _a, g9 = make_batch(np.random.default_rng(1), 16, "fixed", 9)
    if not np.allclose(g9, 9.0):
        raise SystemExit("fixed не держит 9")
    again, _b, _g = make_batch(np.random.default_rng(0), 128, "range")
    if not np.allclose(states, again):
        raise SystemExit("один сид батча дал две физики")
    tapes = []
    starts = []
    for g in (2.0, 8.0, 20.0):
        x, y, vx, vy, actions = start_and_actions(70000, 15)
        tapes.append(actions.copy())
        starts.append((x, vx))
    if not np.array_equal(tapes[0], tapes[1]) or not np.array_equal(tapes[0], tapes[2]):
        raise SystemExit("лента зависит от тяжести")
    if starts[0] != starts[1] or starts[0] != starts[2]:
        raise SystemExit("старт зависит от тяжести")
    scaled = scale_state(fixed[:1])
    _close(scaled[0, 0, 4], fixed[0, 0, 4])
    if scaled.shape[-1] != 5:
        raise SystemExit("в состоянии не пять чисел")
    world = world_vector(np.array([9.0, 4.0]))
    _close(world[0, 0], 1.0)
    _close(world[0, 1], 1.0)
    _close(world[1, 0], np.float32(4.0 / 9.0))
    if abs(float(DT) - 1.0 / 30.0) > 0:
        raise SystemExit("dt")
    if outcome(32, 64, 7.9, True) != "мягкая":
        raise SystemExit("мягкая")
    if outcome(32, 64, 8.0, True) != "удар":
        raise SystemExit("порог включил 8")
    if outcome(10, 64, 1.0, True) != "удар":
        raise SystemExit("мимо площадки")
    if outcome(32, 60, 0.0, False) != "не сел":
        raise SystemExit("зависание названо посадкой")


def check_input():
    torch.manual_seed(0)
    encoder_in = torch.zeros(1, 5)
    z = torch.nn.functional.normalize(torch.zeros(1, 32) + 0.2, dim=-1)
    pair = action_pairs(torch.tensor([1]), z)
    world = torch.tensor([[1.0, 1.0]])
    vis = predictor_input(z, pair, world, "vis")
    hid = predictor_input(z, pair, world, "hid")
    if vis.shape[-1] != 32 + 2 + N_WORLD["vis"]:
        raise SystemExit(vis.shape)
    if hid.shape[-1] != 32 + 2:
        raise SystemExit(hid.shape)
    if not torch.allclose(vis[0, -2:], world[0]):
        raise SystemExit("видимый вход не кончается (g/9, 1)")
    if float(HORIZ_ACCEL) in vis.detach().reshape(-1).tolist():
        raise SystemExit("в видимый вход попало 12")
    if LandPredictor(2).n_world != 2 or LandPredictor(0).n_world != 0:
        raise SystemExit("n_world")
    if encoder_in.shape[-1] != 5:
        raise SystemExit("кодировщик")


def check_tag_and_sources():
    try:
        refuse_tag("numeric-scale")
    except SystemExit:
        pass
    else:
        raise SystemExit("чужой тег прошёл")
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(OUT, "land_range_vis_s0.pt")
    before = os.stat(path) if os.path.isfile(path) else None
    proc = subprocess.run(
        [sys.executable, "-m", "numeric_land.train", "--tag", "numeric-scale"],
        cwd=root, capture_output=True, text=True)
    if proc.returncode == 0:
        raise SystemExit("чужой тег дошёл до обучения")
    after = os.stat(path) if os.path.isfile(path) else None
    created = before is None and after is not None
    touched = (
        before is not None and after is not None
        and (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)
    )
    if created or touched:
        raise SystemExit("чужой тег записал чекпоинт")
    package = os.path.join(root, "numeric_land")
    for name in os.listdir(package):
        if not name.endswith(".py"):
            continue
        text = open(os.path.join(package, name), encoding="utf-8").read()
        if "step_state" in text:
            raise SystemExit(f"step_state упомянут в {name}")
    flat = subprocess.run([sys.executable, "_check_flat.py"], cwd=root)
    if flat.returncode != 0:
        raise SystemExit("плоский мир разошёлся")


def main():
    check_physics()
    check_windows()
    check_input()
    check_tag_and_sources()
    print("numeric land: ок")


if __name__ == "__main__":
    main()

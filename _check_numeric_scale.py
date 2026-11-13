"""Контроль масштаба без обучения. python _check_numeric_scale.py"""
import numpy as np

from numeric_dt.data import step_physics as step_dt
from numeric_scale.data import (
    DT, make_batch, rollout_episode, scale_state, size_of_window, step_physics,
    world_vector,
)
from numeric_scale.models import N_WORLD, ScalePredictor, predictor_input
from numeric_scale.train import refuse_tag


def _close(got, exp, tol=1e-8):
    if abs(float(got) - float(exp)) > tol:
        raise SystemExit(f"{got} != {exp}")


def check_free():
    walls = (0.0, 64.0, 0.0, 64.0)
    pos, vel, n, hit = step_physics([16.0, 16.0], [0.0, 0.0], 5, walls)
    if n != 1 or hit:
        raise SystemExit(f"подшагов {n}, отскок {hit}")
    _close(vel[0], 8.0)
    _close(vel[1], 0.3)
    _close(pos[0] - 16.0, 4.0 / 15.0)
    _close(pos[1] - 16.0, 0.01)
    pdt, vdt, ndt = step_dt([16.0, 16.0], [0.0, 0.0], 5, DT)
    if ndt != 1 or not np.allclose(pos, pdt) or not np.allclose(vel, vdt):
        raise SystemExit("свободный шаг разошёлся с numeric_dt")


def check_widths():
    pos0 = [14.0, 8.0]
    vel0 = [60.0, 0.0]
    p16, v16, n16, hit16 = step_physics(pos0, vel0, 5, (0.0, 16.0, 0.0, 32.0))
    p32, v32, n32, hit32 = step_physics(pos0, vel0, 5, (0.0, 32.0, 0.0, 32.0))
    if n16 != 3 or n32 != 3 or not hit16 or hit32:
        raise SystemExit((n16, n32, hit16, hit32))
    _close(p16[0], 712.0 / 45.0)
    _close(p16[1], 8.0 + 1.0 / 150.0)
    _close(v16[0], -68.0)
    _close(p32[0], 728.0 / 45.0)
    _close(p32[1], 8.0 + 1.0 / 150.0)
    _close(v32[0], 68.0)
    _close(p32[0] - p16[0], 16.0 / 45.0)
    _close(v32[0] - v16[0], 136.0)


def check_scale():
    rel16 = scale_state(np.array([[14.0, 8.0, 60.0, 0.0, DT]]), 16.0, 32.0, "rel")
    rel32 = scale_state(np.array([[28.0, 8.0, 120.0, 0.0, DT]]), 32.0, 32.0, "rel")
    if not np.allclose(rel16, rel32):
        raise SystemExit("одна доля дала два вектора rel")
    _close(rel16[0, 0], 7.0 / 8.0)
    _close(rel16[0, 1], 0.25)
    _close(rel16[0, 2], 0.125)
    _close(rel16[0, 4], 1.0)
    raw = np.array([[14.0, 8.0, 60.0, 0.0, DT]])
    abs16 = scale_state(raw, 16.0, 32.0, "abs")
    abs32 = scale_state(raw, 32.0, 32.0, "abs")
    if not np.allclose(abs16, abs32):
        raise SystemExit("abs увидел ширину")
    _close(abs16[0, 0], 14.0 / 64.0)
    _close(abs16[0, 2], 60.0 / 120.0)
    _close(abs16[0, 4], 1.0)
    world = world_vector(np.array([16.0, 20.0]), np.array([28.0, 24.0]))
    _close(world[0, 0], 16.0 / 64.0)
    _close(world[1, 1], 24.0 / 64.0)
    if not np.allclose(world[:, 2], 1.0):
        raise SystemExit("канал мира не равен 1")


def check_batches():
    _s, _a, widths, heights = make_batch(np.random.default_rng(0), 128)
    if len(set(np.round(widths, 6))) < 2 or len(set(np.round(heights, 6))) < 2:
        raise SystemExit("батч схлопнул размеры")
    if np.any(widths < 16.0) or np.any(widths >= 32.0):
        raise SystemExit("ширина вне [16, 32)")
    if np.any(heights < 16.0) or np.any(heights >= 32.0):
        raise SystemExit("высота вне [16, 32)")
    w, h = size_of_window(7)
    if not (16.0 <= w < 32.0 and 16.0 <= h < 32.0):
        raise SystemExit((w, h))


def check_inputs():
    import torch
    z = torch.zeros(2, 32)
    pair = z[:, :2]
    world = z[:, :3]
    vis = predictor_input(z, pair, world, "vis")
    hid = predictor_input(z, pair, world, "hid")
    if vis.shape[-1] != 37 or hid.shape[-1] != 34:
        raise SystemExit((vis.shape, hid.shape))
    if ScalePredictor(N_WORLD["vis"]).net[0].in_features != 37:
        raise SystemExit("видимый предиктор")
    if ScalePredictor(N_WORLD["hid"]).net[0].in_features != 34:
        raise SystemExit("скрытый предиктор")


def check_tape():
    a = rollout_episode(60000, 12.0, 12.0)
    b = rollout_episode(60000, 24.0, 24.0)
    c = rollout_episode(60000, 48.0, 48.0)
    if not np.array_equal(a[1], b[1]) or not np.array_equal(a[1], c[1]):
        raise SystemExit("лента действий зависит от размера")
    if len(a[1]) != 30:
        raise SystemExit(len(a[1]))
    if a[4] != b[4] or a[4] != c[4]:
        raise SystemExit("монета подхода зависит от размера")
    rect = rollout_episode(66000, 20.0, 28.0)
    if len(rect[1]) != 30:
        raise SystemExit("прямоугольник")


def check_tag():
    try:
        refuse_tag("numeric-walls")
    except SystemExit:
        return
    raise SystemExit("тег numeric-walls принят")


def main():
    check_free()
    check_widths()
    check_scale()
    check_batches()
    check_inputs()
    check_tape()
    check_tag()
    print("numeric_scale: ок")


if __name__ == "__main__":
    main()

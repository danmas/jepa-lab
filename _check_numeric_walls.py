"""Контроль стенок без обучения. python _check_numeric_walls.py"""
import numpy as np

from numeric_dt.data import step_physics as step_dt
from numeric_walls.data import (
    CENTER, DT, SPAN_FIXED, make_batch, rollout_episode, scale_state, scale_walls,
    span_of_window, step_physics, walls_of,
)
from numeric_walls.models import N_WALL, WallPredictor, predictor_input
from numeric_walls.train import refuse_tag


def _close(got, exp, tol=1e-8):
    if abs(float(got) - float(exp)) > tol:
        raise SystemExit(f"{got} != {exp}")


def check_free():
    walls = (5.0, 58.0, 5.0, 58.0)
    pos, vel, n = step_physics([32.0, 32.0], [0.0, 0.0], 5, walls)
    if n != 1:
        raise SystemExit(f"подшагов {n}")
    _close(vel[0], 8.0)
    _close(vel[1], 0.3)
    _close(pos[0] - 32.0, 4.0 / 15.0)
    _close(pos[1] - 32.0, 0.01)
    pdt, vdt, ndt = step_dt([32.0, 32.0], [0.0, 0.0], 5, DT)
    if ndt != 1 or not np.allclose(pos, pdt) or not np.allclose(vel, vdt):
        raise SystemExit("свободный шаг разошёлся с numeric_dt")


def check_widths():
    pos0 = [39.0, 32.0]
    vel0 = [60.0, 0.0]
    p16, v16, _n = step_physics(pos0, vel0, 5, walls_of("W", 16))
    p24, v24, _n = step_physics(pos0, vel0, 5, walls_of("W", 24))
    _close(p16[0], 38.0)
    _close(p16[1], 32.0 + 1.0 / 150.0)
    _close(v16[0], -172.0 / 3.0)
    _close(p24[0], 41.0 + 8.0 / 45.0)
    _close(p24[1], 32.0 + 1.0 / 150.0)
    _close(v24[0], 68.0)
    _close(p24[0] - p16[0], 143.0 / 45.0)
    _close(v24[0] - v16[0], 376.0 / 3.0)


def check_batches():
    rng = np.random.default_rng(0)
    _s, _a, walls, spans = make_batch(rng, "W", "range", 128)
    if len(set(np.round(spans, 6))) < 2:
        raise SystemExit("диапазон схлопнул ширины")
    if np.any(spans < 16.0) or np.any(spans >= 24.0):
        raise SystemExit("ширина вне [16, 24)")
    if not np.allclose(walls[:, 2], 5.0) or not np.allclose(walls[:, 3], 58.0):
        raise SystemExit("W-world сдвинул верх или низ")
    _s, _a, walls_h, spans_h = make_batch(np.random.default_rng(1), "H", "fixed", 128)
    if not np.allclose(spans_h, SPAN_FIXED):
        raise SystemExit("постоянная высота не 20")
    if not np.allclose(walls_h[:, 0], 5.0) or not np.allclose(walls_h[:, 1], 58.0):
        raise SystemExit("H-world сдвинул бока")
    scaled = scale_walls(walls_h)
    if scaled.std(axis=0).max() > 0:
        raise SystemExit("постоянные стенки не константа")
    state = scale_state(_s)
    if not np.allclose(state[..., 4], 1.0):
        raise SystemExit("канал dt не равен 1")


def check_inputs():
    z = torch_zeros()
    pair = z[:, :2]
    walls = z[:, :4]
    vis = predictor_input(z, pair, walls, "vis")
    hid = predictor_input(z, pair, walls, "hid")
    if vis.shape[-1] != 38 or hid.shape[-1] != 34:
        raise SystemExit((vis.shape, hid.shape))
    if WallPredictor(N_WALL["vis"]).net[0].in_features != 38:
        raise SystemExit("видимый предиктор")
    if WallPredictor(N_WALL["hid"]).net[0].in_features != 34:
        raise SystemExit("скрытый предиктор")


def torch_zeros():
    import torch
    return torch.zeros(2, 32)


def check_tape():
    a = rollout_episode(50000, "W", 12.0)[1]
    b = rollout_episode(50000, "W", 32.0)[1]
    c = rollout_episode(50000, "H", 12.0)[1]
    if not np.array_equal(a, b) or not np.array_equal(a, c):
        raise SystemExit("лента действий зависит от размера")
    if len(a) != 30:
        raise SystemExit(len(a))


def check_span_draw():
    if span_of_window(123, "fixed") != SPAN_FIXED:
        raise SystemExit("fixed тянет случайный размер")
    got = span_of_window(123, "range")
    if not (16.0 <= got < 24.0):
        raise SystemExit(got)
    if abs(CENTER - 31.5) > 0:
        raise SystemExit(CENTER)


def check_tag():
    try:
        refuse_tag("numeric-dt")
    except SystemExit:
        return
    raise SystemExit("тег numeric-dt принят")


def main():
    check_free()
    check_widths()
    check_batches()
    check_inputs()
    check_tape()
    check_span_draw()
    check_tag()
    print("numeric_walls: ок")


if __name__ == "__main__":
    main()

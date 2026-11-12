"""Контроль интегратора dt. Обучение не запускает.

python _check_numeric_dt.py
"""
import numpy as np

from data import step_state
from numeric_dt.data import (
    DT_B_HI, DT_B_LO, HI, LO, make_batch, scale_state, step_physics, substep_count,
)
from numeric_dt.train import refuse_tag


def _close(got, exp, tol=1e-4):
    got = np.asarray(got, np.float64)
    exp = np.asarray(exp, np.float64)
    if np.max(np.abs(got - exp)) > tol:
        raise SystemExit(f"ждали {exp}, получили {got}")


def check_one_push():
    pos, vel, n = step_physics([32.0, 32.0], [0.0, 0.0], 5, 1.0 / 30.0)
    if n != 1:
        raise SystemExit(f"толчок вправо должен быть одним подшагом, n={n}")
    _close(vel, [8.0, 0.3])
    _close(pos - np.array([32.0, 32.0]), [4.0 / 15.0, 0.01])


def check_split():
    pos0 = np.array([32.0, 32.0])
    pos, vel, n = step_physics(pos0, [60.0, 0.0], 4, 1.0 / 15.0)
    if n != 5:
        raise SystemExit(f"ждали 5 подшагов, n={n}")
    _close(pos - pos0, [4.0, 0.024])
    _close(vel, [60.0, 0.6])


def check_order():
    def fly(dt, n_steps):
        pos = np.array([32.0, 32.0])
        vel = np.array([0.0, 0.0])
        for _ in range(n_steps):
            pos, vel, n = step_physics(pos, vel, 5, dt)
            if n != 1:
                raise SystemExit(f"на порядке схемы подшаг лишний, dt={dt} n={n}")
        return pos - np.array([32.0, 32.0])

    exact = np.array([8.0 / 15.0, 1.0 / 50.0])
    at_60 = fly(1.0 / 60.0, 4)
    at_120 = fly(1.0 / 120.0, 8)
    _close(at_60, [2.0 / 3.0, 1.0 / 40.0])
    _close(at_120, [3.0 / 5.0, 9.0 / 400.0])
    if not (at_60[0] > exact[0] and at_60[1] > exact[1]):
        raise SystemExit("полунеявный шаг должен быть впереди точного решения")
    if not (at_120[0] > exact[0] and at_120[1] > exact[1]):
        raise SystemExit("шаг 1/120 должен быть впереди точного решения")
    _close(at_120 - exact, 0.5 * (at_60 - exact))


def check_not_step_state():
    pos, vel, _n = step_physics([32.0, 32.0], [0.0, 0.0], 4, 1.0 / 30.0)
    other_pos, other_vel = step_state(
        np.array([32.0, 32.0], np.float32), np.array([0.0, 0.0], np.float32), 4)
    if abs(float(vel[1]) - 0.3) > 1e-4:
        raise SystemExit(f"vy={vel[1]}, ждали 0.3")
    if abs(float(other_vel[1]) - 0.15) > 1e-4:
        raise SystemExit("эталон step_state изменился")
    if abs(float(vel[1]) - float(other_vel[1])) < 1e-3:
        raise SystemExit("шаг dt совпал со step_state")
    if abs(float(pos[1]) - float(other_pos[1])) < 1e-3:
        raise SystemExit("положение dt совпало со step_state")


def check_scale_identity():
    phys = np.array([[10.0, 20.0, 20.0, -60.0, 1.0 / 30.0]], np.float32)
    with_dt = scale_state(phys, "with-dt")
    frame = np.array([[10.0, 20.0, 20.0 / 30.0, -60.0 / 30.0]], np.float32)
    no_dt = scale_state(frame, "no-dt")
    _close(with_dt[0, :4], no_dt[0], tol=1e-6)
    _close(with_dt[0, 4], [1.0], tol=1e-6)


def check_batch_dt():
    rng = np.random.default_rng(0)
    states, _actions, dts = make_batch(rng, "B", "with-dt", batch=128, horizon=6)
    if len(set(np.round(dts, 8))) < 2:
        raise SystemExit("батч руки B взял один dt")
    if np.any(dts < DT_B_LO) or np.any(dts > DT_B_HI):
        raise SystemExit("dt окна вне [1/40, 1/24]")
    for row, dt in zip(states, dts):
        if np.max(np.abs(row[:, 4] - dt)) > 1e-6:
            raise SystemExit("внутри окна dt меняется")


def check_wall():
    pos, vel, _n = step_physics([HI - 0.1, 32.0], [60.0, 0.0], 4, 1.0 / 15.0)
    if pos[0] < LO or pos[0] > HI or pos[1] < LO or pos[1] > HI:
        raise SystemExit(f"центр вышел из комнаты: {pos}")
    if vel[0] >= 0:
        raise SystemExit("от правой стены скорость по x не развернулась")


def check_tag():
    for tag in ("", "v4", "v5-floor", "numeric"):
        try:
            refuse_tag(tag)
        except SystemExit:
            continue
        raise SystemExit(f"тег {tag!r} не отклонён")
    refuse_tag("numeric-dt")


def main():
    check_one_push()
    check_split()
    check_order()
    check_not_step_state()
    check_scale_identity()
    check_batch_dt()
    check_wall()
    check_tag()
    # substep_count импортирован, чтобы формула порога была той же, что в шаге.
    n = substep_count(np.array([60.0, 0.0]), np.zeros(2), 1.0 / 15.0)
    if n != 5:
        raise SystemExit(n)
    print("numeric_dt: ок")


if __name__ == "__main__":
    main()

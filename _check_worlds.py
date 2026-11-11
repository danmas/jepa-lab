"""Проверки режимов до обучения. Без pytest.

    python _check_worlds.py
"""
import sys

import numpy as np
import torch

from data import (
    BUMP_H, FLAT_FLOOR, GRAVITY, NOOP, World, assert_zone_floor,
    blocked_zone_names, bump_centers, bump_height, bump_slope, floor_limit,
    goal_frame, gravity_of, integrate_floor, jitter_gate, paint_floor,
    step_floor, step_other, step_state, zone_floor_margin,
)
def fail(msg):
    print("FAIL", msg)
    return 1


def check_jitter():
    pos = np.array([32.0, 32.0], np.float32)
    vel = np.zeros(2, np.float32)
    slipped = False
    for seed in range(40):
        u = World(seed, world="jitter")
        v = World(seed + 1000, world="jitter")
        u.pos, u.vel = pos.copy(), vel.copy()
        v.pos, v.vel = pos.copy(), vel.copy()
        u.step(5)
        v.step(5)
        if np.hypot(*(u.pos - v.pos)) > 1e-3:
            slipped = True
            break
    if not slipped:
        return fail("jitter: один толчок из одного состояния не разошёлся ни на одной паре сидов")
    same = World(7, world="jitter")
    again = World(7, world="jitter")
    for _ in range(6):
        same.step(5)
        again.step(5)
    if np.hypot(*(same.pos - again.pos)) > 1e-4:
        return fail("jitter: один сид дал две разные последовательности")
    gate = jitter_gate(4.5)
    if gate["train"] or gate["label"] != "не проверка неоднозначности":
        return fail(f"jitter: радиус 4.5 не закрыл обучение: {gate}")
    if not jitter_gate(6.1)["train"]:
        return fail("jitter: радиус 6.1 должен открывать обучение")
    print("jitter ok")
    return 0


def check_noise():
    pos = np.array([20.0, 30.0], np.float32)
    vel = np.array([1.0, -0.5], np.float32)
    flat = World(3, world="flat")
    noisy = World(3, world="noise")
    flat.pos, flat.vel = pos.copy(), vel.copy()
    noisy.pos, noisy.vel = pos.copy(), vel.copy()
    clean = flat.step(5)[0]
    dirty = noisy.step(5)[0]
    if np.max(np.abs(flat.pos - noisy.pos)) > 1e-5:
        return fail("noise: толчок сдвинул мяч иначе, чем плоский мир")
    if np.max(np.abs(clean - dirty)) < 1e-4:
        return fail("noise: кадр модели совпал с чистым")
    if dirty.min() < -1.01 or dirty.max() > 1.01:
        return fail("noise: кадр вышел из [-1, 1]")
    g1 = goal_frame(2, world="noise")
    g2 = goal_frame(2, world="noise")
    if np.max(np.abs(g1 - g2)) > 1e-6:
        return fail("noise: кадр цели зоны меняется между вызовами")
    if np.max(np.abs(g1 - goal_frame(3, world="noise"))) < 1e-6:
        return fail("noise: разные зоны получили один и тот же шум цели")
    print("noise ok")
    return 0


def check_balls():
    w = World(4, world="balls")
    speed0 = float(np.hypot(*w.vel2))
    vel0 = w.vel.copy()
    w.step(5)
    if np.hypot(*(w.vel - vel0)) < 0.5:
        return fail("balls: ненулевой толчок не изменил скорость управляемого мяча")
    if abs(float(np.hypot(*w.vel2)) - speed0) > 1e-4:
        return fail("balls: толчок изменил величину скорости второго мяча")
    pos = np.array([32.0, 32.0], np.float32)
    vel = np.array([2.0, 0.0], np.float32)
    start = float(np.hypot(*vel))
    for _ in range(30):
        pos, vel = step_other(pos, vel)
        if abs(float(pos[1]) - 32) > 1e-3 and abs(float(pos[0]) - 5) < 1e-3:
            pass
    if abs(float(np.hypot(*vel)) - start) > 1e-4:
        return fail("balls: за 30 шагов без угла величина скорости уехала")
    from data import in_zone
    controlled = np.array([14.0, 14.0], np.float32)
    if not in_zone(controlled, 0):
        return fail("balls: управляемый мяч в центре зоны не засчитан")
    print("balls ok")
    return 0


def check_floor():
    deg = float(np.degrees(np.arctan(BUMP_H / 8.0)))
    if deg > 30.0:
        return fail(f"floor: наклон {deg:.1f}° круче 30°")
    pos = np.array([20.0, floor_limit(20.0) - 0.5], np.float64)
    vel = np.array([0.0, 8.0], np.float64)
    new_pos, new_vel = integrate_floor(pos, vel, height=4.0)
    if float(new_vel[1]) >= 0:
        return fail(f"floor: вертикальная скорость не сменила знак: {new_vel}")
    if float(new_pos[1]) > floor_limit(new_pos[0], 4.0) + 1e-3:
        return fail(f"floor: центр ниже пола: y={new_pos[1]} предел={floor_limit(new_pos[0])}")
    margin, zone = zone_floor_margin(4.0)
    try:
        assert_zone_floor(4.0)
    except RuntimeError as exc:
        return fail(f"floor: высота 4 не должна ронять запас: {exc}")
    if margin < 2.0:
        return fail(f"floor: запас {margin} < 2")
    try:
        assert_zone_floor(8.0)
    except RuntimeError:
        pass
    else:
        return fail("floor: высота 8 px не уронила проверку зон")
    pos = np.array([32.0, floor_limit(32.0)], np.float32)
    vel = np.zeros(2, np.float32)
    pos, vel = step_floor(pos, vel, 1)
    peak = FLAT_FLOOR - BUMP_H
    above = float(pos[1]) < peak - 1e-4
    for _ in range(20):
        pos, vel = step_floor(pos, vel, NOOP)
        if float(pos[1]) < peak - 1e-4:
            above = True
        if float(pos[1]) > floor_limit(pos[0]) + 1e-3:
            return fail("floor: долина ушла сквозь пол")
    if not above:
        return fail(f"floor: толчок вверх из долины не поднял центр выше бугров, y={float(pos[1]):.2f}")
    frame = World(0, world="floor").frame()[0]
    if not np.any(np.abs(frame + 0.15) < 1e-5):
        return fail("floor: на кадре не виден пол")
    goal = goal_frame(8, world="floor")[0, 0]
    if not np.any(np.abs(goal + 0.15) < 1e-5):
        return fail("floor: на кадре цели нет пола")
    print(f"floor ok  запас={margin:.2f} px (зона {zone})  наклон={deg:.1f}°")
    return 0


def _roll(height, g, y0, actions, centers):
    pos = np.array([8.0, y0], np.float64)
    vel = np.zeros(2, np.float64)
    for action in actions:
        pos, vel = step_floor(pos, vel, action, g=g, height=height, centers=centers)
        pos = np.asarray(pos, np.float64)
        vel = np.asarray(vel, np.float64)
        if float(pos[1]) > floor_limit(float(pos[0]), height, centers) + 1e-3:
            return None
    return pos


def check_bump_controls():
    want = {
        0: (),
        1: (32.0,),
        2: (20.0, 44.0),
        3: (8.0, 32.0, 56.0),
        4: (8.0, 24.0, 40.0, 56.0),
        5: (8.0, 20.0, 32.0, 44.0, 56.0),
    }
    for n, centers in want.items():
        got = tuple(round(float(x), 5) for x in bump_centers(n))
        if got != tuple(round(float(x), 5) for x in centers):
            return fail(f"центры {n}: {got}")
    trained = bump_centers(2)
    if abs(floor_limit(20.0, 32.0, trained) - (FLAT_FLOOR - 32.0)) > 1e-6:
        return fail("высота 32 не подняла гребень на 32 px")
    overlap = bump_height(14.0, 4.0, centers=bump_centers(5))
    if abs(overlap - 1.0) > 1e-6:
        return fail(f"наложение пяти холмов дало {overlap}, ждали 1")
    if blocked_zone_names(4.0, trained):
        return fail("обученные холмы пересекают зону")
    blocked = blocked_zone_names(7.0, trained)
    if blocked != ["нижний левый", "нижний правый"]:
        return fail(f"высота 7 пересекает не те зоны: {blocked}")
    if gravity_of(3) != float(GRAVITY):
        return fail("три шага ползунка не равны обученной тяжести")
    if abs(gravity_of(20) - 1.0) > 1e-9:
        return fail("двадцать шагов ползунка не дают тяжесть 1")
    low = _roll(4.0, GRAVITY, 40.0, [5] * 8, trained)
    high = _roll(32.0, GRAVITY, 40.0, [5] * 8, trained)
    if low is None or high is None:
        return fail("центр вошёл в тело холма")
    if float(np.hypot(low[0] - high[0], low[1] - high[1])) <= 2.0:
        return fail(f"высокий холм не сдвинул мяч: {low} против {high}")

    def idle(g):
        pos = np.array([32.0, 27.0], np.float64)
        vel = np.zeros(2, np.float64)
        for _ in range(20):
            pos, vel = step_floor(pos, vel, NOOP, g=g, height=4.0, centers=trained)
            pos = np.asarray(pos, np.float64)
            vel = np.asarray(vel, np.float64)
        return pos

    light = idle(float(GRAVITY))
    heavy = idle(1.0)
    if float(np.hypot(light[0] - heavy[0], light[1] - heavy[1])) <= 2.0:
        return fail(f"тяжесть 1 не сдвинула мяч: {light} против {heavy}")
    same_a = World(1, world="floor", gravity=float(GRAVITY)).frame()
    same_b = World(1, world="floor", gravity=1.0).frame()
    if not np.array_equal(same_a, same_b):
        return fail("тяжесть изменила рисунок кадра")
    goal = goal_frame(2, world="floor", bump_h=16, centers=(32.0,))[0, 0]
    blank = paint_floor(np.full((1, 64, 64), -1, np.float32), height=16, centers=(32.0,))[0]
    if abs(float(goal[50, 32]) + 0.15) > 1e-5 or abs(float(goal[50, 32]) - float(blank[50, 32])) > 1e-5:
        return fail("кадр цели рисует не тот холм, что шаг")
    print("bump controls ok")
    return 0


def check_flat_matches_noise_physics():
    # Шум не двигает мяч — уже в check_noise. Здесь плоский шаг против step_state.
    w = World(0)
    pos, vel = w.pos.copy(), w.vel.copy()
    w.step(6)
    pos2, vel2 = step_state(pos, vel, 6)
    if np.max(np.abs(w.pos - pos2)) > 1e-5:
        return fail("flat: World.step разошёлся со step_state")
    print("flat step ok")
    return 0


def check_labels():
    from eval import apply_labels, interval_unresolved, paired_interval
    both = apply_labels(4, 30, 20, 90)
    if not both["лучше_случайных"] or not both["отрыв_15"]:
        return fail(f"ярлыки при запасе 4 и отрыве 20: {both}")
    short = apply_labels(2, 12, 8, 90)
    if not short["лучше_случайных"] or short["отрыв_15"]:
        return fail(f"ярлыки при запасе 2 и отрыве 8: {short}")
    if not interval_unresolved({"ci_low": -1.0, "ci_high": 1.0}):
        return fail("интервал через 0 не просит добавку сидов")
    if not interval_unresolved({"ci_low": 10.0, "ci_high": 20.0}):
        return fail("интервал через 15 не просит добавку сидов")
    if interval_unresolved({"ci_low": 4.0, "ci_high": 10.0}):
        return fail("интервал мимо 0 и 15 не должен расширять выборку")
    episodes = [{"success": i < 70, "rand_success": i < 40} for i in range(90)]
    a = paired_interval(episodes)
    b = paired_interval(episodes)
    if a != b:
        return fail("парный бутстрап не повторяется при сиде 0")
    if "rand_success" not in episodes[0] or a["n"] != 90:
        return fail("у эпизода нет пары или бутстрап потерял размер")
    print(f"labels ok  delta={a['delta_pp']:.1f}")
    return 0


def check_eval_refuses_foreign_world():
    import os
    import subprocess
    path = os.path.join("out", "eval_v4_latent.json")
    before = os.path.getmtime(path) if os.path.exists(path) else None
    proc = subprocess.run(
        [sys.executable, "eval.py", "--tag", "v4", "--world", "floor",
         "--grid", "0", "--seeds", "1", "--zones", "0", "--threads", "4"],
        capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 or "несовпадение" not in text:
        return fail("оценка чужим режимом не отказала: " + text[-400:])
    after = os.path.getmtime(path) if os.path.exists(path) else None
    if before != after:
        return fail("отказ перезаписал файл вердикта")
    print("mismatch ok")
    return 0


def check_cem_plan_repeats():
    import os
    from models import load_model
    from plan import cem_plan
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "jepa_v4.pt")
    if not os.path.exists(path):
        print("cem_plan skip: нет jepa_v4.pt")
        return 0
    model, ema, _ = load_model(path, threads=4)
    ctx = torch.zeros(2, 1, 64, 64)
    z_goal = ema(torch.zeros(1, 1, 64, 64))
    kw = dict(horizon=4, samples=16, iters=2, seed=3)
    first, _, _, _ = cem_plan(model, ema, ctx, z_goal, **kw)
    second, _, _, _ = cem_plan(model, ema, ctx, z_goal, **kw)
    if not torch.equal(first, second):
        return fail("cem_plan с одним сидом дал два плана")
    other, _, _, _ = cem_plan(model, ema, ctx, z_goal, horizon=4, samples=16, iters=2, seed=4)
    if torch.equal(first, other):
        return fail("cem_plan с разными сидами дал один план")
    print("cem_plan ok")
    return 0


def check_cem_seed():
    torch.manual_seed(0)
    model_w = torch.zeros(1)
    # Лёгкая проверка без чекпоинта: multinomial с генератором повторяем.
    probs = torch.full((10, 9), 1.0 / 9)
    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        a = torch.multinomial(probs, 128, replacement=True, generator=g)
        b = torch.multinomial(probs, 128, replacement=True, generator=g)
        return a, b
    a1, b1 = draw(5)
    a2, b2 = draw(5)
    if not torch.equal(a1, a2) or not torch.equal(b1, b2):
        return fail("cem: один сид дал разные выборки")
    c1, _ = draw(6)
    if torch.equal(a1, c1):
        return fail("cem: разные сиды дали одну выборку")
    print("cem seed ok")
    return 0


if __name__ == "__main__":
    errors = 0
    for fn in (check_flat_matches_noise_physics, check_jitter, check_noise, check_balls,
               check_floor, check_bump_controls, check_labels, check_cem_seed,
               check_cem_plan_repeats,
               check_eval_refuses_foreign_world):
        errors += fn()
    if errors:
        print(f"провалено проверок: {errors}")
        sys.exit(1)
    print("режимы сошлись")

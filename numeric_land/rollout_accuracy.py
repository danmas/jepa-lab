"""Многократная раскатка предиктора: насколько он точен на 1, 3, 20 и 40 шагах.

Ровно то, что делает CEM в плане: дано состояние и последовательность действий —
предиктор тащит латент вперёд. Обучение идёт на трёх шагах; план раскатывает
модель на сорок. Здесь измеряется, как растёт ошибка между этими двумя длинами.

Мера — L1 между предсказанным латентом и целью EMA на истинном состоянии (то же,
что в лоссе), усреднённая по x/y/vx/vy/топливу. Сравниваются три семейства на
одних и тех же траекториях правила с наведением.

    python -m numeric_land.rollout_accuracy
"""
import json
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.data import g_of_window
from numeric_land.expert_hover import hover_episode
from numeric_land.field import scale_state
from numeric_land.goal import goal_of_window, world_vector
from numeric_land.models import BallEncoder, LandPredictor, action_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_land_hover_accuracy.json")
FAMILIES = {
    "goal": ("landg_range_vis_s0.pt", "numeric-land-goal"),
    "expert": ("lande_range_vis_s0.pt", "numeric-land-expert"),
    "hover": ("landh_range_vis_s0.pt", "numeric-land-hover"),
}
HORIZONS = (1, 3, 10, 20, 40)
SEED0 = 400_000
N_TRAJ = 40
# обучение смотрит ровно на 3 шага (TRANSITIONS), план — на 40
TRAINED_STEPS = 3


def _load(name):
    fname, tag = FAMILIES[name]
    ck = torch.load(os.path.join(ROOT, "out", fname), map_location="cpu", weights_only=False)
    if ck.get("tag") != tag:
        raise RuntimeError(f"{fname}: тег {ck.get('tag')}")
    enc = BallEncoder()
    pred = LandPredictor(int(ck["n_world"]))
    enc.load_state_dict(ck["encoder"])
    pred.load_state_dict(ck["predictor"])
    ema = EMATarget(enc, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    enc.eval(); pred.eval(); ema.eval()
    return enc, pred, ema


@torch.no_grad()
def unroll_errors(enc, pred, ema, seg_states, seg_actions, g, pad, vt, horizon):
    """Ошибка на каждом шаге раскатки. Возвращает список длиной horizon."""
    world = None
    if pred.n_world:
        world = torch.from_numpy(world_vector(np.array([g]), np.array([pad]),
                                             np.array([vt])).astype(np.float32))
    z = enc(torch.from_numpy(scale_state(seg_states[:1]).astype(np.float32)))
    out = []
    for k in range(horizon):
        a = torch.tensor([int(seg_actions[k])])
        z = pred(z, action_pairs(a, z), world)
        target = ema(torch.from_numpy(scale_state(seg_states[k + 1:k + 2]).astype(np.float32)))
        out.append(float((z - target).abs().mean()))
    return out


def main():
    torch.set_num_threads(8)
    nets = {name: _load(name) for name in FAMILIES}

    trajs = []
    for i in range(N_TRAJ):
        seed = SEED0 + i
        g = g_of_window(seed)
        pad, vt = goal_of_window(seed)
        st, ac, gr = hover_episode(seed, g, pad, vt)
        if gr and len(ac) >= max(HORIZONS):
            trajs.append((g, pad, vt, st, ac))
    print(f"траекторий: {len(trajs)} (правило с наведением, все с посадкой)")
    print(f"обучение смотрит {TRAINED_STEPS} шага, план раскатывает {max(HORIZONS)}\n")

    results = {}
    for name, (enc, pred, ema) in nets.items():
        per_h = {h: [] for h in HORIZONS}
        for g, pad, vt, st, ac in trajs:
            n = len(ac)
            s0 = n - max(HORIZONS)
            seg_states = st[s0:]
            seg_actions = ac[s0:]
            errs = unroll_errors(enc, pred, ema, seg_states, seg_actions, g, pad, vt,
                                 max(HORIZONS))
            for h in HORIZONS:
                per_h[h].append(float(np.mean(errs[:h])))
        results[name] = {h: round(float(np.mean(v)), 6) for h, v in per_h.items()}
        row = "  ".join(f"h{h}={results[name][h]:.5f}" for h in HORIZONS)
        print(f"  {name:7s} ошибка: {row}", flush=True)

    print()
    print("=== рост ошибки от обученных 3 шагов к плановым 40 ===")
    for name in results:
        r3, r40 = results[name][3], results[name][40]
        print(f"  {name:7s}: h3={r3:.5f}  h40={r40:.5f}  рост x{r40/r3:.2f}")

    print()
    print("=== сравнение на 40 шагах ===")
    best = min(results, key=lambda k: results[k][40])
    for name in sorted(results, key=lambda k: results[k][40]):
        print(f"  {name:7s}: {results[name][40]:.5f}  (x{results[name][40]/results[best][40]:.3f} к лучшему {best})")

    payload = {"n_traj": len(trajs), "seed0": SEED0, "horizons": list(HORIZONS),
               "trained_steps": TRAINED_STEPS,
               "metric": "L1(предсказанный латент, цель EMA) в среднем по каналам",
               "families": FAMILIES, "errors": results, "best_h40": best}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n" + OUT_JSON)


if __name__ == "__main__":
    main()

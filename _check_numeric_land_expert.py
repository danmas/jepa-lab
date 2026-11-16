"""Контроль экспертного пакета без обучения. python _check_numeric_land_expert.py"""
import os
import subprocess
import sys

import numpy as np

from numeric_land.expert import (
    POOL_SEED0, expert_episode, expert_pool, make_batch_mixed, rule_action,
)
from numeric_land.field import GROUND, TRANSITIONS
from numeric_land.goal import TRAIN_PAD_HI, TRAIN_PAD_LO, goal_of_window
from numeric_land.data import g_of_window
from numeric_land.train_expert import OUT, refuse_tag


def main():
    # 1. правило садится мягко: при g <= 15 надёжно, при g = 20 заметно чаще половины
    for g, need in ((2.0, 18), (8.0, 18), (15.0, 15), (20.0, 10)):
        soft = 0
        for s in range(20):
            seed = POOL_SEED0 + s
            pad, vt = goal_of_window(seed)
            states, actions, grounded = expert_episode(seed, g, pad, vt)
            if grounded and states[-1][3] < 8.0:
                soft += 1
        if soft < need:
            raise SystemExit(f"правило мягко только {soft}/20 при g={g} (нужно >= {need})")

    # 2. пул набирается, все окна с посадкой, цель/центр в диапазонах
    pool = expert_pool(500)
    if pool["states"].shape[1:] != (TRANSITIONS + 1, 5):
        raise SystemExit(pool["states"].shape)
    if pool["landed_frac"] != 1.0:
        raise SystemExit(f"в пуле не все с посадкой: {pool['landed_frac']}")
    if not (TRAIN_PAD_LO <= pool["pads"].min() and pool["pads"].max() < TRAIN_PAD_HI):
        raise SystemExit("центр пула вне [156, 356)")
    if set(np.unique(pool["vys"])) - {0.0, 3.0, 6.0}:
        raise SystemExit("цель пула вне {0,3,6}")

    # 3. смесь: ровно expert_frac экспертных
    rng = np.random.default_rng(0)
    st, ac, gs, pd, vy = make_batch_mixed(rng, 128, pool, expert_frac=0.5)
    if st.shape != (128, TRANSITIONS + 1, 5) or ac.shape != (128, TRANSITIONS):
        raise SystemExit((st.shape, ac.shape))
    if len(set(np.round(gs, 5))) < 2:
        raise SystemExit("тяжесть не разнообразна")
    # экспертная половина — из пула (по совпадению строк)
    poolset = {tuple(pool["states"][i].ravel().tolist()) for i in range(pool["states"].shape[0])}
    in_pool = sum(1 for i in range(128) if tuple(st[i].ravel().tolist()) in poolset)
    if not (40 <= in_pool <= 100):
        raise SystemExit(f"экспертных окон не около 64, а {in_pool}")
    # expert_frac=0 → пул не нужен
    st0, _ac, _g, _p, _v = make_batch_mixed(np.random.default_rng(1), 32, None, expert_frac=0.0)
    if st0.shape != (32, TRANSITIONS + 1, 5):
        raise SystemExit("expert_frac=0")

    # 4. чужие теги отклоняются
    for bad in ("numeric-land-goal", "numeric-land-pad", "numeric-land-field"):
        try:
            refuse_tag(bad)
        except SystemExit:
            pass
        else:
            raise SystemExit(f"тег {bad} принят")

    # 5. пакет цели не принимает тег экспертного и наоборот
    root = os.path.dirname(os.path.abspath(__file__))
    p1 = subprocess.run([sys.executable, "-m", "numeric_land.train_goal",
                         "--tag", "numeric-land-expert"], cwd=root, capture_output=True, text=True)
    if p1.returncode == 0:
        raise SystemExit("пакет цели принял тег экспертного")
    p2 = subprocess.run([sys.executable, "-m", "numeric_land.train_expert",
                         "--tag", "numeric-land-goal", "--only", "range-vis-0"],
                        cwd=root, capture_output=True, text=True)
    if p2.returncode == 0:
        raise SystemExit("пакет экспертного принял тег цели")
    lande = os.path.join(OUT, "lande_range_vis_s0.pt")
    if os.path.isfile(lande):
        pass  # уже обучен — не наша забота здесь

    # 6. прежние контроли живы
    for chk in ("_check_numeric_land_pad.py", "_check_numeric_land_goal.py"):
        r = subprocess.run([sys.executable, os.path.join(root, chk)], cwd=root)
        if r.returncode != 0:
            raise SystemExit(f"{chk} не прошёл")
    print("numeric land expert: ок")


if __name__ == "__main__":
    main()
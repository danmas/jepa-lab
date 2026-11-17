"""Контроль правила с фазой наведения. python _check_numeric_land_hover.py"""
import os
import subprocess
import sys

import numpy as np

from numeric_land.data import G_HI, MAX_THRUST, g_of_window
from numeric_land.expert_hover import hover_episode, hover_pool
from numeric_land.field import TRANSITIONS
from numeric_land.goal import goal_of_window


def _run(seed, g, pad, vt):
    st, ac, gr = hover_episode(seed, g, pad, vt)
    return st, ac, gr


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # 1. На обученных тяжестях (g < 15) правило садится мягко и в центр
    for g, need in ((2.0, 25), (8.0, 25), (15.0, 25)):
        soft = 0
        for i in range(30):
            seed = 200000 + i
            pad, vt = goal_of_window(seed)
            st, ac, gr = _run(seed, g, pad, vt)
            if gr and abs(st[-1][0] - pad) <= 8 and st[-1][3] < 8.0:
                soft += 1
        if soft < need:
            raise SystemExit(f"g={g}: мягко только {soft}/30 (нужно >= {need})")

    # 2. g=20 вне обучения: мягкость ниже, но на площадку садится
    onpad = 0
    for i in range(30):
        seed = 200000 + i
        pad, vt = goal_of_window(seed)
        st, ac, gr = _run(seed, 20.0, pad, vt)
        if gr and abs(st[-1][0] - pad) <= 8:
            onpad += 1
    if onpad < 25:
        raise SystemExit(f"g=20: на площадке {onpad}/30 (нужно >= 25)")

    # 3. Промах по x мал там, где сел
    misses = []
    for g in (2.0, 8.0, 15.0):
        for i in range(20):
            seed = 200000 + i
            pad, vt = goal_of_window(seed)
            st, ac, gr = _run(seed, g, pad, vt)
            if gr:
                misses.append(abs(st[-1][0] - pad))
    if misses and float(np.mean(misses)) > 3.0:
        raise SystemExit(f"средний промах {np.mean(misses):.2f} — велик")

    # 4. Укладывается в потолок эпизода 450
    for i in range(15):
        seed = 200000 + i
        pad, vt = goal_of_window(seed)
        st, ac, gr = _run(seed, 8.0, pad, vt)
        if len(ac) >= 450:
            raise SystemExit(f"не уложился в потолок: {len(ac)} шагов, сид {seed}")

    # 5. Пул: формат и доля коснувшихся
    p = hover_pool(500, seed=300000)
    if p["states"].shape[1:] != (TRANSITIONS + 1, 5):
        raise SystemExit(p["states"].shape)
    if p["actions"].shape[1] != TRANSITIONS:
        raise SystemExit(p["actions"].shape)
    if p["landed_frac"] < 0.9:
        raise SystemExit(f"доля коснувшихся {p['landed_frac']}")
    if p["gs"].max() >= G_HI:
        raise SystemExit(f"тяжесть пула вышла за обучение: {p['gs'].max()}")
    if set(np.unique(p["vys"])) - {0.0, 3.0, 6.0}:
        raise SystemExit("цель пула вне {0,3,6}")

    # 6. Правило тормозит при любой тяжести: тяга строго больше g в обучении
    if not (MAX_THRUST > G_HI):
        raise SystemExit("тяга не перекрывает всю обученную тяжесть")

    # 7. Пакет обучения наведению: тег, путь, отказ чужим тегам
    from numeric_land import train_hover as TH
    if TH.TAG != "numeric-land-hover":
        raise SystemExit(TH.TAG)
    for other in ("numeric-land-goal", "numeric-land-expert", "numeric-land-pad", ""):
        try:
            TH.refuse_tag(other)
        except SystemExit:
            pass
        else:
            raise SystemExit(f"пакет наведения принял чужой тег {other!r}")
    if not TH.ckpt_path("vis", 0).endswith("landh_range_vis_s0.pt"):
        raise SystemExit(TH.ckpt_path("vis", 0))
    for name in ("landg", "lande"):
        try:
            TH._save(os.path.join(root, "out", name + "_x.pt"), {})
        except RuntimeError:
            pass
        else:
            raise SystemExit("пакет наведения пишет чужое имя файла")

    # 8. Прежние контроли живы
    for chk in ("_check_numeric_land_field.py", "_check_numeric_land_pad.py",
                "_check_numeric_land_goal.py", "_check_numeric_land_expert.py",
                "_check_numeric_land_tracks.py"):
        r = subprocess.run([sys.executable, os.path.join(root, chk)], cwd=root)
        if r.returncode != 0:
            raise SystemExit(f"{chk} не прошёл")
    print("numeric land hover: ок")


if __name__ == "__main__":
    main()

"""Контроль головы режима. python _check_numeric_land_mode.py"""
import os
import subprocess
import sys

from numeric_land.expert_hover import hover_action, hover_mode
from numeric_land.field import step_field
from numeric_land.mode import (
    HELD_SEED0, HELD_N, LOOP_SEED0, TRAIN_N, TRAIN_SEED0,
    feat_latent, feat_state, feat_zblind, n_in_of,
)
from numeric_land import train_mode as TM
from numeric_land import train_hover as TH


def _expect(state, g, pad, vt, mode, action):
    got_mode = hover_mode(state, g, pad, vt)
    got_action = hover_action(state, g, pad, vt)
    if got_mode != mode or got_action != action:
        raise SystemExit(f"{state} pad={pad}: ждали {mode}/{action}, получили {got_mode}/{got_action}")
    if got_mode is not None:
        same = hover_action(state, g, pad, vt, mode=got_mode)
        if same != got_action:
            raise SystemExit("явный режим разошёлся с правилом")


def main():
    root = os.path.dirname(os.path.abspath(__file__))

    # Граница допуска и пустая метка. Индексы посчитаны по веткам правила.
    _expect((100, 400, 0, 0, 10), 8, 256, 6, "approach", 5)
    _expect((256, 400, 0, 0, 10), 8, 256, 6, "land", 1)
    _expect((200, 500, 0, 0, 10), 8, 206, 6, "land", 2)
    _expect((200, 500, 0, 0, 10), 8, 206.1, 6, "approach", 8)
    _expect((256, 512, 0, 0, 10), 8, 256, 6, None, 1)
    _expect((256, 400, 0, 0, 0), 8, 256, 6, None, 1)

    # Тяжесть и v_target режим не двигают.
    state = (180, 450, 1, 0, 10)
    base = hover_mode(state, 8, 256, 6)
    if hover_mode(state, 2, 256, 0) != base or hover_mode(state, 20, 256, 6) != base:
        raise SystemExit("режим зависит от тяжести или v_target")

    # Один и тот же шаг физики не смотрит на метку.
    a = step_field(256, 464, 0, 0, 10, False, 1, 8)
    b = step_field(256, 464, 0, 0, 10, False, 1, 8)
    if a != b:
        raise SystemExit("шаг физики не повторился")

    # Входы: площадка есть у боевых голов и нет у слепого латента. g в вектор не входит.
    import numpy as np
    states = np.zeros((2, 5), np.float64)
    states[:, 0] = 256
    states[:, 4] = 10
    pads = np.asarray([200.0, 406.0])
    z = np.zeros((2, 32), np.float32)
    z[:, 0] = 1
    if feat_latent(z, pads).shape[1] != 33 or n_in_of("latent") != 33:
        raise SystemExit("латентная голова")
    if feat_state(states, pads).shape[1] != 6 or n_in_of("state") != 6:
        raise SystemExit("голова состояния")
    if feat_zblind(z).shape[1] != 32 or n_in_of("zblind") != 32:
        raise SystemExit("слепой латент")
    if abs(float(feat_latent(z, pads)[0, -1]) - 200 / 512) > 1e-6:
        raise SystemExit("канал площадки")
    if feat_zblind(z).shape[1] != feat_latent(z, pads).shape[1] - 1:
        raise SystemExit("слепой латент не короче на канал площадки")

    train = set(range(TRAIN_SEED0, TRAIN_SEED0 + TRAIN_N))
    held = set(range(HELD_SEED0, HELD_SEED0 + HELD_N))
    loop = set(range(LOOP_SEED0, LOOP_SEED0 + 30))
    if train & held or train & loop or held & loop:
        raise SystemExit("сиды пересеклись")
    if 200000 in train or 97000 in train or 200000 in held:
        raise SystemExit("сиды учителя попали в опубликованные диапазоны")

    def _stat(path):
        if not os.path.isfile(path):
            return None
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)

    landm = os.path.join(root, "out", "landm_latent_s0.pt")
    landh = os.path.join(root, "out", "landh_range_vis_s0.pt")
    before_m = _stat(landm)
    before_h = _stat(landh)
    if before_h is None:
        raise SystemExit("нет landh_range_vis_s0.pt")

    bad = subprocess.run(
        [sys.executable, "-m", "numeric_land.train_mode", "--tag", "numeric-land-hover"],
        cwd=root)
    if bad.returncode == 0:
        raise SystemExit("train_mode принял чужой тег")
    if _stat(landm) != before_m:
        raise SystemExit("чужой тег задел landm_latent_s0.pt")

    bad_h = subprocess.run(
        [sys.executable, "-m", "numeric_land.train_hover", "--tag", "numeric-land-mode"],
        cwd=root)
    if bad_h.returncode == 0:
        raise SystemExit("train_hover принял тег головы")
    if _stat(landh) != before_h:
        raise SystemExit("тег головы переписал landh_range_vis_s0.pt")

    try:
        TM._save(os.path.join(root, "out", "landh_x.pt"), {})
    except RuntimeError:
        pass
    else:
        raise SystemExit("голова пишет чужое имя")
    try:
        TH.refuse_tag("numeric-land-mode")
    except SystemExit:
        pass
    else:
        raise SystemExit("пакет наведения не отказал тегу головы")

    nested = subprocess.run(
        [sys.executable, os.path.join(root, "_check_numeric_land_hover.py")],
        cwd=root)
    if nested.returncode != 0:
        raise SystemExit("контроль наведения не прошёл")
    print("numeric land mode: ок")


if __name__ == "__main__":
    main()

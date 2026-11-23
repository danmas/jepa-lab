"""Замкнутый контур латентной головы seed 0: порог 0.7 и запас 12.

    python -m numeric_land.rollout_mode_gate

Старые столбцы читаются из out/numeric_land_mode.json и заново не летают.
Пишется только out/numeric_land_mode_gate.json. Планировщик не вызывается.
Обучения нет. Порог 0.5 в чекпоинте и NAV_TOL не двигаются.
"""
import json
import os

import numpy as np
import torch

from numeric_land.expert_hover import hover_action, hover_mode
from numeric_land.field import scale_state, start_and_actions
from numeric_land.mode import (
    LOOP_N, LOOP_SEED0, PASS_G, PASS_PADS, V_TARGET_LOOP, features_of,
)
from numeric_land.rollout_mode import THREADS, _count_cell, _load_head

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_JSON = os.path.join(ROOT, "out", "numeric_land_mode.json")
OUT_JSON = os.path.join(ROOT, "out", "numeric_land_mode_gate.json")
CKPT = os.path.join(ROOT, "out", "landm_latent_s0.pt")

GATE_P = 0.7
GATE_DX = 12.0
OLD_RULE_SOFT = 180
OLD_LATENT_SOFT = 62
OLD_GAP = 118
HELP_AT = 80


def gate_mode(prob, dx) -> str:
    """Посадка только если оба условия выполнены. Иначе наведение.

    Равенство p = 0.7 и |dx| = 12 разрешает посадку. Ниже порога или дальше
    запаса — наведение. Сравнение идёт по сырой вероятности.
    """
    if float(prob) >= GATE_P and abs(float(dx)) <= GATE_DX:
        return "land"
    return "approach"


def gate_action(state, g, pad_x, prob) -> int:
    """Пустой режим правила даёт действие 1. Иначе прежняя ветка hover_action.

    Вероятность и запас на пустой режим не смотрят. Новой формулы тяги нет.
    """
    if hover_mode(state, g, pad_x, V_TARGET_LOOP) is None:
        return 1
    dx = float(pad_x) - float(state[0])
    return hover_action(state, g, pad_x, V_TARGET_LOOP, mode=gate_mode(prob, dx))


def is_helped(gate_soft: int) -> bool:
    """Истина только от 80 мягких из 180. 79 помощью не считается."""
    return int(gate_soft) >= HELP_AT


def _stamp(path: str):
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _must_keep(before: dict) -> None:
    for path, stamp in before.items():
        if _stamp(path) != stamp:
            raise SystemExit(f"файл изменился до записи отчёта: {os.path.basename(path)}")


def _read_old():
    with open(OLD_JSON, encoding="utf-8") as handle:
        old = json.load(handle)
    rule_soft = int(old["rule_soft_pass"])
    latent_soft = int(old["latent_soft_pass"])
    if rule_soft != OLD_RULE_SOFT or latent_soft != OLD_LATENT_SOFT:
        raise SystemExit(
            f"старая база {rule_soft}/{latent_soft}, ждали {OLD_RULE_SOFT}/{OLD_LATENT_SOFT}. "
            "Отчёт ворот не пишется.")
    if int(old["pass_n"]) != OLD_RULE_SOFT or int(old["loop_n"]) != LOOP_N:
        raise SystemExit("сетка старого отчёта не та. Отчёт ворот не пишется.")
    if float(old["v_target"]) != V_TARGET_LOOP or float(old["threshold"]) != 0.5:
        raise SystemExit("старый отчёт не про порог 0.5 и v_target 6")
    return old


def _copy_cells(old) -> list:
    """Шесть зачётных клеток правила и латента seed 0. Вне зачёта не копируется."""
    found = {}
    for cell in old["cells"]:
        if cell.get("in_pass") is not True:
            continue
        name = cell.get("controller")
        if name not in ("rule", "latent"):
            continue
        key = (float(cell["g"]), float(cell["pad_x"]), name)
        if key in found:
            raise SystemExit(f"дубль клетки {key}")
        soft = int(cell["soft"])
        crash = int(cell["crash"])
        miss = int(cell["miss"])
        if int(cell["of"]) != LOOP_N or soft + crash + miss != LOOP_N:
            raise SystemExit(f"клетка {key} не из {LOOP_N}")
        found[key] = {"soft": soft, "crash": crash, "miss": miss}
    rows = []
    for g in PASS_G:
        for pad in PASS_PADS:
            rule = found.get((float(g), float(pad), "rule"))
            latent = found.get((float(g), float(pad), "latent"))
            if rule is None or latent is None:
                raise SystemExit(f"нет зачётной клетки g={g:g} pad={pad:g}")
            rows.append({
                "g": float(g),
                "pad_x": float(pad),
                "of": LOOP_N,
                "rule": rule,
                "latent": latent,
            })
    rule_sum = sum(row["rule"]["soft"] for row in rows)
    latent_sum = sum(row["latent"]["soft"] for row in rows)
    if rule_sum != OLD_RULE_SOFT or latent_sum != OLD_LATENT_SOFT:
        raise SystemExit(
            f"сумма клеток {rule_sum}/{latent_sum} не сходится с итогом. Отчёт ворот не пишется.")
    return rows


def _check_start(old):
    x, y, _vx, _vy, _actions = start_and_actions(LOOP_SEED0, 0)
    ref = old["start_99000"]
    if abs(float(x) - float(ref["x"])) > 1e-9 or abs(float(y) - float(ref["y"])) > 1e-9:
        raise SystemExit(
            f"старт {LOOP_SEED0} разошёлся с JSON: {(x, y)} vs {(ref['x'], ref['y'])}. "
            "Отчёт ворот не пишется.")
    return float(x), float(y)


def _decide(head, encoder, g, pad_x):
    def decide(state):
        if hover_mode(state, g, pad_x, V_TARGET_LOOP) is None:
            return 1
        state_a = np.asarray(state, np.float64).reshape(1, 5)
        pad_a = np.asarray([pad_x], np.float64)
        with torch.no_grad():
            z = encoder(torch.from_numpy(scale_state(state_a))).numpy()
            feat = features_of("latent", state_a, pad_a, z)
            logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
            prob = float(torch.sigmoid(logit).item())
        return gate_action(state, g, pad_x, prob)
    return decide


def main():
    if not os.path.isfile(CKPT) or not os.path.isfile(OLD_JSON):
        raise SystemExit("нет чекпоинта или старого JSON")
    held = {CKPT: _stamp(CKPT), OLD_JSON: _stamp(OLD_JSON)}
    old = _read_old()
    rows = _copy_cells(old)
    start_x, start_y = _check_start(old)
    _must_keep(held)

    torch.set_num_threads(THREADS)
    _ck, encoder, _ema, head = _load_head("latent", 0)
    _must_keep(held)

    gate_soft = 0
    for row in rows:
        g, pad = row["g"], row["pad_x"]
        soft, crash, miss, starts = _count_cell(
            LOOP_SEED0, LOOP_N, g, pad, _decide(head, encoder, g, pad))
        sx, sy, seed0 = starts[0]
        if seed0 != LOOP_SEED0 or abs(sx - start_x) > 1e-9 or abs(sy - start_y) > 1e-9:
            raise SystemExit("старт внутри сетки разошёлся. Отчёт ворот не пишется.")
        if soft + crash + miss != LOOP_N:
            raise SystemExit("счёт клетки не сошёлся")
        row["gate"] = {"soft": int(soft), "crash": int(crash), "miss": int(miss)}
        gate_soft += int(soft)
        print(
            f"gate g={g:g} pad={pad:g} soft={soft} crash={crash} miss={miss}",
            flush=True,
        )

    _must_keep(held)
    if gate_soft > OLD_RULE_SOFT:
        raise SystemExit(f"мягких {gate_soft} больше зачёта")
    helped = is_helped(gate_soft)
    gap = OLD_RULE_SOFT - gate_soft
    if helped and gap >= OLD_GAP:
        raise SystemExit("флаг помощи без более короткого разрыва")
    payload = {
        "tag": "numeric-land-mode-gate",
        "source_json": "numeric_land_mode.json",
        "checkpoint": "landm_latent_s0.pt",
        "head_seed": 0,
        "kind": "latent",
        "threshold": GATE_P,
        "margin": GATE_DX,
        "recorded_threshold": 0.5,
        "nav_tol": 6.0,
        "vy_soft": 8.0,
        "v_target": V_TARGET_LOOP,
        "loop_seed0": LOOP_SEED0,
        "loop_n": LOOP_N,
        "pass_g": [float(g) for g in PASS_G],
        "pass_pads": [float(pad) for pad in PASS_PADS],
        "help_at": HELP_AT,
        "rule_soft_pass": OLD_RULE_SOFT,
        "latent_soft_pass": OLD_LATENT_SOFT,
        "gate_soft_pass": int(gate_soft),
        "pass_n": OLD_RULE_SOFT,
        "helped": bool(helped),
        "gap_episodes": int(gap),
        "old_gap_episodes": OLD_GAP,
        "planner": False,
        "start_99000": {"x": start_x, "y": start_y},
        "cells": rows,
    }
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, OUT_JSON)
    _must_keep(held)
    print(
        f"{OUT_JSON} gate={gate_soft}/{OLD_RULE_SOFT} helped={helped} gap={gap}",
        flush=True,
    )


if __name__ == "__main__":
    main()

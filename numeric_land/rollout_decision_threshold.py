"""Три заранее заданных порога на уже обученной голове. Обучения нет.

    python -m numeric_land.rollout_decision_threshold

Голова `landd_latent_s0`, кодировщик `landr_scratch_s0`. Пороги `0.5`, `0.65`, `0.7`.
Запаса по расстоянию нет. CEM нет. Старые JSON и чекпоинты только читаются.
Сдвиг — первое пересечение порога на отложенных траекториях учителя `98800…98927`.
Мягкие — замкнутый контур, сиды `99000…99029`, `v_target = 6`,
тяжести `2`, `8`, `15`, центры `256` и `200`.
"""
import os

import numpy as np
import torch

from numeric_land.decision import (
    THRESHOLD, cell_is_result, features_of, timing_of,
)
from numeric_land.expert_hover import hover_action, hover_mode
from numeric_land.field import start_and_actions
from numeric_land.mode import LOOP_N, LOOP_SEED0, PASS_G, PASS_PADS, V_TARGET_LOOP
from numeric_land.relenc import (
    OUT, encode_rel, file_stamp, frames_of, held_seeds, require_soft,
    scan_range, tensors_equal, write_json, MIN_HELD_SOFT,
)
from numeric_land.rollout_decision import (
    LANDM_PATH, MODE_JSON, OUT_JSON as DECISION_JSON, _count_cell, _encoder_from,
    _load_head, _load_json, _probs,
)
from numeric_land.train_decision import ckpt_path
from numeric_land.train_relenc import ckpt_path as relenc_ckpt

THREADS = 8
THRESHOLDS = (0.5, 0.65, 0.7)
SOFT_GAIN_MIN = 15
OUT_PATH = os.path.join(OUT, "numeric_land_decision_threshold.json")
HEAD_KIND = "latent"
HEAD_SEED = 0


def _guard_paths():
    if os.path.abspath(OUT_PATH) == os.path.abspath(DECISION_JSON):
        raise RuntimeError("отчёт порога совпал со старым JSON")
    if OUT_PATH.endswith(".pt"):
        raise RuntimeError(OUT_PATH)
    for threshold in THRESHOLDS:
        if float(threshold) == float(THRESHOLD) and THRESHOLDS[0] != 0.5:
            raise RuntimeError("база порога сдвинута")
    if THRESHOLDS != (0.5, 0.65, 0.7):
        raise RuntimeError(THRESHOLDS)
    if abs(float(THRESHOLD) - 0.5) > 0.0:
        raise RuntimeError("боевой порог в коде больше не 0.5")


def _stamps():
    paths = {
        "head": ckpt_path(HEAD_KIND, HEAD_SEED),
        "encoder": relenc_ckpt("scratch", 0),
        "decision_json": DECISION_JSON,
        "mode_json": MODE_JSON,
        "landm": LANDM_PATH,
    }
    stamps = {key: file_stamp(path) for key, path in paths.items()}
    missing = [key for key, stamp in stamps.items() if stamp is None]
    if missing:
        raise SystemExit(f"нет файла для чтения: {missing}")
    return paths, stamps


def _timing_at(split, prob, threshold: float) -> dict:
    """Тот же расчёт, что timing_of, но порог задан аргументом."""
    step_d, dx_d = [], []
    n = 0
    miss = 0
    seeds = split["seed"]
    for seed in np.unique(seeds):
        mask = seeds == seed
        labels = split["label"][mask]
        if not np.any(labels == 1):
            continue
        n += 1
        order = np.argsort(split["step_i"][mask], kind="mergesort")
        lab = labels[order]
        steps = split["step_i"][mask][order]
        dx = split["dx"][mask][order]
        pp = np.asarray(prob)[mask][order]
        t_at = int(np.argmax(lab == 1))
        crossed = np.flatnonzero(pp >= float(threshold))
        if len(crossed) == 0:
            miss += 1
            continue
        h_at = int(crossed[0])
        step_d.append(float(steps[h_at] - steps[t_at]))
        dx_d.append(float(dx[h_at] - dx[t_at]))
    return {
        "n_teacher": int(n),
        "n_miss": int(miss),
        "n_switched": int(n - miss),
        "step": _stat(step_d),
        "dx": _stat(dx_d),
    }


def _stat(values):
    if not values:
        return None
    arr = np.asarray(values, np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "n": int(len(arr)),
        "n_early": int(np.sum(arr < 0.0)),
        "n_late": int(np.sum(arr > 0.0)),
        "n_tie": int(np.sum(arr == 0.0)),
    }


def _decide(head, encoder, g, pad_x, threshold: float):
    """Пустой режим — действие 1. Иначе ветка по заданному порогу, тягу считает правило."""

    def decide(state):
        if hover_mode(state, g, pad_x, V_TARGET_LOOP) is None:
            return 1
        state_a = np.asarray(state, np.float64).reshape(1, 5)
        pads = np.asarray([pad_x], np.float64)
        z = encode_rel(encoder, state_a, pads)
        feat = features_of("latent", pads, np.asarray([g], np.float64), z)
        with torch.no_grad():
            logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
            prob = float(torch.sigmoid(logit).item())
        mode = "land" if prob >= float(threshold) else "approach"
        return hover_action(state, g, pad_x, V_TARGET_LOOP, mode=mode)

    return decide


def _same_timing(left, right) -> bool:
    if int(left["n_teacher"]) != int(right["n_teacher"]):
        return False
    if int(left["n_miss"]) != int(right["n_miss"]):
        return False
    for key in ("step", "dx"):
        a, b = left[key], right[key]
        if a is None or b is None:
            return False
        if int(a["n"]) != int(b["n"]):
            return False
        if abs(float(a["mean"]) - float(b["mean"])) > 0.0:
            return False
        if abs(float(a["median"]) - float(b["median"])) > 0.0:
            return False
    return True


def _published_match(published_switch, timing, published_cells, cells) -> dict:
    published = published_switch
    step = timing["step"]
    dx = timing["dx"]
    timing_ok = (
        step is not None and dx is not None
        and int(timing["n_miss"]) == int(published["n_miss"])
        and int(timing["n_switched"]) == int(published["n_switched"])
        and int(timing["n_teacher"]) == int(published["n_teacher"])
        and abs(float(step["mean"]) - float(published["step"]["mean"])) == 0.0
        and abs(float(step["median"]) - float(published["step"]["median"])) == 0.0
        and abs(float(dx["mean"]) - float(published["dx"]["mean"])) == 0.0
        and abs(float(dx["median"]) - float(published["dx"]["median"])) == 0.0
    )
    cell_ok = True
    for cell in cells:
        prev = published_cells[(cell["g"], cell["pad_x"])]
        if (cell["soft"], cell["crash"], cell["miss"]) != prev:
            cell_ok = False
    return {"timing": bool(timing_ok), "cells": bool(cell_ok)}


def _reading(base, row) -> str:
    if abs(float(row["threshold"]) - 0.5) == 0.0:
        return "база этого прогона"
    gain = int(row["soft"]) - int(base["soft"])
    base_step = base["timing"]["step"]
    row_step = row["timing"]["step"]
    if base_step is None or row_step is None:
        return "простое поднятие порога недостаточно"
    tail = (
        int(row["timing"]["n_miss"]) <= int(base["timing"]["n_miss"])
        and int(row_step["n_early"]) < int(base_step["n_early"])
        and float(row_step["mean"]) > float(base_step["mean"])
    )
    if gain >= SOFT_GAIN_MIN and tail:
        return "порог — полезный рычаг на этом прогоне"
    if gain >= SOFT_GAIN_MIN and not tail:
        return "мягких больше, ранний хвост не сократился"
    return "простое поднятие порога недостаточно"


def _pass_grid():
    rows = []
    for g in PASS_G:
        for pad in PASS_PADS:
            rows.append((float(g), float(pad)))
    return rows


def main():
    _guard_paths()
    torch.set_num_threads(THREADS)
    paths, before = _stamps()
    published_doc = _load_json(DECISION_JSON)
    if int(published_doc["new_soft"]) != 108 or abs(float(published_doc["threshold"]) - 0.5) > 0.0:
        raise SystemExit("в старом отчёте нет базы 108 при пороге 0.5")
    x0, y0, _vx, _vy, _acts = start_and_actions(LOOP_SEED0, 0)
    mode_json = _load_json(MODE_JSON)
    from numeric_land.decision import assert_start
    assert_start(x0, y0, mode_json["start_99000"])

    held0, held_n = held_seeds()
    episodes = scan_range(held0, held_n, "порог, отложенные")
    require_soft(episodes, MIN_HELD_SOFT, "порог, отложенные")
    split = frames_of(episodes, soft_only=False)
    ck, head = _load_head(HEAD_KIND, HEAD_SEED)
    if os.path.basename(paths["head"]) != "landd_latent_s0.pt":
        raise RuntimeError(paths["head"])
    encoder, ema = _encoder_from(ck, HEAD_KIND)
    source = torch.load(paths["encoder"], map_location="cpu", weights_only=False)
    if file_stamp(paths["encoder"]) != before["encoder"]:
        raise RuntimeError("чтение кодировщика изменило файл")
    if not tensors_equal(encoder.state_dict(), source["encoder"]):
        raise RuntimeError("голова несёт не тот кодировщик")
    if not tensors_equal(ema.state_dict(), source["ema"]):
        raise RuntimeError("голова несёт не ту EMA")
    del source

    prob = _probs(HEAD_KIND, head, encoder, split)
    official = timing_of(split, prob)
    check = _timing_at(split, prob, 0.5)
    if not _same_timing(official, check):
        raise RuntimeError("порог 0.5 не совпал с timing_of")

    published_switch = published_doc["switch"]["all"]["latent"]
    published_cells = {
        (float(cell["g"]), float(cell["pad_x"])): (int(cell["soft"]), int(cell["crash"]), int(cell["miss"]))
        for cell in published_doc["cells"]
        if cell["controller"] == "latent" and cell["in_pass"] is True
    }
    if len(published_cells) != 6:
        raise RuntimeError("в старом отчёте не шесть клеток зачёта")

    rows = []
    grid = _pass_grid()
    if len(grid) * LOOP_N != 180:
        raise RuntimeError("сетка не 180")
    for threshold in THRESHOLDS:
        timing = _timing_at(split, prob, threshold)
        cells = []
        for g, pad in grid:
            decide = _decide(head, encoder, g, pad, threshold)
            soft, crash, miss, start = _count_cell(g, pad, decide)
            assert_start(start[0], start[1], mode_json["start_99000"])
            cells.append({
                "g": g, "pad_x": pad, "soft": int(soft), "crash": int(crash),
                "miss": int(miss), "of": int(LOOP_N),
            })
            print(
                f"порог {threshold:g} g={g:g} pad={pad:g} "
                f"soft={soft} crash={crash} miss={miss}",
                flush=True,
            )
        soft_total = int(sum(cell["soft"] for cell in cells))
        row = {
            "threshold": float(threshold),
            "soft": soft_total,
            "of": 180,
            "timing": timing,
            "cells": cells,
        }
        if abs(float(threshold) - 0.5) == 0.0:
            row["matches_published"] = _published_match(
                published_switch, timing, published_cells, cells)
        rows.append(row)
        step = timing["step"]
        print(
            f"порог {threshold:g} мягких {soft_total}/180  "
            f"шаг mean={None if step is None else step['mean']} "
            f"median={None if step is None else step['median']}  "
            f"n_miss={timing['n_miss']} n_early={None if step is None else step['n_early']}",
            flush=True,
        )

    base = rows[0]
    for row in rows:
        row["soft_minus_0_5"] = int(row["soft"]) - int(base["soft"])
        row["reading"] = _reading(base, row)
        compared = []
        by_key = {(cell["g"], cell["pad_x"]): cell for cell in base["cells"]}
        for cell in row["cells"]:
            prev = by_key[(cell["g"], cell["pad_x"])]
            delta = int(cell["soft"]) - int(prev["soft"])
            compared.append({
                "g": cell["g"],
                "pad_x": cell["pad_x"],
                "soft": cell["soft"],
                "soft_0_5": prev["soft"],
                "delta_soft": delta,
                "counts_as_cell_result": bool(cell_is_result(delta)) if row is not base else False,
            })
        row["cell_vs_0_5"] = compared

    after = {key: file_stamp(path) for key, path in paths.items()}
    if after != before:
        raise SystemExit(f"чтение изменило файл: {before} -> {after}")

    payload = {
        "tag": "numeric-land-decision-threshold",
        "head": "landd_latent_s0.pt",
        "encoder": "landr_scratch_s0.pt",
        "checkpoint_threshold": float(ck.get("threshold")),
        "trained": False,
        "cem": False,
        "distance_margin": False,
        "probe_in_loop": False,
        "skills_choose_action": False,
        "thresholds": [float(item) for item in THRESHOLDS],
        "soft_gain_min": SOFT_GAIN_MIN,
        "early_tail": "сдвиг шага < 0 на траектории учителя; пропуск в n_early не входит",
        "v_target": float(V_TARGET_LOOP),
        "seeds": [int(LOOP_SEED0), int(LOOP_SEED0 + LOOP_N - 1)],
        "g": [float(g) for g in PASS_G],
        "pads": [float(pad) for pad in PASS_PADS],
        "timing_seeds": [int(held0), int(held0 + held_n - 1)],
        "published_soft_0_5": 108,
        "rows": rows,
    }
    write_json(OUT_PATH, payload)
    if file_stamp(paths["head"]) != before["head"]:
        raise SystemExit("запись отчёта задела чекпоинт")
    print(os.path.abspath(OUT_PATH), flush=True)
    for row in rows:
        print(f"итог {row['threshold']:g}: {row['soft']}/180  {row['reading']}", flush=True)


if __name__ == "__main__":
    main()

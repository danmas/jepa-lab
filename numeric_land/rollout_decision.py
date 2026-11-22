"""Замкнутый контур: ветку называет голова, действие берёт правило.

    python -m numeric_land.rollout_decision

Правило и голова режима заново не летают. Probe и CEM не вызываются.
Навык действие не выбирает.
"""
import json
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.decision import (
    THRESHOLD, DecisionHead, abs_ratio, assert_baselines, assert_start,
    better_ok, binary_scores, counts_toward_90, features_of, gap_ok,
    latent_helped, line_90, n_in_of, pass_cells, pilot_passes, subset_split,
    switch_flag, timing_of, cell_is_result,
)
from numeric_land.expert_hover import HOVER_CAP, hover_action, hover_mode
from numeric_land.field import FUEL0, start_and_actions, step_field
from numeric_land.mode import (
    LOOP_N, LOOP_SEED0, V_TARGET_LOOP,
)
from numeric_land.models import BallEncoder
from numeric_land.pad import outcome_at
from numeric_land.relenc import (
    MIN_HELD_SOFT, OUT, PACKAGE_SENTENCE, TAG, RelEncoder, encode_ball, encode_rel,
    file_stamp, frames_of, held_seeds, landf_path, require_soft, scan_range,
    tensors_equal, write_json,
)
from numeric_land.train_decision import ckpt_path
from numeric_land.train_field import EMA_MOMENTUM
from numeric_land.train_relenc import ckpt_path as relenc_ckpt

OUT_JSON = os.path.join(OUT, "numeric_land_decision.json")
RELENC_JSON = os.path.join(OUT, "numeric_land_relenc.json")
MODE_JSON = os.path.join(OUT, "numeric_land_mode.json")
LANDM_PATH = os.path.join(OUT, "landm_latent_s0.pt")
THREADS = 8


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_head(kind: str, seed: int):
    path = ckpt_path(kind, seed)
    name = os.path.basename(path)
    if not name.startswith("landd_") or name.startswith("landm_"):
        raise RuntimeError(name)
    if not os.path.isfile(path):
        raise SystemExit(f"нет {name}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != TAG or ck.get("kind") != kind or int(ck.get("seed")) != int(seed):
        raise RuntimeError(path)
    if float(ck.get("threshold")) != THRESHOLD or int(ck.get("n_in")) != n_in_of(kind):
        raise RuntimeError(path)
    head = DecisionHead(n_in_of(kind))
    head.load_state_dict(ck["head"])
    head.eval()
    return ck, head


def _encoder_from(ck, kind: str):
    if kind == "oldz":
        encoder = BallEncoder()
    else:
        encoder = RelEncoder()
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    return encoder, ema


def _probs(kind, head, encoder, split):
    if kind == "oldz":
        z = encode_ball(encoder, split["state"])
    else:
        z = encode_rel(encoder, split["state"], split["pad_x"])
    feat = features_of(kind, split["pad_x"], split["g"], z)
    if feat.shape[1] != n_in_of(kind):
        raise RuntimeError((kind, feat.shape))
    with torch.no_grad():
        logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
        return torch.sigmoid(logit).numpy()


def _decide(kind, head, encoder, g, pad_x):
    """Пустой режим — действие 1. Иначе ветка по порогу 0.5, тягу считает правило."""

    def decide(state):
        if hover_mode(state, g, pad_x, V_TARGET_LOOP) is None:
            return 1
        state_a = np.asarray(state, np.float64).reshape(1, 5)
        if kind == "oldz":
            z = encode_ball(encoder, state_a)
        else:
            z = encode_rel(encoder, state_a, np.asarray([pad_x], np.float64))
        feat = features_of(kind, np.asarray([pad_x], np.float64), np.asarray([g], np.float64), z)
        with torch.no_grad():
            logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
            prob = float(torch.sigmoid(logit).item())
        mode = "land" if prob >= THRESHOLD else "approach"
        return hover_action(state, g, pad_x, V_TARGET_LOOP, mode=mode)

    return decide


def _episode(seed, g, pad_x, decide):
    x, y, vx, vy, _ = start_and_actions(int(seed), 0)
    fuel = float(FUEL0)
    grounded = False
    start_x, start_y = float(x), float(y)
    steps = 0
    while steps < HOVER_CAP and not grounded:
        state = (float(x), float(y), float(vx), float(vy), float(fuel))
        action = int(decide(state))
        x, y, vx, vy, fuel, grounded = step_field(
            x, y, vx, vy, fuel, grounded, action, float(g))
        steps += 1
    return {
        "seed": int(seed),
        "result": outcome_at(x, y, vy, grounded, pad_x),
        "start_x": start_x,
        "start_y": start_y,
    }


def _count_cell(g, pad_x, decide):
    soft = crash = miss = 0
    start = None
    for seed in range(LOOP_SEED0, LOOP_SEED0 + LOOP_N):
        ep = _episode(seed, g, pad_x, decide)
        if start is None:
            start = (ep["start_x"], ep["start_y"])
        if ep["result"] == "мягкая":
            soft += 1
        elif ep["result"] == "удар":
            crash += 1
        elif ep["result"] == "не сел":
            miss += 1
        else:
            raise RuntimeError(ep["result"])
    return soft, crash, miss, start


def _published_timing(mode_json):
    for head in mode_json["heads"]:
        if head.get("kind") == "latent" and int(head.get("seed")) == 0:
            return head["timing"]
    raise RuntimeError("в базе нет опубликованного сдвига landm latent s0")


def _check_sources(loaded):
    scratch = torch.load(relenc_ckpt("scratch", 0), map_location="cpu", weights_only=False)
    landf = torch.load(landf_path(), map_location="cpu", weights_only=False)
    for kind, ck, encoder, ema in loaded:
        source = landf if kind == "oldz" else scratch
        if not tensors_equal(encoder.state_dict(), source["encoder"]):
            raise RuntimeError(f"кодировщик {kind} не совпал с источником")
        if not tensors_equal(ema.state_dict(), source["ema"]):
            raise RuntimeError(f"EMA {kind} не совпала с источником")


def main():
    torch.set_num_threads(THREADS)
    stamps = {
        "mode": file_stamp(MODE_JSON),
        "landm": file_stamp(LANDM_PATH),
        "landf": file_stamp(landf_path()),
    }
    if stamps["mode"] is None or stamps["landm"] is None or stamps["landf"] is None:
        raise SystemExit("нет базы режима, головы режима или landf")
    if not os.path.isfile(RELENC_JSON):
        raise SystemExit("нет numeric_land_relenc.json, отчёт решения не публикуется")
    mode_json = _load_json(MODE_JSON)
    assert_baselines(mode_json["rule_soft_pass"], mode_json["latent_soft_pass"])
    x0, y0, _vx, _vy, _acts = start_and_actions(LOOP_SEED0, 0)
    assert_start(x0, y0, mode_json["start_99000"])
    published = _published_timing(mode_json)
    relenc = _load_json(RELENC_JSON)

    held0, held_n = held_seeds()
    episodes = scan_range(held0, held_n, "решение, отложенные")
    require_soft(episodes, MIN_HELD_SOFT, "решение, отложенные")
    soft_seeds = [ep["seed"] for ep in episodes if ep["soft"]]
    split_soft = frames_of(episodes, soft_only=True)
    split_all = frames_of(episodes, soft_only=False)

    heads = []
    loaded = []
    bundle = {}
    for kind, seed in (
        ("latent", 0), ("latent", 1), ("latent", 2),
        ("zblind", 0), ("g", 0), ("oldz", 0),
    ):
        ck, head = _load_head(kind, seed)
        encoder, ema = _encoder_from(ck, kind)
        loaded.append((kind, ck, encoder, ema))
        prob = _probs(kind, head, encoder, split_soft)
        scores = binary_scores(split_soft["label"], prob)
        row = {
            "file": os.path.basename(ckpt_path(kind, seed)),
            "kind": kind,
            "seed": int(seed),
            "seconds": ck.get("seconds"),
            "pos_weight": ck.get("pos_weight"),
            "n_in": n_in_of(kind),
            "counts_toward_90": counts_toward_90(kind),
            "land_frac": float(np.mean(split_soft["label"] == 1)),
            **scores,
        }
        heads.append(row)
        bundle[(kind, seed)] = (head, encoder)
        print(f"{row['file']} acc={scores['accuracy']:.4f} f1={scores['f1']:.4f}", flush=True)
    _check_sources(loaded)
    passed_90 = line_90(heads)

    latent_head, latent_enc = bundle[("latent", 0)]
    old_head, old_enc = bundle[("oldz", 0)]
    timing_new = timing_of(split_all, _probs("latent", latent_head, latent_enc, split_all))
    timing_old = timing_of(split_all, _probs("oldz", old_head, old_enc, split_all))
    soft_split = subset_split(split_all, soft_seeds)
    timing_new_soft = timing_of(soft_split, _probs("latent", latent_head, latent_enc, soft_split))
    timing_old_soft = timing_of(soft_split, _probs("oldz", old_head, old_enc, soft_split))
    flag_step = switch_flag(timing_new, timing_old, "step")
    flag_dx = switch_flag(timing_new, timing_old, "dx")
    step_new = None if timing_new["step"] is None else timing_new["step"]["mean"]
    step_old = None if timing_old["step"] is None else timing_old["step"]["mean"]
    dx_new = None if timing_new["dx"] is None else timing_new["dx"]["mean"]
    dx_old = None if timing_old["dx"] is None else timing_old["dx"]["mean"]
    print(f"сдвиг шага {step_new} против {step_old}  флаг {flag_step}", flush=True)
    print(f"сдвиг dx {dx_new} против {dx_old}  флаг {flag_dx}", flush=True)

    cells = []
    start_seen = None
    for g, pad, in_pass in pass_cells():
        for kind in ("latent", "oldz"):
            head, encoder = bundle[(kind, 0)]
            soft, crash, miss, start = _count_cell(g, pad, _decide(kind, head, encoder, g, pad))
            if start_seen is None:
                start_seen = start
                assert_start(start[0], start[1], mode_json["start_99000"])
            elif abs(start[0] - start_seen[0]) > 1e-9 or abs(start[1] - start_seen[1]) > 1e-9:
                raise RuntimeError(f"старт 99000 разъехался: {start} vs {start_seen}")
            cells.append({
                "controller": kind, "g": g, "pad_x": pad, "in_pass": in_pass,
                "soft": int(soft), "crash": int(crash), "miss": int(miss), "of": LOOP_N,
            })
            mark = "" if in_pass else "  вне зачёта"
            print(f"{kind} g={g:g} pad={pad:g} soft={soft} crash={crash} miss={miss}{mark}",
                  flush=True)

    def _soft(controller, in_pass):
        return int(sum(
            cell["soft"] for cell in cells
            if cell["controller"] == controller and cell["in_pass"] is in_pass))

    new_soft = _soft("latent", True)
    oldz_soft = _soft("oldz", True)
    n_pass = sum(1 for cell in cells if cell["controller"] == "latent" and cell["in_pass"]) * LOOP_N
    if n_pass != 180:
        raise RuntimeError(n_pass)
    paired = []
    for cell in cells:
        if cell["controller"] != "latent":
            continue
        other = next(
            item for item in cells
            if item["controller"] == "oldz" and item["g"] == cell["g"] and item["pad_x"] == cell["pad_x"])
        delta = int(cell["soft"]) - int(other["soft"])
        paired.append({
            "g": cell["g"], "pad_x": cell["pad_x"], "in_pass": cell["in_pass"],
            "latent_soft": cell["soft"], "oldz_soft": other["soft"],
            "delta_soft": delta, "counts_as_cell_result": cell_is_result(delta),
        })
    flag_distance = bool(relenc["flag_distance"])
    flag_height = bool(relenc["flag_height"])
    helped = latent_helped(new_soft, oldz_soft)
    payload = {
        "tag": TAG,
        "threshold": THRESHOLD,
        "nav_tol": 6.0,
        "vy_soft": 8.0,
        "v_target": V_TARGET_LOOP,
        "package": PACKAGE_SENTENCE,
        "w_probe": 0.1,
        "cem": False,
        "probe_in_loop": False,
        "skills_choose_action": False,
        "rule_soft_pass": int(mode_json["rule_soft_pass"]),
        "mode_soft_pass": int(mode_json["latent_soft_pass"]),
        "published_landm_latent_s0_timing": published,
        "published_timing_recomputed": False,
        "heads": heads,
        "line_90": bool(passed_90),
        "switch": {
            "flag_step": bool(flag_step),
            "flag_dx": bool(flag_dx),
            "step_mean_new": step_new,
            "step_mean_oldz": step_old,
            "step_ratio": abs_ratio(step_new, step_old),
            "dx_mean_new": dx_new,
            "dx_mean_oldz": dx_old,
            "dx_ratio": abs_ratio(dx_new, dx_old),
            "n_miss_new": timing_new["n_miss"],
            "n_miss_oldz": timing_old["n_miss"],
            "n_switched_new": timing_new["n_switched"],
            "n_switched_oldz": timing_old["n_switched"],
            "n_teacher": timing_new["n_teacher"],
            "all": {"latent": timing_new, "oldz": timing_old},
            "successful_only_not_a_flag": {"latent": timing_new_soft, "oldz": timing_old_soft},
        },
        "probe": {
            "flag_distance": flag_distance,
            "flag_height": flag_height,
            "distance_l1": relenc["scratch"]["0"]["ridge_l1"][0],
            "height_l1": relenc["scratch"]["0"]["ridge_l1"][1],
            "landf_distance_l1": relenc["landf"]["ridge_l1"][0],
            "landf_height_l1": relenc["landf"]["ridge_l1"][1],
            "distance_ratio": relenc["scratch"]["0"]["ridge_ratio"][0],
            "height_ratio": relenc["scratch"]["0"]["ridge_ratio"][1],
        },
        "cells": cells,
        "cell_pairs": paired,
        "new_soft": new_soft,
        "oldz_soft": oldz_soft,
        "pass_n": n_pass,
        "gap_ok": bool(gap_ok(new_soft)),
        "better_ok": bool(better_ok(new_soft)),
        "latent_helped": bool(helped),
        "pilot_pass": bool(pilot_passes(
            flag_distance, flag_height, flag_step, flag_dx, passed_90, new_soft, oldz_soft)),
    }
    for key, stamp in stamps.items():
        path = {"mode": MODE_JSON, "landm": LANDM_PATH, "landf": landf_path()}[key]
        if file_stamp(path) != stamp:
            raise RuntimeError(f"{key} изменился, отчёт не публикуется")
    write_json(OUT_JSON, payload)
    print(OUT_JSON, flush=True)
    print(f"мягких новая {new_soft}/180  oldz {oldz_soft}/180  "
          f"gap_ok {payload['gap_ok']}  better_ok {payload['better_ok']}  "
          f"latent_helped {payload['latent_helped']}  пилот {payload['pilot_pass']}",
          flush=True)


if __name__ == "__main__":
    main()

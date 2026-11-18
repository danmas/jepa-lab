"""Отложенная точность головы и замкнутый контур против явного правила.

    python -m numeric_land.rollout_mode

CEM нет. Порог 0.5 не подбирается. JSON пишется один раз в конце.
"""
import json
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.expert_hover import HOVER_CAP, hover_action, hover_mode
from numeric_land.field import FUEL0, scale_state, start_and_actions, step_field
from numeric_land.mode import (
    EXTRA_G, EXTRA_PAD, HELD_N, HELD_SEED0, LOOP_N, LOOP_SEED0,
    PASS_G, PASS_PADS, THRESHOLD, V_TARGET_LOOP,
    ModeHead, collect_split, encode_rows, features_of, n_in_of,
)
from numeric_land.models import BallEncoder
from numeric_land.pad import outcome_at
from numeric_land.train_mode import TAG, _load_source, _same, ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_land_mode.json")
THREADS = 8
PASS_N = len(PASS_G) * len(PASS_PADS) * LOOP_N
RULE_FLOOR = 0.80
GAP_LIMIT = 0.15


def _load_head(kind: str, seed: int):
    path = ckpt_path(kind, seed)
    if not os.path.isfile(path):
        raise SystemExit(f"нет {os.path.basename(path)}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != TAG or ck.get("kind") != kind or int(ck.get("seed")) != int(seed):
        raise RuntimeError(path)
    if float(ck.get("threshold")) != THRESHOLD:
        raise RuntimeError(ck.get("threshold"))
    if int(ck.get("n_in")) != n_in_of(kind):
        raise RuntimeError(ck.get("n_in"))
    encoder = BallEncoder()
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()
    ema = EMATarget(encoder, momentum=0.99)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    head = ModeHead(n_in_of(kind))
    head.load_state_dict(ck["head"])
    head.eval()
    return ck, encoder, ema, head


def _probs(kind, head, encoder, split):
    if kind == "state":
        z = np.zeros((len(split["label"]), 32), np.float32)
    else:
        z = encode_rows(encoder, split["state"])
    feat = features_of(kind, split["state"], split["pad_x"], z)
    with torch.no_grad():
        logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
        return torch.sigmoid(logit).numpy()


def _binary(labels, prob):
    pred = prob >= THRESHOLD
    truth = labels == 1
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def _timing(split, prob):
    """Первый шаг посадки учителя против первого p ≥ 0.5. Промах остаётся в знаменателе."""
    step_d, height_d, dx_d = [], [], []
    n = 0
    miss = 0
    for seed in np.unique(split["seed"]):
        mask = split["seed"] == seed
        labels = split["label"][mask]
        if not np.any(labels == 1):
            continue
        n += 1
        order = np.argsort(split["step_i"][mask], kind="mergesort")
        lab = labels[order]
        steps = split["step_i"][mask][order]
        height = split["height"][mask][order]
        dx = split["dx"][mask][order]
        pp = prob[mask][order]
        t_at = int(np.argmax(lab == 1))
        crossed = np.flatnonzero(pp >= THRESHOLD)
        if len(crossed) == 0:
            miss += 1
            continue
        h_at = int(crossed[0])
        step_d.append(float(steps[h_at] - steps[t_at]))
        height_d.append(float(height[h_at] - height[t_at]))
        dx_d.append(float(dx[h_at] - dx[t_at]))
    def _stat(values):
        if not values:
            return None
        arr = np.asarray(values, np.float64)
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "n": int(len(arr))}
    return {
        "n_teacher": n,
        "n_miss": miss,
        "n_switched": n - miss,
        "step": _stat(step_d),
        "height": _stat(height_d),
        "dx": _stat(dx_d),
    }


def _cells():
    rows = []
    for g in PASS_G:
        for pad in PASS_PADS:
            rows.append((float(g), float(pad), True))
    for g in PASS_G:
        rows.append((float(g), float(EXTRA_PAD), False))
    for pad in list(PASS_PADS) + [float(EXTRA_PAD)]:
        rows.append((float(EXTRA_G), float(pad), False))
    return rows


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
    result = outcome_at(x, y, vy, grounded, pad_x)
    return {
        "seed": int(seed), "result": result, "steps": int(steps),
        "start_x": start_x, "start_y": start_y,
        "x": float(x), "y": float(y), "vy": float(vy),
    }


def _decide_rule(g, pad_x):
    def decide(state):
        return hover_action(state, g, pad_x, V_TARGET_LOOP)
    return decide


def _decide_head(kind, head, encoder, g, pad_x):
    def decide(state):
        if hover_mode(state, g, pad_x, V_TARGET_LOOP) is None:
            return 1
        state_a = np.asarray(state, np.float64).reshape(1, 5)
        pad_a = np.asarray([pad_x], np.float64)
        if kind == "state":
            z = np.zeros((1, 32), np.float32)
        else:
            with torch.no_grad():
                z = encoder(torch.from_numpy(scale_state(state_a))).numpy()
        feat = features_of(kind, state_a, pad_a, z)
        with torch.no_grad():
            logit = head(torch.from_numpy(np.asarray(feat, np.float32)))
            prob = float(torch.sigmoid(logit).item())
        mode = "land" if prob >= THRESHOLD else "approach"
        return hover_action(state, g, pad_x, V_TARGET_LOOP, mode=mode)
    return decide


def _count_cell(seed0, n, g, pad_x, decide):
    soft = crash = miss = 0
    starts = []
    for seed in range(seed0, seed0 + n):
        ep = _episode(seed, g, pad_x, decide)
        starts.append((ep["start_x"], ep["start_y"], ep["seed"]))
        if ep["result"] == "мягкая":
            soft += 1
        elif ep["result"] == "удар":
            crash += 1
        elif ep["result"] == "не сел":
            miss += 1
        else:
            raise RuntimeError(ep["result"])
    return soft, crash, miss, starts


def _summarize_head(kind, seed, split):
    ck, encoder, ema, head = _load_head(kind, seed)
    source = _load_source()
    if not _same(encoder.state_dict(), source["encoder"]):
        raise RuntimeError(f"кодировщик {kind} s{seed} не совпал с источником")
    if not _same(ema.state_dict(), source["ema"]):
        raise RuntimeError(f"EMA {kind} s{seed} не совпала с источником")
    prob = _probs(kind, head, encoder, split)
    scores = _binary(split["label"], prob)
    timing = _timing(split, prob)
    land_frac = float(np.mean(split["label"] == 1))
    row = {
        "file": os.path.basename(ckpt_path(kind, seed)),
        "kind": kind,
        "seed": int(seed),
        "seconds": ck.get("seconds"),
        "pos_weight": ck.get("pos_weight"),
        "loss_last": ck["log"][-1]["loss"],
        "land_frac": land_frac,
        "counts_toward_90": kind != "zblind",
        **scores,
        "timing": timing,
    }
    print(f"{row['file']} acc={scores['accuracy']:.4f} f1={scores['f1']:.4f} "
          f"miss_switch={timing['n_miss']}/{timing['n_teacher']}", flush=True)
    return row, head, encoder


def main():
    torch.set_num_threads(THREADS)
    print(f"отложенные {HELD_SEED0}…{HELD_SEED0 + HELD_N - 1}", flush=True)
    held = collect_split(HELD_SEED0, HELD_N)
    if set(np.unique(held["seed"])) & set(range(98000, 98512)):
        raise RuntimeError("отложенные пересеклись с обучением")
    heads = []
    loaded = {}
    for kind, seed in (
        ("latent", 0), ("latent", 1), ("latent", 2),
        ("state", 0), ("state", 1), ("state", 2),
        ("zblind", 0),
    ):
        row, head, encoder = _summarize_head(kind, seed, held)
        heads.append(row)
        if (kind, seed) in (("latent", 0), ("state", 0)):
            loaded[(kind, seed)] = (head, encoder)

    cells = []
    start_ref = None
    for g, pad, in_pass in _cells():
        for name, decide in (
            ("rule", _decide_rule(g, pad)),
            ("latent", _decide_head("latent", *loaded[("latent", 0)], g, pad)),
            ("state", _decide_head("state", *loaded[("state", 0)], g, pad)),
        ):
            soft, crash, miss, starts = _count_cell(LOOP_SEED0, LOOP_N, g, pad, decide)
            sx, sy, _ = starts[0]
            if start_ref is None:
                start_ref = (sx, sy)
            elif abs(sx - start_ref[0]) > 1e-9 or abs(sy - start_ref[1]) > 1e-9:
                raise RuntimeError(f"старт 99000 разъехался: {sx, sy} vs {start_ref}")
            cells.append({
                "controller": name, "g": g, "pad_x": pad, "in_pass": in_pass,
                "soft": soft, "crash": crash, "miss": miss, "of": LOOP_N,
            })
            print(f"{name} g={g:g} pad={pad:g} soft={soft} crash={crash} miss={miss}"
                  f"{'' if in_pass else '  вне зачёта'}", flush=True)

    def _soft(controller, in_pass):
        return sum(c["soft"] for c in cells if c["controller"] == controller and c["in_pass"] is in_pass)

    rule_soft = _soft("rule", True)
    latent_soft = _soft("latent", True)
    n_pass_cells = sum(1 for c in cells if c["controller"] == "rule" and c["in_pass"])
    n_pass = n_pass_cells * LOOP_N
    if n_pass != PASS_N:
        raise RuntimeError((n_pass, PASS_N))
    grid_ok = (rule_soft / n_pass) >= RULE_FLOOR
    gap = (rule_soft - latent_soft) / n_pass
    gap_ok = bool(grid_ok and gap <= GAP_LIMIT)
    payload = {
        "tag": TAG,
        "threshold": THRESHOLD,
        "nav_tol": 6.0,
        "vy_soft": 8.0,
        "v_target": V_TARGET_LOOP,
        "held_seed0": HELD_SEED0,
        "held_n": HELD_N,
        "loop_seed0": LOOP_SEED0,
        "loop_n": LOOP_N,
        "pass_g": list(PASS_G),
        "pass_pads": list(PASS_PADS),
        "gap_limit": GAP_LIMIT,
        "rule_floor": RULE_FLOOR,
        "start_99000": {"x": start_ref[0], "y": start_ref[1]},
        "heads": heads,
        "cells": cells,
        "rule_soft_pass": rule_soft,
        "latent_soft_pass": latent_soft,
        "state_soft_pass": _soft("state", True),
        "pass_n": n_pass,
        "grid_ok": grid_ok,
        "gap": gap,
        "gap_ok": gap_ok,
    }
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)
    print(OUT_JSON, flush=True)
    if not grid_ok:
        raise SystemExit(
            f"правило на зачёте {rule_soft}/{n_pass} ниже {RULE_FLOOR}. Порог {GAP_LIMIT} не ослаблялся.")


if __name__ == "__main__":
    main()

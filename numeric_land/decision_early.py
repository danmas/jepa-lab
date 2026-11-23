"""Штраф за раннюю посадку на новой голове. Контур и старые веса не меняются.

    python -m numeric_land.decision_early --tag numeric-land-decision-early

До прогона зафиксировано: margin = 20 пикселей, λ = 0.3, порог контура = 0.5.
Инициализация новая, не из landd_latent_s0. Шагов 1000, как у прошлой головы.
В зачёт замкнутого контура идёт seed 0. Seed 1 — только сдвиг на траекториях учителя.

L = L_BCE + λ * L_early
L_early — среднее по кадрам наведения (|x − pad_x| > 6) от
p_land * relu(|x − pad_x| − margin).
Отдельного порога на p_land нет: высокая вероятность входит множителем.
|x − pad_x| в пикселях и на 512 не делится.
"""
import argparse
import os
import time

import numpy as np
import torch

from numeric_land.decision import (
    THRESHOLD, DecisionHead, assert_start, cell_is_result, features_of, n_in_of,
)
from numeric_land.expert_hover import NAV_TOL
from numeric_land.mode import LOOP_N, LOOP_SEED0, PASS_G, PASS_PADS, V_TARGET_LOOP, class_weight
from numeric_land.relenc import (
    MIN_HELD_SOFT, MIN_TRAIN_SOFT, OUT, RelEncoder, encode_rel, file_stamp,
    frames_of, held_seeds, require_soft, scan_range, tensors_equal, train_seeds,
    write_json,
)
from numeric_land.rollout_decision import _count_cell, _encoder_from, _load_head, _load_json, _probs
from numeric_land.rollout_decision_threshold import _decide, _timing_at
from numeric_land.train_decision import STEPS, BATCH, LR, WEIGHT_DECAY, load_weights
from numeric_land.train_relenc import ckpt_path as relenc_ckpt

TAG = "numeric-land-decision-early"
THREADS = 8
MARGIN = 20.0
LAMBDA = 0.3
EARLY_STEP = 20
SOFT_BASE = 108
SOFT_THRESHOLD_BEST = 112
SOFT_GAIN = 15
STEP_BASE_MEAN = -15.9296875
THRESHOLD_BEST_MEAN = -0.890625
EARLY_RATIO = 0.5
SEEDS = (0, 1)
SCORE_SEED = 0
OUT_PATH = os.path.join(OUT, "numeric_land_decision_early.json")
DECISION_JSON = os.path.join(OUT, "numeric_land_decision.json")
THRESHOLD_JSON = os.path.join(OUT, "numeric_land_decision_threshold.json")
LATENT_PATH = os.path.join(OUT, "landd_latent_s0.pt")


def _refuse(tag: str) -> None:
    if tag != TAG:
        raise SystemExit(
            "нужен тег numeric-land-decision-early. "
            "Чужой тег, включая numeric-land-decision, запрещён.")


def _early_path(seed: int) -> str:
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(seed)
    return os.path.join(OUT, f"landd_early_s{seed}.pt")


def _save_early(path: str, payload: dict) -> None:
    name = os.path.basename(path)
    if name not in ("landd_early_s0.pt", "landd_early_s1.pt"):
        raise RuntimeError(name)
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)


def _stamps():
    paths = {
        "encoder": relenc_ckpt("scratch", 0),
        "latent": LATENT_PATH,
        "decision_json": DECISION_JSON,
        "threshold_json": THRESHOLD_JSON,
    }
    stamps = {key: file_stamp(path) for key, path in paths.items()}
    missing = [key for key, stamp in stamps.items() if stamp is None]
    if missing:
        raise SystemExit(f"нет файла для чтения: {missing}")
    return paths, stamps


def _check_locked_numbers():
    if abs(float(THRESHOLD) - 0.5) > 0.0 or abs(float(NAV_TOL) - 6.0) > 0.0:
        raise RuntimeError("порог контура или допуск учителя сдвинуты")
    if MARGIN != 20.0 or LAMBDA != 0.3 or EARLY_STEP != 20:
        raise RuntimeError("margin, λ или окно раннего шага сдвинуты")
    decision = _load_json(DECISION_JSON)
    threshold = _load_json(THRESHOLD_JSON)
    if int(decision["new_soft"]) != SOFT_BASE or abs(float(decision["threshold"]) - 0.5) > 0.0:
        raise SystemExit("старый отчёт решения не держит 108 при пороге 0.5")
    rows = {float(row["threshold"]): row for row in threshold["rows"]}
    base = rows[0.5]["timing"]["step"]
    best = rows[0.7]["timing"]["step"]
    if abs(float(base["mean"]) - STEP_BASE_MEAN) > 0.0:
        raise SystemExit("база сдвига шага не равна −15.9296875")
    if abs(float(best["mean"]) - THRESHOLD_BEST_MEAN) > 0.0:
        raise SystemExit("сдвиг порога 0.7 не равен −0.890625")
    if int(rows[0.7]["soft"]) != SOFT_THRESHOLD_BEST or int(rows[0.5]["soft"]) != SOFT_BASE:
        raise SystemExit("база мягких 108/112 в отчёте порога не сошлась")


def _labels_match_dx(split) -> None:
    dx = np.asarray(split["dx"], np.float64)
    label = np.asarray(split["label"])
    approach = label == 0
    land = label == 1
    if np.any(approach & (dx <= float(NAV_TOL))) or np.any(land & (dx > float(NAV_TOL))):
        raise RuntimeError("метка режима не совпала с |dx| и допуском 6")


def _early_term(logit, dx, label):
    """Среднее по кадрам наведения. Пустая выборка даёт нуль с градиентом."""
    prob = torch.sigmoid(logit).reshape(-1)
    dist = dx.reshape(-1)
    approach = label.reshape(-1) < 0.5
    term = prob * torch.relu(dist - float(MARGIN))
    if not bool(approach.any()):
        return term.sum() * 0.0
    return term[approach].mean()


def _train_one(seed, features, labels, dx, pos_weight, source, source_name):
    torch.manual_seed(int(seed))
    head = DecisionHead(n_in_of("latent"))
    torch.manual_seed(0)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    feat = torch.from_numpy(np.asarray(features, np.float32))
    target = torch.from_numpy(np.asarray(labels, np.float32))
    dist = torch.from_numpy(np.asarray(dx, np.float32))
    rng = np.random.default_rng(0)
    n = len(target)
    log = []
    t0 = time.perf_counter()
    label = f"early s{seed}"
    print(
        f"[{label}] steps={STEPS} batch={BATCH} margin={MARGIN:g} lambda={LAMBDA:g} "
        f"pos_weight={pos_weight:.4f}",
        flush=True,
    )
    for step in range(1, STEPS + 1):
        pick = rng.integers(0, n, size=BATCH)
        logit = head(feat[pick])
        bce = loss_fn(logit, target[pick])
        early = _early_term(logit, dist[pick], target[pick])
        loss = bce + float(LAMBDA) * early
        if not torch.isfinite(loss):
            raise RuntimeError(f"{label}: лосс не конечен на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "loss": round(float(loss.detach()), 5),
                "bce": round(float(bce.detach()), 5),
                "early": round(float(early.detach()), 5),
            }
            log.append(row)
            elapsed = time.perf_counter() - t0
            print(
                f"  {label} {step:4d}/{STEPS} loss={row['loss']:.4f} "
                f"bce={row['bce']:.4f} early={row['early']:.4f} "
                f"{elapsed / step * 1000:.0f} ms/step",
                flush=True,
            )
    seconds = time.perf_counter() - t0
    payload = {
        "tag": TAG,
        "kind": "early",
        "seed": int(seed),
        "steps": STEPS,
        "batch": BATCH,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "threshold": float(THRESHOLD),
        "margin": float(MARGIN),
        "lambda": float(LAMBDA),
        "init": "fresh",
        "pos_weight": float(pos_weight),
        "n_in": n_in_of("latent"),
        "frozen_encoder": True,
        "source": source_name,
        "seconds": round(float(seconds), 3),
        "encoder": {key: value.detach().cpu().clone() for key, value in source["encoder"].items()},
        "ema": {key: value.detach().cpu().clone() for key, value in source["ema"].items()},
        "head": head.state_dict(),
        "log": log,
    }
    path = _early_path(seed)
    _save_early(path, payload)
    head.eval()
    print(f"[{label}] {path}  ({seconds:.1f} с)", flush=True)
    return head, payload


def _shifts(split, prob, threshold: float):
    step_d = []
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
        pp = np.asarray(prob)[mask][order]
        t_at = int(np.argmax(lab == 1))
        crossed = np.flatnonzero(pp >= float(threshold))
        if len(crossed) == 0:
            miss += 1
            continue
        h_at = int(crossed[0])
        step_d.append(float(steps[h_at] - steps[t_at]))
    return step_d, n, miss


def _timing_row(split, prob):
    official = _timing_at(split, prob, float(THRESHOLD))
    shifts, n, miss = _shifts(split, prob, float(THRESHOLD))
    if n != int(official["n_teacher"]) or miss != int(official["n_miss"]):
        raise RuntimeError("счёт сдвига разошёлся с timing_at")
    step = official["step"]
    if step is None:
        return official, 0
    arr = np.asarray(shifts, np.float64)
    if abs(float(np.mean(arr)) - float(step["mean"])) > 0.0:
        raise RuntimeError("среднее сдвига разошлось")
    if abs(float(np.min(arr)) - float(step["min"])) > 0.0:
        raise RuntimeError("худший сдвиг разошёлся")
    n_early_20 = int(np.sum(arr <= -float(EARLY_STEP)))
    official = dict(official)
    official["n_early_20"] = n_early_20
    return official, n_early_20


def _pass_grid():
    return [(float(g), float(pad)) for g in PASS_G for pad in PASS_PADS]


def _reading(soft, n_early_20, step_mean, baseline_early_20) -> str:
    soft_ok = int(soft) >= SOFT_BASE + SOFT_GAIN
    tail_ok = int(n_early_20) * 2 <= int(baseline_early_20)
    closer_ok = abs(float(step_mean)) < abs(float(THRESHOLD_BEST_MEAN))
    if soft_ok and tail_ok and closer_ok:
        return "штраф сдвинул и хвост, и мягкие"
    if not soft_ok:
        return "штраф в таком виде недостаточен"
    return "мягких прибавилось, ориентир по хвосту или среднему не закрыт"


def _fly(head, encoder, mode_json):
    cells = []
    for g, pad in _pass_grid():
        decide = _decide(head, encoder, g, pad, float(THRESHOLD))
        soft, crash, miss, start = _count_cell(g, pad, decide)
        assert_start(start[0], start[1], mode_json["start_99000"])
        cells.append({
            "g": g, "pad_x": pad, "soft": int(soft), "crash": int(crash),
            "miss": int(miss), "of": int(LOOP_N),
        })
        print(
            f"early s0 g={g:g} pad={pad:g} soft={soft} crash={crash} miss={miss}",
            flush=True,
        )
    if len(cells) * LOOP_N != 180:
        raise RuntimeError("сетка не 180")
    return cells


def _encoder_matches(encoder, ema, source) -> None:
    if not tensors_equal(encoder.state_dict(), source["encoder"]):
        raise RuntimeError("голова несёт не тот кодировщик")
    if not tensors_equal(ema.state_dict(), source["ema"]):
        raise RuntimeError("голова несёт не ту EMA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    _refuse(args.tag)
    _check_locked_numbers()
    torch.set_num_threads(THREADS)
    paths, before = _stamps()
    mode_json = _load_json(os.path.join(OUT, "numeric_land_mode.json"))
    from numeric_land.field import start_and_actions
    x0, y0, _vx, _vy, _acts = start_and_actions(LOOP_SEED0, 0)
    assert_start(x0, y0, mode_json["start_99000"])

    seed0, n_train = train_seeds()
    episodes = scan_range(seed0, n_train, "штраф, обучение")
    require_soft(episodes, MIN_TRAIN_SOFT, "штраф, обучение")
    split = frames_of(episodes, soft_only=True)
    _labels_match_dx(split)
    pos_weight = class_weight(split["label"])
    held0, held_n = held_seeds()
    held = scan_range(held0, held_n, "штраф, отложенные")
    require_soft(held, MIN_HELD_SOFT, "штраф, отложенные")
    held_split = frames_of(held, soft_only=False)

    scratch, stamp_new = load_weights(paths["encoder"])
    if file_stamp(paths["encoder"]) != stamp_new or stamp_new != before["encoder"]:
        raise RuntimeError("чтение кодировщика изменило файл")
    if scratch.get("kind") != "scratch" or int(scratch.get("seed")) != 0:
        raise RuntimeError(paths["encoder"])
    encoder = RelEncoder()
    encoder.load_state_dict(scratch["encoder"])
    encoder.requires_grad_(False)
    encoder.eval()
    z = encode_rel(encoder, split["state"], split["pad_x"])
    features = features_of("latent", split["pad_x"], split["g"], z)
    if features.shape[1] != n_in_of("latent"):
        raise RuntimeError(features.shape)

    base_ck, base_head = _load_head("latent", 0)
    if file_stamp(paths["latent"]) != before["latent"]:
        raise RuntimeError("чтение старой головы изменило файл")
    base_encoder, base_ema = _encoder_from(base_ck, "latent")
    _encoder_matches(base_encoder, base_ema, scratch)
    base_prob = _probs("latent", base_head, base_encoder, held_split)
    base_timing, base_early_20 = _timing_row(held_split, base_prob)
    base_step = base_timing["step"]
    if (
        abs(float(base_step["mean"]) - STEP_BASE_MEAN) > 0.0
        or abs(float(base_step["median"]) - 6.0) > 0.0
        or abs(float(base_step["min"]) - (-157.0)) > 0.0
        or int(base_step["n_early"]) != 43
        or int(base_timing["n_miss"]) != 0
    ):
        raise SystemExit("повтор старой головы не совпал со сдвигом −15.93 / −157")
    print(f"старая голова ранних ≤ −{EARLY_STEP}: {base_early_20}", flush=True)

    trained = []
    for seed in SEEDS:
        head, payload = _train_one(
            seed, features, split["label"], split["dx"], pos_weight, scratch, "landr_scratch_s0.pt")
        trained.append((seed, head, payload))

    published_cells = {
        (float(cell["g"]), float(cell["pad_x"])): (
            int(cell["soft"]), int(cell["crash"]), int(cell["miss"]))
        for cell in _load_json(DECISION_JSON)["cells"]
        if cell["controller"] == "latent" and cell["in_pass"] is True
    }
    rows = []
    score = None
    for seed, head, payload in trained:
        enc, ema = _encoder_from(payload, "latent")
        _encoder_matches(enc, ema, scratch)
        prob = _probs("latent", head, enc, held_split)
        timing, n_early_20 = _timing_row(held_split, prob)
        row = {
            "seed": int(seed),
            "file": os.path.basename(_early_path(seed)),
            "in_loop": seed == SCORE_SEED,
            "timing": timing,
            "n_early_20": int(n_early_20),
            "n_early_20_baseline": int(base_early_20),
        }
        step = timing["step"]
        print(
            f"early s{seed} шаг mean={None if step is None else step['mean']} "
            f"median={None if step is None else step['median']} "
            f"min={None if step is None else step['min']} "
            f"n_early_20={n_early_20} n_miss={timing['n_miss']}",
            flush=True,
        )
        if seed == SCORE_SEED:
            cells = _fly(head, enc, mode_json)
            soft = int(sum(cell["soft"] for cell in cells))
            compared = []
            for cell in cells:
                prev = published_cells[(cell["g"], cell["pad_x"])]
                delta = int(cell["soft"]) - int(prev[0])
                compared.append({
                    "g": cell["g"], "pad_x": cell["pad_x"],
                    "soft": cell["soft"], "crash": cell["crash"], "miss": cell["miss"],
                    "baseline_soft": int(prev[0]),
                    "delta_soft": delta,
                    "counts_as_cell_result": bool(cell_is_result(delta)),
                })
            row["soft"] = soft
            row["of"] = 180
            row["cells"] = cells
            row["cell_vs_108"] = compared
            row["soft_minus_108"] = soft - SOFT_BASE
            row["soft_minus_112"] = soft - SOFT_THRESHOLD_BEST
            row["reading"] = _reading(
                soft, n_early_20, None if step is None else step["mean"], base_early_20)
            score = row
        rows.append(row)

    after = {key: file_stamp(path) for key, path in paths.items()}
    if after != before:
        raise SystemExit("прогон изменил старый файл")
    if score is None or score["timing"]["step"] is None:
        raise RuntimeError("нет зачёта seed 0")
    payload = {
        "tag": TAG,
        "head": "landd_early_s0.pt",
        "also_seed_1": "landd_early_s1.pt",
        "encoder": "landr_scratch_s0.pt",
        "init": "fresh",
        "trained_steps": STEPS,
        "batch": BATCH,
        "threshold": float(THRESHOLD),
        "margin": float(MARGIN),
        "lambda": float(LAMBDA),
        "early_step": EARLY_STEP,
        "early_loss": "mean over approach frames of p_land * relu(|x-pad_x| - margin)",
        "distance_unit": "pixel",
        "cem": False,
        "distance_margin_in_loop": False,
        "skills_choose_action": False,
        "soft_base": SOFT_BASE,
        "soft_threshold_best": SOFT_THRESHOLD_BEST,
        "soft_gain_min": SOFT_GAIN,
        "step_base_mean": STEP_BASE_MEAN,
        "threshold_best_mean": THRESHOLD_BEST_MEAN,
        "early_ratio": EARLY_RATIO,
        "baseline_n_early_20": int(base_early_20),
        "v_target": float(V_TARGET_LOOP),
        "rows": rows,
        "reading": score["reading"],
    }
    write_json(OUT_PATH, payload)
    if {key: file_stamp(path) for key, path in paths.items()} != before:
        raise SystemExit("запись отчёта задела старый файл")
    print(os.path.abspath(OUT_PATH), flush=True)
    print(
        f"итог seed 0: {score['soft']}/180  ранних≤−{EARLY_STEP} {score['n_early_20']} "
        f"против {base_early_20}  {score['reading']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

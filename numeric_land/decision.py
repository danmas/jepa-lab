"""Модель решения: какая ветка правила с наведением включается.

Сеть та же, что у головы режима. Действие берёт ``hover_action``.
Навык и probe в этот лосс и в контур не входят.
"""
import numpy as np

from numeric_land.data import G_LAB
from numeric_land.field import SCALE_XY
from numeric_land.mode import (
    EXTRA_G, EXTRA_PAD, HEAD_HIDDEN, PASS_G, PASS_PADS, THRESHOLD, ModeHead,
)

# 180 * (1 − 0.15) = 153. 152 эту линию не проходит.
GAP_MIN = 153
RECORDED_RULE_SOFT = 180
RECORDED_MODE_SOFT = 62
LATENT_MARGIN = 18
BETTER_MIN = RECORDED_MODE_SOFT + LATENT_MARGIN
CELL_NOISE = 3
LINE_MIN = 0.90
KINDS = ("latent", "zblind", "g", "oldz")
TRAIN_SEEDS = (0, 1, 2)
SWITCH_RATIO = 0.9
HEAD_HIDDEN_WIDTH = HEAD_HIDDEN

DecisionHead = ModeHead


def n_in_of(kind: str) -> int:
    if kind in ("latent", "oldz"):
        return 33
    if kind == "zblind":
        return 32
    if kind == "g":
        return 34
    raise ValueError(kind)


def counts_toward_90(kind: str) -> bool:
    return kind == "latent"


def pad_channel(pads: np.ndarray) -> np.ndarray:
    return (np.asarray(pads, np.float32) / np.float32(SCALE_XY)).reshape(-1, 1)


def feat_latent(z: np.ndarray, pads: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(z, np.float32), pad_channel(pads)], axis=1)


def feat_zblind(z: np.ndarray) -> np.ndarray:
    return np.asarray(z, np.float32)


def feat_g(z: np.ndarray, pads: np.ndarray, gs: np.ndarray) -> np.ndarray:
    g_col = (np.asarray(gs, np.float32) / np.float32(G_LAB)).reshape(-1, 1)
    return np.concatenate([feat_latent(z, pads), g_col], axis=1)


def features_of(kind: str, pads, gs, z) -> np.ndarray:
    if kind in ("latent", "oldz"):
        return feat_latent(z, pads)
    if kind == "zblind":
        return feat_zblind(z)
    if kind == "g":
        return feat_g(z, pads, gs)
    raise ValueError(kind)


def gap_ok(new_soft: int) -> bool:
    return int(new_soft) >= GAP_MIN


def better_ok(new_soft: int) -> bool:
    return int(new_soft) >= BETTER_MIN


def latent_helped(new_soft: int, oldz_soft: int) -> bool:
    return int(new_soft) >= int(oldz_soft) + LATENT_MARGIN


def cell_is_result(delta_soft: int) -> bool:
    """Разница меньше трёх из тридцати на одной клетке результатом не называется."""
    return abs(int(delta_soft)) >= CELL_NOISE


def assert_baselines(rule_soft, latent_soft) -> None:
    if int(rule_soft) != RECORDED_RULE_SOFT or int(latent_soft) != RECORDED_MODE_SOFT:
        raise SystemExit(
            f"база {rule_soft}/{latent_soft} не совпала с {RECORDED_RULE_SOFT}/"
            f"{RECORDED_MODE_SOFT}. Отчёт не публикуется.")


def assert_start(x, y, ref, tol: float = 1e-9) -> None:
    if abs(float(x) - float(ref["x"])) > tol or abs(float(y) - float(ref["y"])) > tol:
        raise SystemExit("старт 99000 не совпал. Отчёт не публикуется.")


def switch_numbers(mean_new, mean_old, n_miss_new, n_miss_old, n_switched, n_teacher) -> bool:
    """Модуль сдвига не выше 0.9 от контроля, промахов не больше, пересечений не меньше половины."""
    if mean_new is None or mean_old is None:
        return False
    if abs(float(mean_old)) <= 0.0:
        return False
    if abs(float(mean_new)) > SWITCH_RATIO * abs(float(mean_old)):
        return False
    if int(n_miss_new) > int(n_miss_old):
        return False
    if int(n_switched) < 0.5 * int(n_teacher):
        return False
    return True


def abs_ratio(mean_new, mean_old):
    if mean_new is None or mean_old is None or abs(float(mean_old)) <= 0.0:
        return None
    return abs(float(mean_new)) / abs(float(mean_old))


def binary_scores(labels, prob) -> dict:
    pred = np.asarray(prob) >= THRESHOLD
    truth = np.asarray(labels) == 1
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
        "accuracy": float(accuracy), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
    }


def line_90(rows) -> bool:
    """Только три сида боевого входа. Контроли эту линию не закрывают."""
    picked = [row for row in rows if row["kind"] == "latent" and int(row["seed"]) in TRAIN_SEEDS]
    if len(picked) != 3:
        return False
    return all(row["accuracy"] >= LINE_MIN and row["f1"] >= LINE_MIN for row in picked)


def timing_of(split, prob) -> dict:
    """Первый шаг посадки учителя против первого p ≥ 0.5. Промах остаётся в знаменателе."""
    step_d, dx_d, height_d = [], [], []
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
        height = split["height"][mask][order]
        pp = np.asarray(prob)[mask][order]
        t_at = int(np.argmax(lab == 1))
        crossed = np.flatnonzero(pp >= THRESHOLD)
        if len(crossed) == 0:
            miss += 1
            continue
        h_at = int(crossed[0])
        step_d.append(float(steps[h_at] - steps[t_at]))
        dx_d.append(float(dx[h_at] - dx[t_at]))
        height_d.append(float(height[h_at] - height[t_at]))

    def _stat(values):
        if not values:
            return None
        arr = np.asarray(values, np.float64)
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "n": int(len(arr))}

    return {
        "n_teacher": int(n),
        "n_miss": int(miss),
        "n_switched": int(n - miss),
        "step": _stat(step_d),
        "dx": _stat(dx_d),
        "height": _stat(height_d),
    }


def switch_flag(new_timing: dict, old_timing: dict, key: str) -> bool:
    new_stat = new_timing.get(key)
    old_stat = old_timing.get(key)
    if new_stat is None or old_stat is None:
        return False
    if int(new_timing["n_teacher"]) != int(old_timing["n_teacher"]):
        return False
    return switch_numbers(
        new_stat["mean"], old_stat["mean"],
        new_timing["n_miss"], old_timing["n_miss"],
        new_timing["n_switched"], new_timing["n_teacher"])


def pass_cells():
    rows = []
    for g in PASS_G:
        for pad in PASS_PADS:
            rows.append((float(g), float(pad), True))
    for g in PASS_G:
        rows.append((float(g), float(EXTRA_PAD), False))
    for pad in list(PASS_PADS) + [float(EXTRA_PAD)]:
        rows.append((float(EXTRA_G), float(pad), False))
    return rows


def pilot_passes(flag_distance, flag_height, flag_step, flag_dx, passed_90,
                 new_soft, oldz_soft) -> bool:
    return bool(
        flag_distance and flag_height and flag_step and flag_dx and passed_90
        and gap_ok(new_soft) and better_ok(new_soft) and latent_helped(new_soft, oldz_soft))


def subset_split(split, seeds) -> dict:
    keep = np.isin(split["seed"], list(seeds))
    return {key: value[keep] for key, value in split.items()}

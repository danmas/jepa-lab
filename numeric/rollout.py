"""Зонд и открытая раскатка шести чекпоинтов в out/numeric_rollout.json.

Зонд — models.fit_ridge. У пиксельной Probe вход это пара латентов, поэтому
fit_ridge от одной матрицы (N, 32) собирает Linear(32, 4). Читаем его .fc,
а не forward пары.
"""
import json
import os

import numpy as np
import torch

from models import EMATarget, fit_ridge
from numeric.data import (
    PROBE_TRAIN, ROLLOUT_N, ROLLOUT_SEED0, ROLLOUT_STEPS, TRAIN_SEEDS,
    episode, probe_states, scale_state, unscale_state,
)
from numeric.models import LATENT, DirectPredictor, NumericEncoder, NumericPredictor, action_pairs
from numeric.train import STEPS, ckpt_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "out", "numeric_rollout.json")
RIDGE_LAM = 1e-2
HORIZON_MARKS = (1, 5, 10, 20)
R2_NAMES = ("x", "y", "vx", "vy")
SHORT_STEP_PX = 1.0
R2_FLOOR = 0.9
SHORT_STEP_MARK = "короткий шаг не развален"


def _r2(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot).cpu().numpy()


def fit_probe(ema, states: np.ndarray):
    """Зонд на z = E_ema(s). Цели — немасштабированные (x, y, vx, vy)."""
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states, np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    if z.shape[1] != LATENT:
        raise RuntimeError(f"латент {z.shape[1]}, ожидался {LATENT}")
    probe = fit_ridge(z[:PROBE_TRAIN], target[:PROBE_TRAIN], lam=RIDGE_LAM)
    if probe.fc.in_features != LATENT:
        raise RuntimeError("fit_ridge собрал не тот вход: нужен один латент 32")
    with torch.no_grad():
        pred = probe.fc(z[PROBE_TRAIN:])
    r2 = _r2(pred, target[PROBE_TRAIN:])
    return probe, r2


def _mean_std(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    if values.shape != (ROLLOUT_N,):
        raise RuntimeError(f"нужен вектор из {ROLLOUT_N} эпизодов, пришло {values.shape}")
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(ddof=1)), 6),
    }


def _euclidean(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    delta = pred.astype(np.float64) - true.astype(np.float64)
    return np.sqrt((delta * delta).sum(axis=-1))


def _load_jepa(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("mode") != "jepa" or int(ck.get("steps", 0)) != STEPS:
        raise RuntimeError(path)
    encoder = NumericEncoder()
    predictor = NumericPredictor()
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()
    return encoder, predictor, ema


def _load_direct(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("mode") != "direct" or int(ck.get("steps", 0)) != STEPS:
        raise RuntimeError(path)
    predictor = DirectPredictor()
    predictor.load_state_dict(ck["predictor"])
    predictor.eval()
    return predictor


def _episodes():
    states = np.empty((ROLLOUT_N, ROLLOUT_STEPS + 1, 4), np.float32)
    actions = np.empty((ROLLOUT_N, ROLLOUT_STEPS), np.int64)
    for i, seed in enumerate(range(ROLLOUT_SEED0, ROLLOUT_SEED0 + ROLLOUT_N)):
        states[i], actions[i] = episode(seed, ROLLOUT_STEPS)
    return states, actions


def _marks(pos_mean_1: float, pos_mean_20: float) -> dict:
    """Порог 1 px — только отметка, что шаг не развален. Отдельного «успеха» нет."""
    return {
        "отметка_шага_1": SHORT_STEP_MARK if pos_mean_1 < SHORT_STEP_PX else None,
        "шаг_20_больше_шага_1": bool(pos_mean_20 > pos_mean_1),
    }


@torch.no_grad()
def rollout_jepa(path, states, actions, probe):
    encoder, predictor, ema = _load_jepa(path)
    scaled = torch.from_numpy(scale_state(states))
    act = torch.from_numpy(actions)
    z = encoder(scaled[:, 0])
    pos_err = {}
    vel_err = {}
    lat_err = {}
    for k in range(1, ROLLOUT_STEPS + 1):
        z = predictor(z, action_pairs(act[:, k - 1], z))
        target = ema(scaled[:, k])
        if k in HORIZON_MARKS:
            lat = (z - target).abs().mean(dim=-1).cpu().numpy()
            # Зонд учился на немасштабированных целях, повторно на 64 не умножаем.
            pred = probe.fc(z).cpu().numpy()
            true = states[:, k]
            pos_err[k] = _euclidean(pred[:, 0:2], true[:, 0:2])
            vel_err[k] = _euclidean(pred[:, 2:4], true[:, 2:4])
            lat_err[k] = lat
    return pos_err, vel_err, lat_err


@torch.no_grad()
def rollout_direct(path, states, actions):
    predictor = _load_direct(path)
    s = torch.from_numpy(scale_state(states[:, 0]))
    act = torch.from_numpy(actions)
    pos_err = {}
    vel_err = {}
    for k in range(1, ROLLOUT_STEPS + 1):
        s = predictor(s, action_pairs(act[:, k - 1], s))
        if k in HORIZON_MARKS:
            pred = unscale_state(s.cpu().numpy())
            true = states[:, k]
            pos_err[k] = _euclidean(pred[:, 0:2], true[:, 0:2])
            vel_err[k] = _euclidean(pred[:, 2:4], true[:, 2:4])
    return pos_err, vel_err


def _series(by_step: dict) -> dict:
    return {str(k): _mean_std(by_step[k]) for k in HORIZON_MARKS}


def _row(pos_err, vel_err, extra=None):
    pos = _series(pos_err)
    vel = _series(vel_err)
    # Сравнение по тем же округлённым средним, которые лежат в JSON.
    row = {
        "положение_px": pos,
        "скорость_px_за_шаг": vel,
    }
    if extra:
        row.update(extra)
    row.update(_marks(pos["1"]["mean"], pos["20"]["mean"]))
    return row


def evaluate() -> dict:
    states, actions = _episodes()
    bank = probe_states()
    report = {
        "ошибка_положения": "евклид_px",
        "ошибка_скорости": "евклид_px_за_шаг",
        "ошибка_латента": "средний_модуль_по_32",
        "шаги": list(HORIZON_MARKS),
        "эпизоды": ROLLOUT_N,
        "сиды_эпизодов": [ROLLOUT_SEED0, ROLLOUT_SEED0 + ROLLOUT_N - 1],
        "сиды_обучения": list(TRAIN_SEEDS),
        "jepa": [],
        "direct": [],
    }
    for seed in TRAIN_SEEDS:
        encoder, predictor, ema = _load_jepa(ckpt_path("jepa", seed))
        del predictor
        probe, r2 = fit_probe(ema, bank)
        del encoder
        r2_stored = {name: round(float(r2[i]), 6) for i, name in enumerate(R2_NAMES)}
        collapsed = any(r2_stored[name] < R2_FLOOR for name in R2_NAMES)
        pos_err, vel_err, lat_err = rollout_jepa(
            ckpt_path("jepa", seed), states, actions, probe)
        extra = {
            "seed": int(seed),
            "checkpoint": os.path.basename(ckpt_path("jepa", seed)),
            "probe_r2": r2_stored,
            "схлопнулось": collapsed,
            "латент_l1": _series(lat_err),
        }
        report["jepa"].append(_row(pos_err, vel_err, extra))
        pos_d, vel_d = rollout_direct(ckpt_path("direct", seed), states, actions)
        direct_row = _row(pos_d, vel_d, None)
        direct_row = {
            "seed": int(seed),
            "checkpoint": os.path.basename(ckpt_path("direct", seed)),
            **direct_row,
        }
        report["direct"].append(direct_row)
        print(f"[rollout] s{seed} jepa r2={extra['probe_r2']} collapsed={collapsed} "
              f"pos1={report['jepa'][-1]['положение_px']['1']['mean']} "
              f"direct pos1={direct_row['положение_px']['1']['mean']}")
    return report


def main():
    torch.set_num_threads(8)
    report = evaluate()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(OUT_JSON)


if __name__ == "__main__":
    main()

"""Ridge двух целей. Обученный probe флаг не закрывает.

    python -m numeric_land.rollout_relenc

landf открыт только на чтение. Сиды 1, 2 и tune в флаг не входят.
"""
import os

import numpy as np
import torch

from numeric_land.models import BallEncoder, fit_readout
from numeric_land.relenc import (
    MIN_HELD_SOFT, MIN_TRAIN_SOFT, OUT, PACKAGE_SENTENCE, TAG, W_PROBE,
    RelEncoder, RelProbe, encode_ball, encode_rel, file_stamp, frames_of,
    held_seeds, l1_ratio, landf_path, probe_pass, probe_target_flag, probe_targets,
    require_soft, scan_range, train_seeds, write_json,
)
from numeric_land.train_relenc import ckpt_path

OUT_JSON = os.path.join(OUT, "numeric_land_relenc.json")
THREADS = 8
LAM = 1e-2


def _metrics(pred, target):
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    l1 = np.mean(np.abs(pred - target), axis=0)
    r2 = []
    for col in range(target.shape[1]):
        ss_res = float(np.sum((pred[:, col] - target[:, col]) ** 2))
        centered = target[:, col] - float(np.mean(target[:, col]))
        ss_tot = float(np.sum(centered ** 2))
        r2.append(None if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot))
    return [float(value) for value in l1], r2


def _ridge_l1(z_train, y_train, z_held, y_held):
    head = fit_readout(torch.from_numpy(z_train), torch.from_numpy(y_train), lam=LAM)
    head.eval()
    with torch.no_grad():
        pred = head(torch.from_numpy(z_held)).numpy()
    return _metrics(pred, y_held)


def _load_rel(kind: str, seed: int):
    path = ckpt_path(kind, seed)
    if not os.path.isfile(path):
        raise SystemExit(f"нет {os.path.basename(path)}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != TAG or ck.get("kind") != kind or int(ck.get("seed")) != int(seed):
        raise RuntimeError(path)
    if float(ck.get("w_probe")) != W_PROBE or int(ck.get("n_in")) != 7:
        raise RuntimeError(path)
    encoder = RelEncoder()
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()
    probe = RelProbe()
    probe.load_state_dict(ck["probe"])
    probe.eval()
    return ck, encoder, probe


def _row_rel(kind, seed, frames_train, y_train, frames_held, y_held, in_flag: bool):
    _ck, encoder, probe = _load_rel(kind, seed)
    z_train = encode_rel(encoder, frames_train["state"], frames_train["pad_x"])
    z_held = encode_rel(encoder, frames_held["state"], frames_held["pad_x"])
    l1, r2 = _ridge_l1(z_train, y_train, z_held, y_held)
    with torch.no_grad():
        trained = probe(torch.from_numpy(z_held)).numpy()
    trained_l1, _trained_r2 = _metrics(trained, y_held)
    return {
        "file": os.path.basename(ckpt_path(kind, seed)),
        "kind": kind,
        "seed": int(seed),
        "in_flag": bool(in_flag),
        "ridge_l1": l1,
        "ridge_r2": r2,
        "checkpoint_probe_l1": trained_l1,
    }


def _row_landf(frames_train, y_train, frames_held, y_held):
    path = landf_path()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != "numeric-land-field" or int(ck.get("seed", -1)) != 0:
        raise RuntimeError(path)
    encoder = BallEncoder()
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()
    z_train = encode_ball(encoder, frames_train["state"])
    z_held = encode_ball(encoder, frames_held["state"])
    l1, r2 = _ridge_l1(z_train, y_train, z_held, y_held)
    return {
        "file": os.path.basename(path),
        "in_flag": False,
        "ridge_l1": l1,
        "ridge_r2": r2,
    }


def main():
    torch.set_num_threads(THREADS)
    stamp = file_stamp(landf_path())
    if stamp is None:
        raise SystemExit("нет landf_range_vis_s0.pt")
    seed0, n = train_seeds()
    train_eps = scan_range(seed0, n, "probe, обучение")
    require_soft(train_eps, MIN_TRAIN_SOFT, "probe, обучение")
    held0, held_n = held_seeds()
    held_eps = scan_range(held0, held_n, "probe, отложенные")
    require_soft(held_eps, MIN_HELD_SOFT, "probe, отложенные")
    frames_train = frames_of(train_eps, soft_only=True)
    frames_held = frames_of(held_eps, soft_only=True)
    y_train = probe_targets(frames_train["state"], frames_train["pad_x"])
    y_held = probe_targets(frames_held["state"], frames_held["pad_x"])
    landf = _row_landf(frames_train, y_train, frames_held, y_held)
    scratch = {}
    for seed in (0, 1, 2):
        scratch[str(seed)] = _row_rel(
            "scratch", seed, frames_train, y_train, frames_held, y_held, seed == 0)
        print(f"scratch s{seed} L1 {scratch[str(seed)]['ridge_l1']}", flush=True)
    tune = _row_rel("tune", 0, frames_train, y_train, frames_held, y_held, False)
    print(f"tune L1 {tune['ridge_l1']}", flush=True)
    print(f"landf L1 {landf['ridge_l1']}", flush=True)
    base = scratch["0"]["ridge_l1"]
    old = landf["ridge_l1"]
    flag_distance = probe_target_flag(base[0], old[0])
    flag_height = probe_target_flag(base[1], old[1])
    for row in list(scratch.values()) + [tune]:
        row["ridge_ratio"] = [l1_ratio(row["ridge_l1"][0], old[0]), l1_ratio(row["ridge_l1"][1], old[1])]
    if file_stamp(landf_path()) != stamp:
        raise RuntimeError("landf изменился")
    payload = {
        "tag": TAG,
        "w_probe": W_PROBE,
        "probe_ratio": 0.9,
        "lam": LAM,
        "package": PACKAGE_SENTENCE,
        "n_soft_train": int(sum(ep["soft"] for ep in train_eps)),
        "n_soft_held": int(sum(ep["soft"] for ep in held_eps)),
        "n_train_frames": int(len(frames_train["label"])),
        "n_held_frames": int(len(frames_held["label"])),
        "targets": ["abs_dx_over_512", "height_over_512"],
        "landf": landf,
        "scratch": scratch,
        "tune": tune,
        "flag_distance": bool(flag_distance),
        "flag_height": bool(flag_height),
        "probe_pass": bool(probe_pass(flag_distance, flag_height)),
        "landf_mtime_ns": stamp[0],
        "landf_size": stamp[1],
    }
    write_json(OUT_JSON, payload)
    print(OUT_JSON, flush=True)
    print(f"флаг расстояния {flag_distance}  флаг высоты {flag_height}  "
          f"отношения {scratch['0']['ridge_ratio']}", flush=True)


if __name__ == "__main__":
    main()

"""L1 третьего шага внутри нового латента. landh не открывается. Горизонт 40 не считается.

    python -m numeric_land.rollout_skills
"""
import os

import numpy as np
import torch

from models import EMATarget
from numeric_land.models import LATENT, LandPredictor, action_pairs
from numeric_land.relenc import (
    MIN_HELD_SOFT, OUT, TAG, W_PROBE, RelEncoder, held_seeds,
    iter_windows, l1_ratio, pack_windows, require_soft, scale_rel, scan_range,
    tensors_equal, write_json,
)
from numeric_land.skills import N_WORLD, skill_flag, world_of
from numeric_land.train_goal import EMA_MOMENTUM
from numeric_land.train_relenc import ckpt_path as relenc_ckpt
from numeric_land.train_skills import ckpt_path

OUT_JSON = os.path.join(OUT, "numeric_land_skills.json")
THREADS = 8
SOURCE_NAME = "landr_scratch_s0.pt"


def _reject(name: str) -> None:
    if (name.startswith("landh_") or name.startswith("landm_") or name.startswith("landf_")
            or name.startswith("landg_") or name.startswith("landp_")
            or name.startswith("lande_") or name.startswith("land_")):
        raise RuntimeError(name)


def _load_source():
    path = relenc_ckpt("scratch", 0)
    _reject(os.path.basename(path))
    if not os.path.isfile(path):
        raise SystemExit(f"нет {SOURCE_NAME}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != TAG or ck.get("kind") != "scratch" or int(ck.get("seed")) != 0:
        raise RuntimeError(path)
    return ck


def _load_skill(phase: str, seed: int, source):
    path = ckpt_path(phase, seed)
    name = os.path.basename(path)
    _reject(name)
    if not name.startswith("landk_"):
        raise RuntimeError(name)
    if not os.path.isfile(path):
        raise SystemExit(f"нет {name}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != TAG or ck.get("phase") != phase or int(ck.get("seed")) != int(seed):
        raise RuntimeError(path)
    if int(ck.get("n_world")) != N_WORLD:
        raise RuntimeError(ck.get("n_world"))
    encoder = RelEncoder()
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()
    if not tensors_equal(encoder.state_dict(), source["encoder"]):
        raise RuntimeError(f"кодировщик {name} не совпал с {SOURCE_NAME}")
    ema = EMATarget(encoder, momentum=EMA_MOMENTUM)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    if not tensors_equal(ema.state_dict(), source["ema"]):
        raise RuntimeError(f"EMA {name} не совпала")
    predictor = LandPredictor(N_WORLD)
    predictor.load_state_dict(ck["predictor"])
    predictor.eval()
    return encoder, predictor, ema


@torch.no_grad()
def third_step_l1(encoder, predictor, ema, windows) -> float:
    """Средний |z − EMA| на третьем шаге. Дальше горизонта нет."""
    states = windows["states"]
    actions = windows["actions"]
    pads = windows["pads"]
    total = 0.0
    n = 0
    step = 2048
    for start in range(0, len(states), step):
        stop = start + step
        st = torch.from_numpy(np.ascontiguousarray(scale_rel(states[start:stop], pads[start:stop])))
        ac = torch.from_numpy(actions[start:stop].astype(np.int64))
        world = torch.from_numpy(np.ascontiguousarray(
            world_of(windows["gs"][start:stop], pads[start:stop], windows["vys"][start:stop])))
        if st.shape[-1] != 7 or world.shape[-1] != N_WORLD:
            raise RuntimeError((st.shape, world.shape))
        z = encoder(st[:, 0])
        if z.shape[-1] != LATENT:
            raise RuntimeError(z.shape)
        err = None
        for k in range(3):
            z = predictor(z, action_pairs(ac[:, k], z), world)
            target = ema(st[:, k + 1])
            if k == 2:
                err = (z - target).abs().mean(dim=-1)
        total += float(err.sum())
        n += int(err.shape[0])
    if n <= 0:
        raise RuntimeError("нет окон фазы")
    return total / n


def _phase_windows(episodes, bucket: str):
    rows = [window for window in iter_windows(episodes, soft_only=True) if window["bucket"] == bucket]
    packed = pack_windows(rows)
    packed["bucket"] = bucket
    return packed


def _phase_block(bucket, windows, source):
    shared_enc, shared_pred, shared_ema = _load_skill("shared", 0, source)
    shared_l1 = third_step_l1(shared_enc, shared_pred, shared_ema, windows)
    seeds = {}
    for seed in (0, 1, 2):
        encoder, predictor, ema = _load_skill(bucket, seed, source)
        seeds[str(seed)] = third_step_l1(encoder, predictor, ema, windows)
        print(f"{bucket} s{seed} L1 {seeds[str(seed)]:.6f}", flush=True)
    flag = skill_flag(seeds["0"], shared_l1)
    print(f"{bucket} shared L1 {shared_l1:.6f}  флаг {flag}", flush=True)
    return {
        "n_windows": int(len(windows["states"])),
        "shared_l1": float(shared_l1),
        "s0": float(seeds["0"]),
        "s1": float(seeds["1"]),
        "s2": float(seeds["2"]),
        "ratio_s0": l1_ratio(seeds["0"], shared_l1),
        "flag": bool(flag),
    }


def main():
    torch.set_num_threads(THREADS)
    # Горизонт 40 и landh в этот замер не входят. Файлы не открываются.
    held0, held_n = held_seeds()
    episodes = scan_range(held0, held_n, "навыки, отложенные")
    require_soft(episodes, MIN_HELD_SOFT, "навыки, отложенные")
    source = _load_source()
    approach = _phase_windows(episodes, "approach")
    land = _phase_windows(episodes, "land")
    payload = {
        "tag": TAG,
        "threshold": 0.5,
        "w_probe": W_PROBE,
        "horizon_40": False,
        "cem": False,
        "landh_opened": False,
        "source": SOURCE_NAME,
        "approach": _phase_block("approach", approach, source),
        "land": _phase_block("land", land, source),
    }
    # Флаг навыка порог и линии контура не двигает.
    if payload["threshold"] != 0.5 or payload["w_probe"] != W_PROBE:
        raise RuntimeError("порог или вес probe сдвинулись")
    write_json(OUT_JSON, payload)
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()

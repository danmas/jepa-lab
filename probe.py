"""Линейный probe: латент -> физическое состояние (x, y, vx, vy).

Отвечает на вопрос «понимает ли модель, где мяч и куда он летит» в терминах, которые
можно произнести словами. Обучается закрытой формой (ridge), не градиентом, и только
на замороженном энкодере — то есть измеряет то, что уже есть в латенте.

    python probe.py --tag v3
"""
import argparse
import json
import os

import numpy as np
import torch

from data import WORLDS, make_batch
from models import fit_ridge, load_model, r2_score, world_of

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def collect(model, samples=6000, horizon=12, seed=0, world="flat"):
    """Пары латентов (z_{t-1}, z_t) и истинное состояние (x, y, vx, vy).

    Скорость = смещение между кадрами (pos_t − pos_{t-1}) — ровно то, что видит физика.
    """
    rng = np.random.default_rng(seed)
    Z, Y = [], []
    got = 0
    while got < samples:
        frames, _, pos = make_batch(64, rng, horizon, world=world)
        with torch.no_grad():
            z = model.encoder(torch.from_numpy(frames.reshape(-1, 1, 64, 64)))
        z = z.view(frames.shape[0], horizon, -1)
        zp, zc = z[:, :-1], z[:, 1:]                       # (B, H-1, 48)
        vel = pos[:, 1:] - pos[:, :-1]
        Y.append(torch.from_numpy(np.concatenate(
            [pos[:, 1:].reshape(-1, 2), vel.reshape(-1, 2)], axis=1).astype(np.float32)))
        Z.append(torch.cat([zp.reshape(-1, z.shape[-1]), zc.reshape(-1, z.shape[-1])], dim=1))
        got += Z[-1].shape[0]
    return torch.cat(Z), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--world", default=None, choices=WORLDS,
                    help="если задан и не совпадает с чекпоинтом — отказ")
    args = ap.parse_args()

    path = os.path.join(OUT, f"jepa_{args.tag}.pt")
    model, ema, ck = load_model(path, threads=args.threads)
    world = world_of(ck)
    if args.world is not None and args.world != world:
        raise SystemExit(f"несовпадение режима: чекпоинт {world}, запрошен {args.world}")
    Z, Y = collect(model, args.samples, world=world)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Z.shape[0], generator=g)
    n_tr = int(0.8 * len(perm))
    tr, te = perm[:n_tr], perm[n_tr:]

    probe = fit_ridge(Z[tr], Y[tr], lam=args.lam)
    pred = probe.fc(Z[te])   # Z уже склеен как [z_prev | z_cur]
    r2 = r2_score(pred, Y[te])
    mae = (pred - Y[te]).abs().mean(0).tolist()

    torch.save({"probe": probe.state_dict(), "r2": r2, "world": world},
               os.path.join(OUT, f"probe_{args.tag}.pt"))
    res = {"tag": args.tag, "world": world, "samples": int(Z.shape[0]), "r2": r2,
           "mae": [round(v, 3) for v in mae],
           "columns": ["x", "y", "vx", "vy"]}
    with open(os.path.join(OUT, f"probe_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print(f"probe на {Z.shape[0]} латентах (R², 1.0 = идеально):")
    for name, r, m in zip(res["columns"], r2, mae):
        print(f"  {name:>3}: R²={r:>7.4f}   MAE={m:>6.2f}")


if __name__ == "__main__":
    main()

"""Диагностика: предсказывает ли латентная стоимость РЕАЛЬНЫЙ промах плана.

Сравниваем два вопроса:
1. Корреляция по всем решениям сразу (абсолютная калибровка) — обычно шумная,
   потому что у каждого контекста свой «уровень» стоимости.
2. Корреляция ВНУТРИ одного решения (из 128 сэмплов CEM) — именно это использует
   планировщик: ему нужно правильно ранжировать варианты из одного контекста.

    python _cost_check.py --tag v4
"""
import argparse
import json
import os

import numpy as np
import torch

from data import N_ACTIONS, ZONE_CENTERS, World, step_state, zone_frame
from models import load_model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--contexts", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    model, ema, _ = load_model(os.path.join(OUT, f"jepa_{args.tag}.pt"), threads=args.threads)
    rng = np.random.default_rng(0)

    all_cost, all_miss, per_context = [], [], []
    for c in range(args.contexts):
        w = World(seed=c + 1)
        w.step(4)
        frames = [w.frame(), w.step(4)]
        ctx = torch.from_numpy(np.stack(frames))
        z_prev = model.encoder(ctx[0:1])
        z_cur = model.encoder(ctx[1:2])
        zone = c % 9
        z_goal = ema(torch.from_numpy(zone_frame(zone)))
        cx, cy = ZONE_CENTERS[zone]

        u = torch.from_numpy(rng.integers(0, N_ACTIONS, size=(args.samples, args.horizon)))
        zp = z_prev.expand(args.samples, -1).contiguous()
        zc = z_cur.expand(args.samples, -1).contiguous()
        ds = []
        for k in range(args.horizon):
            zn = model.predictor(zp, zc, u[:, k])
            ds.append((zn - z_goal).pow(2).mean(1))
            zp, zc = zc, zn
        cost = (ds[-1] + 0.3 * torch.stack(ds[-3:]).mean(0)).numpy()

        miss = np.zeros(args.samples, np.float32)
        for i in range(args.samples):
            p, v = w.pos.copy(), w.vel.copy()
            for a in u[i].tolist():
                p, v = step_state(p, v, a)
            miss[i] = np.hypot(p[0] - cx, p[1] - cy)

        rho = spearman(cost, miss)
        per_context.append(rho)
        all_cost += cost.tolist(); all_miss += miss.tolist()
        print(f"  контекст {c} (зона {zone}): Спирмен(стоимость, промах) = {rho:+.3f}")

    glob = spearman(all_cost, all_miss)
    res = {"tag": args.tag, "spearman_global": glob,
           "spearman_per_context_mean": float(np.nanmean(per_context)),
           "spearman_per_context": [round(float(x), 3) for x in per_context],
           "contexts": args.contexts, "samples": args.samples}
    print(f"\nглобально (все решения вместе): {glob:+.3f}")
    print(f"внутри решения, среднее: {res['spearman_per_context_mean']:+.3f}")
    with open(os.path.join(OUT, f"cost_check_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

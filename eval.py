"""Оценка планировщика: успех на реальном симуляторе, время решения, сравнение с пикселями.

    python eval.py --tag v3                      # латентный планировщик + baseline
    python eval.py --tag v3 --pixel-episodes 3   # плюс настоящий пиксельный CEM

Главное: успех всегда проверяется в НАСТОЯЩЕМ мире (физика), а не в латенте —
иначе получилось бы «модель довольна собой».
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from data import (N_ACTIONS, ZONE_CENTERS, ZONES, World, in_zone, zone_frame)
from models import load_model
from plan import cem_plan, execute

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def median(xs):
    return float(np.median(xs)) if xs else float("nan")


def latent_eval(model, ema, zones, seeds, horizon, max_steps, cem_kw,
                probe=None, probe_weight=0.0):
    """Прогон MPC по зонам и стартам -> (эпизоды, стоимости, промахи плана, времена)."""
    episodes, costs, shadow_misses, plan_times = [], [], [], []
    for z in zones:
        for s in seeds:
            r = execute(model, ema, z, seed=s, horizon=horizon, max_steps=max_steps,
                        cem_kw=cem_kw, verbose=False, save_gif=False,
                        probe=probe, probe_weight=probe_weight)
            episodes.append(r)
            plan_times += [st["plan_time"] for st in r["steps"]]
            for st in r["steps"]:
                if "shadow_miss" in st:
                    costs.append(st["cost"])
                    shadow_misses.append(st["shadow_miss"])
    return episodes, costs, shadow_misses, plan_times


def random_baseline(zones, seeds, max_steps, rng):
    ok = 0
    total = 0
    for z in zones:
        for s in seeds:
            w = World(s)
            hit = False
            for _ in range(max_steps):
                w.step(int(rng.integers(0, N_ACTIONS)))
                if in_zone(w.pos, z):
                    hit = True
                    break
            ok += int(hit)
            total += 1
    return ok / max(total, 1)


@torch.no_grad()
def cem_plan_pixel(model, context, goal_frame, horizon=10, samples=128, iters=8,
                   elite=16, seed=0):
    """Тот же CEM, но стоимость считается в пикселях (нужен декодер на каждом шаге)."""
    t0 = time.perf_counter()
    z_prev = model.encoder(context[0:1])
    z_cur = model.encoder(context[1:2])
    probs = torch.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    best_seq, best_cost = None, float("inf")
    for _ in range(iters):
        u = torch.multinomial(probs, samples, replacement=True).T.contiguous()
        zp = z_prev.expand(samples, -1).contiguous()
        zc = z_cur.expand(samples, -1).contiguous()
        ds = []
        for k in range(horizon):
            zn = model.predictor(zp, zc, u[:, k])
            f = model.decoder(zn)                       # <- декодер в цикле поиска
            ds.append((f - goal_frame).pow(2).mean((1, 2, 3)))
            zp, zc = zc, model.encoder(f)               # <- и энкодер обратно
        cost = ds[-1] + 0.3 * torch.stack(ds[-3:]).mean(0)
        i = int(cost.argmin())
        if float(cost[i]) < best_cost:
            best_cost, best_seq = float(cost[i]), u[i].clone()
        idx = cost.argsort()[:elite]
        counts = torch.zeros_like(probs)
        for e in idx:
            counts[torch.arange(horizon), u[e]] += 1
        probs = 0.7 * probs + 0.3 * (counts + 0.5) / (elite + 0.5 * N_ACTIONS)
    return best_seq, best_cost, time.perf_counter() - t0


def pixel_eval(model, zones, seeds, horizon, max_steps, cem_kw):
    """Настоящее пиксельное планирование (дорого) — чтобы сравнение было честным."""
    n = cem_kw.get("samples", 128)
    episodes = []
    for z in zones:
        for s in seeds:
            w = World(s)
            f_a = w.frame()
            f_b = w.step(4)
            goal = torch.from_numpy(zone_frame(z))
            prev_f, cur_f = f_a, f_b
            ok, times = False, []
            for _ in range(max_steps):
                ctx = torch.from_numpy(np.stack([prev_f, cur_f]))
                seq, cost, dt = cem_plan_pixel(model, ctx, goal, horizon=horizon,
                                               seed=s, **{k: v for k, v in cem_kw.items()
                                                          if k in ("samples", "iters")})
                times.append(dt)
                new_f = w.step(int(seq[0]))
                if in_zone(w.pos, z):
                    ok = True
                    break
                prev_f, cur_f = cur_f, new_f
            episodes.append({"zone": z, "seed": s, "success": ok,
                             "plan_time_median": median(times), "steps": len(times)})
    return episodes


def make_grid(model, ema, seed, horizon, max_steps, cem_kw, path):
    """3×3: по одному эпизоду на каждую зону, финальный кадр с рамкой цели."""
    from PIL import Image, ImageDraw
    from plan import draw_zone, frame_to_img

    W = 3 * (64 * 3 + 8) + 8
    H = 3 * (64 * 3 + 26) + 8
    canvas = Image.new("RGB", (W, H), "white")
    ok = 0
    for z in range(9):
        r = execute(model, ema, z, seed=seed, horizon=horizon, max_steps=max_steps,
                    cem_kw=cem_kw, verbose=False, save_gif=False, keep_frames=True)
        frames = r.pop("frames")
        tile = draw_zone(frame_to_img(frames[-1]), z)
        x = 8 + (z % 3) * (64 * 3 + 8)
        y = 8 + (z // 3) * (64 * 3 + 26)
        canvas.paste(tile, (x, y))
        d = ImageDraw.Draw(canvas)
        mark = "OK" if r["success"] else f"мимо {r['miss_px']:.0f}px"
        d.text((x, y + 64 * 3 + 2), f"{ZONES[z]['name']}: {mark} ({r['steps_used']} шагов)",
               fill="green" if r["success"] else "red")
        ok += int(r["success"])
    canvas.save(path)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--zones", default="0,2,4,6,8", help="какие зоны проверять")
    ap.add_argument("--seeds", type=int, default=4, help="сколько стартов на зону")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pixel-episodes", type=int, default=0,
                    help=">0 — запустить настоящий пиксельный CEM на стольких стартах")
    ap.add_argument("--cost", choices=["latent", "mixed"], default="latent",
                    help="latent: только латент; mixed: латент + позиция мяча по probe")
    ap.add_argument("--probe-weight", type=float, default=None,
                    help="вес probe-члена в mixed (по умолчанию 1.0)")
    ap.add_argument("--grid", type=int, default=1, help="1: собрать out/plan_grid.png по всем 9 зонам")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    zones = [int(z) for z in args.zones.split(",")]
    seeds = list(range(1, args.seeds + 1))
    cem_kw = {"samples": args.samples, "iters": args.iters}

    model, ema, _ = load_model(os.path.join(OUT, f"jepa_{args.tag}.pt"), threads=args.threads)

    print(f"латентный планировщик: {len(zones)} зон × {len(seeds)} стартов, "
          f"H={args.horizon}, {args.samples}×{args.iters} CEM")
    if args.probe_weight is None:
        args.probe_weight = 1.0 if args.cost == "mixed" else 0.0
    from plan import load_probe
    probe = load_probe(args.tag) if args.probe_weight else None
    print(f"стоимость: {args.cost}" + (f" (probe_weight={args.probe_weight})" if args.cost == "mixed" else ""))
    episodes, costs, shadow_misses, plan_times = latent_eval(
        model, ema, zones, seeds, args.horizon, args.steps, cem_kw,
        probe=probe, probe_weight=args.probe_weight)
    succ = sum(e["success"] for e in episodes) / len(episodes)
    print(f"  успех: {succ*100:.0f}%  шагов в среднем: "
          f"{np.mean([e['steps_used'] for e in episodes]):.1f}  "
          f"время решения (медиана): {median(plan_times)*1000:.0f} мс, "
          f"p90: {np.percentile(plan_times, 90)*1000:.0f} мс")

    rnd = random_baseline(zones, seeds, args.steps, np.random.default_rng(0))
    print(f"  случайные действия: {rnd*100:.0f}%")

    rho = spearman(costs, shadow_misses)
    print(f"  корреляция латентная стоимость ↔ реальный промах (Спирмен): {rho:.3f} "
          f"на {len(costs)} решениях")

    result = {"tag": args.tag, "zones": zones, "seeds": seeds,
              "cost": args.cost, "probe_weight": args.probe_weight,
              "latent": {"success_rate": succ,
                         "plan_time_median": median(plan_times),
                         "plan_time_p90": float(np.percentile(plan_times, 90)),
                         "random_baseline": rnd,
                         "cost_vs_miss_spearman": rho,
                         "decisions": len(costs)},
              "episodes": episodes}

    # пиксельное сравнение
    if args.pixel_episodes:
        pmodel, _, _ = load_model(os.path.join(OUT, f"pixel_{args.tag}.pt"), threads=args.threads)
        print(f"пиксельный планировщик: {args.pixel_episodes} стартов (это долго)")
        pez = zones[:1]
        pseeds = seeds[:args.pixel_episodes]
        t0 = time.perf_counter()
        pepisodes = pixel_eval(pmodel, pez, pseeds, args.horizon, min(args.steps, 10), cem_kw)
        psucc = sum(e["success"] for e in pepisodes) / len(pepisodes)
        ptime = median([e["plan_time_median"] for e in pepisodes])
        print(f"  успех: {psucc*100:.0f}%  время решения (медиана): {ptime*1000:.0f} мс  "
              f"[всего {time.perf_counter()-t0:.0f} с]")
        result["pixel"] = {"success_rate": psucc, "plan_time_median": ptime,
                           "episodes": pepisodes}

    if args.grid:
        gok = make_grid(model, ema, seeds[0], args.horizon, args.steps, cem_kw,
                        os.path.join(OUT, "plan_grid.png"))
        print(f"  сетка 9 зон (seed={seeds[0]}): {gok}/9 попаданий -> out/plan_grid.png")
        result["grid_success"] = gok

    name = f"eval_{args.tag}_{args.cost}.json"
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\nсохранено: out/{name}")


if __name__ == "__main__":
    main()

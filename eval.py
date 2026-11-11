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

from data import (N_ACTIONS, NOISE_SIGMA, SLIP, WORLDS, ZONES, World, goal_frame,
                  in_zone, make_batch, add_obs_noise)
from models import load_model, world_of
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
                probe=None, probe_weight=0.0, world="flat", slip=SLIP, paired=False):
    """Прогон MPC по зонам и стартам -> (эпизоды, стоимости, промахи плана, времена)."""
    episodes, costs, shadow_misses, plan_times = [], [], [], []
    for z in zones:
        for s in seeds:
            r = execute(model, ema, z, seed=s, horizon=horizon, max_steps=max_steps,
                        cem_kw=cem_kw, verbose=False, save_gif=False,
                        probe=probe, probe_weight=probe_weight, world=world, slip=slip)
            if paired:
                r["rand_success"] = bool(random_success(z, s, max_steps, world, slip))
            episodes.append(r)
            plan_times += [st["plan_time"] for st in r["steps"]]
            for st in r["steps"]:
                if "shadow_miss" in st:
                    costs.append(st["cost"])
                    shadow_misses.append(st["shadow_miss"])
    return episodes, costs, shadow_misses, plan_times


def random_baseline(zones, seeds, max_steps, rng):
    """Старый общий поток случайных толчков. Для парного вердикта не используется."""
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


def random_success(zone, seed, max_steps, world="flat", slip=SLIP, rng=None):
    """Случайные толчки из того же старта, что и план. Свой поток, если rng не передан.

    Нулевой шаг как у планировщика, чтобы сравнивать политику, а не разный старт.
    """
    w = World(int(seed), world=world, slip=slip)
    w.step(4)
    draw = rng if rng is not None else np.random.default_rng(int(seed))
    for _ in range(max_steps):
        w.step(int(draw.integers(0, N_ACTIONS)))
        if in_zone(w.pos, zone):
            return True
    return False


def apply_labels(lo: float, hi: float, delta: float, n: int = 0) -> dict:
    """лучше_случайных — только нижняя граница > 0. отрыв_15 — только точечный отрыв ≥ 15."""
    return {
        "delta_pp": float(delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "лучше_случайных": bool(lo > 0.0),
        "отрыв_15": bool(delta >= 15.0),
        "n": int(n),
    }


def paired_interval(episodes):
    """Бутстрап пар: 2000 выборок, numpy сид 0, перцентили 2.5 и 97.5, в пунктах."""
    diff = np.array(
        [float(bool(e["success"])) - float(bool(e["rand_success"])) for e in episodes],
        np.float64)
    rng = np.random.default_rng(0)
    n = len(diff)
    idx = rng.integers(0, n, size=(2000, n))
    means = diff[idx].mean(axis=1) * 100.0
    lo, hi = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    delta = float(diff.mean() * 100.0)
    return apply_labels(lo, hi, delta, n)


def interval_unresolved(stats) -> bool:
    lo, hi = stats["ci_low"], stats["ci_high"]
    return (lo <= 0.0 <= hi) or (lo <= 15.0 <= hi)


@torch.no_grad()
def cem_plan_pixel(model, context, goal_frame, horizon=10, samples=128, iters=8,
                   elite=16, seed=0):
    """Тот же CEM, но стоимость считается в пикселях (нужен декодер на каждом шаге)."""
    t0 = time.perf_counter()
    z_prev = model.encoder(context[0:1])
    z_cur = model.encoder(context[1:2])
    probs = torch.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    g = torch.Generator().manual_seed(int(seed))
    best_seq, best_cost = None, float("inf")
    for _ in range(iters):
        u = torch.multinomial(probs, samples, replacement=True, generator=g).T.contiguous()
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


def pixel_eval(model, zones, seeds, horizon, max_steps, cem_kw, world="flat", slip=SLIP):
    """Настоящее пиксельное планирование (дорого) — чтобы сравнение было честным."""
    episodes = []
    for z in zones:
        for s in seeds:
            w = World(s, world=world, slip=slip)
            f_a = w.frame()
            f_b = w.step(4)
            goal = torch.from_numpy(goal_frame(z, world=world))
            prev_f, cur_f = f_a, f_b
            ok, times = False, []
            for _ in range(max_steps):
                ctx = torch.from_numpy(np.stack([prev_f, cur_f]))
                seq, cost, dt = cem_plan_pixel(
                    model, ctx, goal, horizon=horizon, seed=s,
                    **{k: v for k, v in cem_kw.items() if k in ("samples", "iters")})
                times.append(dt)
                new_f = w.step(int(seq[0]))
                if in_zone(w.pos, z):
                    ok = True
                    break
                prev_f, cur_f = cur_f, new_f
            episodes.append({"zone": z, "seed": s, "success": ok,
                             "plan_time_median": median(times), "steps": len(times)})
    return episodes


def make_grid(model, ema, seed, horizon, max_steps, cem_kw, path, world="flat", slip=SLIP):
    """3×3: по одному эпизоду на каждую зону, финальный кадр с рамкой цели."""
    from PIL import Image, ImageDraw
    from plan import draw_zone, frame_to_img

    W = 3 * (64 * 3 + 8) + 8
    H = 3 * (64 * 3 + 26) + 8
    canvas = Image.new("RGB", (W, H), "white")
    ok = 0
    for z in range(9):
        r = execute(model, ema, z, seed=seed, horizon=horizon, max_steps=max_steps,
                    cem_kw=cem_kw, verbose=False, save_gif=False, keep_frames=True,
                    world=world, slip=slip)
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


@torch.no_grad()
def denoise_report(model, ema, n=256):
    """Одношаговое предсказание рядом с чистым и с шумным кадром. Ярлыки зон не трогает."""
    frames, acts, _ = make_batch(n, np.random.default_rng(20261111), horizon=6, world="flat")
    noisy = add_obs_noise(frames, np.random.default_rng(20261112), NOISE_SIGMA)
    z_pred = model.predictor(
        model.encoder(torch.from_numpy(noisy[:, 0])),
        model.encoder(torch.from_numpy(noisy[:, 1])),
        torch.from_numpy(acts[:, 1]))
    z_clean = ema(torch.from_numpy(frames[:, 2]))
    z_noisy = ema(torch.from_numpy(noisy[:, 2]))
    dist_clean = float((z_pred - z_clean).norm(dim=1).mean())
    dist_noisy = float((z_pred - z_noisy).norm(dim=1).mean())
    return {
        "dist_clean": dist_clean,
        "dist_noisy": dist_noisy,
        "label": "гасит шум" if dist_clean < dist_noisy else "копирует шум",
    }


def probe_r2(tag, world, model):
    """R² probe этого режима. Чужой или пустой файл пересчитывается и записывается."""
    path = os.path.join(OUT, f"probe_{tag}.json")
    cols = ["x", "y", "vx", "vy"]
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if (doc.get("world") or "flat") == world and doc.get("r2"):
            names = doc.get("columns") or cols
            return {name: doc["r2"][i] for i, name in enumerate(names)}
    from models import fit_ridge, r2_score
    from probe import collect

    Z, Y = collect(model, world=world)
    perm = torch.randperm(Z.shape[0], generator=torch.Generator().manual_seed(0))
    n_tr = int(0.8 * len(perm))
    tr, te = perm[:n_tr], perm[n_tr:]
    fitted = fit_ridge(Z[tr], Y[tr], lam=1e-2)
    pred = fitted.fc(Z[te])
    scores = r2_score(pred, Y[te])
    mae = (pred - Y[te]).abs().mean(0).tolist()
    torch.save({"probe": fitted.state_dict(), "r2": scores, "world": world},
               os.path.join(OUT, f"probe_{tag}.pt"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "world": world, "samples": int(Z.shape[0]), "r2": scores,
                   "mae": [round(float(v), 3) for v in mae], "columns": cols},
                  f, ensure_ascii=False, indent=1)
    return {name: scores[i] for i, name in enumerate(cols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--world", default=None, choices=WORLDS,
                    help="если задан и не совпадает с чекпоинтом — отказ, файл вердикта не пишется")
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

    model, ema, ck = load_model(os.path.join(OUT, f"jepa_{args.tag}.pt"), threads=args.threads)
    world = world_of(ck)
    if args.world is not None and args.world != world:
        raise SystemExit(
            f"несовпадение режима: чекпоинт {world}, запрошен {args.world}. Вердикт не записан.")
    slip = float(ck.get("slip") or SLIP)
    paired = world != "flat"

    print(f"латентный планировщик [{world}]: {len(zones)} зон × {len(seeds)} стартов, "
          f"H={args.horizon}, {args.samples}×{args.iters} CEM")
    if args.probe_weight is None:
        args.probe_weight = 1.0 if args.cost == "mixed" else 0.0
    from plan import load_probe
    probe = load_probe(args.tag) if args.probe_weight else None
    print(f"стоимость: {args.cost}" + (f" (probe_weight={args.probe_weight})" if args.cost == "mixed" else ""))
    episodes, costs, shadow_misses, plan_times = latent_eval(
        model, ema, zones, seeds, args.horizon, args.steps, cem_kw,
        probe=probe, probe_weight=args.probe_weight, world=world, slip=slip, paired=paired)
    succ = sum(e["success"] for e in episodes) / len(episodes)
    print(f"  успех: {succ*100:.0f}%  шагов в среднем: "
          f"{np.mean([e['steps_used'] for e in episodes]):.1f}  "
          f"время решения (медиана): {median(plan_times)*1000:.0f} мс, "
          f"p90: {np.percentile(plan_times, 90)*1000:.0f} мс")

    rho = spearman(costs, shadow_misses)
    print(f"  корреляция латентная стоимость ↔ реальный промах (Спирмен): {rho:.3f} "
          f"на {len(costs)} решениях")

    result = {"tag": args.tag, "world": world, "zones": zones, "seeds": seeds,
              "cost": args.cost, "probe_weight": args.probe_weight,
              "latent": {"success_rate": succ,
                         "plan_time_median": median(plan_times),
                         "plan_time_p90": float(np.percentile(plan_times, 90)) if plan_times else None,
                         "cost_vs_miss_spearman": rho,
                         "decisions": len(costs)},
              "episodes": episodes}

    if paired:
        labels = paired_interval(episodes)
        expanded = False
        if seeds == list(range(1, 11)) and interval_unresolved(labels):
            extra = list(range(11, 23))
            print("  интервал накрывает 0 или 15 — один раз добавляю сиды 11…22")
            more, more_costs, more_miss, more_times = latent_eval(
                model, ema, zones, extra, args.horizon, args.steps, cem_kw,
                probe=probe, probe_weight=args.probe_weight, world=world, slip=slip, paired=True)
            episodes.extend(more)
            costs.extend(more_costs)
            shadow_misses.extend(more_miss)
            plan_times.extend(more_times)
            labels = paired_interval(episodes)
            expanded = True
            result["episodes"] = episodes
            result["seeds"] = seeds + extra
            succ = sum(e["success"] for e in episodes) / len(episodes)
            result["latent"]["success_rate"] = succ
            result["latent"]["plan_time_median"] = median(plan_times)
            result["latent"]["plan_time_p90"] = float(np.percentile(plan_times, 90))
            result["latent"]["cost_vs_miss_spearman"] = spearman(costs, shadow_misses)
            result["latent"]["decisions"] = len(costs)
        labels["expanded"] = expanded
        rand = sum(e["rand_success"] for e in episodes) / len(episodes)
        result["latent"]["random_baseline"] = rand
        result["labels"] = labels
        print(f"  случайные действия: {rand*100:.1f}%  "
              f"delta={labels['delta_pp']:.1f} п.п.  "
              f"интервал [{labels['ci_low']:.1f}, {labels['ci_high']:.1f}]  "
              f"лучше_случайных={labels['лучше_случайных']}  отрыв_15={labels['отрыв_15']}")
        result["probe_r2"] = probe_r2(args.tag, world, model)
        print("  probe R²: " + ", ".join(f"{k}={v:.3f}" for k, v in result["probe_r2"].items()))
        if world == "noise":
            result["noise"] = denoise_report(model, ema)
            print(f"  шум: {result['noise']['label']}  "
                  f"до чистого {result['noise']['dist_clean']:.4f}  "
                  f"до шумного {result['noise']['dist_noisy']:.4f}")
    else:
        rnd = random_baseline(zones, seeds, args.steps, np.random.default_rng(0))
        print(f"  случайные действия: {rnd*100:.0f}%")
        result["latent"]["random_baseline"] = rnd

    # пиксельное сравнение: время чужого мира в отчёт не подставляется
    if args.pixel_episodes:
        pixel_path = os.path.join(OUT, f"pixel_{args.tag}.pt")
        if not os.path.exists(pixel_path):
            if paired:
                print("  пиксельного чекпоинта этого мира нет — поле времени не пишу")
            else:
                raise SystemExit(f"нет чекпоинта {pixel_path}")
        else:
            pmodel, _, pck = load_model(pixel_path, threads=args.threads)
            if world_of(pck) != world:
                print(f"  пиксельный чекпоинт мира {world_of(pck)} не подходит к {world} — время не пишу")
            else:
                print(f"пиксельный планировщик: {args.pixel_episodes} стартов (это долго)")
                pez = zones[:1]
                pseeds = seeds[:args.pixel_episodes]
                t0 = time.perf_counter()
                step_cap = args.steps if paired else min(args.steps, 10)
                pepisodes = pixel_eval(
                    pmodel, pez, pseeds, args.horizon, step_cap, cem_kw, world=world, slip=slip)
                ptime = median([e["plan_time_median"] for e in pepisodes])
                print(f"  время решения (медиана): {ptime*1000:.0f} мс  "
                      f"[всего {time.perf_counter()-t0:.0f} с]")
                if paired:
                    result["pixel_plan_time_median"] = ptime
                else:
                    psucc = sum(e["success"] for e in pepisodes) / len(pepisodes)
                    print(f"  успех: {psucc*100:.0f}%")
                    result["pixel"] = {"success_rate": psucc, "plan_time_median": ptime,
                                       "episodes": pepisodes}

    if args.grid:
        grid_path = os.path.join(OUT, "plan_grid.png" if world == "flat" else f"plan_grid_{args.tag}.png")
        gok = make_grid(model, ema, seeds[0], args.horizon, args.steps, cem_kw,
                        grid_path, world=world, slip=slip)
        print(f"  сетка 9 зон (seed={seeds[0]}): {gok}/9 попаданий -> {os.path.basename(grid_path)}")
        result["grid_success"] = gok

    name = f"eval_{args.tag}_{args.cost}.json"
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\nсохранено: out/{name}")


if __name__ == "__main__":
    main()

"""Планировщик: CEM (cross-entropy method) + MPC, целиком в латентном пространстве.

Цель задаётся латентом кадра-цели: z_goal = E_ema(мяч в центре зоны).
Стоимость плана: ||z_H − z_goal||² + 0.3 · среднее по последним трём шагам.

    python plan.py --goal "правый верхний" --seed 1
    python plan.py --zone 0 --steps 30 --dry

Пиксели в поиске не участвуют. Пиксели нужны только чтобы показать результат.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from data import (ACTION_NAMES, N_ACTIONS, ZONE_CENTERS, ZONE_HALF, ZONES, World,
                  find_zone, in_zone, step_state, zone_frame)
from models import EMATarget, Probe, WorldModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SCALE = 4


def load_probe(tag):
    """Загружает ridge-probe, если он обучен (нужен для смешанной стоимости)."""
    path = os.path.join(OUT, f"probe_{tag}.pt")
    if not os.path.exists(path):
        return None
    probe = Probe()
    probe.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["probe"])
    probe.eval()
    return probe


def load(tag="v3", mode="jepa", threads=8):
    torch.set_num_threads(threads)
    path = os.path.join(OUT, f"{mode}{('_' + tag) if tag else ''}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel(n_actions=ck.get("n_actions", 0))
    model.load_state_dict(ck["model"])
    model.eval()
    ema = EMATarget(model.encoder)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    return model, ema


@torch.no_grad()
def cem_plan(model, ema, context, z_goal, horizon=10, samples=128, iters=8,
             elite=16, seed=0, log_probe=None, goal_xy=None, probe_weight=0.0):
    """Ищет последовательность действий в латенте.

    context: (2,1,64,64) два последних настоящих кадра.
    Возвращает (действия (H,) int64, стоимость float, время сек, инфо dict).
    """
    t0 = time.perf_counter()
    z_prev = model.encoder(context[0:1])
    z_cur = model.encoder(context[1:2])
    probs = torch.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    g = torch.Generator().manual_seed(seed)

    best_seq, best_cost = None, float("inf")
    for _ in range(iters):
        u = torch.multinomial(probs, samples, replacement=True).T.contiguous()  # (S,H)
        zp = z_prev.expand(samples, -1).contiguous()
        zc = z_cur.expand(samples, -1).contiguous()
        ds = []
        for k in range(horizon):
            zn = model.predictor(zp, zc, u[:, k])
            ds.append((zn - z_goal).pow(2).mean(1))
            zp, zc = zc, zn
        cost = ds[-1] + 0.3 * torch.stack(ds[-3:]).mean(0)          # (S,)
        if probe_weight and log_probe is not None and goal_xy is not None:
            # смешанная стоимость: латент + позиция мяча по probe (в пикселях, /100)
            xy = log_probe(zp, zc)
            gxy = torch.tensor(goal_xy, dtype=xy.dtype)
            cost = cost + probe_weight * ((xy[:, :2] - gxy) ** 2).mean(1) / 100.0

        i = int(cost.argmin())
        if float(cost[i]) < best_cost:
            best_cost, best_seq = float(cost[i]), u[i].clone()

        idx = cost.argsort()[:elite]
        counts = torch.zeros_like(probs)
        for e in idx:
            counts[torch.arange(horizon), u[e]] += 1
        dist = (counts + 0.5) / (elite + 0.5 * N_ACTIONS)
        probs = 0.7 * probs + 0.3 * dist

    info = {}
    if log_probe is not None:
        # оценка цели по probe: другой (не-латентный) взгляд на то же решение
        zp, zc = z_prev.clone(), z_cur.clone()
        for k in range(horizon):
            zn = model.predictor(zp, zc, best_seq[k].view(1))
            zp, zc = zc, zn
        xy = log_probe(zp, zc)[0].numpy()
        info["probe_xy"] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
    return best_seq, best_cost, time.perf_counter() - t0, info


def execute(model, ema, zone_index, seed=1, horizon=10, max_steps=30,
            cem_kw=None, probe=None, verbose=True, save_gif=True, tag="", shadow=True,
            keep_frames=False, probe_weight=0.0):
    """MPC: план -> выполняем первое действие -> перепланируем от настоящих кадров."""
    cem_kw = cem_kw or {}
    w = World(seed)
    f_a = w.frame()
    f_b = w.step(4)                       # нулевой шаг, чтобы получить пару кадров
    z_goal = ema(torch.from_numpy(zone_frame(zone_index))).detach()
    cx, cy = ZONE_CENTERS[zone_index]

    steps, frames = [], [f_a[0], f_b[0]]
    prev_frame, cur_frame = f_a, f_b
    success = False
    for t in range(max_steps):
        ctx = torch.from_numpy(np.stack([prev_frame, cur_frame]))
        seq, cost, dt, info = cem_plan(model, ema, ctx, z_goal, horizon=horizon,
                                       log_probe=probe, goal_xy=(cx, cy),
                                       probe_weight=probe_weight, **cem_kw)
        action = int(seq[0])
        pos_before = w.pos.copy()
        if shadow:
            # открытая петля на настоящей физике: куда мяч придёт, если выполнить план целиком.
            # нужно, чтобы проверить, предсказывает ли латентная стоимость реальный промах
            sp, sv = w.pos.copy(), w.vel.copy()
            for a in seq.tolist():
                sp, sv = step_state(sp, sv, a)
            info["shadow_miss"] = round(float(np.hypot(sp[0] - cx, sp[1] - cy)), 2)
        new_frame = w.step(action)
        frames.append(new_frame[0])
        steps.append({"step": t + 1, "action": ACTION_NAMES[action], "cost": round(cost, 4),
                      "plan_time": round(dt, 3), "pos": [round(float(v), 1) for v in w.pos],
                      **info})
        if verbose:
            print(f"  шаг {t+1}: {ACTION_NAMES[action]:<12} "
                  f"мяч ({pos_before[0]:.0f},{pos_before[1]:.0f}) -> "
                  f"({w.pos[0]:.0f},{w.pos[1]:.0f})  cost={cost:.3f}  {dt*1000:.0f} мс")
        if in_zone(w.pos, zone_index):
            success = True
            break
        prev_frame, cur_frame = cur_frame, new_frame

    cx, cy = ZONE_CENTERS[zone_index]
    miss = float(np.hypot(w.pos[0] - cx, w.pos[1] - cy))
    result = {"zone": zone_index, "zone_name": ZONES[zone_index]["name"], "seed": seed,
              "success": success, "steps_used": len(steps), "miss_px": round(miss, 2),
              "steps": steps, "final_pos": [round(float(v), 1) for v in w.pos]}
    if save_gif:
        path = os.path.join(OUT, f"plan_{zone_index}_{seed}{tag}.gif")
        save_trajectory(frames, path, zone_index, result)
        result["gif"] = path
    if keep_frames:
        result["frames"] = frames
    return result


def draw_zone(img: Image.Image, zone_index: int):
    d = ImageDraw.Draw(img)
    cx, cy = ZONE_CENTERS[zone_index]
    x0 = (cx - ZONE_HALF) * SCALE
    y0 = (cy - ZONE_HALF) * SCALE
    d.rectangle([x0, y0, x0 + 2 * ZONE_HALF * SCALE, y0 + 2 * ZONE_HALF * SCALE],
                outline=(90, 200, 90), width=2)
    return img


def frame_to_img(fr: np.ndarray) -> Image.Image:
    a = ((np.clip(fr, -1, 1) + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(a, "L").convert("RGB").resize((64 * SCALE, 64 * SCALE), Image.BILINEAR)


def save_trajectory(frames, path, zone_index, result):
    imgs = []
    for i, fr in enumerate(frames):
        img = draw_zone(frame_to_img(fr), zone_index)
        canvas = Image.new("RGB", (img.width + 160, img.height), "white")
        canvas.paste(img, (160, 0))
        d = ImageDraw.Draw(canvas)
        d.text((6, 8), f"цель: {ZONES[zone_index]['name']}", fill="black")
        d.text((6, 30), f"кадр {i}", fill="black")
        d.text((6, 52), "OK" if result["success"] else "промах", fill="green" if result["success"] else "red")
        imgs.append(canvas)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=150, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default=None, help="фраза, напр. 'правый верхний угол'")
    ap.add_argument("--zone", type=int, default=None, help="индекс зоны 0..8")
    ap.add_argument("--seed", type=int, default=1, help="сид стартового состояния мира")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30, help="максимум реальных шагов")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--dry", action="store_true", help="только план, без исполнения")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cost", choices=["latent", "mixed"], default="latent",
                    help="latent: стоимость только в латенте; mixed: латент + позиция мяча по probe")
    ap.add_argument("--probe-weight", type=float, default=None,
                    help="вес probe-члена в mixed (по умолчанию 1.0)")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    zone = args.zone
    if zone is None:
        if args.goal is None:
            ap.error("нужен --goal или --zone")
        zone = find_zone(args.goal)
        if zone is None:
            print("не понял цель. Доступные зоны: " + ", ".join(f"{i}={z['name']}" for i, z in enumerate(ZONES)))
            return

    if args.probe_weight is None:
        args.probe_weight = 1.0 if args.cost == "mixed" else 0.0
    model, ema = load(args.tag, threads=args.threads)
    probe = load_probe(args.tag) if args.probe_weight else None
    if args.probe_weight and probe is None:
        print("probe не найден — считаю без него (python probe.py --tag " + args.tag + ")")
        args.probe_weight = 0.0
    print(f"стоимость: {args.cost}" + (f" (probe_weight={args.probe_weight})" if args.cost == "mixed" else ""))
    cem_kw = {"samples": args.samples, "iters": args.iters, "seed": args.seed}
    print(f"цель: {ZONES[zone]['name']} {ZONE_CENTERS[zone]}, старт seed={args.seed}")
    t0 = time.perf_counter()
    res = execute(model, ema, zone, seed=args.seed, horizon=args.horizon,
                  max_steps=args.steps, cem_kw=cem_kw, probe=probe,
                  probe_weight=args.probe_weight, save_gif=not args.no_gif)
    res.pop("frames", None)
    total = time.perf_counter() - t0
    res["total_time"] = round(total, 2)
    with open(os.path.join(OUT, f"plan_{zone}_{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nитог: {'ДОСТИГНУТА' if res['success'] else 'промах'} за {res['steps_used']} шагов, "
          f"промах {res['miss_px']} px, всего {total:.1f} с")
    if res.get("gif"):
        print("gif:", res["gif"])


if __name__ == "__main__":
    main()

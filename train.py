"""Обучение action-conditioned world model: --mode jepa | --mode pixel.

    python train.py --mode jepa  --steps 2500 --tag v3
    python train.py --mode pixel --steps 2500 --tag v3

Обе модели видят одни и те же данные (кадры + случайные действия) и имеют одинаковую
архитектуру. Разница только в том, что является целью предиктора: эмбеддинг или кадр.

Лосс считается на ТРЁХ шагах раскатки, каждый следующий — по своим же предсказаниям.
Это то, что будет делать планировщик, и без такого обучения он дрейфует.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import N_ACTIONS, NOISE_SIGMA, SLIP, WORLDS, make_batch, measure_jitter_ambiguity
from models import EMATarget, WorldModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
HORIZON = 6  # x0..x5: два кадра контекста + три цели, нужно 5 действий


def build_loss(mode, model, ema, frames, acts):
    """Возвращает (total_loss, {компоненты}).

    frames: x0..x5, acts: a0..a4 (действие i переводит кадр i в кадр i+1).
    Три шага предсказания: x1->x2 (действие a1), x2->x3 (a2), x3->x4 (a3).
    """
    x = [frames[:, i] for i in range(5)]          # x0..x4
    z0, z1 = model.encoder(x[0]), model.encoder(x[1])

    # реконструкция текущего кадра: не даёт латенту коллапсировать + даёт декодер
    rec = F.mse_loss(model.decoder(z1), x[1])

    if mode == "jepa":
        p2 = model.predictor(z0, z1, acts[:, 1])
        p3 = model.predictor(z1, p2, acts[:, 2])
        p4 = model.predictor(p2, p3, acts[:, 3])
        with torch.no_grad():
            t2, t3, t4 = ema(x[2]), ema(x[3]), ema(x[4])
        l1 = F.mse_loss(p2, t2)
        l2 = F.mse_loss(p3, t3)
        l3 = F.mse_loss(p4, t4)
        # VICReg-штраф не нужен: латент на единичной сфере, а реконструкция не даёт коллапсу
        total = l1 + 0.5 * l2 + 0.25 * l3 + 0.5 * rec
        return total, {"l1": l1.item(), "l2": l2.item(), "l3": l3.item(), "rec": rec.item()}

    # pixel-режим: предсказываем сами кадры, свой кадр возвращаем в энкодер
    f2 = model.decoder(model.predictor(z0, z1, acts[:, 1]))
    l1 = F.mse_loss(f2, x[2])
    z2 = model.encoder(f2)
    f3 = model.decoder(model.predictor(z1, z2, acts[:, 2]))
    l2 = F.mse_loss(f3, x[3])
    z3 = model.encoder(f3)
    f4 = model.decoder(model.predictor(z2, z3, acts[:, 3]))
    l3 = F.mse_loss(f4, x[4])
    total = l1 + 0.5 * l2 + 0.25 * l3 + 0.5 * rec
    return total, {"l1": l1.item(), "l2": l2.item(), "l3": l3.item(), "rec": rec.item()}


@torch.no_grad()
def quick_eval(mode, model, ema, seed=12345, world="flat"):
    """Одношаговая ошибка на фиксированном батче: в пикселях и в латенте."""
    rng = np.random.default_rng(seed)
    frames, acts, _ = make_batch(128, rng, HORIZON, world=world)
    frames, acts = torch.from_numpy(frames), torch.from_numpy(acts)
    z = model.predictor(model.encoder(frames[:, 0]), model.encoder(frames[:, 1]), acts[:, 1])
    pix = F.mse_loss(model.decoder(z), frames[:, 2]).item()
    z_true = ema(frames[:, 2]) if mode == "jepa" else model.encoder(frames[:, 2])
    return {"pixel_mse": pix, "latent_mse": F.mse_loss(z, z_true).item()}


def ckpt_path(mode, tag):
    return os.path.join(OUT, f"{mode}{('_' + tag) if tag else ''}.pt")


def refuse_tag(world, tag):
    """Неплоский режим не имеет права затереть v4 и архив v0.2."""
    if world == "flat":
        return
    if not tag or tag == "v4":
        raise SystemExit(
            f"режим {world}: нужен непустой тег и не v4, например v5-{world}. "
            "Файлы out/jepa.pt, out/pixel.pt и чекпоинты v4 не перезаписываются.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["jepa", "pixel"], required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="суффикс имени чекпоинта, напр. v3")
    ap.add_argument("--world", choices=WORLDS, default="flat")
    ap.add_argument("--unit-norm", type=int, default=1, help="1: латент на единичной сфере")
    ap.add_argument("--threads", type=int, default=8)  # 12 потоков на 6 ядрах медленнее
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)
    refuse_tag(args.world, args.tag)
    if args.world == "jitter":
        gate = measure_jitter_ambiguity()
        gate_path = os.path.join(OUT, "ambiguity_jitter.json")
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(gate, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"[jitter] радиус разброса {gate['radius_px']:.2f} px "
              f"(порог {gate['threshold_px']}) -> {gate_path}")
        if not gate["train"]:
            raise SystemExit(
                "jitter не обучается: радиус не больше 6 px, это не проверка неоднозначности. "
                "Вероятность 0.25 не поднимается.")

    model = WorldModel(n_actions=N_ACTIONS, unit_norm=bool(args.unit_norm))
    ema = EMATarget(model.encoder, momentum=0.99)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    print(f"[{args.mode}] params={model.params_count()/1e6:.2f}M steps={args.steps} "
          f"batch={args.batch} actions={N_ACTIONS} horizon={HORIZON} world={args.world}")
    log, t_start = [], time.perf_counter()

    for step in range(1, args.steps + 1):
        frames, acts, _ = make_batch(args.batch, rng, HORIZON, world=args.world)
        frames, acts = torch.from_numpy(frames), torch.from_numpy(acts)

        loss, parts = build_loss(args.mode, model, ema, frames, acts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if args.mode == "jepa":
            ema.update(model.encoder)

        if step % 100 == 0 or step == 1:
            ev = quick_eval(args.mode, model, ema, world=args.world)
            row = {"step": step, "loss": round(loss.item(), 5),
                   **{k: round(v, 5) for k, v in parts.items()},
                   **{k: round(v, 5) for k, v in ev.items()}}
            log.append(row)
            el = time.perf_counter() - t_start
            print(f"  {step:5d}/{args.steps} loss={row['loss']:.4f} "
                  f"l1={row['l1']:.4f} l3={row['l3']:.4f} rec={row['rec']:.4f} | "
                  f"eval pix={row['pixel_mse']:.4f} lat={row['latent_mse']:.4f} | "
                  f"{el/step*1000:.0f} ms/step, осталось ~{el/step*(args.steps-step)/60:.1f} мин")

    path = ckpt_path(args.mode, args.tag)
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "args": vars(args), "n_actions": N_ACTIONS,
                "horizon": HORIZON, "log": log,
                "world": args.world,
                "sigma": NOISE_SIGMA if args.world == "noise" else None,
                "slip": SLIP if args.world == "jitter" else None}, path)
    with open(os.path.join(OUT, f"{args.mode}{('_' + args.tag) if args.tag else ''}_log.json"),
              "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"[{args.mode}] сохранено: {path}  ({(time.perf_counter()-t_start)/60:.1f} мин)")


if __name__ == "__main__":
    main()

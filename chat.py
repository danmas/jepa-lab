"""Чат с world model: слова -> цель -> планирование в латенте -> отчёт словами.

    python chat.py

Где здесь LLM (и где его нет):
  * слова -> зона            — детерминированная таблица (в реальной системе это LLM);
  * динамика и поиск         — JEPA-модель + CEM, целиком в латенте;
  * латент -> слова          — линейный probe (R² в out/probe_*.json).

Ни одна строчка этого файла не требует GPU и не зовёт внешние API.
"""
import argparse
import os
import sys

import numpy as np
import torch

from data import SLIP, ZONE_CENTERS, ZONES, find_zone
from models import Probe, load_model, world_of
from plan import execute

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load_probe(tag):
    path = os.path.join(OUT, f"probe_{tag}.pt")
    if not os.path.exists(path):
        return None
    probe = Probe()
    probe.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["probe"])
    probe.eval()
    return probe


def dir_word(v, plus, minus, neutral):
    return plus if v > 0.6 else minus if v < -0.6 else neutral


def describe_state(model, probe, frames, true_pos=None):
    """Пара последних кадров -> слова (через probe: позиция + скорость)."""
    if probe is None:
        return "probe не обучен — запустите python probe.py"
    with torch.no_grad():
        x = torch.from_numpy(np.stack(frames).astype(np.float32))[:, None]
        z = model.encoder(x)
        xy = probe(z[0:1], z[1:2])[0].numpy()
    vx, vy = float(xy[2]), float(xy[3])
    words = (f"мяч примерно в ({xy[0]:.0f}, {xy[1]:.0f}), "
             f"летит {dir_word(vx, 'вправо', 'влево', 'по вертикали')}"
             f"/{dir_word(vy, 'вниз', 'вверх', 'без вертикали')}")
    if true_pos is not None:
        words += f"  [истина: ({true_pos[0]:.0f}, {true_pos[1]:.0f})]"
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cost", choices=["latent", "mixed"], default="latent",
                    help="latent — только латентная стоимость; mixed — латент + позиция по probe")
    ap.add_argument("--probe-weight", type=float, default=None,
                    help="вес probe-члена в mixed (по умолчанию 1.0)")
    args = ap.parse_args()

    model, ema, ck = load_model(os.path.join(OUT, f"jepa_{args.tag}.pt"), threads=args.threads)
    world = world_of(ck)
    slip = float(ck.get("slip") or SLIP)
    if args.probe_weight is None:
        args.probe_weight = 1.0 if args.cost == "mixed" else 0.0
    probe = load_probe(args.tag)
    cem_kw = {"samples": args.samples, "iters": args.iters, "seed": args.seed}

    print(f"Чат с world model, мир «{world}». Примеры: «положи мяч в правый верхний угол», "
          "«где мяч», «зоны», «выход».")
    print("Модель ничего не знает про слова: фразу разбирает таблица зон, "
          "планирование идёт в латенте.\n")

    last = None
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        low = text.lower()
        if low in ("выход", "exit", "quit", "q"):
            break
        if "зон" in low and find_zone(text) is None:
            print("Доступные зоны (пиши любую из формулировок):")
            for i, z in enumerate(ZONES):
                print(f"  {i}: {z['name']:<14} {', '.join([z['name'], *z['syn']])}")
            continue
        if "где" in low or "положение" in low or "скорост" in low:
            if last is None:
                print("Мяч ещё не появлялся — сначала поставь цель.")
            else:
                print(describe_state(model, probe, last["frames"][-2:], last["pos"]))
            continue

        zone = find_zone(text)
        if zone is None:
            print("Не понял цель. Скажи, например: «в верхний правый угол» или "
                  "«в центр». Список зон: «зоны».")
            continue

        print(f"[слой 1] фраза -> зона «{ZONES[zone]['name']}» {ZONE_CENTERS[zone]}")
        res = execute(model, ema, zone, seed=args.seed, horizon=args.horizon,
                      max_steps=args.steps, cem_kw=cem_kw, probe=probe,
                      probe_weight=args.probe_weight,
                      verbose=True, save_gif=True, tag="_chat", keep_frames=True,
                      world=world, slip=slip)
        frames = res.pop("frames")
        last = {"frames": frames, "pos": res["final_pos"], "zone": zone}
        print(f"[итог] {'достигнута' if res['success'] else 'промах'} за "
              f"{res['steps_used']} шагов, промах {res['miss_px']} px")
        print(f"[слой 3] {describe_state(model, probe, frames[-2:], res['final_pos'])}")
        if res.get("gif"):
            print(f"[файл] {res['gif']}")
        print()


if __name__ == "__main__":
    main()

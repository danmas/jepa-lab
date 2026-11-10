"""«Воображение»: раскатываем мир вперёд и смотрим, кто дольше не врёт.

JEPA-модель крутит шаги ЦЕЛИКОМ в латентном пространстве (пиксели трогает
только чтобы показать нам результат). Пиксельная модель вынуждена каждый шаг
генерировать кадр и скармливать его себе же.

    python imagine.py                 # K=30 шагов, все артефакты в out/

Результаты: out/rollout.png (сетка), out/rollout.gif, out/error_curve.png,
out/latent_pca.png, out/metrics.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from data import make_episode
from models import EMATarget, WorldModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SCALE = 3  # увеличение кадров для наглядности


def load(mode):
    ck = torch.load(os.path.join(OUT, f"{mode}.pt"), map_location="cpu", weights_only=False)
    model = WorldModel()
    model.load_state_dict(ck["model"])
    model.eval()
    ema = EMATarget(model.encoder)
    ema.load_state_dict(ck["ema"])
    ema.eval()
    return model, ema, ck.get("log", [])


@torch.no_grad()
def rollout(model, frames: torch.Tensor, steps: int, mode: str):
    """frames: (T,1,64,64). Возвращает (декодированные кадры (steps,1,64,64), латенты)."""
    x_prev, x_cur = frames[0:1], frames[1:2]
    z_prev, z_cur = model.encoder(x_prev), model.encoder(x_cur)
    frames_out, latents = [], []
    for _ in range(steps):
        z_next = model.predictor(z_prev, z_cur)
        frames_out.append(model.decoder(z_next))
        latents.append(z_next)
        if mode == "pixel":
            x_prev, x_cur = x_cur, frames_out[-1]  # авторегрессия по своим кадрам
            z_prev, z_cur = model.encoder(x_prev), model.encoder(x_cur)
        else:
            z_prev, z_cur = z_cur, z_next  # чистая латентная раскатка, пиксели не участвуют
    return torch.cat(frames_out), torch.cat(latents)


def to_img(t: torch.Tensor) -> Image.Image:
    a = t[0].clamp(-1, 1).numpy()
    a = ((a + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(a, "L").convert("RGB").resize((64 * SCALE, 64 * SCALE), Image.BILINEAR)


def sharpness(t: torch.Tensor) -> float:
    """Средний градиент кадра: честный индикатор «размылось в кашу»."""
    dx = (t[..., 1:] - t[..., :-1]).abs().mean().item()
    dy = (t[..., 1:, :] - t[..., :-1, :]).abs().mean().item()
    return (dx + dy) / 2


def centroid(t: torch.Tensor):
    """Центр масс яркости кадра (x, y) — оценка позиции мяча."""
    w = (t.squeeze() + 1) / 2                      # -> (H, W)
    s = w.sum().clamp(min=1e-6)
    ys, xs = torch.meshgrid(torch.arange(t.shape[-2], dtype=torch.float32),
                            torch.arange(t.shape[-1], dtype=torch.float32), indexing="ij")
    return torch.stack([(w * xs).sum() / s, (w * ys).sum() / s])


def label_column(texts, h):
    w = 150
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(texts):
        d.text((6, i * (h // len(texts)) + 6), t, fill="black")
    return img


def make_grid(real, jepa, pixel, cols, title_rows=("реальные кадры", "JEPA (в латенте)", "пиксельная модель")):
    """Сетка: строки — модели, столбцы — горизонты."""
    cw = 64 * SCALE
    pad, top = 8, 26
    W = 150 + len(cols) * (cw + pad) + pad
    H = top + 3 * (cw + pad) + pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for j, t in enumerate(cols):
        d.text((150 + j * (cw + pad) + pad, 6), f"шаг +{t}", fill="black")
    for i, (row, name) in enumerate(zip((real, jepa, pixel), title_rows)):
        y = top + i * (cw + pad)
        d.text((6, y + cw // 2), name, fill="black")
        for j, t in enumerate(cols):
            canvas.paste(to_img(row[t - 1]), (150 + j * (cw + pad) + pad, y))
    return canvas


def make_gif(real, jepa, pixel, path):
    n = len(jepa)
    frames = []
    for t in range(n):
        canvas = Image.new("RGB", (3 * 64 * 2 + 16, 64 * 2 + 18), "white")
        d = ImageDraw.Draw(canvas)
        for i, name in enumerate(("реальное", "JEPA", "пиксели")):
            d.text((i * (128 + 8) + 4, 3), name, fill="black")
        for i, f in enumerate((real[t], jepa[t], pixel[t])):
            canvas.paste(to_img(f).resize((128, 128), Image.BILINEAR), (i * (128 + 8), 16))
        d.text((4, 64 * 2 + 4), f"шаг +{t + 1}", fill="black")
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)


def plot_curves(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    steps = np.arange(1, len(metrics["jepa"]["pixel_mse"]) + 1)
    keys = (("pixel_mse", "Ошибка кадра (MSE, лог)"),
            ("centroid_err", "Промах мяча, пикселей"),
            ("sharpness", "Резкость (больше = чётче)"))
    for i, (key, title) in enumerate(keys):
        for mode, style in (("jepa", "o-"), ("pixel", "s--")):
            ax[i].plot(steps, metrics[mode][key], style, label=mode, ms=3)
        ax[i].set_title(title, fontsize=10)
        ax[i].set_xlabel("шаг раскатки")
        ax[i].grid(alpha=.3)
        ax[i].legend(fontsize=8)
    ax[0].set_yscale("log")
    if "real_sharpness" in metrics:
        ax[2].axhline(metrics["real_sharpness"], color="k", ls=":", lw=1, label="реальный кадр")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_latent(model, ema, episode_frames, jepa_lat, pixel_lat, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        # облако настоящих латентов: много эпизодов
        cloud = []
        for s in range(6):
            fr, _ = make_episode(40, seed=500 + s)
            cloud.append(model.encoder(torch.from_numpy(fr)).numpy())
        cloud = np.concatenate(cloud)
        real_roll = ema(episode_frames[2:2 + len(jepa_lat)]).numpy()

    allp = np.concatenate([cloud, real_roll, jepa_lat.numpy(), pixel_lat.numpy()])
    mu = allp.mean(0)
    _, _, vt = np.linalg.svd(allp - mu, full_matrices=False)
    proj = lambda a: (a - mu) @ vt[:2].T

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(*proj(cloud).T, s=4, alpha=.25, c="tab:gray", label="латент настоящих кадров")
    r = proj(real_roll)
    ax.plot(*r.T, "o-", ms=3, c="tab:green", alpha=.7, label="истинная траектория")
    ax.plot(*proj(np.concatenate([jepa_lat.numpy()])).T, "x--", ms=5, c="tab:blue", label="JEPA: раскатка в латенте")
    ax.plot(*proj(pixel_lat.numpy()).T, "+--", ms=6, c="tab:red", label="пиксельная: латент её же кадров")
    ax.set_title("Латентное пространство (PCA, 2 компоненты)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)

    jm, jema, _ = load("jepa")
    pm, pema, _ = load("pixel")

    frames_np, pos = make_episode(2 + args.steps, seed=args.seed)
    frames = torch.from_numpy(frames_np)

    jepa_frames, jepa_lat = rollout(jm, frames, args.steps, "jepa")
    pixel_frames, pixel_lat = rollout(pm, frames, args.steps, "pixel")
    real = frames[2:2 + args.steps]

    with torch.no_grad():
        true_lat = jema(real)

    metrics = {"seed": args.seed, "steps": args.steps, "jepa": {}, "pixel": {}}
    for name, out, lat in (("jepa", jepa_frames, jepa_lat), ("pixel", pixel_frames, pixel_lat)):
        metrics[name]["pixel_mse"] = [round(F.mse_loss(o, r).item(), 5) for o, r in zip(out, real)]
        metrics[name]["sharpness"] = [round(sharpness(o), 5) for o in out]
        metrics[name]["latent_mse"] = [round(F.mse_loss(l, t).item(), 5) for l, t in zip(lat, true_lat)]
        metrics[name]["centroid_err"] = [round(torch.dist(centroid(o), centroid(r)).item(), 3)
                                         for o, r in zip(out, real)]
    metrics["real_sharpness"] = round(sharpness(real), 5)

    cols = [c for c in (1, 3, 5, 10, 15, 20, 25, 30) if c <= args.steps]
    grid = make_grid(real, jepa_frames, pixel_frames, cols)
    grid.save(os.path.join(OUT, "rollout.png"))
    make_gif(real, jepa_frames, pixel_frames, os.path.join(OUT, "rollout.gif"))
    plot_curves(metrics, os.path.join(OUT, "error_curve.png"))
    plot_latent(jm, jema, real, jepa_lat, pixel_lat, os.path.join(OUT, "latent_pca.png"))

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)

    print(f"резкость реального кадра: {metrics['real_sharpness']:.4f}")
    print(f"{'шаг':>4} | {'JEPA MSE':>8} {'дрейф px':>8} {'резк':>6} | {'PIX MSE':>8} {'дрейф px':>8} {'резк':>6}")
    for i in range(args.steps):
        if (i + 1) in cols:
            j, p = metrics["jepa"], metrics["pixel"]
            print(f"{i+1:>4} | {j['pixel_mse'][i]:>8.4f} {j['centroid_err'][i]:>8.2f} {j['sharpness'][i]:>6.4f} | "
                  f"{p['pixel_mse'][i]:>8.4f} {p['centroid_err'][i]:>8.2f} {p['sharpness'][i]:>6.4f}")
    print("\nготово: out/rollout.png, out/rollout.gif, out/error_curve.png, out/latent_pca.png")


if __name__ == "__main__":
    main()

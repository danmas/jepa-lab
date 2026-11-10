"""Страница проверки: фраза -> план в латенте -> мяч на экране.

    python demo_ui.py
    открыть http://127.0.0.1:8765

Фразу разбирает таблица зон. Модель будущее кадрами не рисует:
CEM перебирает толчки в 48-мерном латенте, в симулятор уходит
только первое действие, затем план считается заново.
"""
import argparse
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from chat import describe_state
from data import ZONE_CENTERS, ZONE_HALF, ZONES, find_zone
from models import load_model
from plan import load_probe, execute

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
VIEW = 384  # 64 * 6, как в глазах, не в поиске

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jepa-lab — цель словами</title>
<style>
  :root {
    --bg: #f3efe6;
    --ink: #1c1915;
    --muted: #5e584e;
    --line: #ddd4c4;
    --card: #fffdf8;
    --ok: #1d6b45;
    --bad: #8d2f2f;
    --zone: #2f9e62;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font: 16px/1.45 "Segoe UI", sans-serif;
  }
  main {
    max-width: 980px;
    margin: 0 auto;
    padding: 28px 20px 48px;
  }
  h1 { font-size: 28px; font-weight: 650; margin: 0 0 8px; letter-spacing: -0.03em; }
  .lead { color: var(--muted); max-width: 62ch; margin: 0 0 22px; }
  .bench {
    display: grid;
    grid-template-columns: 384px 1fr;
    gap: 22px;
    align-items: start;
  }
  .stage {
    background: #111;
    border-radius: 12px;
    padding: 12px;
  }
  canvas { display: block; width: 384px; height: 384px; background: #111; border-radius: 8px; }
  .play {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
    color: #d9d3c5;
    font-size: 13px;
  }
  .play button, .go, .zones button, select, input {
    font: inherit;
    color: var(--ink);
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 8px;
  }
  .play button { color: var(--ink); padding: 4px 10px; cursor: pointer; }
  .play input[type="range"] { flex: 1; }
  .panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 16px;
  }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  textarea, select, input[type="number"] {
    width: 100%;
    padding: 8px 10px;
    margin-bottom: 12px;
  }
  textarea { resize: vertical; min-height: 64px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .zones { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 12px; }
  .zones button { padding: 8px 6px; cursor: pointer; font-size: 13px; }
  .zones button:hover, .go:hover { border-color: #1c1915; }
  .go {
    width: 100%;
    padding: 10px 12px;
    background: #1c1915;
    color: #f3efe6;
    cursor: pointer;
    font-weight: 650;
  }
  .go:disabled { opacity: 0.55; cursor: wait; }
  .report { margin-top: 14px; white-space: pre-wrap; }
  .report strong.ok { color: var(--ok); }
  .report strong.bad { color: var(--bad); }
  .steps { margin: 8px 0 0; padding-left: 18px; }
  .steps li { margin: 2px 0; }
  .note { color: var(--muted); font-size: 13px; margin-top: 14px; }
  @media (max-width: 860px) {
    .bench { grid-template-columns: 1fr; }
    .stage, canvas { width: min(384px, 100%); height: auto; }
    canvas { height: auto; aspect-ratio: 1; }
  }
</style>
</head>
<body>
<main>
  <h1>Мяч в указанную зону</h1>
  <p class="lead">Фраза попадает в таблицу зон, не в сеть. Дальше модель не рисует будущее: она перебирает толчки в 48 числах, толкает мяч в симуляторе и на каждом шаге считает план заново. Картинка слева нужна только вам.</p>
  <div class="bench">
    <div class="stage">
      <canvas id="view" width="384" height="384"></canvas>
      <div class="play">
        <button id="toggle" type="button">пауза</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <span id="frame-label">кадр 0</span>
      </div>
    </div>
    <div class="panel">
      <label for="phrase">Фраза</label>
      <textarea id="phrase">мяч в верхний левый угол</textarea>
      <label>Зона</label>
      <div class="zones" id="zones"></div>
      <div class="row">
        <div>
          <label for="seed">Стартовый сид</label>
          <input id="seed" type="number" value="1" min="0" step="1">
        </div>
        <div>
          <label for="cost">Стоимость плана</label>
          <select id="cost">
            <option value="latent" selected>latent — только 48 чисел</option>
            <option value="mixed">mixed — плюс позиция по probe</option>
          </select>
        </div>
      </div>
      <button class="go" id="go" type="button">Довести мяч</button>
      <div class="report" id="report">Нажмите кнопку или выберите зону. Один прогон — до пары секунд на CPU.</div>
      <p class="note">Потолок 20 реальных шагов, поиск 128×8, горизонт 10. На 90 стартах latent попадает примерно в 8 случаях из 10. Промах — нормальный исход, не ошибка страницы.</p>
    </div>
  </div>
</main>
<script>
const view = document.getElementById("view");
const ctx = view.getContext("2d");
const report = document.getElementById("report");
const scrub = document.getElementById("scrub");
const frameLabel = document.getElementById("frame-label");
const toggle = document.getElementById("toggle");
const go = document.getElementById("go");
let frames = [];
let zone = null;
let timer = null;
let playing = true;

function draw(i) {
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, 384, 384);
  if (frames[i]) ctx.drawImage(frames[i], 0, 0);
  if (zone) {
    const s = 6, half = 8;
    ctx.strokeStyle = "#3dd68c";
    ctx.lineWidth = 2;
    ctx.strokeRect((zone[0] - half) * s, (zone[1] - half) * s, half * 2 * s, half * 2 * s);
  }
  frameLabel.textContent = "кадр " + i + (frames.length ? " / " + (frames.length - 1) : "");
  scrub.value = String(i);
}

function stop() {
  if (timer) cancelAnimationFrame(timer);
  timer = null;
}

function play() {
  stop();
  if (!playing || frames.length < 2) return;
  let last = 0;
  const step = (t) => {
    if (!playing) return;
    if (t - last > 180) {
      last = t;
      let i = Number(scrub.value) + 1;
      if (i >= frames.length) i = 0;
      draw(i);
    }
    timer = requestAnimationFrame(step);
  };
  timer = requestAnimationFrame(step);
}

scrub.addEventListener("input", () => { playing = false; toggle.textContent = "играть"; draw(Number(scrub.value)); stop(); });
toggle.addEventListener("click", () => {
  playing = !playing;
  toggle.textContent = playing ? "пауза" : "играть";
  if (playing) play(); else stop();
});

async function loadZones() {
  const zones = await (await fetch("/api/zones")).json();
  const box = document.getElementById("zones");
  for (const z of zones) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = z.name;
    b.addEventListener("click", () => {
      document.getElementById("phrase").value = "мяч в " + z.name;
      run();
    });
    box.appendChild(b);
  }
}

function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

async function run() {
  const text = document.getElementById("phrase").value.trim();
  go.disabled = true;
  report.textContent = "Перебирает толчки в латенте…";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text,
        seed: Number(document.getElementById("seed").value),
        cost: document.getElementById("cost").value
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "ошибка");
    frames = await Promise.all(data.frames.map(async (b64) => {
      const img = new Image();
      img.src = "data:image/png;base64," + b64;
      await img.decode();
      return img;
    }));
    zone = data.center;
    scrub.max = String(Math.max(0, frames.length - 1));
    playing = true;
    toggle.textContent = "пауза";
    draw(0);
    play();
    const klass = data.success ? "ok" : "bad";
    const head = data.success ? "цель достигнута" : "промах";
    const items = data.steps.map((s) =>
      "<li>шаг " + s.step + ": " + esc(s.action) +
      " → (" + s.pos[0] + ", " + s.pos[1] + "), " + Math.round(s.plan_time * 1000) + " мс</li>"
    ).join("");
    report.innerHTML =
      "<div>фраза → зона «" + esc(data.zone_name) + "» (" + data.center[0] + ", " + data.center[1] + ")</div>" +
      "<div><strong class='" + klass + "'>" + head + "</strong> за " + data.steps_used +
      " шагов, до центра " + data.miss_px + " px. Поиск в сумме " + data.think_ms + " мс.</div>" +
      "<div>" + esc(data.report) + "</div>" +
      "<ol class='steps'>" + items + "</ol>";
  } catch (err) {
    report.textContent = String(err.message || err);
  } finally {
    go.disabled = false;
  }
}

document.getElementById("go").addEventListener("click", run);
loadZones().catch((err) => { report.textContent = String(err); });
draw(0);
</script>
</body>
</html>
"""


LOCK = threading.Lock()
MODEL = None
EMA = None
PROBE = None


def png_b64(frame: np.ndarray) -> str:
    a = ((np.clip(frame, -1, 1) + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(a, "L").resize((VIEW, VIEW), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_phrase(text: str, seed: int, cost: str) -> dict:
    zone = find_zone(text)
    if zone is None:
        names = ", ".join(z["name"] for z in ZONES)
        raise ValueError("Не понял цель. Например: «мяч в верхний левый угол». Зоны: " + names)
    weight = 1.0 if cost == "mixed" else 0.0
    if weight and PROBE is None:
        raise ValueError("Для mixed нужен out/probe_v4.pt — сначала python probe.py --tag v4")
    with LOCK:
        with torch.no_grad():
            res = execute(
                MODEL, EMA, zone, seed=seed, horizon=10, max_steps=20,
                cem_kw={"samples": 128, "iters": 8, "seed": seed},
                probe=PROBE, probe_weight=weight, verbose=False,
                save_gif=False, keep_frames=True, shadow=False,
            )
            frames = res.pop("frames")
            report = describe_state(MODEL, PROBE, frames[-2:], res["final_pos"])
    think_ms = int(round(sum(s["plan_time"] for s in res["steps"]) * 1000))
    status = "цель достигнута" if res["success"] else "промах"
    xy = res["final_pos"]
    return {
        "zone": zone,
        "zone_name": res["zone_name"],
        "center": list(ZONE_CENTERS[zone]),
        "half": ZONE_HALF,
        "success": res["success"],
        "steps_used": res["steps_used"],
        "miss_px": res["miss_px"],
        "final_pos": xy,
        "think_ms": think_ms,
        "report": f"мяч в ({xy[0]:.0f}, {xy[1]:.0f}), {status}. {report}",
        "steps": res["steps"],
        "frames": [png_b64(fr) for fr in frames],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[ui]", fmt % args)

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/zones":
            payload = [{"i": i, "name": z["name"], "center": list(z["center"])} for i, z in enumerate(ZONES)]
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/run":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
            text = str(data.get("text", "")).strip()
            seed = int(data.get("seed", 1))
            cost = data.get("cost", "latent")
            if cost not in ("latent", "mixed"):
                raise ValueError("cost: latent или mixed")
            if not text:
                raise ValueError("пустая фраза")
            payload = run_phrase(text, seed, cost)
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200, raw, "application/json; charset=utf-8")
        except ValueError as exc:
            raw = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(400, raw, "application/json; charset=utf-8")
        except Exception as exc:
            print("[ui] ошибка прогона:", exc)
            raw = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(500, raw, "application/json; charset=utf-8")


def main():
    global MODEL, EMA, PROBE
    ap = argparse.ArgumentParser(description="Страница проверки планирования")
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    path = os.path.join(OUT, f"jepa_{args.tag}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"нет чекпоинта {path}. Сначала: python train.py --mode jepa --steps 2500 --tag {args.tag}")
    print(f"гружу {path} …")
    MODEL, EMA, _ = load_model(path, threads=args.threads)
    PROBE = load_probe(args.tag)
    if PROBE is None:
        print("probe не найден — отчёт без слов о скорости, mixed недоступен")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"открыть http://127.0.0.1:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

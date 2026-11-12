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
from data import (
    ACTIONS, ACTION_NAMES, GRAVITY, IMPULSE, SLIP, ZONE_CENTERS, ZONE_HALF, ZONES,
    blocked_zone_names, bump_centers, find_zone, gravity_of,
)
from models import load_model, world_of
from plan import load_probe, execute
from numeric.plan import execute as execute_numeric
from numeric_dt.plan import execute as execute_dt

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
  .caption { color: #d9d3c5; font-size: 13px; line-height: 1.4; margin: 10px 2px 0; }
  .note { color: var(--muted); font-size: 13px; margin-top: 14px; }
  .knobs { margin-bottom: 4px; }
  .knobs input[type="range"] {
    width: 100%;
    margin: 0 0 12px;
    padding: 0;
    border: none;
    background: transparent;
  }
  .knob-note { font-size: 13px; color: var(--muted); margin: 0 0 8px; line-height: 1.35; }
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
      <p class="caption" id="caption">Вид сбоку, не сверху: верх картинки — верх. Мяч тянет вниз слабая тяжесть, стены его отталкивают. Кадр — тот, который получает модель. Жёлтая стрелка — выбранный толчок в масштабе поля, без подмены пробуксовкой. Прямой короче косого, «на месте» стрелки не даёт. Тяжесть слабее прямого толчка примерно в 27 раз, поэтому отдельной стрелкой в том же масштабе её не разглядеть.</p>
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
      <label for="model">Модель</label>
      <select id="model">
        <option value="pixel" selected>пиксельная</option>
        <option value="numeric">числовая</option>
        <option value="numeric-dt">с шагом dt</option>
      </select>
      <p class="knob-note" id="model-note" hidden>Числовая JEPA, сид обучения 0, только плоский мир. В сеть идут четыре числа, не кадр. Поиск мерит положение мяча по зонду. Это не замер пиксельной v4.</p>
      <div id="dt-row" hidden>
        <label for="dt">Шаг dt</label>
        <select id="dt">
          <option value="1/60">1/60 с — экстраполяция</option>
          <option value="1/30" selected>1/30 с — внутри обучения</option>
          <option value="1/15">1/15 с — экстраполяция</option>
        </select>
      </div>
      <label for="world">Мир</label>
      <select id="world">
        <option value="flat" selected>плоский</option>
        <option value="jitter">пробуксовка</option>
        <option value="noise">помехи на кадре</option>
        <option value="balls">два мяча</option>
        <option value="floor">бугры</option>
      </select>
      <div class="knobs" id="floor-knobs" hidden>
        <label for="bump-h">Высота холмов <span id="bump-h-val">4</span> px</label>
        <input id="bump-h" type="range" min="0" max="40" step="1" value="4">
        <label for="bump-n">Число холмов <span id="bump-n-val">2</span></label>
        <input id="bump-n" type="range" min="0" max="5" step="1" value="2">
        <label for="g-steps">Тяжесть <span id="g-val">0.15</span></label>
        <input id="g-steps" type="range" min="0" max="80" step="1" value="3">
        <p class="knob-note" id="centers-line">Холмы в x = 20, 44.</p>
        <p class="knob-note" id="train-warn" hidden></p>
        <p class="knob-note" id="zone-warn" hidden></p>
      </div>
      <div class="row">
        <div>
          <label for="seed">Сид прогона</label>
          <input id="seed" type="text" value="" readonly>
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
      <p class="note" id="run-note">Каждое нажатие берёт новый сид, поэтому та же фраза идёт другим путём. Потолок 20 реальных шагов, поиск 128×8, горизонт 10. На плоском мире latent попадает примерно в 8 случаях из 10. Промах — нормальный исход, не ошибка страницы.</p>
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
let marks = [];
let zone = null;
let outcome = null;
let timer = null;
let playing = true;
const SCALE = 6;
const ZONE_TABLE = [
  ["верхний левый", 14, 14],
  ["верхний центр", 32, 14],
  ["верхний правый", 50, 14],
  ["средний левый", 14, 32],
  ["центр", 32, 32],
  ["средний правый", 50, 32],
  ["нижний левый", 14, 50],
  ["нижний центр", 32, 50],
  ["нижний правый", 50, 50]
];
const CAPTION_PIXEL = "Вид сбоку, не сверху: верх картинки — верх. Мяч тянет вниз слабая тяжесть, стены его отталкивают. Кадр — тот, который получает модель. Жёлтая стрелка — выбранный толчок в масштабе поля, без подмены пробуксовкой. Прямой короче косого, «на месте» стрелки не даёт. ";
const CAPTION_NUMERIC = "Вид сбоку, не сверху: верх картинки — верх. Мяч тянет вниз слабая тяжесть, стены его отталкивают. Кадр для вас: числовая модель получает четыре числа, не картинку. Жёлтая стрелка — выбранный толчок в масштабе поля. Прямой короче косого, «на месте» стрелки не даёт. ";
const CAPTION_DT = "Вид сбоку, не сверху: верх картинки — верх. Мяч тянет вниз тяжесть 9 px/с², толчок — 240 px/с². Кадр для вас: модель получает пять чисел, положение, скорость в пикселях в секунду и dt, не картинку. Жёлтая стрелка — направление толчка, не путь за этот dt. ";
const NOTE_PIXEL = "Каждое нажатие берёт новый сид, поэтому та же фраза идёт другим путём. Потолок 20 реальных шагов, поиск 128×8, горизонт 10. На плоском мире latent попадает примерно в 8 случаях из 10. Промах — нормальный исход, не ошибка страницы.";
const NOTE_DT = "Каждое нажатие берёт новый сид, поэтому та же фраза идёт другим путём. Потолок — 20 шагов выбранного dt, не одна секунда. Поиск 128×8, горизонт 10. Это показ руки B, не замер. Промах — нормальный исход, не ошибка страницы.";
const NOTE_NUMERIC = "Числовая JEPA, сид обучения 0, только плоский мир. В сеть идут четыре числа, не кадр. Поиск мерит положение мяча по зонду. Это не замер пиксельной v4.";
const NOTE_DT_MODEL = "JEPA с шагом dt, рука B, сид обучения 0. В сеть идут пять чисел, скорость в пикселях в секунду. Физика — новый интегратор, не шаг пиксельного кадра. Поиск мерит положение по зонду. Это не замер.";

function centersOf(n) {
  if (n === 0) return [];
  if (n === 1) return [32];
  if (n === 2) return [20, 44];
  const out = [];
  for (let i = 0; i < n; i++) out.push(8 + i * 48 / (n - 1));
  return out;
}

function floorLimitAt(x, h, centers) {
  let best = 0;
  for (const cx of centers) {
    const d = Math.abs(x - cx);
    if (d < 8) best = Math.max(best, h * (1 - d / 8));
  }
  return 58 - best;
}

function blockedNames(h, centers) {
  const names = [];
  for (const row of ZONE_TABLE) {
    let margin = Infinity;
    for (let i = 0; i < 65; i++) {
      const x = row[1] - 8 + (16 * i) / 64;
      margin = Math.min(margin, floorLimitAt(x, h, centers) - row[2]);
    }
    if (margin < 2) names.push(row[0]);
  }
  return names;
}

function gravitySentence(steps) {
  if (steps === 0) return "Вниз не тянет.";
  if (steps === 80) return "Тяжесть не слабее прямого толчка.";
  const n = Math.round(4 / (steps * 0.05));
  let text = "Тяжесть слабее прямого толчка примерно в " + n + " раз";
  if (steps === 3) text += ", поэтому отдельной стрелкой в том же масштабе её не разглядеть";
  return text + ".";
}

function syncModel() {
  const mode = document.getElementById("model").value;
  const numeric = mode === "numeric" || mode === "numeric-dt";
  const world = document.getElementById("world");
  for (const opt of world.options) opt.disabled = numeric && opt.value !== "flat";
  if (numeric && world.value !== "flat") world.value = "flat";
  document.getElementById("cost").disabled = numeric;
  const note = document.getElementById("model-note");
  note.hidden = mode === "pixel";
  note.textContent = mode === "numeric-dt" ? NOTE_DT_MODEL : NOTE_NUMERIC;
  document.getElementById("dt-row").hidden = mode !== "numeric-dt";
  document.getElementById("run-note").textContent = mode === "numeric-dt" ? NOTE_DT : NOTE_PIXEL;
  syncKnobs();
}

function dtSentence() {
  const v = document.getElementById("dt").value;
  if (v === "1/60") return "Сейчас dt = 1/60 с: это экстраполяция, 0.5× от 1/30.";
  if (v === "1/15") return "Сейчас dt = 1/15 с: это экстраполяция, 2× от 1/30.";
  return "Сейчас dt = 1/30 с: внутри отрезка, на котором училась рука B.";
}

function syncKnobs() {
  const mode = document.getElementById("model").value;
  const numeric = mode === "numeric";
  const floor = document.getElementById("world").value === "floor";
  document.getElementById("floor-knobs").hidden = !floor;
  const h = Number(document.getElementById("bump-h").value);
  const n = Number(document.getElementById("bump-n").value);
  const gs = Number(document.getElementById("g-steps").value);
  document.getElementById("bump-h-val").textContent = String(h);
  document.getElementById("bump-n-val").textContent = String(n);
  document.getElementById("g-val").textContent = (gs * 0.05).toFixed(2);
  const centers = centersOf(n);
  document.getElementById("centers-line").textContent = n === 0
    ? "Холмов нет, пол ровный."
    : "Холмы в x = " + centers.join(", ") + ".";
  const trained = h === 4 && n === 2 && gs === 3;
  const trainWarn = document.getElementById("train-warn");
  trainWarn.hidden = !floor || trained;
  trainWarn.textContent = "Модель училась на двух холмах высотой 4 и тяжести 0.15. Этот прогон не является замером v5-floor.";
  const zones = blockedNames(h, centers);
  const zoneWarn = document.getElementById("zone-warn");
  zoneWarn.hidden = !floor || zones.length === 0;
  zoneWarn.textContent = zones.length ? "Холм заходит в зоны: " + zones.join(", ") + "." : "";
  const head = mode === "numeric-dt" ? CAPTION_DT : (numeric ? CAPTION_NUMERIC : CAPTION_PIXEL);
  const tail = mode === "numeric-dt" ? dtSentence() : gravitySentence(floor ? gs : 3);
  document.getElementById("caption").textContent = head + tail;
}

function drawArrow(x, y, dx, dy) {
  const len = Math.hypot(dx, dy);
  if (len < 2) return;
  const tipX = x + dx;
  const tipY = y + dy;
  const ang = Math.atan2(dy, dx);
  const head = Math.min(11, len * 0.38);
  ctx.strokeStyle = "#f2c14e";
  ctx.fillStyle = "#f2c14e";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(tipX - Math.cos(ang) * head * 0.6, tipY - Math.sin(ang) * head * 0.6);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - head * Math.cos(ang - 0.45), tipY - head * Math.sin(ang - 0.45));
  ctx.lineTo(tipX - head * Math.cos(ang + 0.45), tipY - head * Math.sin(ang + 0.45));
  ctx.closePath();
  ctx.fill();
}

function drawTrajectory(color) {
  if (marks.length < 2) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  marks.forEach((m, idx) => {
    const x = m.x * SCALE;
    const y = m.y * SCALE;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  marks.forEach((m) => {
    ctx.beginPath();
    ctx.arc(m.x * SCALE, m.y * SCALE, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawPushVectors() {
  marks.forEach((m) => {
    drawArrow(m.x * SCALE, m.y * SCALE, m.push[0] * SCALE, m.push[1] * SCALE);
  });
}

function draw(i) {
  const last = frames.length ? frames.length - 1 : 0;
  const atEnd = frames.length > 0 && i === last;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, 384, 384);
  if (frames[i]) ctx.drawImage(frames[i], 0, 0);
  if (zone) {
    const half = 8;
    ctx.strokeStyle = atEnd && outcome === "ok" ? "#4da3ff"
      : atEnd && outcome === "bad" ? "#ff5a5a"
      : "#3dd68c";
    ctx.lineWidth = 3;
    ctx.strokeRect((zone[0] - half) * SCALE, (zone[1] - half) * SCALE, half * 2 * SCALE, half * 2 * SCALE);
  }
  if (atEnd && outcome) {
    drawTrajectory(outcome === "ok" ? "#4da3ff" : "#ff5a5a");
    drawPushVectors();
  }
  const m = marks[i];
  if (m && !(atEnd && outcome)) drawArrow(m.x * SCALE, m.y * SCALE, m.push[0] * SCALE, m.push[1] * SCALE);
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
      if (i >= frames.length) {
        playing = false;
        toggle.textContent = "играть";
        draw(frames.length - 1);
        return;
      }
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
  if (playing && frames.length && Number(scrub.value) >= frames.length - 1) draw(0);
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

function freshSeed() {
  const el = document.getElementById("seed");
  const prev = el.value;
  let n;
  do {
    n = Math.floor(Math.random() * 1000000000);
  } while (String(n) === prev);
  el.value = String(n);
  return n;
}

async function run() {
  const text = document.getElementById("phrase").value.trim();
  const seed = freshSeed();
  go.disabled = true;
  report.textContent = "Перебирает толчки в латенте…";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text,
        seed,
        cost: document.getElementById("cost").value,
        model: document.getElementById("model").value,
        dt: document.getElementById("dt").value,
        world: document.getElementById("world").value,
        bump_h: Number(document.getElementById("bump-h").value),
        bump_n: Number(document.getElementById("bump-n").value),
        g_steps: Number(document.getElementById("g-steps").value)
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
    marks = data.marks || [];
    outcome = data.success ? "ok" : "bad";
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
      "<div>фраза → зона «" + esc(data.zone_name) + "» (" + data.center[0] + ", " + data.center[1] + "), сид " + esc(data.seed) + "</div>" +
      "<div><strong class='" + klass + "'>" + head + "</strong> за " + data.steps_used +
      " шагов, до центра " + data.miss_px + " px. Поиск в сумме " + data.think_ms + " мс.</div>" +
      "<div>" + esc(data.report) + "</div>" +
      "<ol class='steps'>" + items + "</ol>";
  } catch (err) {
    const msg = String(err.message || err);
    report.textContent = msg === "Failed to fetch"
      ? "Программа расчёта не запущена. В каталоге лаборатории выполните python demo_ui.py и обновите страницу."
      : msg;
  } finally {
    go.disabled = false;
  }
}

document.getElementById("go").addEventListener("click", run);
document.getElementById("model").addEventListener("change", syncModel);
document.getElementById("dt").addEventListener("change", syncKnobs);
document.getElementById("world").addEventListener("change", syncKnobs);
["bump-h", "bump-n", "g-steps"].forEach((id) => {
  document.getElementById(id).addEventListener("input", syncKnobs);
});
syncModel();
loadZones().catch((err) => { report.textContent = String(err); });
draw(0);
</script>
</body>
</html>
"""


LOCK = threading.Lock()
THREADS = 8
CACHE = {}
TAGS = {
    "flat": "v4",
    "jitter": "v5-jitter",
    "noise": "v5-noise",
    "balls": "v5-balls",
    "floor": "v5-floor",
}


PUSH = {name: [float(ACTIONS[i][0]), float(ACTIONS[i][1])] for i, name in enumerate(ACTION_NAMES)}


def blob_center(frame: np.ndarray) -> list:
    """Центр светлого пятна на кадре 64×64. Ось y вниз, как на картинке."""
    w = (np.clip(frame, -1, 1) + 1) / 2
    total = float(w.sum())
    if total < 1e-6:
        return [32.0, 32.0]
    ys, xs = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    return [round(float((w * xs).sum() / total), 2), round(float((w * ys).sum() / total), 2)]


def frame_marks(frames, steps, ball=None) -> list:
    """Толчок, который привёл к этому кадру. Длина — номинальный импульс, не подмена.

    Точки — управляемый мяч. Центр пятна на кадре с двумя мячами или с полом врёт.
    """
    if ball is None:
        centers = list(map(blob_center, frames))
    else:
        centers = [[float(p[0]), float(p[1])] for p in ball]
    pushes = [[0.0, 0.0]]
    if len(frames) > 1:
        pushes.append([0.0, 0.0])
    for step in steps:
        pushes.append(PUSH[step["action"]])
    pushes = pushes[:len(centers)]
    while len(pushes) < len(centers):
        pushes.append([0.0, 0.0])
    return [{"x": c[0], "y": c[1], "push": p} for c, p in zip(centers, pushes)]


def bundle(world: str):
    if world not in TAGS:
        raise ValueError("неизвестный мир")
    if world not in CACHE:
        tag = TAGS[world]
        path = os.path.join(OUT, f"jepa_{tag}.pt")
        if not os.path.exists(path):
            raise ValueError(
                f"нет чекпоинта для мира «{world}». Сначала: "
                f"python train.py --mode jepa --steps 2500 --tag {tag} --world {world}")
        model, ema, ck = load_model(path, threads=THREADS)
        got = world_of(ck)
        if got != world:
            raise ValueError(f"чекпоинт {tag} обучен в мире «{got}», запрошен «{world}»")
        CACHE[world] = (model, ema, load_probe(tag), tag, float(ck.get("slip") or SLIP))
    return CACHE[world]


def png_b64(frame: np.ndarray) -> str:
    a = ((np.clip(frame, -1, 1) + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(a, "L").resize((VIEW, VIEW), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _int_field(data: dict, key: str, default: int, lo: int, hi: int) -> int:
    if key not in data or data[key] is None:
        return default
    val = data[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{key}: целое от {lo} до {hi}")
    if val < lo or val > hi:
        raise ValueError(f"{key}: целое от {lo} до {hi}")
    return val


def floor_request(data: dict) -> dict:
    """Ползунки только для мира бугров. Нет поля — обученные 4, 2 и тяжесть 0.15."""
    bump_h = _int_field(data, "bump_h", 4, 0, 40)
    bump_n = _int_field(data, "bump_n", 2, 0, 5)
    g_steps = _int_field(data, "g_steps", 3, 0, 80)
    centers = bump_centers(bump_n)
    return {
        "bump_h": bump_h,
        "bump_n": bump_n,
        "centers": centers,
        "g_steps": g_steps,
        "gravity": gravity_of(g_steps),
        "blocked": blocked_zone_names(float(bump_h), centers),
    }


def _dt_report(res: dict) -> str:
    status = "цель достигнута" if res["success"] else "промах"
    xy = res["final_pos"]
    extra = ""
    if res["dt"] == "1/60":
        extra = " Это экстраполяция: 0.5× от 1/30."
    elif res["dt"] == "1/15":
        extra = " Это экстраполяция: 2× от 1/30."
    return (
        f"мяч в ({xy[0]:.0f}, {xy[1]:.0f}), {status}. "
        "JEPA с шагом dt, рука B, сид обучения 0. "
        f"dt = {res['dt']} с. Поиск ведёт мяч по положению из зонда. Это не замер."
        + extra
    )


def run_phrase(text: str, seed: int, cost: str, world: str, knobs=None,
               model: str = "pixel", dt_key: str = "1/30") -> dict:
    zone = find_zone(text)
    if zone is None:
        names = ", ".join(z["name"] for z in ZONES)
        raise ValueError("Не понял цель. Например: «мяч в верхний левый угол». Зоны: " + names)
    if model == "numeric":
        if world != "flat":
            raise ValueError("Числовая модель училась только на плоском мире.")
        with LOCK:
            with torch.no_grad():
                res = execute_numeric(zone, seed=seed, horizon=10, max_steps=20)
                frames = res.pop("frames")
                ball = res.pop("ball", None)
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
            "report": (
                f"мяч в ({xy[0]:.0f}, {xy[1]:.0f}), {status}. "
                "Числовая JEPA, сид обучения 0. Поиск ведёт мяч по положению из зонда."
            ),
            "steps": res["steps"],
            "marks": frame_marks(frames, res["steps"], ball),
            "world": world,
            "model": "numeric",
            "seed": seed,
            "train_seed": 0,
            "gravity_down": GRAVITY,
            "impulse": IMPULSE,
            "frames": [png_b64(fr) for fr in frames],
        }
    if model == "numeric-dt":
        if world != "flat":
            raise ValueError("Модель с шагом dt не ведёт пиксельные миры.")
        with LOCK:
            with torch.no_grad():
                res = execute_dt(zone, seed=seed, dt_key=dt_key, horizon=10, max_steps=20)
                frames = res.pop("frames")
                ball = res.pop("ball", None)
        think_ms = int(round(sum(s["plan_time"] for s in res["steps"]) * 1000))
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
            "report": _dt_report(res),
            "steps": res["steps"],
            "marks": frame_marks(frames, res["steps"], ball),
            "world": world,
            "model": "numeric-dt",
            "seed": seed,
            "train_seed": 0,
            "arm": "B",
            "dt": res["dt"],
            "frames": [png_b64(fr) for fr in frames],
        }
    weight = 1.0 if cost == "mixed" else 0.0
    model, ema, probe, tag, slip = bundle(world)
    if weight and probe is None:
        raise ValueError(f"Для mixed нужен out/probe_{tag}.pt — сначала python probe.py --tag {tag}")
    floor = {}
    if world == "floor":
        floor = knobs or floor_request({})
    with LOCK:
        with torch.no_grad():
            res = execute(
                model, ema, zone, seed=seed, horizon=10, max_steps=20,
                cem_kw={"samples": 128, "iters": 8, "seed": seed},
                probe=probe, probe_weight=weight, verbose=False,
                save_gif=False, keep_frames=True, shadow=False,
                world=world, slip=slip,
                bump_h=float(floor.get("bump_h", 4)),
                centers=floor.get("centers"),
                gravity=float(floor.get("gravity", GRAVITY)),
            )
            frames = res.pop("frames")
            ball = res.pop("ball", None)
            report = describe_state(model, probe, frames[-2:], res["final_pos"])
    think_ms = int(round(sum(s["plan_time"] for s in res["steps"]) * 1000))
    status = "цель достигнута" if res["success"] else "промах"
    xy = res["final_pos"]
    out = {
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
        "marks": frame_marks(frames, res["steps"], ball),
        "world": world,
        "gravity_down": GRAVITY,
        "impulse": IMPULSE,
        "model": "pixel",
        "seed": seed,
        "frames": [png_b64(fr) for fr in frames],
    }
    if floor:
        out["bump_h"] = floor["bump_h"]
        out["bump_n"] = floor["bump_n"]
        out["centers"] = [int(c) if float(c).is_integer() else round(float(c), 2) for c in floor["centers"]]
        out["g_steps"] = floor["g_steps"]
        out["gravity"] = 0.15 if floor["g_steps"] == 3 else round(floor["g_steps"] * 0.05, 2)
        out["blocked"] = floor["blocked"]
        out["gravity_down"] = out["gravity"]
    return out


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
            world = data.get("world", "flat")
            model_name = data.get("model", "pixel")
            if model_name not in ("pixel", "numeric", "numeric-dt"):
                raise ValueError("model: pixel, numeric или numeric-dt")
            dt_key = data.get("dt", "1/30")
            if model_name == "numeric-dt" and dt_key not in ("1/60", "1/30", "1/15"):
                raise ValueError("dt: 1/60, 1/30 или 1/15")
            if cost not in ("latent", "mixed"):
                raise ValueError("cost: latent или mixed")
            if world not in TAGS:
                raise ValueError("world: flat, jitter, noise, balls или floor")
            if not text:
                raise ValueError("пустая фраза")
            if model_name == "numeric" and world != "flat":
                raise ValueError("Числовая модель училась только на плоском мире.")
            if model_name == "numeric-dt" and world != "flat":
                raise ValueError("Модель с шагом dt не ведёт пиксельные миры.")
            knobs = floor_request(data) if world == "floor" else None
            payload = run_phrase(text, seed, cost, world, knobs, model_name, dt_key)
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
    global THREADS
    ap = argparse.ArgumentParser(description="Страница проверки планирования")
    ap.add_argument("--tag", default="v4", help="тег плоского мира; остальные миры берутся по своим тегам v5")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    THREADS = args.threads
    if args.tag != "v4":
        TAGS["flat"] = args.tag
    print(f"гружу плоский мир, тег {TAGS['flat']} …")
    bundle("flat")
    if CACHE["flat"][2] is None:
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

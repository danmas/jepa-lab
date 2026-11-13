"""Страница большого поля. План — сеть landf_range_vis_s0. Правило в файле — не сеть.

    python _land_demo.py
    открыть http://127.0.0.1:8766

Физика — step_field. demo_ui.py эта страница не меняет.
"""
import argparse
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from numeric_land.data import MAX_THRUST
from numeric_land.field import (
    EPISODE_CAP, FIELD_W, FUEL0, GROUND, PAD_CENTER, outcome as outcome_of,
    start_and_actions, step_field as step_raw,
)

HORIZON = 40  # дефолт горизонта плана; переопределяется из интерфейса

EPISODE = 450


def apply_step(state, action, g):
    """Обёртка словаря над интегратором пакета."""
    x, y, vx, vy, fuel, grounded = step_raw(
        state["x"], state["y"], state["vx"], state["vy"], state["fuel"],
        bool(state["grounded"]), int(action), g)
    out = dict(state)
    out.update(x=x, y=y, vx=vx, vy=vy, fuel=fuel, grounded=grounded, action=int(action))
    return out


def start_of(seed, g, start_xy=None):
    """Старт от сида (розыгрыш спеки) либо по выбранной кликом точке (vx=vy=0)."""
    if start_xy is not None:
        x, y = float(start_xy[0]), float(start_xy[1])
        vx, vy = 0.0, 0.0
    else:
        x, y, vx, vy, _actions = start_and_actions(int(seed), 0)
    return {
        "x": x, "y": y, "vx": vx, "vy": vy,
        "fuel": FUEL0, "grounded": False, "action": 1, "g": float(g),
    }


def choose_action(state, g):
    """Довести x к середине площадки и включить тягу, если удар уже не погасить."""
    if state["grounded"]:
        return 1
    x, y, vx, vy = state["x"], state["y"], state["vx"], state["vy"]
    height = max(0.0, GROUND - y)
    aim = max(-8.0, min(8.0, (PAD_CENTER - x) * 1.5))
    if vx < aim - 1.2:
        ix = 2
    elif vx > aim + 1.2:
        ix = 0
    else:
        ix = 1
    iy = 0
    if state["fuel"] > 0.0 and height > 0.0:
        pred = vy * vy + 2.0 * (float(g) - MAX_THRUST) * height
        if pred > 36.0:
            iy = 2
        elif pred > 16.0 and height < 5.0:
            iy = 1
    return iy * 3 + ix


def outcome(state):
    return outcome_of(state["x"], state["y"], state["vy"], state["grounded"])


# --- живой показ: один шаг плана за запрос -----------------------------

_LIVE = {}
_LIVE_SEQ = itertools.count(1)


def _start_live(seed, g, horizon=HORIZON, start_xy=None):
    if start_xy is not None:
        x, y = float(start_xy[0]), float(start_xy[1])
        vx, vy = 0.0, 0.0
    else:
        x, y, vx, vy, _actions = start_and_actions(int(seed), 0)
    run_id = f"run{next(_LIVE_SEQ)}"
    _LIVE[run_id] = {
        "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy),
        "fuel": FUEL0, "grounded": False, "g": float(g), "seed": int(seed),
        "horizon": int(horizon), "t": 0, "frames": [],
    }
    ep = _live_episode(run_id)
    return run_id, ep


def _live_episode(run_id):
    """Собрать ответ из текущего состояния сессии."""
    s = _LIVE[run_id]
    last = s["frames"][-1] if s["frames"] else _public(s)
    result = None
    if s["grounded"]:
        result = outcome_of(s["x"], s["y"], s["vy"], True)
    elif s["t"] >= EPISODE_CAP:
        result = "не сел"
    return {
        "g": s["g"], "seed": s["seed"], "mode": "plan-live",
        "horizon": s["horizon"], "result": result, "steps": s["t"],
        "frame": last, "frames": s["frames"],
    }


def _live_plan_step(run_id):
    """Один шаг CEM: план сети -> первое действие -> интегратор."""
    s = _LIVE[run_id]
    if s["grounded"] or s["t"] >= EPISODE_CAP:
        return _live_episode(run_id)
    from numeric_land.plan_field import bundle, cem_plan
    encoder, predictor, probe = bundle()
    seq, _cost, _elapsed = cem_plan(
        encoder, predictor, probe, s["x"], s["y"], s["vx"], s["vy"], s["fuel"],
        s["g"], horizon=int(s["horizon"]), seed=int(s["seed"]) + s["t"])
    action = int(seq[0])
    nxt = _public(apply_step(s, action, s["g"]))
    s["x"], s["y"], s["vx"], s["vy"], s["fuel"], s["grounded"] = (
        nxt["x"], nxt["y"], nxt["vx"], nxt["vy"], nxt["fuel"], nxt["grounded"])
    s["t"] += 1
    s["frames"].append({**nxt, "action": action})
    if len(s["frames"]) > 1200:
        del s["frames"][:-1200]
    return _live_episode(run_id)
def rollout(seed, g, mode, horizon=HORIZON, start_xy=None):
    if mode == "plan":
        from numeric_land.plan_field import run_episode
        episode = run_episode(int(seed), float(g), horizon=int(horizon),
                              start_xy=start_xy)
        episode["mode"] = "plan"
        episode["horizon"] = int(horizon)
        return episode
    state = start_of(seed, g, start_xy=start_xy)
    frames = [_public(state)]
    for _ in range(EPISODE):
        if state["grounded"]:
            break
        if mode == "fall":
            action = 1
        else:
            action = choose_action(state, g)
        state = apply_step(state, action, g)
        frames.append(_public(state))
    return {"g": float(g), "seed": int(seed), "mode": mode,
            "result": outcome(state), "frames": frames}


def _public(state):
    return {
        "x": round(state["x"], 4),
        "y": round(state["y"], 4),
        "vx": round(state["vx"], 4),
        "vy": round(state["vy"], 4),
        "fuel": round(state["fuel"], 4),
        "grounded": bool(state["grounded"]),
        "action": int(state.get("action", 1)),
    }


def check_spec():
    """Контрольные дроби спеки. Страница их не показывает."""
    def close(got, exp, tol=1e-9):
        if abs(float(got) - float(exp)) > tol:
            raise SystemExit(f"{got} != {exp}")

    base = {"x": 256.0, "y": 256.0, "vx": 0.0, "vy": 0.0, "fuel": 10.0, "grounded": False}
    free = apply_step(dict(base), 1, 9)
    close(free["vy"], 0.3)
    close(free["y"] - 256.0, 0.01)
    close(free["fuel"], 10.0)
    slow = apply_step(dict(base), 1, 2)
    fast = apply_step(dict(base), 1, 15)
    close(fast["vy"] - slow["vy"], 13.0 / 30.0)
    close(fast["y"] - slow["y"], 13.0 / 900.0)
    touch = apply_step(
        {"x": 256.0, "y": 25599.0 / 50.0, "vx": 0.0, "vy": 1.0, "fuel": 10.0, "grounded": False},
        1, 9)
    close(touch["y"], 512.0)
    close(touch["vy"], 1.3)
    close(touch["vx"], 0.0)
    frozen = apply_step(touch, 8, 9)
    close(frozen["y"], 512.0)
    close(frozen["vy"], 1.3)
    side = apply_step(
        {"x": -1.0, "y": 400.0, "vx": -1.0, "vy": 0.0, "fuel": 10.0, "grounded": False},
        1, 9)
    close(side["x"], -31.0 / 30.0)
    close(side["vx"], -1.0)
    if side["y"] == 512.0:
        raise SystemExit("открытый x назван землёй")
    empty = apply_step(
        {"x": 256.0, "y": 400.0, "vx": 0.0, "vy": 0.0, "fuel": 0.0, "grounded": False},
        8, 9)
    close(empty["vx"], 0.0)
    close(empty["vy"], 0.3)
    full = apply_step(dict(base), 8, 9)
    close(full["vx"], 0.4)
    close(full["vy"], -11.0 / 30.0)
    close(full["fuel"], 9.98)
    print("land demo physics: ок")


PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jepa-lab — посадка, большое поле</title>
<style>
  :root {
    --bg: #f3efe6; --ink: #1c1915; --muted: #5e584e;
    --line: #ddd4c4; --card: #fffdf8; --ok: #1d6b45; --bad: #8d2f2f;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.45 "Segoe UI", sans-serif; }
  main { max-width: 1120px; margin: 0 auto; padding: 28px 20px 48px; }
  h1 { font-size: 28px; font-weight: 650; margin: 0 0 8px; letter-spacing: -0.03em; }
  .lead { color: var(--muted); max-width: 78ch; margin: 0 0 22px; }
  .bench { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 22px; align-items: start; }
  .stage { background: #111; border-radius: 12px; padding: 12px; min-width: 0; }
  canvas { display: block; width: 100%; height: auto; aspect-ratio: 960 / 496;
    background: #111; border-radius: 8px; }
  .play { display: flex; gap: 8px; align-items: center; margin-top: 10px;
    color: #d9d3c5; font-size: 13px; }
  .play input[type="range"] { flex: 1; }
  button, select, input[type="number"] {
    font: inherit; color: var(--ink); background: var(--card);
    border: 1px solid var(--line); border-radius: 8px; }
  .play button, .go, .actions button { padding: 8px 10px; cursor: pointer; }
  .panel { background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 16px; }
  label { display: block; font-size: 13px; color: var(--muted); margin: 0 0 6px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  input[type="number"], select { width: 100%; padding: 8px 10px; margin-bottom: 12px; }
  .go { width: 100%; background: #1c1915; color: #f3efe6; font-weight: 650; margin-bottom: 8px; }
  .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 8px 0 12px; }
  .report { white-space: pre-wrap; margin-top: 8px; }
  .report strong.ok { color: var(--ok); }
  .report strong.bad { color: var(--bad); }
  .note { color: var(--muted); font-size: 13px; margin-top: 12px; }
  .caption { color: #d9d3c5; font-size: 13px; line-height: 1.4; margin: 10px 2px 0; }
  @media (max-width: 860px) {
    .bench { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<main>
  <h1>Посадка, большое поле</h1>
  <p class="lead">Земля на y = 512, по x стены нет. В кадре не всё поле, а окно полёта: x от 160 до 352, y от 416 до 512, масштаб по осям один. Зелёная черта — площадка x от 248 до 264, высота черты 4 пикселя. «План сети» — CEM landf_range_vis_s0, горизонт 40, потолок 450. Один запуск может занять десятки секунд. «Правило» — короткое правило в коде, не сеть. «Падение» тягу не включает.</p>
  <div class="bench">
    <div class="stage">
      <canvas id="view" width="960" height="496"></canvas>
      <p class="caption" id="caption">Кадр появится после кнопки.</p>
      <div class="play">
        <button id="toggle" type="button">пауза</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <span id="frame-label">кадр 0</span>
      </div>
    </div>
    <div class="panel">
      <div class="row">
        <div>
          <label for="g">Тяжесть g, px/с²</label>
          <input id="g" type="number" min="2" max="20" step="1" value="8">
        </div>
        <div>
                  <label for="seed">Сид старта</label>
                  <input id="seed" type="number" min="0" max="999999" step="1" value="89000">
                </div>
              </div>
              <label for="horizon">Горизонт плана (шагов)</label>
                            <input id="horizon" type="number" min="1" max="450" step="1" value="40" title="На сколько шагов вперёд сеть смотрит, выбирая действие. Больше — дальше видит, но дольше считает.">
                    <label>Старт по клику на экране</label>
                    <div class="row">
                      <button class="go" id="clear-start" type="button">сбросить старт</button>
                      <span id="start-note" style="align-self:center;font-size:13px;color:var(--muted);">старт от сида (клик по канвасу — поставить точку)</span>
                    </div>
                    <button class="go" id="plan" type="button">план сети</button>
            <button class="go" id="live" type="button">живой план (по шагам)</button>
            <button class="go" id="example" type="button">правило в коде</button>
            <button class="go" id="fall" type="button">просто падать</button>
      <label>Ручной шаг, три на три</label>
      <div class="actions" id="actions"></div>
      <div class="report" id="report">Старт ещё не построен.</div>
      <p class="note">Мягкая посадка: в момент касания x внутри площадки и скорость вниз меньше 8. Бак начинается с 10. Горизонталь топливо не тратит. Пустой бак гасит и боковую тягу. После касания кадр замирает. Сиды 89000–89029 — старты замера плана. Если сеть ещё не обучена, кнопка плана напишет, что файла нет.</p>
    </div>
  </div>
</main>
<script>
const X0 = 160;
const Y0 = 416;
const GROUND = 512;
const S = 5;
const VW = (352 - X0) * S;
const GROUND_PAD = 16;
const VH = (GROUND - Y0) * S + GROUND_PAD;
const SHIP = 14;
const NAMES = [
  "влево", "пусто", "вправо",
  "влево + половина", "половина", "вправо + половина",
  "влево + полная", "полная", "вправо + полная"
];
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
let frames = [];
let cursor = 0;
let playing = false;
let timer = null;
let clickStart = null;   // стартовая точка {x, y} по клику, в мировых координатах (canvas)

function gValue() { return Number(document.getElementById("g").value); }
function seedValue() { return Number(document.getElementById("seed").value); }
function horizonValue() { return Number(document.getElementById("horizon").value); }

function wx(x) { return (x - X0) * S; }
function wy(y) { return (y - Y0) * S; }

// клик по канвасу -> стартовая точка (px -> мировые координаты, vx=vy=0 на сервере)
canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (VW / rect.width);
  const my = (e.clientY - rect.top) * (VH / rect.height);
  const x = Math.round((X0 + mx / S) * 2) / 2;
  const y = Math.round((Y0 + my / S) * 2) / 2;
  clickStart = { x, y };
  document.getElementById("start-note").textContent =
    "старт: x " + x + ", y " + y + " (клик — сменить, «сбросить» — к сиду)";
  if (frames.length) draw(frames[cursor], frames.slice(0, cursor + 1));
});

function clearStart() {
  clickStart = null;
  document.getElementById("start-note").textContent =
    "старт от сида (клик по канвасу — поставить точку)";
  if (frames.length) draw(frames[cursor], frames.slice(0, cursor + 1));
}

function startPayload() {
  if (clickStart) return {start_x: clickStart.x, start_y: clickStart.y};
  return {};
}

function draw(frame, trail) {
  ctx.clearRect(0, 0, VW, VH);
  ctx.fillStyle = "#14181f";
  ctx.fillRect(0, 0, VW, VH);
  ctx.strokeStyle = "#2a3140";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 160; x <= 352; x += 16) {
    ctx.moveTo(wx(x), 0);
    ctx.lineTo(wx(x), wy(GROUND));
  }
  for (let y = Y0; y <= GROUND; y += 16) {
    ctx.moveTo(0, wy(y));
    ctx.lineTo(VW, wy(y));
  }
  ctx.stroke();
  const groundY = wy(GROUND);
  ctx.strokeStyle = "#3c4a40";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, groundY);
  ctx.lineTo(VW, groundY);
  ctx.stroke();
  ctx.fillStyle = "#2f9e62";
  ctx.fillRect(wx(248), groundY - 4, 16 * S, 4);
  ctx.strokeStyle = "#8fd0a8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < trail.length; i++) {
    const p = trail[i];
    const cmd = i === 0 ? "moveTo" : "lineTo";
    ctx[cmd](wx(p.x), wy(p.y));
  }
  ctx.stroke();
  const x = wx(frame.x);
  const y = wy(frame.y);
  ctx.fillStyle = frame.grounded ? "#f2d48a" : "#f7f1e4";
  ctx.beginPath();
  ctx.moveTo(x, y - SHIP);
  ctx.lineTo(x - SHIP * 0.55, y);
  ctx.lineTo(x + SHIP * 0.55, y);
  ctx.closePath();
  ctx.fill();
  const a = frame.action | 0;
  const iy = Math.floor(a / 3);
  const ix = a % 3;
  const sx = ix - 1;
  const frac = [0, 0.5, 1][iy];
  if (frac > 0 || sx !== 0) {
    ctx.strokeStyle = "#e2b340";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - sx * SHIP, y + frac * SHIP);
    ctx.stroke();
  }
  if (clickStart) {
    const csx = wx(clickStart.x);
    const csy = wy(clickStart.y);
    ctx.strokeStyle = "#4da3ff";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.arc(csx, csy, 13, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#4da3ff";
    ctx.beginPath();
    ctx.arc(csx, csy, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}
function show(index) {
  if (!frames.length) return;
  cursor = Math.max(0, Math.min(frames.length - 1, index));
  document.getElementById("scrub").max = String(frames.length - 1);
  document.getElementById("scrub").value = String(cursor);
  document.getElementById("frame-label").textContent = "кадр " + cursor;
  draw(frames[cursor], frames.slice(0, cursor + 1));
  const f = frames[cursor];
  const result = document.getElementById("report").dataset.result || "";
  document.getElementById("caption").textContent =
    "x " + f.x.toFixed(2) + "  y " + f.y.toFixed(2) +
    "  vx " + f.vx.toFixed(2) + "  vy " + f.vy.toFixed(2) +
    "  топливо " + f.fuel.toFixed(2) +
    "  действие " + NAMES[f.action] +
    (f.grounded ? "  касание" : "");
}

function stop() {
  playing = false;
  if (timer) clearInterval(timer);
  timer = null;
  document.getElementById("toggle").textContent = "пуск";
}

function play() {
  if (frames.length < 2) return;
  playing = true;
  document.getElementById("toggle").textContent = "пауза";
  if (cursor >= frames.length - 1) cursor = 0;
  timer = setInterval(() => {
    if (cursor >= frames.length - 1) { stop(); return; }
    show(cursor + 1);
  }, 33);
}

function setReport(data) {
  const last = data.frames[data.frames.length - 1];
  const cls = data.result === "мягкая" ? "ok" : "bad";
  const box = document.getElementById("report");
  box.dataset.result = data.result;
  box.innerHTML = "<strong class='" + cls + "'>" + data.result + "</strong>\\n"
    + "шагов " + (data.frames.length - 1)
    + ", g " + data.g + ", сид " + data.seed + "\\n"
    + "финиш x " + last.x.toFixed(2) + ", vy " + last.vy.toFixed(2)
    + ", топливо " + last.fuel.toFixed(2);
}

function setBusy(mode, busy) {
  for (const id of ["plan", "live", "example", "fall"]) {
    const btn = document.getElementById(id);
    btn.disabled = busy;
  }
  const plan = document.getElementById("plan");
  if (busy) {
    plan.dataset.label = plan.textContent;
    plan.textContent = "план сети — считаю…";
  } else if (plan.dataset.label) {
    plan.textContent = plan.dataset.label;
    delete plan.dataset.label;
  }
}

let liveActive = null;

async function runLive() {
  stop();
  const reportBox = document.getElementById("report");
  setBusy("live", true);
  let runId = null;
  try {
    const start = await fetch("/api/live/start", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({g: gValue(), seed: seedValue(), horizon: horizonValue(), ...startPayload()})
        });
    const startData = await start.json();
    if (!start.ok) { reportBox.textContent = startData.error || "ошибка"; return; }
    runId = startData.run_id;
    frames = startData.frames.length ? startData.frames : [startData.frame];
    reportBox.textContent = "живой план — шаг 0";
    show(0);
    for (let t = 1; ; t++) {
      const res = await fetch("/api/live/step", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({run_id: runId})
      });
      const data = await res.json();
      if (!res.ok) { reportBox.textContent = data.error || "ошибка"; break; }
      frames = data.frames;
      reportBox.textContent = "живой план — шаг " + data.steps;
      show(frames.length - 1);
      if (data.result) {
        setReport({frames, result: data.result, g: data.g, seed: data.seed});
        break;
      }
    }
  } catch (e) {
    reportBox.textContent = "ошибка: " + e;
  } finally {
    liveActive = null;
    setBusy("live", false);
  }
}

async function run(mode) {
  stop();
  const reportBox = document.getElementById("report");
  if (mode === "plan") {
    reportBox.textContent = "Строю план сети… это может занять до минуты.";
    reportBox.innerHTML = "<span style='color:#1c1915;'>Строю план сети… это может занять до минуты.</span>";
  } else {
    reportBox.textContent = "Считаю…";
  }
  setBusy(mode, true);
  try {
    const res = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({g: gValue(), seed: seedValue(), mode, horizon: horizonValue(), ...startPayload()})
        });
    const data = await res.json();
    if (!res.ok) { reportBox.textContent = data.error || "ошибка"; return; }
    frames = data.frames;
    setReport(data);
    show(0);
    play();
  } finally {
    setBusy(mode, false);
  }
}

async function manual(action) {
  stop();
  const res = await fetch("/api/step", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({g: gValue(), seed: seedValue(), action, frames})
  });
  const data = await res.json();
  if (!res.ok) { document.getElementById("report").textContent = data.error || "ошибка"; return; }
  frames = data.frames;
  setReport(data);
  show(frames.length - 1);
}

document.getElementById("plan").onclick = () => run("plan");
document.getElementById("live").onclick = () => runLive();
document.getElementById("example").onclick = () => run("example");
document.getElementById("fall").onclick = () => run("fall");
document.getElementById("clear-start").onclick = () => clearStart();
document.getElementById("toggle").onclick = () => playing ? stop() : play();
document.getElementById("scrub").oninput = (e) => { stop(); show(Number(e.target.value)); };
const box = document.getElementById("actions");
NAMES.forEach((name, i) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = name;
  b.onclick = () => manual(i);
  box.appendChild(b);
});
run("plan");
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            path = self.path.split("?", 1)[0]
            start_xy = None
            if "start_x" in payload and "start_y" in payload:
                sx = float(payload["start_x"])
                sy = float(payload["start_y"])
                if not (0 <= sx <= FIELD_W and 0 < sy < GROUND):
                    raise ValueError("старт вне поля")
                start_xy = (sx, sy)
            if path == "/api/run":
                mode = payload.get("mode", "example")
                if mode not in ("example", "fall", "plan"):
                    raise ValueError("mode: plan, example или fall")
                g = float(payload.get("g", 8))
                if not 0 < g <= 40:
                    raise ValueError("g вне 0…40")
                horizon = int(payload.get("horizon", HORIZON))
                if not 1 <= horizon <= 450:
                    raise ValueError("горизонт вне 1…450")
                data = rollout(int(payload.get("seed", 0)), g, mode,
                               horizon=horizon, start_xy=start_xy)
            elif path == "/api/live/start":
                g = float(payload.get("g", 8))
                if not 0 < g <= 40:
                    raise ValueError("g вне 0…40")
                horizon = int(payload.get("horizon", HORIZON))
                if not 1 <= horizon <= 450:
                    raise ValueError("горизонт вне 1…450")
                run_id, data = _start_live(int(payload.get("seed", 0)), g,
                                           horizon=horizon, start_xy=start_xy)
                data["run_id"] = run_id
            elif path == "/api/live/step":
                run_id = str(payload.get("run_id", ""))
                if run_id not in _LIVE:
                    raise ValueError("нет такой сессии")
                data = _live_plan_step(run_id)
            elif path == "/api/step":
                data = _manual(payload)
            else:
                raise ValueError("неизвестный путь")
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        except (ValueError, KeyError, TypeError) as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(400, body, "application/json; charset=utf-8")


def _manual(payload):
    g = float(payload.get("g", 8))
    if not 0 < g <= 40:
        raise ValueError("g вне 0…40")
    action = int(payload["action"])
    frames = payload.get("frames") or []
    if not frames:
        state = start_of(int(payload.get("seed", 0)), g)
    else:
        last = frames[-1]
        state = {
            "x": float(last["x"]), "y": float(last["y"]),
            "vx": float(last["vx"]), "vy": float(last["vy"]),
            "fuel": float(last["fuel"]), "grounded": bool(last["grounded"]),
            "action": int(last.get("action", 1)),
        }
    if not frames:
        frames = [_public(state)]
    nxt = apply_step(state, action, g)
    frames = list(frames) + [_public(nxt)]
    return {"g": g, "seed": int(payload.get("seed", 0)), "mode": "manual",
            "result": outcome(nxt), "frames": frames}


def main():
    ap = argparse.ArgumentParser(description="Страница симулятора посадки")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check_spec()
        return
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

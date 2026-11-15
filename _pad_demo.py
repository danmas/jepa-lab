"""Демо площадки: сети площадки (numeric-land-pad) и сети цели (numeric-land-goal).

    python _pad_demo.py
    открыть http://127.0.0.1:8767

Кодировщик заморожен из out/landf_range_vis_s0.pt. План смотрит в текущий центр.
Сеть площадки (pad) видит тройку (g/9, 1, pad_x/512); сеть цели (goal) — ещё и желаемую vy.
"""
import argparse
import itertools
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from models import EMATarget
from numeric_land.field import GROUND, PAD_CENTER, FUEL0, scale_state
from numeric_land.models import BallEncoder, LandPredictor, fit_readout
from numeric_land.pad import (
    PAD_HALF, PLAN_PADS, TRAIN_PAD_HI, TRAIN_PAD_LO,
    outcome_at, probe_states, same_start,
)
from numeric_land.train_pad import OUT, ckpt_path

# --- загрузка моделей ----------------------------------------------------

_RIDGE_LAM = 1e-2
_PROBE_TRAIN = 3072


# Семейство сети: площадка (pad, тройка мира) или цель (goal, четвёрка с желаемой vy).
_FAMILIES = {
    "pad-vis": ("pad", "vis"),
    "pad-hid": ("pad", "hid"),
    "goal-vis": ("goal", "vis"),
    "goal-hid": ("goal", "hid"),
}
_TRAIN_MODULE = {"pad": "pad", "goal": "goal"}
_TAG = {"pad": "numeric-land-pad", "goal": "numeric-land-goal"}


def _parts(variant: str):
    if variant not in _FAMILIES:
        raise ValueError("сеть: pad-vis, pad-hid, goal-vis или goal-hid")
    return _FAMILIES[variant]


def _load(variant: str):
    family, sub = _parts(variant)
    if family == "pad":
        from numeric_land.train_pad import ckpt_path as path_of
        pg = __import__("numeric_land.pad", fromlist=["probe_states"])
    else:
        from numeric_land.train_goal import ckpt_path as path_of
        pg = __import__("numeric_land.goal", fromlist=["probe_states"])
    tag = _TAG[family]
    path = path_of(sub, 0)
    if not os.path.isfile(path):
        raise ValueError(
            f"нет {os.path.basename(path)}. Сначала:\n"
            f"python -m numeric_land.train_{_TRAIN_MODULE[family]} --tag {tag}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("tag") != tag or ck.get("variant") != sub:
        raise ValueError(path)
    encoder = BallEncoder()
    predictor = LandPredictor(int(ck["n_world"]))
    encoder.load_state_dict(ck["encoder"])
    predictor.load_state_dict(ck["predictor"])
    ema = EMATarget(encoder, momentum=float(ck.get("ema_momentum", 0.99)))
    ema.load_state_dict(ck["ema"])
    encoder.eval()
    predictor.eval()
    ema.eval()

    states = pg.probe_states()
    scaled = torch.from_numpy(scale_state(states))
    target = torch.from_numpy(np.array(states, np.float32, copy=True))
    with torch.no_grad():
        z = ema(scaled)
    probe = fit_readout(z[:_PROBE_TRAIN], target[:_PROBE_TRAIN], lam=_RIDGE_LAM)
    return encoder, predictor, probe, int(ck["n_world"])


# --- симулятор ------------------------------------------------------------

def apply_step(state, action, g):
    from numeric_land.field import step_field
    x, y, vx, vy, fuel, grounded = step_field(
        state["x"], state["y"], state["vx"], state["vy"], state["fuel"],
        bool(state["grounded"]), int(action), g)
    out = dict(state)
    out.update(x=x, y=y, vx=vx, vy=vy, fuel=fuel, grounded=grounded,
               action=int(action))
    return out


def start_of(seed, start_xy=None):
    if start_xy is not None:
        x, y = float(start_xy[0]), float(start_xy[1])
        vx, vy = 0.0, 0.0
    else:
        x, y, vx, vy, _ = same_start(int(seed))
    return {"x": x, "y": y, "vx": vx, "vy": vy, "fuel": FUEL0,
            "grounded": False, "action": 1}


def _public(state):
    return {
        "x": round(state["x"], 4), "y": round(state["y"], 4),
        "vx": round(state["vx"], 4), "vy": round(state["vy"], 4),
        "fuel": round(state["fuel"], 4), "grounded": bool(state["grounded"]),
        "action": int(state.get("action", 1)),
    }


def rollout(seed, g, pad, variant, mode, horizon=40, start_xy=None, desired_vy=0.0):
    enc, pred, probe, _nw = _load(variant)
    models = (enc, pred, probe)
    family, _sub = _parts(variant)
    if mode == "plan":
        if family == "goal":
            from numeric_land.plan_goal import run_episode
            ep = run_episode(int(seed), float(g), float(pad), float(desired_vy),
                             models, max_steps=450)
            ep["desired_vy"] = float(desired_vy)
        else:
            from numeric_land.plan_pad import run_episode
            ep = run_episode(int(seed), float(g), float(pad), models, max_steps=450)
        ep["mode"] = "plan"
        ep["horizon"] = horizon
        ep["variant"] = variant
        return ep
    state = start_of(seed, start_xy=start_xy)
    frames = [_public(state)]
    cap = 450
    for _ in range(cap):
        if state["grounded"]:
            break
        if mode == "fall":
            action = 1
        elif mode == "example":
            action = _choose_action(state, g, pad)
        else:
            raise ValueError(mode)
        state = apply_step(state, action, g)
        frames.append(_public(state))
    return {"g": float(g), "seed": int(seed), "pad_x": float(pad),
            "variant": variant, "mode": mode, "horizon": horizon,
            "result": outcome_at(state["x"], state["y"], state["vy"],
                                 state["grounded"], pad),
            "steps": len(frames) - 1, "frames": frames}


def _choose_action(state, g, pad):
    """Короткое правило: подрулить к центру pad и тормозить к земле."""
    if state["grounded"]:
        return 1
    x, y, vx, vy = state["x"], state["y"], state["vx"], state["vy"]
    height = max(0.0, GROUND - y)
    aim = max(-8.0, min(8.0, (float(pad) - x) * 1.5))
    if vx < aim - 1.2:
        ix = 2
    elif vx > aim + 1.2:
        ix = 0
    else:
        ix = 1
    iy = 0
    from numeric_land.data import MAX_THRUST
    if state["fuel"] > 0.0 and height > 0.0:
        pred = vy * vy + 2.0 * (float(g) - MAX_THRUST) * height
        if pred > 36.0:
            iy = 2
        elif pred > 16.0 and height < 5.0:
            iy = 1
    return iy * 3 + ix


# --- живой показ ----------------------------------------------------------

_LIVE = {}
_LIVE_SEQ = itertools.count(1)


def _start_live(seed, g, pad, variant, horizon, start_xy=None, desired_vy=0.0):
    st = start_of(seed, start_xy=start_xy)
    run_id = f"run{next(_LIVE_SEQ)}"
    _LIVE[run_id] = {"x": st["x"], "y": st["y"], "vx": st["vx"], "vy": st["vy"],
                     "fuel": st["fuel"], "grounded": st["grounded"],
                     "g": float(g), "pad": float(pad), "variant": variant,
                     "desired_vy": float(desired_vy),
                     "seed": int(seed), "horizon": int(horizon),
                     "t": 0, "frames": []}
    return run_id, _live_episode(run_id)


def _live_episode(run_id):
    s = _LIVE[run_id]
    last = s["frames"][-1] if s["frames"] else {
        "x": s["x"], "y": s["y"], "vx": s["vx"], "vy": s["vy"],
        "fuel": s["fuel"], "grounded": s["grounded"], "action": 1}
    result = None
    if s["grounded"]:
        result = outcome_at(s["x"], s["y"], s["vy"], True, s["pad"])
    elif s["t"] >= 450:
        result = "не сел"
    return {"g": s["g"], "seed": s["seed"], "pad_x": s["pad"],
            "variant": s["variant"], "mode": "plan-live",
            "desired_vy": s.get("desired_vy", 0.0),
            "horizon": s["horizon"], "result": result, "steps": s["t"],
            "frame": last, "frames": s["frames"]}


def _live_plan_step(run_id):
    s = _LIVE[run_id]
    if s["grounded"] or s["t"] >= 450:
        return _live_episode(run_id)
    family, _sub = _parts(s["variant"])
    enc, pred, probe, _nw = _load(s["variant"])
    if family == "goal":
        from numeric_land.plan_goal import cem_plan
        seq, _c, _e = cem_plan(
            enc, pred, probe, s["x"], s["y"], s["vx"], s["vy"], s["fuel"],
            s["g"], s["pad"], s.get("desired_vy", 0.0),
            horizon=int(s["horizon"]), seed=int(s["seed"]) + s["t"])
    else:
        from numeric_land.plan_pad import cem_plan
        seq, _c, _e = cem_plan(
            enc, pred, probe, s["x"], s["y"], s["vx"], s["vy"], s["fuel"],
            s["g"], s["pad"], horizon=int(s["horizon"]), seed=int(s["seed"]) + s["t"])
    action = int(seq[0])
    frame = {"x": s["x"], "y": s["y"], "vx": s["vx"], "vy": s["vy"],
             "fuel": s["fuel"], "grounded": s["grounded"]}
    nxt = _public(apply_step(frame, action, s["g"]))
    s.update(x=nxt["x"], y=nxt["y"], vx=nxt["vx"], vy=nxt["vy"],
             fuel=nxt["fuel"], grounded=nxt["grounded"])
    s["t"] += 1
    s["frames"].append({**nxt, "action": action})
    if len(s["frames"]) > 1200:
        del s["frames"][:-1200]
    return _live_episode(run_id)


# --- страница -------------------------------------------------------------

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jepa-lab — площадка как параметр окна</title>
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
    background: #111; border-radius: 8px; cursor: crosshair; }
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
  @media (max-width: 860px) { .bench { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Площадка как параметр окна</h1>
  <p class="lead">Центр площадки — параметр окна, интегратор его не видит. Сеть площадки видит тройку мира <code>(g/9, 1, pad_x/512)</code>, её скрытый близнец — нет. Сеть цели <b>дополнительно</b> видит желаемую скорость посадки <code>desired_vy/120</code> — четвёрку <code>(g/9, 1, pad_x/512, desired_vy/120)</code>; её скрытый близнец тройку и цель не видит. Покрути «желаемую vy» и сравни <b>goal-vis</b> с <b>goal-hid</b>: видимая сеть реагирует на цель, скрытая — нет. «План сети» — CEM, горизонт 40, потолок 450. «Живой план» — один шаг за запрос. Клик по экрану — точка старта.</p>
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
      <label for="pad">Центр площадки</label>
      <input id="pad" type="number" min="106" max="406" step="1" value="256">
      <label for="g">Тяжесть g, px/с²</label>
      <input id="g" type="number" min="2" max="20" step="1" value="8">
      <label for="seed">Сид старта</label>
      <input id="seed" type="number" min="0" max="999999" step="1" value="94000">
      <label for="horizon">Горизонт плана (шагов)</label>
      <input id="horizon" type="number" min="1" max="450" step="1" value="40">
      <label for="variant">Сеть</label>
      <select id="variant">
        <option value="pad-vis">площадка, видимая (тройка мира)</option>
        <option value="pad-hid" selected>площадка, скрытая</option>
        <option value="goal-vis">цель, видимая (тройка + желаемая vy)</option>
        <option value="goal-hid">цель, скрытая (контроль)</option>
      </select>
      <label for="desired_vy">Желаемая vy при посадке (только для сети цели)</label>
      <input id="desired_vy" type="number" min="0" max="120" step="1" value="0">
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
      <p id="phase-line" style="margin:6px 0 0;font-size:13px;color:var(--muted);">
        фаза: —&nbsp;&nbsp;синий = прицел&nbsp;|&nbsp;оранжевый = торможение (высота &lt; 40)</p>
      <p class="note">Мягкая посадка: первое касание в [pad−8, pad+8] и vy&lt;8. Порог 8 — «удар». Сети площадки обучены на сидах 94000…, сети цели — на 97000…. Ноль мягких — допустимый итог, победителя нет: у сетей цели выигрыш виден в предсказании (реакция на цель), а не в числе посадок на g=20.</p>
    </div>
  </div>
</main>
<script>
const X0 = 160, Y0 = 416;
const S = 5;
const VW = (352 - X0) * S, GROUND = 512, GROUND_PAD = 16;
const VH = (GROUND - Y0) * S + GROUND_PAD;
const SHIP = 14;
const NAMES = ["влево","пусто","вправо","влево + половина","половина",
  "вправо + половина","влево + полная","полная","вправо + полная"];
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
let frames = [], cursor = 0, playing = false, timer = null;
let clickStart = null;

function padValue() { return Number(document.getElementById("pad").value); }
function gValue() { return Number(document.getElementById("g").value); }
function seedValue() { return Number(document.getElementById("seed").value); }
function horizonValue() { return Number(document.getElementById("horizon").value); }
function variantValue() { return document.getElementById("variant").value; }
function desiredVyValue() { return Number(document.getElementById("desired_vy").value) || 0; }

function wx(x) { return (x - X0) * S; }
function wy(y) { return (y - Y0) * S; }

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
  ctx.fillStyle = "#14181f"; ctx.fillRect(0, 0, VW, VH);
  const pad = padValue();
  ctx.strokeStyle = "#2a3140"; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = 160; x <= 352; x += 16) { ctx.moveTo(wx(x), 0); ctx.lineTo(wx(x), wy(GROUND)); }
  for (let y = Y0; y <= GROUND; y += 16) { ctx.moveTo(0, wy(y)); ctx.lineTo(VW, wy(y)); }
  ctx.stroke();
  const groundY = wy(GROUND);
  // тормозная зона: ниже этой высоты план должен гасить vy
  const BRAKE_ALT = 40;
  ctx.strokeStyle = "#e2b340"; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(0, wy(GROUND - BRAKE_ALT)); ctx.lineTo(VW, wy(GROUND - BRAKE_ALT)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "#3c4a40"; ctx.lineWidth = 2; ctx.beginPath();
  ctx.moveTo(0, groundY); ctx.lineTo(VW, groundY); ctx.stroke();
  // площадка в текущем центре pad
  ctx.fillStyle = "#2f9e62";
  ctx.fillRect(wx(pad - 8), groundY - 4, 16 * S, 4);
  // границы площадки
  ctx.strokeStyle = "#8fd0a8"; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(wx(pad - 8), groundY - 10); ctx.lineTo(wx(pad - 8), groundY);
  ctx.moveTo(wx(pad + 8), groundY - 10); ctx.lineTo(wx(pad + 8), groundY);
  ctx.stroke(); ctx.setLineDash([]);
  // след
  ctx.strokeStyle = "#8fd0a8"; ctx.lineWidth = 2; ctx.beginPath();
  trail.forEach((p, i) => { const c = i === 0 ? "moveTo" : "lineTo"; ctx[c](wx(p.x), wy(p.y)); });
  ctx.stroke();
  // корпус
  const h = GROUND - frame.y;
  const phase = frame.grounded ? "касание" : (h > 40 ? "прицел" : "торможение");
  const phaseColor = phase === "касание" ? "#f2d48a" : (phase === "прицел" ? "#4da3ff" : "#ff9d3c");
  const a = frame.action | 0, iy = Math.floor(a/3), ix = a % 3;
    const x = wx(frame.x), y = wy(frame.y);
  ctx.fillStyle = phaseColor;
    ctx.beginPath(); ctx.moveTo(x, y - SHIP); ctx.lineTo(x - SHIP*0.55, y);
    ctx.lineTo(x + SHIP*0.55, y); ctx.closePath(); ctx.fill();
  const sx = ix - 1, frac = [0,0.5,1][iy];
  if (frac > 0 || sx !== 0) {
    ctx.strokeStyle = "#e2b340"; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x - sx*SHIP, y + frac*SHIP); ctx.stroke();
  }
  // маркер старта
  if (clickStart) {
    ctx.strokeStyle = "#4da3ff"; ctx.lineWidth = 2; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.arc(wx(clickStart.x), wy(clickStart.y), 13, 0, Math.PI*2);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#4da3ff"; ctx.beginPath();
    ctx.arc(wx(clickStart.x), wy(clickStart.y), 3, 0, Math.PI*2); ctx.fill();
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
  document.getElementById("caption").textContent =
    "x " + f.x.toFixed(2) + "  y " + f.y.toFixed(2) +
    "  vx " + f.vx.toFixed(2) + "  vy " + f.vy.toFixed(2) +
    "  топливо " + f.fuel.toFixed(2) + "  действие " + NAMES[f.action] +
    (f.grounded ? "  касание" : "") + "   фаза: " + (GROUND - f.y > 40 ? "прицел" : "торможение");
  const ph = f.grounded ? "касание" : (GROUND - f.y > 40 ? "прицел" : "торможение");
  const phColor = ph === "касание" ? "#1d6b45" : (ph === "прицел" ? "#4da3ff" : "#ff9d3c");
  document.getElementById("phase-line").innerHTML = 'фаза: <b style="color:' + phColor + '">' + ph + '</b>  синий = прицел | оранжевый = торможение (высота < 40)';
}
function stop() { playing = false; if (timer) clearInterval(timer); timer = null;
  document.getElementById("toggle").textContent = "пуск"; }
function play() {
  if (frames.length < 2) return;
  playing = true; document.getElementById("toggle").textContent = "пауза";
  if (cursor >= frames.length - 1) cursor = 0;
  timer = setInterval(() => { if (cursor >= frames.length - 1) { stop(); return; } show(cursor + 1); }, 33);
}

function setReport(data) {
  const last = data.frames[data.frames.length - 1];
  const cls = data.result === "мягкая" ? "ok" : "bad";
  const box = document.getElementById("report");
  box.dataset.result = data.result;
  const goalNote = (data.variant || "").indexOf("goal") === 0
    ? ", желаемая vy " + (data.desired_vy == null ? 0 : data.desired_vy) : "";
  box.innerHTML = "<strong class='" + cls + "'>" + data.result + "</strong>\\n"
    + "шагов " + (data.frames.length - 1) + ", g " + data.g +
    ", центр " + data.pad_x + ", сеть " + data.variant + goalNote + "\\n"
    + "финиш x " + last.x.toFixed(2) + ", vy " + last.vy.toFixed(2) +
    ", топливо " + last.fuel.toFixed(2);
}

function setBusy(mode, busy) {
  for (const id of ["plan","live","example","fall"]) document.getElementById(id).disabled = busy;
  const plan = document.getElementById("plan");
  if (busy) { plan.dataset.label = plan.textContent; plan.textContent = "план сети — считаю…"; }
  else if (plan.dataset.label) { plan.textContent = plan.dataset.label; delete plan.dataset.label; }
}

async function run(mode) {
  stop(); const reportBox = document.getElementById("report");
  reportBox.innerHTML = mode === "plan" ? "<span style='color:#1c1915;'>Строю план сети… это может занять до минуты.</span>" : "Считаю…";
  setBusy(mode, true);
  try {
    const res = await fetch("/api/run", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({g:gValue(), seed:seedValue(), mode,
        pad:padValue(), variant:variantValue(), horizon:horizonValue(),
        desired_vy:desiredVyValue(), ...startPayload()})});
    const data = await res.json();
    if (!res.ok) { reportBox.textContent = data.error || "ошибка"; return; }
    frames = data.frames; runResult = data;
    setReport(data); show(0); play();
  } finally { setBusy(mode, false); }
}

let runResult = null;

async function runLive() {
  stop(); const reportBox = document.getElementById("report");
  setBusy("live", true); let runId = null;
  try {
    const start = await fetch("/api/live/start", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({g:gValue(), seed:seedValue(), pad:padValue(),
        variant:variantValue(), horizon:horizonValue(),
        desired_vy:desiredVyValue(), ...startPayload()})});
    const startData = await start.json();
    if (!start.ok) { reportBox.textContent = startData.error || "ошибка"; return; }
    runId = startData.run_id;
    frames = startData.frames.length ? startData.frames : [startData.frame];
    reportBox.textContent = "живой план — шаг 0";
    show(0);
    for (let t = 1; ; t++) {
      const res = await fetch("/api/live/step", {method:"POST",
        headers:{"Content-Type":"application/json"}, body: JSON.stringify({run_id:runId})});
      const data = await res.json();
      if (!res.ok) { reportBox.textContent = data.error || "ошибка"; break; }
      frames = data.frames; reportBox.textContent = "живой план — шаг " + data.steps;
      show(frames.length - 1);
      if (data.result) { setReport({frames, result: data.result, g: data.g, pad_x: data.pad_x, variant: data.variant}); break; }
    }
  } catch (e) { reportBox.textContent = "ошибка: " + e; }
  finally { setBusy("live", false); }
}

async function manual(action) {
  stop();
  const res = await fetch("/api/step", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({g:gValue(), seed:seedValue(), action, frames,
      pad:padValue(), variant:variantValue(), desired_vy:desiredVyValue()})});
  const data = await res.json();
  if (!res.ok) { document.getElementById("report").textContent = data.error || "ошибка"; return; }
  frames = data.frames; setReport(data); show(frames.length - 1);
}

document.getElementById("plan").onclick = () => run("plan");
document.getElementById("live").onclick = () => runLive();
document.getElementById("example").onclick = () => run("example");
document.getElementById("fall").onclick = () => run("fall");
document.getElementById("clear-start").onclick = () => clearStart();
document.getElementById("toggle").onclick = () => playing ? stop() : play();
document.getElementById("scrub").oninput = (e) => { stop(); show(Number(e.target.value)); };
const box = document.getElementById("actions");
NAMES.forEach((name, i) => { const b = document.createElement("button");
  b.type="button"; b.textContent=name; b.onclick=()=>manual(i); box.appendChild(b); });
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
                sx = float(payload["start_x"]); sy = float(payload["start_y"])
                if not (0 <= sx <= 512 and 0 < sy < GROUND):
                    raise ValueError("старт вне поля")
                start_xy = (sx, sy)
            pad = float(payload.get("pad", 256))
            if not 106 <= pad <= 406:
                raise ValueError("центр вне 106…406")
            variant = str(payload.get("variant", "pad-hid"))
            if variant not in _FAMILIES:
                raise ValueError("сеть: pad-vis, pad-hid, goal-vis или goal-hid")
            desired_vy = float(payload.get("desired_vy", 0.0))
            if not 0.0 <= desired_vy <= 120.0:
                raise ValueError("желаемая vy вне 0…120")
            if path == "/api/run":
                mode = payload.get("mode", "example")
                if mode not in ("example", "fall", "plan"):
                    raise ValueError("mode: plan, example или fall")
                g = float(payload.get("g", 8))
                if not 0 < g <= 40:
                    raise ValueError("g вне 0…40")
                horizon = int(payload.get("horizon", 40))
                if not 1 <= horizon <= 450:
                    raise ValueError("горизонт вне 1…450")
                data = rollout(int(payload.get("seed", 0)), g, pad, variant,
                               mode, horizon=horizon, start_xy=start_xy,
                               desired_vy=desired_vy)
            elif path == "/api/live/start":
                g = float(payload.get("g", 8))
                if not 0 < g <= 40:
                    raise ValueError("g вне 0…40")
                horizon = int(payload.get("horizon", 40))
                if not 1 <= horizon <= 450:
                    raise ValueError("горизонт вне 1…450")
                run_id, data = _start_live(int(payload.get("seed", 0)), g, pad,
                                           variant, horizon,
                                           start_xy=start_xy, desired_vy=desired_vy)
                data["run_id"] = run_id
            elif path == "/api/live/step":
                run_id = str(payload.get("run_id", ""))
                if run_id not in _LIVE:
                    raise ValueError("нет такой сессии")
                data = _live_plan_step(run_id)
            elif path == "/api/step":
                payload.setdefault("desired_vy", desired_vy)
                data = _manual(payload, pad, variant)
            else:
                raise ValueError("неизвестный путь")
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        except (ValueError, KeyError, TypeError) as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(400, body, "application/json; charset=utf-8")


def _manual(payload, pad, variant):
    g = float(payload.get("g", 8))
    if not 0 < g <= 40:
        raise ValueError("g вне 0…40")
    action = int(payload["action"])
    frames = payload.get("frames") or []
    if not frames:
        state = start_of(int(payload.get("seed", 0)))
    else:
        last = frames[-1]
        state = {"x": float(last["x"]), "y": float(last["y"]),
                 "vx": float(last["vx"]), "vy": float(last["vy"]),
                 "fuel": float(last["fuel"]),
                 "grounded": bool(last["grounded"]),
                 "action": int(last.get("action", 1))}
    if not frames:
        frames = [_public(state)]
    nxt = apply_step(state, action, g)
    frames = list(frames) + [_public(nxt)]
    return {"g": g, "seed": int(payload.get("seed", 0)), "pad_x": float(pad),
            "variant": variant, "mode": "manual",
            "desired_vy": float(payload.get("desired_vy", 0.0)),
            "result": outcome_at(nxt["x"], nxt["y"], nxt["vy"],
                                 nxt["grounded"], pad),
            "frames": frames}


def main():
    ap = argparse.ArgumentParser(description="Демо площадки")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()
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
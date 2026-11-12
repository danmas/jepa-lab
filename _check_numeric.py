"""Проверки числового мира до и помимо обучения.

Траектория совпадает с step_state и с плоским World. Постоянный кодировщик
получает штраф. Открытая раскатка лосса не подглядывает в истину. Чужой тег
отказывается и не трогает пиксельные чекпоинты.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import World, step_state
from numeric.data import follow, scale_state, start_state, unscale_state
from models import EMATarget
from numeric.models import (
    DirectPredictor, NumericEncoder, NumericPredictor, collapse_penalty, direct_loss, jepa_loss,
)
from numeric.rollout import _euclidean
from numeric.train import refuse_tag

ROOT = os.path.dirname(os.path.abspath(__file__))


def check_trajectory():
    seed = 7
    actions = [0, 5, 7, 4, 1, 8]
    world = World(seed, world="flat")
    pos0, vel0 = world.state()
    rng = np.random.default_rng(seed)
    pos, vel = start_state(rng)
    if not (np.allclose(pos, pos0) and np.allclose(vel, vel0)):
        raise SystemExit("старт числового генератора не совпал с плоским World")
    got = follow(pos0, vel0, actions)
    pos, vel = np.array(pos0, np.float32), np.array(vel0, np.float32)
    for i, action in enumerate(actions):
        pos, vel = step_state(pos, vel, int(action))
        if not np.allclose(got[i + 1, 0:2], pos, atol=1e-4):
            raise SystemExit("генератор разошёлся с прямым step_state")
        if not np.allclose(got[i + 1, 2:4], vel, atol=1e-4):
            raise SystemExit("скорость генератора разошлась с step_state")
    for action in actions:
        world.step(int(action))
    wpos, wvel = world.state()
    if not np.allclose(got[-1, 0:2], wpos, atol=1e-4):
        raise SystemExit("конец траектории разошёлся с World")
    if not np.allclose(got[-1, 2:4], wvel, atol=1e-4):
        raise SystemExit("скорость конца разошлась с World")
    if got.shape[-1] != 4:
        raise SystemExit("в состоянии больше четырёх чисел")
    raw = np.array([[32, 10, 1.5, -0.5]], np.float32)
    back = unscale_state(scale_state(raw))
    if not np.allclose(back, raw, atol=1e-5):
        raise SystemExit("масштаб 64 и 4 не обращается")
    err = _euclidean(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
    if abs(float(err[0]) - 5.0) > 1e-9:
        raise SystemExit("евклидова ошибка положения не равна гипотенузе")


def check_penalty_and_open_loop():
    class ConstEnc(nn.Module):
        def forward(self, state):
            z = torch.zeros(*state.shape[:-1], 32, dtype=state.dtype)
            z[..., 0] = 1
            return z

    class SpyPred(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, z, pair):
            self.inputs.append(z.detach().clone())
            return F.normalize(z + 0.01, dim=-1)

    class SpyEma(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, state):
            self.seen.append(state.detach().clone())
            return torch.zeros(state.shape[0], 32)

    states = torch.randn(8, 6, 4)
    actions = torch.randint(0, 9, (8, 5))
    enc, pred, ema = ConstEnc(), SpyPred(), SpyEma()
    _loss, parts = jepa_loss(enc, pred, ema, states, actions)
    if not (parts["penalty"] > 0):
        raise SystemExit("постоянный единичный кодировщик не получил штраф")
    const = torch.zeros(16, 32)
    const[:, 0] = 1
    if not (float(collapse_penalty(const)) > 0):
        raise SystemExit("штраф константы не положителен")
    first_out = F.normalize(pred.inputs[0] + 0.01, dim=-1)
    if not torch.allclose(pred.inputs[1], first_out, atol=1e-6):
        raise SystemExit("второй шаг читает не предыдущее предсказание")
    if not torch.allclose(pred.inputs[2], F.normalize(pred.inputs[1] + 0.01, dim=-1), atol=1e-6):
        raise SystemExit("третий шаг читает не предыдущее предсказание")
    true_next = enc(states[:, 1])
    if torch.allclose(pred.inputs[1], true_next, atol=1e-5):
        raise SystemExit("второй шаг совпал с кодировкой истинного состояния")
    for k in range(3):
        if not torch.allclose(ema.seen[k], states[:, k + 1]):
            raise SystemExit("цель шага — не истинное состояние этого шага")

    class SpyDirect(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, state, pair):
            self.inputs.append(state.detach().clone())
            return state + 0.1

    direct = SpyDirect()
    direct_loss(direct, states, actions)
    if not torch.allclose(direct.inputs[1], states[:, 0] + 0.1):
        raise SystemExit("direct на втором шаге читает истину, а не предсказание")


def check_backward():
    states = torch.randn(16, 6, 4)
    actions = torch.randint(0, 9, (16, 5))
    encoder, predictor = NumericEncoder(), NumericPredictor()
    ema = EMATarget(encoder, momentum=0.99)
    loss, _ = jepa_loss(encoder, predictor, ema, states, actions)
    loss.backward()
    grads = [p.grad for p in list(encoder.parameters()) + list(predictor.parameters())]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise SystemExit("градиент JEPA не конечен")
    direct = DirectPredictor()
    loss_d, _ = direct_loss(direct, states, actions)
    loss_d.backward()
    grads_d = [p.grad for p in direct.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in grads_d):
        raise SystemExit("градиент direct не конечен")


def check_refuse():
    for bad in ("", "v4", "v5-floor", "v5-jitter"):
        try:
            refuse_tag(bad)
        except SystemExit:
            pass
        else:
            raise SystemExit(f"тег {bad!r} не отклонён")
    refuse_tag("numeric")
    before = _snapshot()
    for bad in ("", "v4", "v5-floor"):
        proc = subprocess.run(
            [sys.executable, "-m", "numeric.train", "--tag", bad],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            raise SystemExit(f"тег {bad!r} не завершился отказом")
        text = proc.stdout + proc.stderr
        if "ms/step" in text or "AdamW" in text:
            raise SystemExit(f"тег {bad!r} дошёл до обучения")
    after = _snapshot()
    if before != after:
        raise SystemExit("отказ тега изменил файлы в out/")


def _snapshot():
    paths = glob.glob(os.path.join(ROOT, "out", "*"))
    return {p: (os.path.getmtime(p), os.path.getsize(p)) for p in paths if os.path.isfile(p)}


def main():
    check_trajectory()
    check_penalty_and_open_loop()
    check_backward()
    check_refuse()
    print("numeric checks ok")


if __name__ == "__main__":
    main()

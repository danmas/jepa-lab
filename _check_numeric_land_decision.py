"""Контроль решения и относительного кодировщика. python _check_numeric_land_decision.py"""
import inspect
import os
import subprocess
import sys

import numpy as np
import torch

from models import EMATarget
from numeric_land.decision import (
    BETTER_MIN, GAP_MIN, LATENT_MARGIN, THRESHOLD, assert_baselines, assert_start,
    better_ok, binary_scores, counts_toward_90, features_of, gap_ok, latent_helped,
    line_90, n_in_of, pilot_passes, switch_numbers,
)
from numeric_land.expert_hover import NAV_TOL, hover_action, hover_mode
from numeric_land.field import step_field
from numeric_land.goal import outcome_at
from numeric_land.models import LOSS_WEIGHTS as MODEL_WEIGHTS
from numeric_land.models import COLLAPSE_WEIGHT as MODEL_COLLAPSE
from numeric_land.models import BallEncoder, LandPredictor
from numeric_land.relenc import (
    N_IN, PACKAGE_SENTENCE, W_PROBE, RelEncoder, RelProbe, accepted, bucket_of,
    expand_encoder, iter_windows, pools_of, probe_pass, probe_target_flag,
    probe_targets, rel_loss, scale_rel,
)
from numeric_land.skills import skill_flag, skill_loss
from numeric_land.train_goal import COLLAPSE_WEIGHT, LOSS_WEIGHTS, W_TERM
from numeric_land import train_decision as TD
from numeric_land import train_relenc as TR
from numeric_land import train_skills as TS
from numeric_land.field import scale_state


def _stamp(path):
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _run(root, args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        args, cwd=root, timeout=120, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env)
    return proc


def _episode(states, soft=True):
    return {
        "seed": 1,
        "g": 8.0,
        "pad_x": 256.0,
        "v_target": 6.0,
        "states": np.asarray(states, np.float64),
        "actions": np.asarray([1, 1, 1], np.int64),
        "grounded": True,
        "soft": soft,
    }


def _expect_tag(root, args, needle, names, before):
    proc = _run(root, args)
    if proc.returncode == 0:
        raise SystemExit(f"чужой тег принят: {args}\n{proc.stdout}\n{proc.stderr}")
    text = (proc.stdout or "") + (proc.stderr or "")
    if needle not in text:
        raise SystemExit(f"отказ не назвал тег {needle}: {args}\n{text}")
    for key, path in names.items():
        if _stamp(path) != before[key]:
            raise SystemExit(f"чужой тег задел {path}")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    hover_path = os.path.join(root, "numeric_land", "expert_hover.py")
    hover_before = _stamp(hover_path)

    if NAV_TOL != 6.0 or THRESHOLD != 0.5 or W_PROBE != 0.1:
        raise SystemExit("допуск, порог или вес probe сдвинуты")
    if GAP_MIN != 153 or BETTER_MIN != 80 or LATENT_MARGIN != 18:
        raise SystemExit("линии контура сдвинуты")
    if W_TERM != 0.25 or LOSS_WEIGHTS != (1.0, 0.5, 0.25) or COLLAPSE_WEIGHT != 0.1:
        raise SystemExit("лосс навыка разошёлся с целью")
    if MODEL_WEIGHTS != LOSS_WEIGHTS or MODEL_COLLAPSE != COLLAPSE_WEIGHT:
        raise SystemExit("лосс JEPA разошёлся")
    if N_IN != 7 or TR.THREADS != 8 or TS.THREADS != 8 or TD.THREADS != 8:
        raise SystemExit("ширина или число потоков")
    if TR.STEPS != 3000 or TS.STEPS != 3000 or TD.STEPS != 1000:
        raise SystemExit("число шагов")
    if TR.EXPERT_FRAC != 0.5 or TS.EXPERT_FRAC != 0.5 or TD.BATCH != 256:
        raise SystemExit("батч или доля эксперта")
    if TR.N_WORLD != 2 or TS.N_WORLD != 4:
        raise SystemExit("длина канала мира")

    params = list(inspect.signature(step_field).parameters)
    if "pad_x" in params or "mode" in params or "pad" in params:
        raise SystemExit(f"физика видит лишнее: {params}")
    try:
        step_field(256, 464, 0, 0, 10, False, 1, 8, pad_x=200)
    except TypeError:
        pass
    else:
        raise SystemExit("шаг принял pad_x")
    first = step_field(256, 464, 0, 0, 10, False, 1, 8)
    second = step_field(256, 464, 0, 0, 10, False, 1, 8)
    if first != second:
        raise SystemExit("шаг физики не повторился")

    land = (250.0, 400.0, 0.0, 0.0, 10.0)
    approach = (249.9, 400.0, 0.0, 0.0, 10.0)
    if hover_mode(land, 8, 256, 6) != "land":
        raise SystemExit("граница 6 не посадка")
    if hover_mode(approach, 8, 256, 6) != "approach":
        raise SystemExit("граница 6.1 не наведение")
    if hover_mode((256, 512, 0, 0, 10), 8, 256, 6) is not None:
        raise SystemExit("земля не пустой режим")
    if hover_action((256, 512, 0, 0, 10), 8, 256, 6) != 1:
        raise SystemExit("на земле действие не 1")
    if hover_mode((256, 400, 0, 0, 0), 8, 256, 6) is not None:
        raise SystemExit("пустой бак не пустой режим")
    if hover_action((256, 400, 0, 0, 0), 8, 256, 6) != 1:
        raise SystemExit("пустой бак, действие не 1")

    if outcome_at(256, 512, 8, True, 256) != "удар":
        raise SystemExit("vy = 8 не удар")
    if outcome_at(256, 512, 7.9, True, 256) != "мягкая":
        raise SystemExit("vy < 8 внутри площадки не мягкая")

    state = np.array([[256.0, 464.0, 0.0, 0.0, 10.0]], np.float64)
    rel = scale_rel(state, np.array([200.0]))
    base = scale_state(state)
    if rel.shape[-1] != 7 or not np.allclose(rel[0, :5], base[0]):
        raise SystemExit("первые пять каналов разошлись со scale_state")
    if abs(float(rel[0, 5]) - (256.0 - 200.0) / 512.0) > 1e-6:
        raise SystemExit("знаковый канал")
    if abs(float(rel[0, 6]) - (512.0 - 464.0) / 512.0) > 1e-6:
        raise SystemExit("канал высоты")
    if float(rel[0, 6]) != float(np.float32(1.0) - base[0, 1]):
        raise SystemExit("высота не равна 1 − y/512")
    odd = np.array([[10.0, 470.3, 1.0, -2.0, 4.0]], np.float64)
    odd_rel = scale_rel(odd, np.array([300.0]))
    odd_base = scale_state(odd)
    if float(odd_rel[0, 6]) != float(np.float32(1.0) - odd_base[0, 1]):
        raise SystemExit("высота не повторяет y при произвольном y")
    left = np.array([[100.0, 464.0, 0.0, 0.0, 10.0]], np.float64)
    left_rel = scale_rel(left, np.array([200.0]))
    left_tgt = probe_targets(left, np.array([200.0]))
    if not (float(left_rel[0, 5]) < 0.0):
        raise SystemExit("канал слева от площадки не знаковый")
    if abs(float(left_tgt[0, 0]) - abs(float(left_rel[0, 5]))) > 1e-6:
        raise SystemExit("цель probe не модуль знакового канала")
    if abs(float(left_tgt[0, 1]) - float(left_rel[0, 6])) > 1e-6:
        raise SystemExit("цель высоты не седьмой канал")

    if BallEncoder().net[0].in_features != 5 or RelEncoder().net[0].in_features != 7:
        raise SystemExit("ширина входа кодировщика")
    enc5 = BallEncoder()
    enc7 = RelEncoder()
    expand_encoder(enc7, enc5.state_dict())
    if not torch.equal(enc7.net[0].weight[:, :5], enc5.net[0].weight):
        raise SystemExit("старые столбцы не скопированы")
    if int(torch.count_nonzero(enc7.net[0].weight[:, 5:])) != 0:
        raise SystemExit("новые столбцы не нули")
    if not torch.equal(enc7.net[0].bias, enc5.net[0].bias):
        raise SystemExit("смещение первого слоя")
    if not torch.equal(enc7.net[2].weight, enc5.net[2].weight):
        raise SystemExit("скрытый слой не скопирован")

    if probe_target_flag(0.91, 1.0) or not probe_target_flag(0.9, 1.0):
        raise SystemExit("порог 0.9 у probe")
    if probe_target_flag(0.0, 0.0):
        raise SystemExit("нулевой L1 старого кодировщика закрыл флаг")
    if probe_pass(True, False) or not probe_pass(True, True):
        raise SystemExit("probe требует обе цели")
    if W_PROBE != 0.1:
        raise SystemExit("вес probe сдвинулся после флага")

    far = [200.0, 400.0, 0.0, 0.0, 10.0]
    near = [256.0, 400.0, 0.0, 0.0, 10.0]
    ground = [200.0, 512.0, 0.0, 0.0, 10.0]
    pure = list(iter_windows([_episode([far, far, far, far])], True))
    if len(pure) != 1 or pure[0]["bucket"] != "approach":
        raise SystemExit("чистое наведение")
    if not accepted("approach", "approach") or not accepted("approach", "shared"):
        raise SystemExit("чистое наведение не в обоих пулах")
    if accepted("approach", "land"):
        raise SystemExit("наведение попало в посадку")
    mixed = list(iter_windows([_episode([far, far, far, near])], True))
    if len(mixed) != 1 or mixed[0]["bucket"] != "mixed":
        raise SystemExit("стык не смешанный")
    if pools_of(("approach", "approach", "land", "land")) != frozenset({"shared"}):
        raise SystemExit("стык попал в чистый пул")
    if accepted("mixed", "approach") or accepted("mixed", "land") or not accepted("mixed", "shared"):
        raise SystemExit("стык разошёлся с общим пулом")
    if not accepted("mixed", "encoder"):
        raise SystemExit("стык выпал из пула кодировщика")
    if list(iter_windows([_episode([far, ground, far, far])], True)):
        raise SystemExit("пустой режим остался в окне")
    if list(iter_windows([_episode([far, far, far, far], soft=False)], True)):
        raise SystemExit("неуспешный эпизод попал в пул")
    if bucket_of(("land", "land", "land", "land")) != "land":
        raise SystemExit("чистая посадка")

    if skill_flag(0.91, 1.0) or not skill_flag(0.9, 1.0):
        raise SystemExit("порог навыка")
    if THRESHOLD != 0.5:
        raise SystemExit("флаг навыка сдвинул порог")

    z = np.zeros((2, 32), np.float32)
    pads = np.array([200.0, 256.0])
    gs = np.array([8.0, 2.0])
    if features_of("latent", pads, gs, z).shape != (2, 33) or n_in_of("latent") != 33:
        raise SystemExit("боевой вход")
    if features_of("oldz", pads, gs, z).shape != (2, 33) or n_in_of("oldz") != 33:
        raise SystemExit("вход старого латента")
    if features_of("zblind", pads, gs, z).shape != (2, 32) or n_in_of("zblind") != 32:
        raise SystemExit("вход без площадки")
    if features_of("g", pads, gs, z).shape != (2, 34) or n_in_of("g") != 34:
        raise SystemExit("вход с тяжестью")
    if abs(float(features_of("latent", pads, gs, z)[0, -1]) - 200.0 / 512.0) > 1e-6:
        raise SystemExit("канал площадки")
    if abs(float(features_of("g", pads, gs, z)[0, -1]) - 8.0 / 9.0) > 1e-6:
        raise SystemExit("канал тяжести")
    if counts_toward_90("latent") is not True or counts_toward_90("oldz") or counts_toward_90("zblind") or counts_toward_90("g"):
        raise SystemExit("линия 90 % закрывает не тот вход")
    rows = [
        {"kind": "latent", "seed": 0, "accuracy": 0.90, "f1": 0.90},
        {"kind": "latent", "seed": 1, "accuracy": 0.91, "f1": 0.95},
        {"kind": "latent", "seed": 2, "accuracy": 0.90, "f1": 0.90},
        {"kind": "oldz", "seed": 0, "accuracy": 0.50, "f1": 0.40},
        {"kind": "zblind", "seed": 0, "accuracy": 0.10, "f1": 0.0},
        {"kind": "g", "seed": 0, "accuracy": 0.20, "f1": 0.0},
    ]
    if not line_90(rows):
        raise SystemExit("три боевых сида на 0.90 не проходят линию")
    rows[0]["accuracy"] = 0.89
    if line_90(rows):
        raise SystemExit("0.89 закрыл линию 90 %")
    if binary_scores(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.4]))["f1"] != 0.0:
        raise SystemExit("нулевой знаменатель F1")

    if switch_numbers(-40.0, -30.0, 0, 0, 5, 10):
        raise SystemExit("сдвиг −40 против −30 прошёл")
    old = -30.0
    new = -0.9 * abs(old)
    if not switch_numbers(new, old, 0, 0, 5, 10):
        raise SystemExit("сдвиг ровно 0.9 без новых промахов не прошёл")
    if switch_numbers(new, old, 2, 1, 5, 10):
        raise SystemExit("рост промахов прошёл")
    if switch_numbers(new, old, 0, 0, 4, 10):
        raise SystemExit("меньше половины пересечений прошло")

    if gap_ok(152) or not gap_ok(153):
        raise SystemExit("линия 153")
    if not better_ok(153) or not better_ok(80) or better_ok(79):
        raise SystemExit("линия 80")
    if latent_helped(153, 140) or not latent_helped(158, 140):
        raise SystemExit("прибавка 18")
    if pilot_passes(True, True, True, True, True, 153, 140):
        raise SystemExit("153 против 140 назван новым латентом")
    if pilot_passes(True, True, True, True, True, 152, 100):
        raise SystemExit("152 закрыли пилот")
    if THRESHOLD != 0.5 or NAV_TOL != 6.0:
        raise SystemExit("после линий сдвинулись порог или допуск")
    try:
        assert_baselines(180, 61)
    except SystemExit:
        pass
    else:
        raise SystemExit("чужая база не остановила прогон")
    assert_baselines(180, 62)
    try:
        assert_start(0.0, 0.0, {"x": 1.0, "y": 2.0})
    except SystemExit:
        pass
    else:
        raise SystemExit("чужой старт не остановил прогон")

    if "два добавленных канала" not in PACKAGE_SENTENCE or "0.1" not in PACKAGE_SENTENCE:
        raise SystemExit("оговорка про пакет каналов и probe")
    loss_src = inspect.getsource(rel_loss)
    if "W_PROBE" not in loss_src or "mse" in loss_src.lower() or "W_TERM" in loss_src:
        raise SystemExit("лосс кодировщика")
    scratch_src = inspect.getsource(TR.train_scratch)
    if "torch.load" in scratch_src or "landf" in scratch_src or "_load_landf" in scratch_src:
        raise SystemExit("боевой scratch читает поле")
    tune_src = inspect.getsource(TR.train_tune)
    if "_load_landf_readonly" not in tune_src:
        raise SystemExit("дообучение не читает источник")
    if "torch.load" not in inspect.getsource(TR._load_landf_readonly):
        raise SystemExit("чтение источника не через torch.load")
    decision_src = inspect.getsource(TD.load_weights)
    if "landm" in inspect.getsource(TD.train_one):
        raise SystemExit("обучение головы ссылается на старый файл")

    torch.manual_seed(0)
    batch = 4
    encoder = RelEncoder()
    predictor = LandPredictor(2)
    probe = RelProbe()
    ema = EMATarget(encoder, momentum=0.99)
    states = torch.randn(batch, 4, 7)
    actions = torch.randint(0, 9, (batch, 3))
    world = torch.randn(batch, 2)
    targets = torch.rand(batch, 4, 2)
    loss, _parts = rel_loss(encoder, predictor, probe, ema, states, actions, world, targets)
    loss.backward()
    if encoder.net[0].weight.grad is None or probe.fc.weight.grad is None:
        raise SystemExit("градиент probe не дошёл до кодировщика")
    if predictor.net[0].weight.grad is None:
        raise SystemExit("предиктор не получил лосс JEPA")
    if any(param.grad is not None for param in ema.parameters()):
        raise SystemExit("EMA получила градиент")

    encoder_f = RelEncoder()
    encoder_f.requires_grad_(False)
    pred_s = LandPredictor(4)
    ema_s = EMATarget(encoder_f, momentum=0.99)
    loss_s, _parts_s = skill_loss(
        encoder_f, pred_s, ema_s, torch.randn(batch, 4, 7), actions, torch.randn(batch, 4),
        np.full(batch, 200.0), np.full(batch, 6.0), np.full(batch, 40.0))
    loss_s.backward()
    if encoder_f.net[0].weight.grad is not None:
        raise SystemExit("навык учит кодировщик")
    if pred_s.net[0].weight.grad is None:
        raise SystemExit("навык не учит предиктор")

    names = {
        "landr": os.path.join(root, "out", "landr_scratch_s0.pt"),
        "landk": os.path.join(root, "out", "landk_approach_s0.pt"),
        "landd": os.path.join(root, "out", "landd_latent_s0.pt"),
        "landh": os.path.join(root, "out", "landh_range_vis_s0.pt"),
        "landm": os.path.join(root, "out", "landm_latent_s0.pt"),
        "landf": os.path.join(root, "out", "landf_range_vis_s0.pt"),
    }
    before = {key: _stamp(path) for key, path in names.items()}
    if before["landh"] is None or before["landf"] is None or before["landm"] is None:
        raise SystemExit("нет эталонных чекпоинтов")
    py = sys.executable
    _expect_tag(root, [py, "-m", "numeric_land.train_relenc", "--tag", "numeric-land-field"],
                "numeric-land-decision", names, before)
    _expect_tag(root, [py, "-m", "numeric_land.train_decision", "--tag", "numeric-land-mode"],
                "numeric-land-decision", names, before)
    _expect_tag(root, [py, "-m", "numeric_land.train_skills", "--tag", "numeric-land-hover"],
                "numeric-land-decision", names, before)
    _expect_tag(root, [py, "-m", "numeric_land.train_hover", "--tag", "numeric-land-decision"],
                "numeric-land-hover", names, before)
    _expect_tag(root, [py, "-m", "numeric_land.train_mode", "--tag", "numeric-land-decision"],
                "numeric-land-mode", names, before)
    landm_before = _stamp(names["landm"])
    try:
        TD.load_weights(names["landm"])
    except RuntimeError:
        pass
    else:
        raise SystemExit("загрузка открыла старую голову")
    if _stamp(names["landm"]) != landm_before:
        raise SystemExit("отказ от старой головы задел файл")
    if _stamp(hover_path) != hover_before:
        raise SystemExit("expert_hover.py изменился")

    note = os.path.join(root, "KB", "README_LAND_DECISION.md")
    if os.path.isfile(note):
        text = open(note, encoding="utf-8").read()
        for needle in ("0.1", "0.5", "153", "80", "18", "landr_scratch_s0", "CEM",
                       "Победителя нет", "пакет", "два добавленных канала"):
            if needle not in text:
                raise SystemExit(f"в заметке нет «{needle}»")
        if "новый вес" in text:
            raise SystemExit("заметка предлагает новый вес")
        index = open(os.path.join(root, "KB", "README_INDEX.md"), encoding="utf-8").read()
        if "README_LAND_DECISION.md" not in index:
            raise SystemExit("в индексе нет ссылки")
        if index.count("README_LAND_DECISION.md") != 1:
            raise SystemExit("в индексе больше одной ссылки")

    del decision_src
    print("ok", flush=True)


if __name__ == "__main__":
    main()

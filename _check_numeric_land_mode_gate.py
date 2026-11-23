"""Контроль ворот 0.7 и запаса 12. python _check_numeric_land_mode_gate.py

Чекпоинт и старый JSON здесь только сверяются по времени и размеру.
Сетка из 180 эпизодов этим скриптом не летает.
"""
import ast
import os


def _stamp(path):
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    watched = [
        os.path.join(root, "out", "landm_latent_s0.pt"),
        os.path.join(root, "out", "numeric_land_mode.json"),
        os.path.join(root, "numeric_land", "rollout_mode.py"),
        os.path.join(root, "numeric_land", "expert_hover.py"),
        os.path.join(root, "out", "numeric_land_mode_gate.json"),
    ]
    before = {}
    for path in watched:
        stamp = _stamp(path)
        if stamp is None and not path.endswith("numeric_land_mode_gate.json"):
            raise SystemExit(f"нет {os.path.basename(path)}")
        before[path] = stamp

    from numeric_land.expert_hover import NAV_TOL, hover_action, hover_mode
    from numeric_land.field import GROUND
    from numeric_land.pad import PAD_HALF, VY_SOFT
    from numeric_land.rollout_mode_gate import (
        GATE_DX, GATE_P, HELP_AT, OLD_GAP, OLD_LATENT_SOFT, OLD_RULE_SOFT,
        gate_action, gate_mode, is_helped,
    )

    if (GATE_P, GATE_DX, HELP_AT, OLD_RULE_SOFT, OLD_LATENT_SOFT, OLD_GAP) != (
        0.7, 12.0, 80, 180, 62, 118,
    ):
        raise SystemExit("константы ворот сдвинуты")
    if NAV_TOL != 6.0 or GROUND != 512 or PAD_HALF != 8.0 or VY_SOFT != 8.0:
        raise SystemExit("учитель или земля сдвинуты")
    if is_helped(79) or not is_helped(80):
        raise SystemExit("порог помощи: 79 ложь, 80 истина")
    if OLD_RULE_SOFT - 80 >= OLD_GAP:
        raise SystemExit("восемьдесят мягких не укорачивают разрыв")

    if gate_mode(0.7, 12) != "land" or gate_mode(0.7, -12) != "land":
        raise SystemExit("граница 0.7 и 12 должна быть посадкой")
    if gate_mode(0.7, 12.1) != "approach" or gate_mode(0.7, -12.1) != "approach":
        raise SystemExit("12.1 — наведение")
    if gate_mode(0.699, 0) != "approach" or gate_mode(0.699, 12) != "approach":
        raise SystemExit("ниже 0.7 — наведение")

    g, pad, vt = 8.0, 256.0, 6.0
    near = (250.0, 400.0, 0.0, 0.0, 10.0)
    far = (249.9, 400.0, 0.0, 0.0, 10.0)
    if hover_mode(near, g, pad, vt) != "land":
        raise SystemExit("|dx| = 6 должно оставаться посадкой учителя")
    if hover_mode(far, g, pad, vt) != "approach":
        raise SystemExit("|dx| = 6.1 должно оставаться наведением учителя")

    wide = (246.0, 400.0, 0.0, 0.0, 10.0)
    if abs(pad - wide[0]) != 10.0:
        raise SystemExit("тест |dx| = 10 собран неверно")
    if hover_mode(wide, g, pad, vt) != "approach":
        raise SystemExit("на |dx| = 10 hover_mode остаётся наведением")
    land_branch = hover_action(wide, g, pad, vt, mode="land")
    approach_branch = hover_action(wide, g, pad, vt, mode="approach")
    if land_branch == approach_branch:
        raise SystemExit("ветки на |dx| = 10 совпали, проверка их не различает")
    if gate_action(wide, g, pad, 0.7) != land_branch:
        raise SystemExit("ворота на |dx| = 10 должны звать ветку посадки")
    if gate_action(wide, g, pad, 0.699) != approach_branch:
        raise SystemExit("ниже 0.7 на |dx| = 10 остаётся ветка наведения")

    edge = (pad - 12.0, 400.0, 0.0, 0.0, 10.0)
    over = (pad - 12.1, 400.0, 0.0, 0.0, 10.0)
    if abs(pad - edge[0]) != 12.0 or not (abs(pad - over[0]) > 12.0):
        raise SystemExit("тест границы запаса собран неверно")
    if hover_mode(edge, g, pad, vt) != "approach":
        raise SystemExit("запас 12 не должен менять hover_mode")
    if gate_action(edge, g, pad, 0.7) != hover_action(edge, g, pad, vt, mode="land"):
        raise SystemExit("на |dx| = 12 и p = 0.7 исполняется ветка посадки")
    if gate_action(over, g, pad, 0.7) != hover_action(over, g, pad, vt, mode="approach"):
        raise SystemExit("на |dx| = 12.1 исполняется наведение")

    for state in (
        (256.0, 512.0, 0.0, 0.0, 10.0),
        (256.0, 520.0, 0.0, 0.0, 10.0),
        (256.0, 400.0, 0.0, 0.0, 0.0),
        (256.0, 400.0, 0.0, 0.0, -1.0),
    ):
        if hover_mode(state, g, pad, vt) is not None:
            raise SystemExit(f"ждали пустой режим: {state}")
        if gate_action(state, g, pad, 1.0) != 1:
            raise SystemExit(f"пустой режим не заглушил голову: {state}")

    gate_path = os.path.join(root, "numeric_land", "rollout_mode_gate.py")
    source = open(gate_path, encoding="utf-8").read()
    lowered = source.lower()
    if "torch.save" in lowered or "cem" in lowered or "train_mode" in lowered:
        raise SystemExit("прогон ворот тянет запись, планировщик или обучение")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "plan" in node.module.split("."):
                raise SystemExit("прогон ворот импортирует план")

    for path, stamp in before.items():
        if _stamp(path) != stamp:
            raise SystemExit(f"контроль записал {os.path.basename(path)}")
    print("gate check ok", flush=True)


if __name__ == "__main__":
    main()

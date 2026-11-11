"""Плоский мир не должен уехать от эталона.

    python _check_flat.py

Файл tests/flat_world_seed0.json снят до правки физики. Допуск 1e-4.
Расхождение любой точки — ненулевой код выхода. Pytest не нужен.
"""
import json
import os
import sys

from data import World

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "tests", "flat_world_seed0.json")


def main():
    with open(PATH, encoding="utf-8") as f:
        doc = json.load(f)
    tol = float(doc.get("tol", 1e-4))
    actions = [int(a) for a in doc["actions"]]
    want = doc["positions"]
    if len(want) != len(actions) + 1:
        print(f"в эталоне {len(want)} точек на {len(actions)} толчков")
        return 1
    world = World(int(doc["seed"]), world="flat")
    got = [world.pos.astype(float)]
    for action in actions:
        world.step(action)
        got.append(world.pos.astype(float))
    bad = 0
    for i, (g, w) in enumerate(zip(got, want)):
        err = max(abs(float(g[0]) - float(w[0])), abs(float(g[1]) - float(w[1])))
        if err > tol:
            bad += 1
            print(f"точка {i}: ошибка {err:.3e} > {tol:.1e}  было {w}  стало {[float(g[0]), float(g[1])]}")
    if bad:
        print(f"плоский эталон не сошёлся: {bad} точек")
        return 1
    print(f"плоский эталон совпал: сид {doc['seed']}, {len(actions)} толчков, допуск {tol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

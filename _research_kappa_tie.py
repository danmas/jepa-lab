"""Тай-брейк между 0.7 и 0.6: 16 сидов на трёх клетках. Замер, ничего не меняет."""
import json
import os

import numpy as np
import torch

from numeric_land.relenc import OUT, write_json
from numeric_land.skill_assemble import load_bundle, take_stamps
from numeric_land.skill_land import pair_table
from _research_kappa import fly

GAINS = (0.7, 0.6, 0.5)
CELLS = (("15/406", 15.0, 406.0), ("20/356", 20.0, 356.0), ("20/406", 20.0, 406.0))
SEEDS = range(210_000, 210_016)


def main():
    torch.set_num_threads(8)
    before = take_stamps()
    bundle = load_bundle()
    actions, step_pairs = pair_table()
    report = []
    for gain in GAINS:
        for name, g, pad in CELLS:
            eps = [fly(seed, g, pad, gain, bundle, actions, step_pairs) for seed in SEEDS]
            stats = {"gain": gain, "cell": name, "n": len(eps),
                     "soft": sum(1 for e in eps if e["soft"]),
                     "dry": sum(1 for e in eps if e["dry"]),
                     "steps_median": float(np.median([e["steps"] for e in eps])),
                     "drift_median": round(float(np.median([e["drift"] for e in eps])), 1),
                     "vx_approach": round(float(np.mean([e["vx_approach"] for e in eps
                                                         if e["vx_approach"] is not None])), 2)}
            report.append(stats)
            print(json.dumps(stats, ensure_ascii=False), flush=True)
    if take_stamps() != before:
        raise SystemExit("замер сдвинул штамп чекпоинта")
    path = os.path.join(OUT, "research_kappa_tie.json")
    if os.path.isfile(path):
        raise SystemExit(f"уже есть {path}")
    write_json(path, {"gains": list(GAINS), "seeds": [SEEDS[0], SEEDS[-1]],
                      "report": report, "heads_trained": False, "stamps_unchanged": True})
    print(path, flush=True)


if __name__ == "__main__":
    main()

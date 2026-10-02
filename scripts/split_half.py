# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Two-fold cross-validated tuning against the ISLES 2018 RAPID maps.

Cases are split at random into halves A and B. The parameter grid is scored on A, the best
configuration (highest mean Dice for Tmax > 6 s plus core) is evaluated on B, and vice versa.
The two held-out evaluations are what should be quoted; the in-sample numbers are not.

usage: python scripts/split_half.py ROOT OUT_DIR [--seed 0]
"""
from __future__ import annotations

import sys, json, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_isles2018 import run_config
from openperfusion.io_isles2018 import find_cases
from openperfusion.pipeline import PipelineConfig
from openperfusion.validate import summarize_cases


def grid():
    engines = [dict(method="bcsvd", lam=0.05), dict(method="bcsvd", lam=0.10), dict(method="bcsvd", lam=0.15),
               dict(method="osvd", oi_threshold=0.05)]
    for eng, sig in itertools.product(engines, [1.5, 2.0, 2.5]):
        yield {**eng, "spatial_sigma_mm": sig}


def score(summ):
    return summ["hypo_dice"] + summ["core_dice"]


def main():
    root, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    cases = find_cases(root)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cases))
    halves = {"A": [cases[i] for i in sorted(perm[: len(cases) // 2])], "B": [cases[i] for i in sorted(perm[len(cases) // 2:])]}
    json.dump({k: [c.name for c in v] for k, v in halves.items()}, open(out / "split.json", "w"), indent=1)
    tuning_rows, held_out = [], []
    for tune, test in (("A", "B"), ("B", "A")):
        best, best_s = None, -1
        for cfg in grid():
            t0 = time.time()
            summ, _ = run_config(halves[tune], PipelineConfig(**cfg))
            row = {"tune_half": tune, "config": json.dumps(cfg), "score": score(summ), **summ, "seconds": round(time.time() - t0, 1)}
            tuning_rows.append(row)
            pd.DataFrame(tuning_rows).to_csv(out / "tuning.csv", index=False)
            print(f"[tune {tune}] {cfg} score={row['score']:.3f} hypo_dice={summ['hypo_dice']:.3f} core_dice={summ['core_dice']:.3f}", flush=True)
            if row["score"] > best_s:
                best, best_s = cfg, row["score"]
        summ, df = run_config(halves[test], PipelineConfig(**best))
        df.to_csv(out / f"heldout_{test}_cases.csv", index=False)
        held_out.append({"test_half": test, "config": json.dumps(best), **summ})
        pd.DataFrame(held_out).to_csv(out / "heldout.csv", index=False)
        print(f"[held-out {test}] best={best} -> " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if not isinstance(v, dict)}), flush=True)
    # pooled held-out (each case scored exactly once, by a config it did not tune)
    pooled = pd.concat([pd.read_csv(out / f"heldout_{h}_cases.csv") for h in ("A", "B")])
    pooled.to_csv(out / "heldout_pooled_cases.csv", index=False)
    ps = summarize_cases(pooled.to_dict("records"))
    json.dump(ps, open(out / "heldout_pooled_summary.json", "w"), indent=2, default=str)
    print("[pooled held-out] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ps.items()}, default=str), flush=True)


if __name__ == "__main__":
    main()

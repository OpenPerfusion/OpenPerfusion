# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Run the three best 11-case configurations on all staged ISLES 2018 cases."""
import sys, json, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_isles2018 import run_config
from openperfusion.io_isles2018 import find_cases, load_case
from openperfusion.pipeline import PipelineConfig

root, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
configs = [dict(method="bcsvd", spatial_sigma_mm=2.0, lam=0.10),
           dict(method="osvd", spatial_sigma_mm=2.0, oi_threshold=0.05),
           dict(method="osvd", spatial_sigma_mm=2.0, oi_threshold=0.095)]
cases = find_cases(root)          # paths; loaded one at a time inside run_config
print(len(cases), "cases", flush=True)
rows = []
for cfg in configs:
    t0 = time.time()
    summ, df = run_config(cases, PipelineConfig(**cfg))
    tag = "_".join(f"{k}{v}" for k, v in cfg.items())
    df.to_csv(out / f"cases_{tag}.csv", index=False)
    summ = {"config": json.dumps(cfg), **summ, "seconds": round(time.time() - t0, 1)}
    rows.append(summ)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)

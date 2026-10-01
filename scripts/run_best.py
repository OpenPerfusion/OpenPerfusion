# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
import sys, json, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_isles2018 import run_config
from openperfusion.io_isles2018 import find_cases
from openperfusion.pipeline import PipelineConfig
root, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cfg = dict(method="bcsvd", spatial_sigma=2.0, lam=0.10)
t0 = time.time(); summ, df = run_config(find_cases(root), PipelineConfig(**cfg))
df.to_csv(out / "cases_best.csv", index=False)
json.dump({"config": cfg, **summ, "seconds": round(time.time() - t0, 1)}, open(out / "summary_best.json", "w"), indent=2, default=str)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}))

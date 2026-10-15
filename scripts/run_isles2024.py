# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Run the ISLES 2018 agreement harness on ISLES'24 subjects (icobrain cva reference maps, Siemens / Philips).

    python scripts/run_isles2024.py /path/to/isles24_unpacked results/isles24 [--max-cases N]
    python scripts/run_isles2024.py /path/to/isles24/data results/isles24 --parquet      # straight from the shards
"""
import sys, json, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sweep_isles2018 import run_config
from openperfusion.io_isles2024 import find_subjects, load_subject, find_shards, load_subject_from_parquet
from openperfusion.pipeline import PipelineConfig
root, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
parquet = "--parquet" in sys.argv
subs = find_shards(root) if parquet else find_subjects(root)
loader = load_subject_from_parquet if parquet else load_subject
if "--max-cases" in sys.argv:
    subs = subs[: int(sys.argv[sys.argv.index("--max-cases") + 1])]
cfg = {}
t0 = time.time(); summ, df = run_config(subs, PipelineConfig(**cfg), loader=loader)
df.to_csv(out / "cases.csv", index=False)
json.dump({"config": cfg, **summ, "seconds": round(time.time() - t0, 1)}, open(out / "summary.json", "w"), indent=2, default=str)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}))

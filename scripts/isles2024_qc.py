# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""QC-only pass over ISLES'24 parquet shards: load each subject without registration, run the pipeline, and
record the AIF QC flags per subject. Fast (no registration, no reference comparison), resumable, so it can be
run in short time slices. Newer `run_config` rows carry the same flags; this fills them in for runs made before.

    python scripts/isles2024_qc.py /path/to/isles24/data results/isles24_50 [--budget SECONDS]

Writes <out>/qc.csv; `scripts/isles2024_summary.py` merges it into cases.csv.
"""
import sys, time
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.io_isles2024 import find_shards, load_subject_from_parquet
from openperfusion.pipeline import PipelineConfig, run_pipeline
from openperfusion.aif import second_peak_fraction

FLAGS = ("aif_primary_weak", "aif_fallback", "aif_last_resort", "aif_weak", "truncated", "aif_reselected_lead", "aif_reselected_vein")


def main(data: Path, out: Path, budget: float):
    out.mkdir(parents=True, exist_ok=True); f = out / "qc.csv"; t0 = time.time()
    done = set(pd.read_csv(f).case) if f.exists() else set()
    todo = []
    for sh in find_shards(data):
        try:
            sid = pq.read_table(str(sh), columns=["subject_id"]).column("subject_id")[0].as_py()
        except Exception:
            continue                                   # shard still downloading / incomplete
        if sid not in done:
            todo.append((sid, sh))
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    for sid, sh in todo:
        if time.time() - t0 > budget:
            break
        try:
            d = load_subject_from_parquet(sh, load_4d=True, register=False)
            res = run_pipeline(d["ctp4d"], d["dt"], d["voxel_size"], PipelineConfig())
            q = res.aif.qc
            row = {"case": sid, "shape": str(d["ctp4d"].shape), **{k: bool(q.get(k, False)) for k in FLAGS},
                   "aif_peak": q.get("aif_peak"), "n_aif_voxels": q.get("n_aif_voxels"), "aif_ttp": q.get("aif_ttp"),
                   "tissue_ttp": q.get("tissue_ttp"), "aif_lead": q.get("aif_lead"), "aif_second_peak": float(second_peak_fraction(res.aif.aif_raw[None])[0]),
                   "seconds": res.qc.get("seconds")}
            pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(f"{sid}: peak {row['aif_peak']:.0f} HU, ttp {row['aif_ttp']:.1f}/{row['tissue_ttp']:.1f} s, "
                  f"second-peak {row['aif_second_peak']:.2f} {[k for k in FLAGS if row[k]]}", flush=True)
        except Exception as e:
            print(f"{sid}: failed: {e}", flush=True)
    print(f"elapsed {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), float(a[a.index("--budget") + 1]) if "--budget" in a else 1e9)

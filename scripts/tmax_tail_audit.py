# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Where do OpenPerfusion and the reference disagree at the Tmax > 6 s boundary, and what is that tissue like?

Per case: Tmax percentiles (ours vs reference) in tissue neither calls hypoperfused; volumes called only by us
and only by the reference; and the CBV, enhancement SNR and reference Tmax of the only-ours voxels. Used to
diagnose the bilateral white-matter over-calls seen on ISLES'24.

    python scripts/tmax_tail_audit.py isles2018 /path/to/TRAINING results/audit/isles2018.csv [--sigma 1.5] [--max N]
    python scripts/tmax_tail_audit.py isles2024 /path/to/shards results/audit/isles2024.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.maps import threshold_maps


def audit(d: dict, cfg: PipelineConfig) -> dict:
    res = run_pipeline(d["ctp4d"], d["dt"], d["voxel_size"], cfg)
    vox_ml = float(np.prod(d["voxel_size"])) / 1000
    valid = res.mask & np.isfinite(d["tmax"]) & (d["cbf"] > 0)
    t_o = threshold_maps(res.maps, d["voxel_size"], baseline=res.baseline, reference=cfg.reference,
                         min_cluster_ml=cfg.min_cluster_ml, restrict_core_to_hypo=cfg.restrict_core_to_hypo)
    ours6 = t_o.hypo & valid; ref6 = (d["tmax"] > 6) & valid
    oo = ours6 & ~ref6; orf = ref6 & ~ours6; normal = valid & ~ours6 & ~ref6
    hu = d["ctp4d"]; nb = res.n_baseline
    base = hu[..., :nb].mean(-1); noise = np.maximum(hu[..., :nb].std(-1), 1.0); peak = (hu - base[..., None]).max(-1)
    q = lambda a, m, p: float(np.percentile(a[m], p)) if m.any() else np.nan
    return {"case": d["case"], "slices": int(ours6.shape[2]), "normal_ml": float(normal.sum() * vox_ml),
            "ours_ml": float(ours6.sum() * vox_ml), "ref_ml": float(ref6.sum() * vox_ml),
            "only_ours_ml": float(oo.sum() * vox_ml), "only_ref_ml": float(orf.sum() * vox_ml),
            "tmax_ours_p50_normal": q(res.maps.tmax, normal, 50), "tmax_ref_p50_normal": q(d["tmax"], normal, 50),
            "tmax_ours_p90_normal": q(res.maps.tmax, normal, 90), "tmax_ref_p90_normal": q(d["tmax"], normal, 90),
            "tmax_ours_p99_normal": q(res.maps.tmax, normal, 99), "tmax_ref_p99_normal": q(d["tmax"], normal, 99),
            "oo_tmax_ours_p50": q(res.maps.tmax, oo, 50), "oo_tmax_ref_p50": q(d["tmax"], oo, 50), "oo_tmax_ours_p90": q(res.maps.tmax, oo, 90),
            "oo_cbv_p50": q(res.maps.cbv, oo, 50), "normal_cbv_p50": q(res.maps.cbv, normal, 50),
            "oo_snr_p50": q(peak / noise, oo, 50), "normal_snr_p50": q(peak / noise, normal, 50),
            "aif_peak": float(res.aif.aif_raw.max()), "seconds": res.qc["seconds"]}


def main(kind: str, root: str, out: Path, sigma: float, max_n: int, budget: float = 1e9, skip: set = frozenset()):
    if kind == "isles2018":
        from openperfusion.io_isles2018 import find_cases, load_case
        items = find_cases(root); loader = load_case
    else:
        from openperfusion.io_isles2024 import find_shards, load_subject_from_parquet
        items = find_shards(root); loader = load_subject_from_parquet
    cfg = PipelineConfig(spatial_sigma_mm=sigma)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = (set(pd.read_csv(out).case) if out.exists() else set()) | set(skip)   # resumable: one row appended per case
    t0 = time.time()
    for it in items[:max_n]:
        if time.time() - t0 > budget:
            break
        if kind == "isles2018" and Path(it).name in done:
            continue
        if kind != "isles2018":
            import pyarrow.parquet as pq
            if pq.read_table(str(it), columns=["subject_id"]).column("subject_id")[0].as_py() in done:
                continue
        try:
            d = loader(it)
            r = audit(d, cfg)
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{r['case']}: only ours {r['only_ours_ml']:.0f} mL, only ref {r['only_ref_ml']:.0f} mL; normal p90 ours {r['tmax_ours_p90_normal']:.1f} vs ref {r['tmax_ref_p90_normal']:.1f}", flush=True)
        except Exception as e:
            print(f"{it}: failed: {e}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], Path(a[2]), float(a[a.index("--sigma") + 1]) if "--sigma" in a else 1.5, int(a[a.index("--max") + 1]) if "--max" in a else 10**6,
         float(a[a.index("--budget") + 1]) if "--budget" in a else 1e9,
         set(a[a.index("--skip") + 1].split(",")) if "--skip" in a else frozenset())

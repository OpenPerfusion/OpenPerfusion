"""Parameter sweep against the ISLES 2018 RAPID maps.

For each configuration, run every staged case and report agreement with RAPID for the
Tmax > 6 s and rCBF < 30 % regions, plus the Dice of (our Tmax > t) vs (RAPID Tmax > 6)
for t = 4..10 to find the threshold on our maps that best matches RAPID's 6 s boundary.

usage: python scripts/sweep_isles2018.py ROOT OUT.csv [--max-cases N]
"""
from __future__ import annotations

import sys, json, itertools, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.io_isles2018 import find_cases, load_case
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.maps import threshold_maps, PerfusionMaps
from openperfusion.validate import dice, icc21, bland_altman, compare_to_reference, summarize_cases


def run_config(cases, cfg: PipelineConfig, tmax_grid=(4, 5, 6, 7, 8, 9, 10), core_grid=(0.25, 0.30, 0.35, 0.40)):
    rows = []
    for d in cases:
        if not isinstance(d, dict):            # a path: load lazily, one case in memory at a time
            try:
                d = load_case(d)
            except Exception as e:
                print(f"{d}: load failed: {e}", flush=True); continue
            if d["ctp4d"] is None:
                continue
        try:
            res = run_pipeline(d["ctp4d"], d["dt"], d["voxel_size"], cfg)
        except Exception as e:
            print(f"{d['case']}: pipeline failed: {e}", flush=True); continue
        ref = {k: d[k] for k in ("cbf", "cbv", "mtt", "tmax") if d.get(k) is not None}
        cmp = compare_to_reference(res.maps, ref, d["voxel_size"], baseline=res.baseline, lesion=d.get("lesion"),
                                   reference=cfg.reference, min_cluster_ml=cfg.min_cluster_ml,
                                   restrict_core_to_hypo=cfg.restrict_core_to_hypo)
        # threshold matching: RAPID's own Tmax>6 and rCBF<30 regions vs ours at other cut-offs
        valid = res.mask & np.isfinite(ref["tmax"]) & (ref["cbf"] > 0)
        ref_maps = PerfusionMaps(cbf=ref["cbf"], cbv=ref["cbv"], mtt=ref["mtt"], tmax=ref["tmax"], ttp=res.maps.ttp,
                                 mask=valid, dt=d["dt"], method="reference")
        ours_v = PerfusionMaps(cbf=res.maps.cbf, cbv=res.maps.cbv, mtt=res.maps.mtt, tmax=res.maps.tmax, ttp=res.maps.ttp,
                               mask=valid, dt=d["dt"], method=cfg.method)
        t_ref = threshold_maps(ref_maps, d["voxel_size"], baseline=res.baseline, reference=cfg.reference,
                               min_cluster_ml=cfg.min_cluster_ml, restrict_core_to_hypo=cfg.restrict_core_to_hypo)
        vox_ml = float(np.prod(d["voxel_size"])) / 1000
        extra = {}
        for t in tmax_grid:
            t_o = threshold_maps(ours_v, d["voxel_size"], baseline=res.baseline, tmax_thr=t, reference=cfg.reference,
                                 min_cluster_ml=cfg.min_cluster_ml, restrict_core_to_hypo=cfg.restrict_core_to_hypo)
            extra[f"dice_hypo_t{t}"] = dice(t_o.hypo, t_ref.hypo)
            extra[f"hypo_ml_t{t}"] = t_o.hypo_ml
        for cf in core_grid:
            t_o = threshold_maps(ours_v, d["voxel_size"], baseline=res.baseline, core_frac=cf, reference=cfg.reference,
                                 min_cluster_ml=cfg.min_cluster_ml, restrict_core_to_hypo=cfg.restrict_core_to_hypo)
            extra[f"dice_core_c{int(cf*100)}"] = dice(t_o.core, t_ref.core)
            extra[f"core_ml_c{int(cf*100)}"] = t_o.core_ml
        rows.append({"case": d["case"], "shape": str(d["ctp4d"].shape), **cmp, **extra, "aif_n": res.aif.qc.get("n_aif_voxels"),
                     "aif_peak": res.aif.qc.get("aif_peak"), "k_av": res.aif.k_av, "n_baseline": res.n_baseline,
                     "seconds": res.qc["seconds"]})
        print(f"{d['case']}: r_tmax={cmp.get('r_tmax', float('nan')):.2f} hypo {cmp['hypo_ml_ours']:.0f}/{cmp['hypo_ml_ref']:.0f} dice {cmp['dice_hypo']:.2f} core {cmp['core_ml_ours']:.0f}/{cmp['core_ml_ref']:.0f} {res.qc['seconds']}s", flush=True)
        del d
    df = pd.DataFrame(rows)
    summ = summarize_cases(rows)
    out = {"n": len(df), "r_cbf": df.r_cbf.mean(), "r_cbv": df.r_cbv.mean(), "r_tmax": df.r_tmax.mean(), "sp_tmax": df.spearman_tmax.mean(),
           "hypo_dice": df.dice_hypo.mean(), "hypo_icc": summ["hypo_icc"], "hypo_bias": summ["hypo_ba"]["bias"],
           "core_dice": df.dice_core.mean(), "core_icc": summ["core_icc"], "core_bias": summ["core_ba"]["bias"],
           "defuse3_kappa": summ.get("defuse3_kappa")}
    for t in tmax_grid:
        out[f"hypo_dice_t{t}"] = df[f"dice_hypo_t{t}"].mean()
        out[f"hypo_bias_t{t}"] = (df[f"hypo_ml_t{t}"] - df["hypo_ml_ref"]).mean()
    for cf in core_grid:
        out[f"core_dice_c{int(cf*100)}"] = df[f"dice_core_c{int(cf*100)}"].mean()
    return out, df


def main():
    root = sys.argv[1]; out_csv = sys.argv[2]
    max_cases = int(sys.argv[sys.argv.index("--max-cases") + 1]) if "--max-cases" in sys.argv else 0
    cases = find_cases(root)
    if max_cases:
        cases = cases[:max_cases]
    print(f"{len(cases)} cases loaded")
    configs = []
    for method, sig in itertools.product(["osvd", "bcsvd", "fourier"], [1.0, 2.0, 3.0]):
        configs.append(dict(method=method, spatial_sigma=sig))
    configs += [dict(method="osvd", spatial_sigma=2.0, temporal_sigma=1.0),
                dict(method="osvd", spatial_sigma=2.0, oi_threshold=0.05),
                dict(method="osvd", spatial_sigma=2.0, oi_threshold=0.15),
                dict(method="bcsvd", spatial_sigma=2.0, lam=0.10),
                dict(method="bcsvd", spatial_sigma=2.0, lam=0.25),
                dict(method="osvd", spatial_sigma=2.0, reference="global_median"),
                dict(method="osvd", spatial_sigma=2.0, min_cluster_ml=3.0),
                dict(method="osvd", spatial_sigma=2.0, restrict_core_to_hypo=True)]
    results = []
    for c in configs:
        t0 = time.time()
        summ, df = run_config(cases, PipelineConfig(**c))
        summ = {"config": json.dumps(c), **summ, "seconds": round(time.time() - t0, 1)}
        results.append(summ)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}))
        pd.DataFrame(results).to_csv(out_csv, index=False)
        df.to_csv(Path(out_csv).with_suffix("") .as_posix() + "_" + "_".join(f"{k}{v}" for k, v in c.items()) + ".csv", index=False)


if __name__ == "__main__":
    main()

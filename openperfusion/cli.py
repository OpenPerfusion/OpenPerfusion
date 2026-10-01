# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Command line: `openperfusion phantom` and `openperfusion isles2018 <root>`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .phantom import make_phantom
from .pipeline import run_pipeline, PipelineConfig
from .validate import phantom_report, compare_to_reference, summarize_cases
from .report import phantom_figure, case_figure
from .maps import threshold_maps, PerfusionMaps


def _cfg_from_args(a) -> PipelineConfig:
    return PipelineConfig(method=a.method, spatial_sigma=a.spatial_sigma, temporal_sigma=a.temporal_sigma,
                          oi_threshold=a.oi, lam=a.lam, pr=a.pr, reference=a.reference,
                          min_cluster_ml=a.min_cluster_ml, restrict_core_to_hypo=a.restrict_core)


def _add_common(p):
    p.add_argument("--method", default="bcsvd", choices=["bcsvd", "osvd", "fourier"])
    p.add_argument("--spatial-sigma", type=float, default=2.0)
    p.add_argument("--temporal-sigma", type=float, default=0.0)
    p.add_argument("--oi", type=float, default=0.095, help="oSVD oscillation-index target")
    p.add_argument("--lam", type=float, default=0.10, help="bcSVD truncation fraction")
    p.add_argument("--pr", type=float, default=0.15, help="Fourier regularisation (Straka 2010)")
    p.add_argument("--reference", default="contralateral", choices=["contralateral", "global_median"])
    p.add_argument("--min-cluster-ml", type=float, default=1.0)
    p.add_argument("--restrict-core", action="store_true", help="core only inside Tmax>6 region")
    p.add_argument("--out", default="results")


def cmd_phantom(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hu, truth = make_phantom(noise_sd=a.noise, residue=a.residue, seed=a.seed)
    res = run_pipeline(hu, truth.dt, truth.voxel_size, _cfg_from_args(a))
    rep = phantom_report(res.maps, truth)
    rep["table"].to_csv(out / "phantom_tiles.csv", index=False)
    summary = {**rep["summary"], "aif_in_true_aif": float(truth.aif_mask[res.aif.aif_mask].mean()),
               "vof_in_true_vof": float(truth.vof_mask[res.aif.vof_mask].mean()) if res.aif.vof_mask is not None else None,
               "k_av_est": res.aif.k_av, "k_av_true": 1 / 0.7, "qc": res.qc,
               "core_ml": res.thresholds.core_ml, "hypo_ml": res.thresholds.hypo_ml}
    (out / "phantom_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    phantom_figure(res, truth, z=hu.shape[2] - 1, path=str(out / "phantom_maps.png"))
    print(json.dumps({k: v for k, v in summary.items() if k != "qc"}, indent=2, default=str))
    print(f"wrote {out}/phantom_tiles.csv, phantom_summary.json, phantom_maps.png")


def cmd_isles2018(a):
    from .io_isles2018 import find_cases, load_case
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = find_cases(a.root)
    if a.max_cases:
        cases = cases[: a.max_cases]
    rows = []
    for case in cases:
        d = load_case(case)
        if d["ctp4d"] is None:
            print(f"{case.name}: no 4D CTP, skipped"); continue
        cfg = _cfg_from_args(a)
        res = run_pipeline(d["ctp4d"], d["dt"], d["voxel_size"], cfg)
        ref = {k: d[k] for k in ("cbf", "cbv", "mtt", "tmax") if d.get(k) is not None}
        cmp = compare_to_reference(res.maps, ref, d["voxel_size"], baseline=res.baseline, lesion=d.get("lesion"),
                                   reference=cfg.reference, min_cluster_ml=cfg.min_cluster_ml,
                                   restrict_core_to_hypo=cfg.restrict_core_to_hypo)
        row = {"case": case.name, "dt": d["dt"], "shape": str(d["ctp4d"].shape), **cmp,
               "seconds": res.qc["seconds"], "k_av": res.aif.k_av, "n_baseline": res.n_baseline,
               "truncated": res.aif.qc.get("truncated")}
        rows.append(row)
        print(f"{case.name}: r_cbf={cmp.get('r_cbf', float('nan')):.2f} r_tmax={cmp.get('r_tmax', float('nan')):.2f} "
              f"core {cmp['core_ml_ours']:.0f}/{cmp['core_ml_ref']:.0f} mL (dice {cmp['dice_core']:.2f}) "
              f"Tmax>6 {cmp['hypo_ml_ours']:.0f}/{cmp['hypo_ml_ref']:.0f} mL (dice {cmp['dice_hypo']:.2f}) {res.qc['seconds']}s")
        if a.figures:
            valid = res.mask & np.isfinite(ref.get("tmax", res.maps.tmax)) & (ref.get("cbf", res.maps.cbf) > 0)
            ref_maps = PerfusionMaps(cbf=ref.get("cbf", res.maps.cbf), cbv=ref.get("cbv", res.maps.cbv), mtt=ref.get("mtt", res.maps.mtt),
                                     tmax=ref.get("tmax", res.maps.tmax), ttp=res.maps.ttp, mask=valid, dt=d["dt"], method="reference")
            t_ref = threshold_maps(ref_maps, d["voxel_size"], baseline=res.baseline, reference=cfg.reference,
                                   min_cluster_ml=cfg.min_cluster_ml, restrict_core_to_hypo=cfg.restrict_core_to_hypo)
            z = int(np.argmax(res.thresholds.hypo.sum(axis=(0, 1)))) if res.thresholds.hypo.any() else d["ctp4d"].shape[2] // 2
            case_figure(res.maps, ref, res.thresholds, t_ref, z, str(out / f"{case.name}.png"), lesion=d.get("lesion"), title=case.name)
    df = pd.DataFrame(rows)
    df.to_csv(out / "isles2018_cases.csv", index=False)
    summ = summarize_cases(rows) if rows else {}
    (out / "isles2018_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))


def cmd_dicom(a):
    """Run the pipeline on a DICOM 4D CTP directory (vendor export), optional motion correction."""
    from .io_dicom import read_ctp_dicom
    from .io_isles2018 import save_nifti
    import nibabel as nib
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = read_ctp_dicom(a.directory, dt=a.dt)
    print(f"loaded {c.n_files} images: {c.hu.shape} voxel {tuple(round(v, 2) for v in c.voxel_size)} dt {c.dt}s shuttle={c.meta['shuttle_mode']}")
    hu = c.hu
    if a.motion:
        from .motion import motion_correct
        mr = motion_correct(hu, voxel_size=c.voxel_size)
        hu = mr.hu
        print(f"motion correction: max in-plane shift {mr.max_shift_mm:.1f} mm, max rotation {abs(mr.theta).max():.1f} deg")
    res = run_pipeline(hu, c.dt, c.voxel_size, _cfg_from_args(a))
    t = res.thresholds
    aff = np.diag([c.voxel_size[0], c.voxel_size[1], c.voxel_size[2], 1.0])
    for k in ("cbf", "cbv", "mtt", "tmax", "ttp"):
        save_nifti(getattr(res.maps, k), aff, out / f"{k}.nii.gz")
    save_nifti(t.core.astype(np.uint8), aff, out / "core_rcbf30.nii.gz")
    save_nifti(t.hypo.astype(np.uint8), aff, out / "hypoperfusion_tmax6.nii.gz")
    summary = {"core_ml": t.core_ml, "hypoperfusion_ml": t.hypo_ml, "mismatch_ml": t.mismatch_ml, "mismatch_ratio": t.mismatch_ratio,
               "tmax_ml": t.tmax_ml, "hir": t.hir, "defuse3_target_profile": t.defuse3_target, "qc": res.qc, "dicom": c.meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("qc", "dicom")}, indent=2, default=str))
    if res.aif.qc.get("aif_weak"):
        print("WARNING: weak arterial input function — maps unreliable; check the AIF location")


def main(argv=None):
    p = argparse.ArgumentParser(prog="openperfusion", description="Open-source CT perfusion (research use only).")
    sub = p.add_subparsers(dest="cmd", required=True)
    ph = sub.add_parser("phantom", help="run the pipeline on the digital phantom and report recovery")
    ph.add_argument("--noise", type=float, default=3.0, help="HU noise sd")
    ph.add_argument("--residue", default="exp", choices=["exp", "box"])
    ph.add_argument("--seed", type=int, default=0)
    _add_common(ph); ph.set_defaults(func=cmd_phantom)
    isl = sub.add_parser("isles2018", help="run on ISLES 2018 cases and compare with the RAPID maps")
    isl.add_argument("root"); isl.add_argument("--max-cases", type=int, default=0); isl.add_argument("--figures", action="store_true")
    _add_common(isl); isl.set_defaults(func=cmd_isles2018)
    dc = sub.add_parser("dicom", help="run on a directory of DICOM CTP images and write NIfTI maps + summary.json")
    dc.add_argument("directory"); dc.add_argument("--dt", type=float, default=1.0, help="uniform time grid (s)")
    dc.add_argument("--motion", action="store_true", help="rigid in-plane motion correction before processing")
    _add_common(dc); dc.set_defaults(func=cmd_dicom)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

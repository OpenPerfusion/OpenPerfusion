# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Frozen analysis for the paired OpenPerfusion-vs-RAPID cohort (the pre-specified analysis plan, as code).

The cohort is a directory of cases. Each case has the CTP export (a DICOM directory, or a NIfTI 4D file) and the
reference package's numbers; optionally its maps. A manifest CSV lists them:

    case,ctp,vendor,ref_core_ml,ref_tmax6_ml,ref_mismatch_ml,ref_mismatch_ratio,ref_hir,ref_defuse3,ref_tmax_map,ref_cbf_map,followup_infarct_ml,reperfused
    001,cases/001/dicom,Siemens,12.3,88.0,75.7,7.2,0.31,1,,,,

(`ctp`: path to the DICOM directory or .nii(.gz); `ref_*` columns from the RAPID report; `ref_tmax_map` /
`ref_cbf_map`: optional NIfTI maps on the CTP grid for voxelwise Dice; `followup_infarct_ml` and `reperfused`:
optional, for the clinical anchor; empty cells allowed.)

    python scripts/cohort_compare.py manifest.csv results/cohort [--motion] [--dt 1.0] [--max N]

What it does, in the order the analysis plan states it:
  1. runs every case with the frozen defaults of the tagged release (no per-case tuning; the config is written to
     the output so it can be checked against the tag), with the QC verdict recorded;
  2. primary endpoints: Tmax > 6 s and rCBF < 30 % volumes vs the reference, ICC(2,1) with bootstrap 95 % CI,
     Bland-Altman bias and limits with CIs; secondary: DEFUSE-3 target-profile agreement and kappa, mismatch
     ratio, HIR, voxelwise Dice where maps exist, per-vendor subgroups, follow-up infarct anchor;
  3. everything three ways: all cases, excluding `reject`, excluding `reject` and `review`;
  4. writes per-case rows (cases.csv), the summary (summary.json) and the agreement figure.
Only de-identified per-case numbers leave the machine; no images are written.
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.qc import qc_verdict
from openperfusion.validate import icc21, bland_altman, dice

FROZEN = PipelineConfig()          # the release defaults; any deviation is a protocol amendment


def bootstrap_ci(stat, a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(a); b = np.asarray(b); n = len(a); vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            vals.append(stat(a[i], b[i]))
        except Exception:
            pass
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))] if vals else [np.nan, np.nan]


def kappa(x, y):
    x = np.asarray(x, bool); y = np.asarray(y, bool); po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def load_ctp(path: str, dt: float, motion: bool):
    p = Path(path)
    if p.is_dir():
        from openperfusion.io_dicom import read_ctp_dicom
        c = read_ctp_dicom(p, dt=dt); hu, vox, meta, mr = c.hu, c.voxel_size, c.meta, None
        if motion:
            from openperfusion.motion import motion_correct
            mr = motion_correct(hu, voxel_size=vox); hu = mr.hu
        return hu, float(c.dt), vox, meta, mr
    import nibabel as nib
    img = nib.load(str(p)); z = img.header.get_zooms()
    return np.asarray(img.dataobj, np.float32), float(z[3]) if len(z) > 3 and z[3] > 0 else dt, tuple(float(v) for v in z[:3]), {}, None


def run_case(row, dt, motion):
    hu, dt_c, vox, meta, mr = load_ctp(row["ctp"], dt, motion)
    res = run_pipeline(hu, dt_c, vox, FROZEN); t = res.thresholds; v = qc_verdict(res, meta=meta, motion=mr)
    out = {"case": row["case"], "vendor": row.get("vendor"), "shape": str(hu.shape), "dt": dt_c, "status": v["status"], "warnings": " | ".join(v["warnings"]),
           "core_ml": t.core_ml, "tmax6_ml": t.hypo_ml, "mismatch_ml": t.mismatch_ml, "mismatch_ratio": t.mismatch_ratio, "hir": t.hir, "defuse3": int(t.defuse3_target),
           "tmax4_ml": t.tmax_ml[4], "tmax8_ml": t.tmax_ml[8], "tmax10_ml": t.tmax_ml[10], "aif_peak": v["aif_peak_hu"],
           "motion_mm": float(getattr(mr, "max_shift_mm", np.nan)) if mr is not None else np.nan, "seconds": res.qc["seconds"]}
    for k in ("ref_core_ml", "ref_tmax6_ml", "ref_mismatch_ml", "ref_mismatch_ratio", "ref_hir", "ref_defuse3", "followup_infarct_ml", "reperfused"):
        out[k] = pd.to_numeric(row.get(k), errors="coerce")
    # voxelwise Dice if the reference maps are given on the CTP grid
    if isinstance(row.get("ref_tmax_map"), str) and row["ref_tmax_map"]:
        import nibabel as nib
        rt = np.asarray(nib.load(row["ref_tmax_map"]).dataobj, np.float32)
        if rt.shape == t.hypo.shape:
            out["dice_tmax6"] = dice(t.hypo, (rt > 6) & t.tissue)
    return out


def block(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["ref_tmax6_ml", "ref_core_ml"])
    out = {"n": int(len(d))}
    if len(d) < 3:
        return out
    for ours, ref, name in (("tmax6_ml", "ref_tmax6_ml", "tmax6"), ("core_ml", "ref_core_ml", "core")):
        a, b = d[ours].values, d[ref].values
        ba = bland_altman(a, b)
        out[name] = {"icc": icc21(a, b), "icc_ci": bootstrap_ci(icc21, a, b), "bias": ba["bias"], "loa": [ba["loa_low"], ba["loa_high"]],
                     "bias_ci": bootstrap_ci(lambda x, y: bland_altman(x, y)["bias"], a, b),
                     "loa_low_ci": bootstrap_ci(lambda x, y: bland_altman(x, y)["loa_low"], a, b), "loa_high_ci": bootstrap_ci(lambda x, y: bland_altman(x, y)["loa_high"], a, b),
                     "within_10ml": float((np.abs(a - b) <= 10).mean()), "pearson": float(np.corrcoef(a, b)[0, 1])}
    dd = d.dropna(subset=["ref_defuse3"])
    if len(dd) >= 3:
        x, y = dd.defuse3.values.astype(bool), dd.ref_defuse3.values.astype(bool)
        out["defuse3"] = {"n": int(len(dd)), "agreement": float((x == y).mean()), "agreement_ci": bootstrap_ci(lambda p, q: (p == q).mean(), x, y),
                          "kappa": kappa(x, y), "kappa_ci": bootstrap_ci(kappa, x, y), "ours_positive": int(x.sum()), "ref_positive": int(y.sum())}
    if "dice_tmax6" in d and d.dice_tmax6.notna().any():
        out["dice_tmax6_mean"] = float(d.dice_tmax6.mean()); out["dice_tmax6_median"] = float(d.dice_tmax6.median())
    fu = d.dropna(subset=["followup_infarct_ml"])
    if len(fu) >= 3:
        rep = fu[fu.reperfused == 1] if "reperfused" in fu else fu
        if len(rep) >= 3:
            out["followup_anchor_reperfused"] = {"n": int(len(rep)), "ours_core_vs_infarct_icc": icc21(rep.core_ml.values, rep.followup_infarct_ml.values),
                                                 "ref_core_vs_infarct_icc": icc21(rep.ref_core_ml.values, rep.followup_infarct_ml.values)}
    return out


def main(manifest: Path, out: Path, dt: float, motion: bool, max_n: int):
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(manifest, dtype=str).fillna("")
    f = out / "cases.csv"; done = set(pd.read_csv(f).case.astype(str)) if f.exists() else set()
    t0 = time.time()
    for _, row in man.head(max_n).iterrows():
        if str(row["case"]) in done:
            continue
        try:
            r = run_case(row, dt, motion)
        except Exception as e:
            r = {"case": row["case"], "vendor": row.get("vendor"), "status": "failed", "warnings": str(e)[:200]}
        pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"{r['case']}: {r.get('status')} core {r.get('core_ml', float('nan')):.0f}/{r.get('ref_core_ml', float('nan')):.0f} tmax6 {r.get('tmax6_ml', float('nan')):.0f}/{r.get('ref_tmax6_ml', float('nan')):.0f}", flush=True)
    df = pd.read_csv(f)
    ok = df[df.status != "failed"]
    summ = {"config": FROZEN.__dict__, "n_manifest": int(len(man)), "n_failed": int((df.status == "failed").sum()),
            "status_counts": df.status.value_counts().to_dict(),
            "all": block(ok), "excluding_reject": block(ok[ok.status != "reject"]), "excluding_reject_and_review": block(ok[ok.status == "ok"]),
            "by_vendor": {v: block(g[g.status != "reject"]) for v, g in ok.groupby("vendor")} if "vendor" in ok and ok.vendor.notna().any() else {},
            "seconds": round(time.time() - t0, 1)}
    json.dump(summ, open(out / "summary.json", "w"), indent=1, default=float)
    try:
        import subprocess
        fig_csv = out / "cases_for_figure.csv"
        ok.rename(columns={"tmax6_ml": "hypo_ml_ours", "ref_tmax6_ml": "hypo_ml_ref", "core_ml": "core_ml_ours", "ref_core_ml": "core_ml_ref"}).assign(
            flagged=lambda x: x.status == "reject").dropna(subset=["hypo_ml_ref", "core_ml_ref"]).to_csv(fig_csv, index=False)
        subprocess.run([sys.executable, str(Path(__file__).with_name("agreement_figure.py")), str(fig_csv), str(out / "agreement.png"), "--ref", "RAPID", "--flags", "flagged"], check=False)
    except Exception:
        pass
    for name in ("all", "excluding_reject", "excluding_reject_and_review"):
        b = summ[name]
        if "tmax6" in b:
            print(f"{name:28s} n={b['n']:3d}  Tmax>6 ICC {b['tmax6']['icc']:.2f} [{b['tmax6']['icc_ci'][0]:.2f}, {b['tmax6']['icc_ci'][1]:.2f}] bias {b['tmax6']['bias']:+.1f} LoA {b['tmax6']['loa'][0]:.0f}/{b['tmax6']['loa'][1]:+.0f} | "
                  f"core ICC {b['core']['icc']:.2f} LoA {b['core']['loa'][0]:.0f}/{b['core']['loa'][1]:+.0f}" + (f" | DEFUSE-3 {b['defuse3']['agreement']:.2f} kappa {b['defuse3']['kappa']:.2f}" if "defuse3" in b else ""))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), float(a[a.index("--dt") + 1]) if "--dt" in a else 1.0, "--motion" in a, int(a[a.index("--max") + 1]) if "--max" in a else 10**6)

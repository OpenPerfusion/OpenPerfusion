# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""How the deconvolution regularisation should depend on the acquisition's native sampling interval.

The bcSVD truncation (`lam`, 10 % of the largest singular value) was chosen on ISLES 2018, which is sampled at
1 s. Vendor exports arrive at 0.5 s (GE cine), 1-2 s (most Siemens/Philips), or 2.8 s per position in shuttle
mode, and the DICOM reader resamples them onto a uniform 1 s grid by linear interpolation. On the UniToBrain
shuttle patient, absolute CBF varied 13 / 20 / 32 mL/100 g/min at lam 0.10 / 0.05 / 0.02, so the choice matters.

This script simulates each acquisition from the digital phantom (noise added at the native frames, then the
same linear resampling the reader applies), sweeps lam and the oSVD alternative, and scores each against the
phantom's known CBF, CBV, MTT and Tmax, including Tmax classification at the 6 s boundary.

    python scripts/phantom_sampling.py results/phantom_sampling
"""
import sys, json, itertools, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.phantom import make_phantom
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.validate import phantom_report

DELAYS = (0.0, 2.0, 4.0, 5.0, 7.0, 9.0)        # tiles both sides of the 6 s boundary
GRID_DT = 1.0                                  # the reader's uniform grid


def resample_to_grid(hu_native: np.ndarray, times_native, t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of every voxel's series onto t_grid, clamped at the ends (as io_dicom does)."""
    out = np.empty(hu_native.shape[:3] + (len(t_grid),), np.float32)
    flat = hu_native.reshape(-1, hu_native.shape[-1])
    for i, tg in enumerate(t_grid):
        j = np.searchsorted(times_native, tg, side="right") - 1
        j = np.clip(j, 0, len(times_native) - 2)
        w = (tg - times_native[j]) / (times_native[j + 1] - times_native[j])
        w = float(np.clip(w, 0.0, 1.0))
        out[..., i] = ((1 - w) * flat[:, j] + w * flat[:, j + 1]).reshape(hu_native.shape[:3])
    return out


def acquire(native: str, noise_sd: float, seed: int):
    """Return (hu on the 1 s grid, truth, native_dt) for one simulated acquisition."""
    duration = 60.0
    t_grid = np.arange(0.0, duration, GRID_DT)
    if native == "shuttle2.8":
        # two table positions alternating: every slice sampled every 2.8 s, position B offset by 1.4 s
        hu, truth = make_phantom(delays=DELAYS, dt=1.4, duration=duration, noise_sd=noise_sd, seed=seed)
        t_all = np.arange(0.0, duration, 1.4)
        out = np.empty(hu.shape[:3] + (len(t_grid),), np.float32)
        for z in range(hu.shape[2]):
            sel = np.arange(z % 2, len(t_all), 2)                       # A: even frames, B: odd frames
            out[:, :, z] = resample_to_grid(hu[:, :, z:z + 1, sel], t_all[sel], t_grid)[:, :, 0]
        return out, truth, 2.8
    dt = float(native)
    hu, truth = make_phantom(delays=DELAYS, dt=dt, duration=duration, noise_sd=noise_sd, seed=seed)
    t_nat = np.arange(0.0, duration, dt)
    return (hu if dt == GRID_DT else resample_to_grid(hu, t_nat, t_grid)), truth, dt


def score(res, truth, ref_table: pd.DataFrame | None = None) -> dict:
    """Truth-based errors, plus agreement with the calibrated reference configuration (1 s native, bcSVD 0.10,
    the setting validated against RAPID): the scale RAPID agreement was obtained at is the target, not the
    phantom's nominal values (truncated SVD biases absolute CBF and Tmax for every package in the same way)."""
    rep = phantom_report(res.maps, truth); df = rep["table"]; s = rep["summary"]
    good = df[df.cbf_true >= 30]
    rel = {}
    if ref_table is not None:
        m = good.merge(ref_table[["delay_s", "cbf_true", "mtt_true", "cbf_est", "tmax_est"]], on=["delay_s", "cbf_true", "mtt_true"], suffixes=("", "_ref"))
        rel = {"cbf_ratio_vs_ref": float((m.cbf_est / m.cbf_est_ref).mean()),
               "cbf_abs_logratio_vs_ref": float(np.abs(np.log(m.cbf_est / m.cbf_est_ref)).mean()),
               "tmax_diff_vs_ref_s": float((m.tmax_est - m.tmax_est_ref).mean()),
               "tmax_rmsd_vs_ref_s": float(np.sqrt(((m.tmax_est - m.tmax_est_ref) ** 2).mean())),
               "tmax6_agreement_vs_ref": float(((m.tmax_est > 6) == (m.tmax_est_ref > 6)).mean())}
    # Tmax classification at 6 s on tiles with CBF >= 30 (truth: delay > 6)
    cls = good.assign(true6=good.delay_s > 6, est6=good.tmax_est > 6)
    acc = float((cls.true6 == cls.est6).mean())
    # near-boundary tiles (5 and 7 s) are the hard ones
    nb = cls[cls.delay_s.isin([5.0, 7.0])]
    acc_nb = float((nb.true6 == nb.est6).mean()) if len(nb) else np.nan
    return {"cbf_rel_err": s["cbf_rel_err_mean_cbf30plus"], "cbf_rel_err_all": s["cbf_rel_err_mean"],
            "cbf_abs_rel_err": float(good.cbf_rel_err.abs().mean()),
            "cbv_rel_err": s["cbv_rel_err_mean"], "mtt_rel_err": s["mtt_rel_err_mean"],
            "tmax_bias_s": s["tmax_bias_s"], "tmax_rmse_s": float(np.sqrt((good.tmax_err_s ** 2).mean())),
            "tmax6_accuracy": acc, "tmax6_accuracy_near": acc_nb,
            "normal_cbf_est": s["normal_cbf_est"], "normal_tmax_est": s["normal_tmax_est"],
            "pearson_cbf": s["pearson_cbf"], "delay_response": s["delay_response"], **rel, "_table": df}


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    natives = ["0.5", "1.0", "2.0", "2.8", "shuttle2.8"]
    noises = [3.0, 6.0]
    configs = [("bcsvd", lam) for lam in (0.02, 0.05, 0.10, 0.15, 0.20)] + [("osvd", None)]
    rows = []; t0 = time.time(); ref_tables = {}
    for noise in noises:                      # reference per noise level: 1 s native, bcSVD 0.10 (the ISLES-validated setting)
        hu, truth, _ = acquire("1.0", noise, seed=0)
        res = run_pipeline(hu, GRID_DT, truth.voxel_size, PipelineConfig(method="bcsvd", lam=0.10))
        ref_tables[noise] = score(res, truth)["_table"]
    for native, noise in itertools.product(natives, noises):
        hu, truth, ndt = acquire(native, noise, seed=0)
        for method, lam in configs:
            cfg = PipelineConfig(method=method, **({"lam": lam} if lam is not None else {}))
            res = run_pipeline(hu, GRID_DT, truth.voxel_size, cfg)
            sc = score(res, truth, ref_tables[noise]); sc.pop("_table")
            r = {"native": native, "native_dt": ndt, "noise_sd": noise, "method": method, "lam": lam if lam is not None else np.nan,
                 "aif_peak": float(res.aif.aif_raw.max()), **sc}
            rows.append(r)
            print(f"{native:>10s} noise {noise:.0f} {method} lam {lam}: CBF x{r['cbf_ratio_vs_ref']:.2f} of ref, Tmax {r['tmax_diff_vs_ref_s']:+.2f} s vs ref, "
                  f"6 s agreement {r['tmax6_agreement_vs_ref']:.2f} | truth: CBF err {r['cbf_rel_err']:+.2f}, Tmax rmse {r['tmax_rmse_s']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
    df = pd.DataFrame(rows)
    # per native sampling: the bcSVD lam that best reproduces the reference scale (CBF log-ratio and Tmax RMSD vs ref)
    best = {}
    for native in natives:
        sub = df[(df.native == native) & (df.method == "bcsvd")].groupby("lam").agg(cbf=("cbf_abs_logratio_vs_ref", "mean"), tmax=("tmax_rmsd_vs_ref_s", "mean"), agree=("tmax6_agreement_vs_ref", "mean"), cbf_ratio=("cbf_ratio_vs_ref", "mean"))
        sub["cost"] = sub.cbf + sub.tmax / 2.0          # 0.1 in log-CBF (10 %) weighs like 0.2 s of Tmax
        best[native] = {"lam": float(sub.cost.idxmin()), **{k: float(v) for k, v in sub.loc[sub.cost.idxmin()].items()}}
    osvd = {native: {k: float(v) for k, v in df[(df.native == native) & (df.method == "osvd")][["cbf_ratio_vs_ref", "tmax_diff_vs_ref_s", "tmax6_agreement_vs_ref"]].mean().items()} for native in natives}
    json.dump({"best_lam_by_native": best, "osvd_by_native": osvd, "seconds": round(time.time() - t0, 1)}, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({"best_lam_by_native": best, "osvd_by_native": osvd}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Run OpenPerfusion on every raw UniToBrain patient under a root and compare with the dataset's NLR maps.

    python scripts/unitobrain_batch.py unitobrain/DeepHealth_IEEE --out results/unitobrain --motion

Expects per-patient folders MOL-xxx (raw DICOM) and, optionally, MOL-xxx_Registered_Filtered_3mm_20HU_Maps.
Writes <out>/cases.csv (one row per patient: acquisition, AIF, motion, volumes, agreement with the NLR maps),
<out>/<patient>/summary.json and overview.png, and <out>/montage.png.
"""
import argparse, glob, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from scipy.stats import pearsonr, spearmanr

from openperfusion.io_dicom import read_ctp_dicom
from openperfusion.motion import motion_correct
from openperfusion.pipeline import run_pipeline, PipelineConfig


def load_maps(folder, name):
    vols = []
    for f in sorted(glob.glob(str(Path(folder) / name / "*.dcm"))):
        d = pydicom.dcmread(f)
        a = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
        vols.append((float(d.ImagePositionPatient[2]), a.T))
    vols.sort(key=lambda t: t[0])
    return np.stack([v for _, v in vols], axis=-1)


def one(raw, maps, out, motion, dt):
    out.mkdir(parents=True, exist_ok=True)
    row = dict(patient=raw.name)
    t0 = time.time(); c = read_ctp_dicom(raw, dt=dt); row["read_s"] = round(time.time() - t0, 1)
    row.update(model=c.meta.get("model"), kvp=c.meta.get("kvp"), n_slices=c.meta["n_slices"], n_frames=c.meta["n_times"],
               shuttle=c.meta["shuttle_mode"], native_dt=round(float(np.median(np.diff(c.raw_times[0]))), 2) if len(c.raw_times[0]) > 1 else None,
               n_files=c.n_files, dx_mm=round(c.voxel_size[0], 3), dz_mm=round(c.voxel_size[2], 2))
    hu = c.hu
    if motion:
        t1 = time.time(); m = motion_correct(hu, voxel_size=c.voxel_size); hu = m.hu
        row.update(motion_s=round(time.time() - t1, 1), max_shift_mm=round(m.max_shift_mm, 2), max_rot_deg=round(float(abs(m.theta).max()), 2),
                   p95_shift_mm=round(float(np.percentile(np.hypot(m.dx * c.voxel_size[0], m.dy * c.voxel_size[1]), 95)), 2))
    t2 = time.time(); r = run_pipeline(hu, c.dt, c.voxel_size, PipelineConfig()); row["pipeline_s"] = round(time.time() - t2, 1)
    th = r.thresholds; q = r.aif.qc
    row.update(aif_peak=round(float(r.aif.aif_raw.max()), 1), aif_n=int(r.aif.aif_mask.sum()), aif_ttp=q.get("aif_ttp"), tissue_ttp=q.get("tissue_ttp"),
               aif_slices=",".join(str(z) for z in sorted(set(np.argwhere(r.aif.aif_mask)[:, 2].tolist()))), k_av=round(float(r.aif.k_av), 2),
               vof_peak=round(float(q.get("vof_peak", 0) or 0), 1), aif_fallback=bool(q.get("aif_fallback", False)), aif_weak=bool(q.get("aif_weak", False)),
               aif_last_resort=bool(q.get("aif_last_resort", False)), n_baseline=r.n_baseline,
               core_ml=round(th.core_ml, 1), hypo_ml=round(th.hypo_ml, 1), mismatch_ratio=round(th.mismatch_ratio, 2), hir=round(th.hir, 2),
               defuse3=bool(th.defuse3_target), tmax4_ml=round(th.tmax_ml[4], 1), tmax8_ml=round(th.tmax_ml[8], 1), tmax10_ml=round(th.tmax_ml[10], 1),
               cbf_med=round(float(np.median(r.maps.cbf[r.mask])), 1), cbv_med=round(float(np.median(r.maps.cbv[r.mask])), 2),
               mtt_med=round(float(np.median(r.maps.mtt[r.mask])), 1), tmax_med=round(float(np.median(r.maps.tmax[r.mask])), 1),
               brain_ml=round(float(r.mask.sum() * np.prod(c.voxel_size) / 1000.0), 0))
    # hemispheric asymmetry of Tmax>6 (which side, how lateralised)
    X = r.mask.shape[0]; xs = np.arange(X)[:, None, None]
    left = th.hypo & (xs < X // 2); right = th.hypo & (xs >= X // 2)
    row.update(hypo_left_ml=round(float(left.sum() * np.prod(c.voxel_size) / 1000.0), 1), hypo_right_ml=round(float(right.sum() * np.prod(c.voxel_size) / 1000.0), 1))
    ref = None
    if maps is not None and maps.exists():
        ref = {n: load_maps(maps, n) for n in ("NLR_CBF", "NLR_CBV", "NLR_MTT", "NLR_Delay")}
        if ref["NLR_CBF"].shape == r.mask.shape:
            valid = r.mask & (ref["NLR_CBF"] > 0)
            for ours, name, key in ((r.maps.cbf, "NLR_CBF", "cbf"), (r.maps.cbv, "NLR_CBV", "cbv"), (r.maps.mtt, "NLR_MTT", "mtt"), (r.maps.tmax, "NLR_Delay", "delay")):
                v = valid & np.isfinite(ours)
                row[f"r_{key}"] = round(float(pearsonr(ours[v], ref[name][v])[0]), 3)
                row[f"rho_{key}"] = round(float(spearmanr(ours[v], ref[name][v])[0]), 3)
                row[f"ref_{key}_med"] = round(float(np.median(ref[name][v])), 2)
            # does the reference delay map lateralise the same way?
            d = ref["NLR_Delay"]; dl = float(np.median(d[r.mask & (xs < X // 2)])); dr = float(np.median(d[r.mask & (xs >= X // 2)]))
            row.update(ref_delay_left_med=round(dl, 2), ref_delay_right_med=round(dr, 2),
                       same_side=bool(np.sign(row["hypo_left_ml"] - row["hypo_right_ml"]) == np.sign(dl - dr)) if abs(dl - dr) > 0.2 and abs(row["hypo_left_ml"] - row["hypo_right_ml"]) > 5 else None)
        else:
            row["ref_shape_mismatch"] = str(ref["NLR_CBF"].shape)
    json.dump(row, open(out / "summary.json", "w"), indent=2, default=str)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        base = hu[..., :max(3, r.n_baseline)].mean(-1); Z = hu.shape[2]; z = Z // 2; m = r.mask[:, :, z]
        fig, ax = plt.subplots(1, 4 if ref is not None else 3, figsize=(16, 4.2))
        ax[0].imshow(base[:, :, z].T, cmap="gray", vmin=0, vmax=80, origin="lower"); ax[0].set_title(f"{raw.name} z{z} baseline, AIF green")
        am = r.aif.aif_mask[:, :, z]
        if am.any(): ax[0].contour(am.T, levels=[0.5], colors="lime", linewidths=1)
        ax[1].imshow(np.where(m, r.maps.tmax[:, :, z], np.nan).T, cmap="jet", vmin=0, vmax=12, origin="lower"); ax[1].set_title("Tmax (s)")
        ax[2].imshow(base[:, :, z].T, cmap="gray", vmin=0, vmax=80, origin="lower")
        ov = np.zeros(m.shape + (4,)); ov[th.hypo[:, :, z]] = (1, 0, 0, 0.5); ov[th.core[:, :, z]] = (0, 0.3, 1, 0.8)
        ax[2].imshow(ov.transpose(1, 0, 2), origin="lower"); ax[2].set_title(f"Tmax>6 {th.hypo_ml:.0f} mL, core {th.core_ml:.0f} mL")
        if ref is not None and "NLR_Delay" in ref and ref["NLR_Delay"].shape == r.mask.shape:
            ax[3].imshow(np.where(m, ref["NLR_Delay"][:, :, z], np.nan).T, cmap="jet", vmin=-2, vmax=10, origin="lower"); ax[3].set_title("UniToBrain NLR delay (s)")
        for x in ax: x.axis("off")
        plt.tight_layout(); plt.savefig(out / "overview.png", dpi=60); plt.close(fig)
    except Exception as e:                     # pragma: no cover
        row["figure_error"] = str(e)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", default="results/unitobrain")
    ap.add_argument("--motion", action="store_true"); ap.add_argument("--dt", type=float, default=1.0); ap.add_argument("--only", nargs="*")
    a = ap.parse_args(); root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raws = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"[A-Z]+-\d+", p.name))
    if a.only: raws = [p for p in raws if p.name in a.only]
    rows = []
    for raw in raws:
        maps = root / f"{raw.name}_Registered_Filtered_3mm_20HU_Maps"
        try:
            row = one(raw, maps, out / raw.name, a.motion, a.dt)
        except Exception as e:
            row = dict(patient=raw.name, error=repr(e))
        rows.append(row)
        print({k: row.get(k) for k in ("patient", "shuttle", "max_shift_mm", "aif_peak", "aif_ttp", "tissue_ttp", "core_ml", "hypo_ml", "defuse3", "r_cbf", "r_delay", "same_side", "error")}, flush=True)
        pd.DataFrame(rows).to_csv(out / "cases.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.image as mpimg
        imgs = [(r["patient"], out / r["patient"] / "overview.png") for r in rows if (out / r["patient"] / "overview.png").exists()]
        if imgs:
            fig, ax = plt.subplots(len(imgs), 1, figsize=(16, 4.2 * len(imgs)))
            ax = np.atleast_1d(ax)
            for a_, (name, f) in zip(ax, imgs): a_.imshow(mpimg.imread(f)); a_.axis("off")
            plt.tight_layout(); plt.savefig(out / "montage.png", dpi=60); plt.close(fig)
    except Exception as e:
        print("montage skipped:", e)


if __name__ == "__main__":
    main()

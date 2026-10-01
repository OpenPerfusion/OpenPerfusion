# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Run OpenPerfusion on one raw UniToBrain patient (GE LightSpeed VCT, shuttle mode) and compare
with the dataset's own NLR maps for that patient.

    python scripts/unitobrain_compare.py unitobrain/DeepHealth_IEEE/MOL-001 \
        unitobrain/DeepHealth_IEEE/MOL-001_Registered_Filtered_3mm_20HU_Maps --out results/unitobrain/MOL-001

Writes summary.json (volumes, QC, agreement with the NLR maps) and overview.png (baseline CT, Tmax,
thresholded masks, reference delay map for three slices).
"""
import argparse, glob, json, time
from pathlib import Path
import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("maps", nargs="?"); ap.add_argument("--out", default="results/unitobrain")
    ap.add_argument("--motion", action="store_true"); ap.add_argument("--dt", type=float, default=1.0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); c = read_ctp_dicom(a.raw, dt=a.dt); t_read = time.time() - t0
    hu = c.hu; motion = None
    if a.motion:
        t1 = time.time(); m = motion_correct(hu, voxel_size=c.voxel_size); hu = m.hu
        motion = dict(seconds=round(time.time() - t1, 1), max_shift_mm=round(m.max_shift_mm, 2), max_rot_deg=round(float(abs(m.theta).max()), 2))
    t2 = time.time(); r = run_pipeline(hu, c.dt, c.voxel_size, PipelineConfig()); t_pipe = time.time() - t2
    th = r.thresholds
    summary = dict(dicom=c.meta, read_seconds=round(t_read, 1), pipeline_seconds=round(t_pipe, 1), motion=motion,
                   aif=dict(peak_hu=round(float(r.aif.aif_raw.max()), 1), n_voxels=int(r.aif.aif_mask.sum()), k_av=round(float(r.aif.k_av), 2),
                            slices=sorted(set(np.argwhere(r.aif.aif_mask)[:, 2].tolist())), qc={k: v for k, v in r.aif.qc.items()}),
                   core_ml=round(th.core_ml, 1), hypoperfusion_ml=round(th.hypo_ml, 1), mismatch_ratio=round(th.mismatch_ratio, 2),
                   tmax_ml={k: round(v, 1) for k, v in th.tmax_ml.items()}, hir=round(th.hir, 2), defuse3=bool(th.defuse3_target),
                   brain_median=dict(cbf=round(float(np.median(r.maps.cbf[r.mask])), 1), cbv=round(float(np.median(r.maps.cbv[r.mask])), 2),
                                     mtt=round(float(np.median(r.maps.mtt[r.mask])), 1), tmax=round(float(np.median(r.maps.tmax[r.mask])), 1)))
    if a.maps:
        ref = {n: load_maps(a.maps, n) for n in ("NLR_CBF", "NLR_CBV", "NLR_MTT", "NLR_Delay")}
        valid = r.mask & (ref["NLR_CBF"] > 0)
        agree = {}
        for ours, name in ((r.maps.cbf, "NLR_CBF"), (r.maps.cbv, "NLR_CBV"), (r.maps.mtt, "NLR_MTT"), (r.maps.tmax, "NLR_Delay")):
            v = valid & np.isfinite(ours)
            agree[name] = dict(ours_median=round(float(np.median(ours[v])), 2), ref_median=round(float(np.median(ref[name][v])), 2),
                               pearson=round(float(pearsonr(ours[v], ref[name][v])[0]), 3), spearman=round(float(spearmanr(ours[v], ref[name][v])[0]), 3))
        summary["agreement_with_nlr_maps"] = agree
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=str)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        base = hu[..., :max(3, r.n_baseline)].mean(-1); Z = hu.shape[2]; zs = sorted({Z // 5, Z // 2, 4 * Z // 5})
        fig, ax = plt.subplots(len(zs), 4 if a.maps else 3, figsize=(16, 4 * len(zs)))
        for row, z in enumerate(zs):
            m = r.mask[:, :, z]
            ax[row, 0].imshow(base[:, :, z].T, cmap="gray", vmin=0, vmax=80, origin="lower"); ax[row, 0].set_title(f"z{z} baseline CT, AIF in green")
            am = r.aif.aif_mask[:, :, z]
            if am.any(): ax[row, 0].contour(am.T, levels=[0.5], colors="lime", linewidths=1)
            ax[row, 1].imshow(np.where(m, r.maps.tmax[:, :, z], np.nan).T, cmap="jet", vmin=0, vmax=12, origin="lower"); ax[row, 1].set_title("Tmax (s)")
            ax[row, 2].imshow(base[:, :, z].T, cmap="gray", vmin=0, vmax=80, origin="lower")
            ov = np.zeros(m.shape + (4,)); ov[th.hypo[:, :, z]] = (1, 0, 0, 0.5); ov[th.core[:, :, z]] = (0, 0.3, 1, 0.8)
            ax[row, 2].imshow(ov.transpose(1, 0, 2), origin="lower"); ax[row, 2].set_title("Tmax>6 s (red), rCBF<30 % (blue)")
            if a.maps:
                ax[row, 3].imshow(np.where(m, ref["NLR_Delay"][:, :, z], np.nan).T, cmap="jet", vmin=-2, vmax=10, origin="lower"); ax[row, 3].set_title("UniToBrain NLR delay (s)")
            for x in ax[row]: x.axis("off")
        plt.tight_layout(); plt.savefig(out / "overview.png", dpi=70)
    except Exception as e:                     # pragma: no cover
        print("figure skipped:", e)
    print(json.dumps({k: summary[k] for k in ("core_ml", "hypoperfusion_ml", "mismatch_ratio", "defuse3", "brain_median", "aif")}, default=str))


if __name__ == "__main__":
    main()

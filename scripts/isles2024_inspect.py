# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Inspect one ISLES'24 subject where OpenPerfusion and icobrain disagree: curves, AIF/VOF, Tmax maps, and the
disputed region (ours > 6 s where icobrain <= 6 s, and the reverse).

    python scripts/isles2024_inspect.py /path/to/shard.parquet results/inspect/sub-stroke0074
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.io_isles2024 import load_subject_from_parquet
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.maps import threshold_maps
from openperfusion.aif import curve_features


def main(shard: str, out: Path, cfg: PipelineConfig = None):
    out.mkdir(parents=True, exist_ok=True)
    d = load_subject_from_parquet(Path(shard), load_4d=True, register=True)
    cfg = cfg or PipelineConfig()
    res = run_pipeline(d["ctp4d"], d["dt"], d["voxel_size"], cfg)
    vox_ml = float(np.prod(d["voxel_size"])) / 1000
    valid = res.mask & np.isfinite(d["tmax"]) & (d["cbf"] > 0)
    t_o = threshold_maps(res.maps, d["voxel_size"], baseline=res.baseline, reference=cfg.reference,
                         min_cluster_ml=cfg.min_cluster_ml, restrict_core_to_hypo=cfg.restrict_core_to_hypo)
    ours6 = t_o.hypo & valid
    ref6 = (d["tmax"] > 6) & valid
    only_ours = ours6 & ~ref6; only_ref = ref6 & ~ours6; both = ours6 & ref6
    normal = valid & ~ours6 & ~ref6
    conc = res.conc if hasattr(res, "conc") else None
    T = d["ctp4d"].shape[-1]; t = np.arange(T) * d["dt"]
    aif = res.aif.aif_raw; vof = res.aif.vof
    fa = curve_features(aif[None], d["dt"])
    info = {"case": d["case"], "shape": list(d["ctp4d"].shape), "dt": d["dt"], "voxel_size": d["voxel_size"],
            "registration": d["registration"], "aif_qc": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.aif.qc.items()},
            "aif_peak": float(aif.max()), "aif_ttp": float(fa.ttp[0]), "aif_width": float(fa.width[0]),
            "ours_ml": float(ours6.sum() * vox_ml), "ref_ml": float(ref6.sum() * vox_ml),
            "only_ours_ml": float(only_ours.sum() * vox_ml), "only_ref_ml": float(only_ref.sum() * vox_ml), "both_ml": float(both.sum() * vox_ml),
            "tmax_ours_in_only_ours": [float(np.percentile(res.maps.tmax[only_ours], p)) for p in (10, 50, 90)] if only_ours.any() else None,
            "tmax_ref_in_only_ours": [float(np.percentile(d["tmax"][only_ours], p)) for p in (10, 50, 90)] if only_ours.any() else None,
            "tmax_ours_in_normal": [float(np.percentile(res.maps.tmax[normal], p)) for p in (10, 50, 90)],
            "tmax_ref_in_normal": [float(np.percentile(d["tmax"][normal], p)) for p in (10, 50, 90)],
            "tmax_ours_in_both": [float(np.percentile(res.maps.tmax[both], p)) for p in (10, 50, 90)] if both.any() else None,
            "tmax_ref_in_both": [float(np.percentile(d["tmax"][both], p)) for p in (10, 50, 90)] if both.any() else None,
            "cbf_ours_median_normal": float(np.median(res.maps.cbf[normal])), "cbf_ref_median_normal": float(np.median(d["cbf"][normal])),
            "per_slice_only_ours_ml": [float(only_ours[..., z].sum() * vox_ml) for z in range(ours6.shape[2])],
            "per_slice_ref_ml": [float(ref6[..., z].sum() * vox_ml) for z in range(ours6.shape[2])]}
    # hemisphere split (left/right halves of x) of the disputed region
    X = ours6.shape[0]
    info["only_ours_left_right_ml"] = [float(only_ours[: X // 2].sum() * vox_ml), float(only_ours[X // 2:].sum() * vox_ml)]
    info["ref_left_right_ml"] = [float(ref6[: X // 2].sum() * vox_ml), float(ref6[X // 2:].sum() * vox_ml)]
    json.dump(info, open(out / "info.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in info.items() if not k.startswith("per_slice")}, indent=1, default=str))

    # curves: mean concentration in each region (recompute concentration from HU minus baseline)
    hu = d["ctp4d"]; base = hu[..., : res.n_baseline].mean(-1, keepdims=True)
    def mean_curve(m):
        return (hu[m] - base[m]).mean(0) if m.any() else np.zeros(T)
    c_only_ours, c_ref, c_both, c_norm = mean_curve(only_ours), mean_curve(only_ref), mean_curve(both), mean_curve(normal)
    z = int(np.argmax(info["per_slice_only_ours_ml"]))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    a = ax[0, 0]; a.plot(t, aif, "r", label=f"AIF (peak {aif.max():.0f} HU, ttp {fa.ttp[0]:.0f} s)")
    if vof is not None: a.plot(t, vof, "b", label="VOF")
    a.legend(); a.set_title("arterial / venous"); a.set_xlabel("s"); a.set_ylabel("HU")
    a = ax[0, 1]
    a.plot(t, c_norm, "k", label=f"normal ({normal.sum() * vox_ml:.0f} mL)")
    a.plot(t, c_both, "m", label=f"both > 6 s ({both.sum() * vox_ml:.0f} mL)")
    a.plot(t, c_only_ours, "r", label=f"only ours > 6 ({only_ours.sum() * vox_ml:.0f} mL)")
    a.plot(t, c_ref, "y", label=f"only icobrain > 6 ({only_ref.sum() * vox_ml:.0f} mL)")
    a.legend(); a.set_title("mean tissue curves"); a.set_xlabel("s"); a.set_ylabel("HU")
    a = ax[0, 2]
    for m, lab, col in ((normal, "normal", "k"), (both, "both", "m"), (only_ours, "only ours", "r"), (only_ref, "only icobrain", "y")):
        if m.any():
            a.hist(res.maps.tmax[m], bins=np.arange(0, 20, 0.5), histtype="step", color=col, label=f"ours, {lab}", density=True)
            a.hist(d["tmax"][m], bins=np.arange(0, 20, 0.5), histtype="step", color=col, ls="--", density=True)
    a.axvline(6, color="grey"); a.legend(fontsize=8); a.set_title("Tmax distribution (solid ours, dashed icobrain)"); a.set_xlabel("s")
    sl = lambda m: m[..., z].T
    a = ax[1, 0]; a.imshow(sl(hu[..., : res.n_baseline].mean(-1)), cmap="gray", vmin=0, vmax=80, origin="lower")
    a.contour(sl(res.aif.aif_mask).astype(float), levels=[0.5], colors="lime"); a.set_title(f"baseline, slice {z}, AIF green"); a.axis("off")
    a = ax[1, 1]; a.imshow(np.where(sl(valid), sl(res.maps.tmax), np.nan), cmap="jet", vmin=0, vmax=12, origin="lower"); a.set_title(f"OpenPerfusion Tmax ({info['ours_ml']:.0f} mL > 6 s)"); a.axis("off")
    a = ax[1, 2]; a.imshow(np.where(sl(valid), sl(d["tmax"]), np.nan), cmap="jet", vmin=0, vmax=12, origin="lower"); a.set_title(f"icobrain Tmax ({info['ref_ml']:.0f} mL > 6 s)"); a.axis("off")
    for a in (ax[1, 1], ax[1, 2]):
        a.contour(sl(ours6).astype(float), levels=[0.5], colors="r", linewidths=0.8)
        a.contour(sl(ref6).astype(float), levels=[0.5], colors="y", linewidths=0.8)
    plt.suptitle(f"{d['case']}: ours {info['ours_ml']:.0f} mL vs icobrain {info['ref_ml']:.0f} mL (red ours > 6 s, yellow icobrain > 6 s)")
    plt.tight_layout(); plt.savefig(out / "inspect.png", dpi=100); print("wrote", out / "inspect.png")
    return d, res, info


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Validation harness.

Two levels:
1. Phantom: recovery of known CBF / CBV / MTT / Tmax by tile, and delay-insensitivity
   (does Tmax track the imposed delay while CBF stays put?).
2. Reference software (e.g. RAPID maps shipped with ISLES 2018): voxelwise agreement of
   maps, Dice of the Tmax > 6 s and rCBF < 30 % regions, and volume agreement across
   cases (bias, limits of agreement, ICC), plus overlap with a follow-up lesion when present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .maps import PerfusionMaps, threshold_maps
from .phantom import PhantomTruth


# ----------------------------------------------------------------------------- phantom
def phantom_report(maps: PerfusionMaps, truth: PhantomTruth) -> dict:
    """Tile-wise recovery table + delay response + normal-tissue summary."""
    rows = []
    for z in range(truth.cbf.shape[2]):
        delay = float(np.unique(truth.delay[:, :, z][truth.tile_mask[:, :, z]])[0])
        for cbf in np.unique(truth.cbf[truth.tile_mask]):
            for mtt in np.unique(truth.mtt[truth.tile_mask]):
                sel = truth.tile_mask[:, :, z] & (truth.cbf[:, :, z] == cbf) & (truth.mtt[:, :, z] == mtt)
                if not sel.any():
                    continue
                rows.append(dict(
                    delay_s=delay, cbf_true=cbf, mtt_true=mtt, cbv_true=cbf * mtt / 60,
                    cbf_est=float(maps.cbf[:, :, z][sel].mean()), cbf_sd=float(maps.cbf[:, :, z][sel].std()),
                    cbv_est=float(maps.cbv[:, :, z][sel].mean()),
                    mtt_est=float(maps.mtt[:, :, z][sel].mean()),
                    tmax_est=float(maps.tmax[:, :, z][sel].mean()), tmax_sd=float(maps.tmax[:, :, z][sel].std()),
                ))
    df = pd.DataFrame(rows)
    df["cbf_rel_err"] = df.cbf_est / df.cbf_true - 1
    df["cbv_rel_err"] = df.cbv_est / df.cbv_true - 1
    df["mtt_rel_err"] = df.mtt_est / df.mtt_true - 1
    df["tmax_err_s"] = df.tmax_est - df.delay_s

    # delay response: mean Tmax per delay for well-perfused tiles, relative to delay 0
    good = df[df.cbf_true >= 30]
    resp = good.groupby("delay_s").tmax_est.mean()
    delay_response = (resp - resp.iloc[0]).round(2).to_dict()
    cbf_by_delay = good.groupby("delay_s").cbf_rel_err.mean().round(3).to_dict()

    nm = truth.normal_mask
    summary = dict(
        n_tiles=len(df),
        cbf_rel_err_mean=float(df.cbf_rel_err.mean()), cbf_rel_err_mean_cbf30plus=float(good.cbf_rel_err.mean()),
        cbv_rel_err_mean=float(df.cbv_rel_err.mean()),
        mtt_rel_err_mean=float(df.mtt_rel_err.mean()),
        tmax_bias_s=float(good.tmax_err_s.mean()), tmax_sd_s=float(good.tmax_err_s.std()),
        delay_response=delay_response, cbf_rel_err_by_delay=cbf_by_delay,
        normal_cbf_est=float(maps.cbf[nm].mean()), normal_cbv_est=float(maps.cbv[nm].mean()),
        normal_mtt_est=float(maps.mtt[nm].mean()), normal_tmax_est=float(maps.tmax[nm].mean()),
        pearson_cbf=float(np.corrcoef(df.cbf_true, df.cbf_est)[0, 1]),
        pearson_cbv=float(np.corrcoef(df.cbv_true, df.cbv_est)[0, 1]),
        pearson_mtt=float(np.corrcoef(df.mtt_true, df.mtt_est)[0, 1]),
    )
    return {"table": df, "summary": summary}


# ----------------------------------------------------------------------------- reference software
def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s > 0 else float("nan")


def _corr(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_to_reference(ours: PerfusionMaps, ref: dict, voxel_size, baseline=None, lesion: np.ndarray | None = None,
                         tmax_thr: float = 6.0, core_frac: float = 0.30, **thr_kw) -> dict:
    """Compare our maps with reference maps (dict with any of cbf, cbv, mtt, tmax; same grid).

    Thresholded regions for the reference are derived from the reference maps with the SAME
    thresholding code (same rCBF reference-region rule, cluster filter), so the comparison
    isolates the map computation from the post-processing choices.
    """
    valid = ours.mask.copy()
    for k in ("cbf", "tmax"):
        if k in ref:
            valid &= np.isfinite(ref[k]) & (ref[k] > 0)
    out = {"n_voxels": int(valid.sum())}
    for k in ("cbf", "cbv", "mtt", "tmax"):
        if k in ref:
            a, b = getattr(ours, k)[valid], ref[k][valid]
            out[f"r_{k}"] = _corr(a, b)
            out[f"spearman_{k}"] = _corr(pd.Series(a).rank().values, pd.Series(b).rank().values)
    # thresholded regions, same post-processing for both
    ref_maps = PerfusionMaps(cbf=ref.get("cbf", ours.cbf), cbv=ref.get("cbv", ours.cbv), mtt=ref.get("mtt", ours.mtt),
                             tmax=ref.get("tmax", ours.tmax), ttp=ours.ttp, mask=valid, dt=ours.dt, method="reference")
    ours_v = PerfusionMaps(cbf=ours.cbf, cbv=ours.cbv, mtt=ours.mtt, tmax=ours.tmax, ttp=ours.ttp, mask=valid,
                           dt=ours.dt, method=ours.method)
    t_ours = threshold_maps(ours_v, voxel_size, baseline=baseline, core_frac=core_frac, tmax_thr=tmax_thr, **thr_kw)
    t_ref = threshold_maps(ref_maps, voxel_size, baseline=baseline, core_frac=core_frac, tmax_thr=tmax_thr, **thr_kw)
    out.update(
        core_ml_ours=t_ours.core_ml, core_ml_ref=t_ref.core_ml, dice_core=dice(t_ours.core, t_ref.core),
        hypo_ml_ours=t_ours.hypo_ml, hypo_ml_ref=t_ref.hypo_ml, dice_hypo=dice(t_ours.hypo, t_ref.hypo),
        mismatch_ratio_ours=t_ours.mismatch_ratio, mismatch_ratio_ref=t_ref.mismatch_ratio,
        defuse3_ours=t_ours.defuse3_target, defuse3_ref=t_ref.defuse3_target,
        ref_cbf_ours=t_ours.reference_cbf, ref_cbf_ref=t_ref.reference_cbf,
    )
    if lesion is not None:
        les = lesion.astype(bool)
        vox_ml = float(np.prod(voxel_size)) / 1000
        out.update(lesion_ml=float(les.sum() * vox_ml), dice_core_lesion_ours=dice(t_ours.core, les),
                   dice_core_lesion_ref=dice(t_ref.core, les), dice_hypo_lesion_ours=dice(t_ours.hypo, les),
                   dice_hypo_lesion_ref=dice(t_ref.hypo, les))
    return out


def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    return dict(bias=float(d.mean()), sd=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                loa_low=float(d.mean() - 1.96 * d.std(ddof=1)) if len(d) > 1 else float("nan"),
                loa_high=float(d.mean() + 1.96 * d.std(ddof=1)) if len(d) > 1 else float("nan"))


def icc21(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single measurement."""
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    if n < 2:
        return float("nan")
    gm = x.mean()
    ms_r = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    ss_e = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def summarize_cases(per_case: list[dict]) -> dict:
    df = pd.DataFrame(per_case)
    out = {"n_cases": len(df)}
    for k in ("core", "hypo"):
        a, b = df[f"{k}_ml_ours"].values, df[f"{k}_ml_ref"].values
        out[f"{k}_icc"] = icc21(a, b)
        out[f"{k}_ba"] = bland_altman(a, b)
        out[f"{k}_dice_mean"] = float(df[f"dice_{k}"].mean())
    for k in ("cbf", "cbv", "mtt", "tmax"):
        if f"r_{k}" in df:
            out[f"r_{k}_mean"] = float(df[f"r_{k}"].mean())
    if "defuse3_ours" in df:
        agree = (df.defuse3_ours == df.defuse3_ref)
        out["defuse3_agreement"] = float(agree.mean())
        # Cohen's kappa
        po = agree.mean(); p1 = df.defuse3_ours.mean(); p2 = df.defuse3_ref.mean()
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        out["defuse3_kappa"] = float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
    return out

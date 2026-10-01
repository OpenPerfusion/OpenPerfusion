# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Perfusion maps in physiological units, and RAPID-style thresholded volumes."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage as ndi

from .deconvolve import deconvolve_bcsvd, deconvolve_osvd, deconvolve_fourier
from .phantom import RHO, K_H


@dataclass
class PerfusionMaps:
    cbf: np.ndarray    # mL/100 g/min
    cbv: np.ndarray    # mL/100 g
    mtt: np.ndarray    # s
    tmax: np.ndarray   # s
    ttp: np.ndarray    # s
    mask: np.ndarray
    dt: float
    method: str
    qc: dict = field(default_factory=dict)


def perfusion_maps(conc: np.ndarray, mask: np.ndarray, aif: np.ndarray, dt: float,
                   method: str = "osvd", **kw) -> PerfusionMaps:
    """Deconvolve every voxel in `mask` and scale to physiological units.

    Absolute scaling uses the conventional density/haematocrit factor k_H/rho; relative
    values (rCBF) are unaffected by it. Tmax is in seconds and is not rescaled.
    """
    curves = conc[mask]
    if method == "bcsvd":
        r = deconvolve_bcsvd(curves, aif, dt, **kw)
    elif method == "osvd":
        r = deconvolve_osvd(curves, aif, dt, **kw)
    elif method == "fourier":
        r = deconvolve_fourier(curves, aif, dt, **kw)
    else:
        raise ValueError(method)
    scale = K_H / RHO
    cbf_v = scale * r.kmax * 6000.0
    # no clipping: clipping noisy tissue curves at zero biases their area upward
    auc_a = np.trapezoid(aif, dx=dt)
    auc_t = np.trapezoid(curves, dx=dt, axis=1)
    cbv_v = scale * (auc_t / max(auc_a, 1e-9)) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mtt_v = np.where(cbf_v > 1e-6, cbv_v / cbf_v * 60.0, 0.0)
    ttp_v = curves.argmax(axis=1) * dt

    def full(v):
        out = np.zeros(mask.shape, np.float32); out[mask] = v; return out

    qc = {"method": method}
    if r.lam is not None:
        qc["osvd_lambda_median"] = float(np.median(r.lam))
    return PerfusionMaps(cbf=full(cbf_v), cbv=full(cbv_v), mtt=full(mtt_v), tmax=full(r.tmax),
                         ttp=full(ttp_v), mask=mask, dt=dt, method=method, qc=qc)


@dataclass
class ThresholdResult:
    tissue: np.ndarray        # analysed voxels (mask minus vessels / CSF)
    core: np.ndarray          # rCBF < core_frac
    hypo: np.ndarray          # Tmax > tmax_thr
    rcbf: np.ndarray
    reference_cbf: float
    reference_side: str
    core_ml: float
    hypo_ml: float
    mismatch_ml: float
    mismatch_ratio: float
    tmax_ml: dict             # {4: mL, 6: mL, 8: mL, 10: mL}
    hir: float                # Tmax>10 / Tmax>6
    defuse3_target: bool
    params: dict


def _remove_small(m: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1 or not m.any():
        return m
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, np.where(sizes >= min_voxels)[0] + 1)
    return keep & m


def threshold_maps(maps: PerfusionMaps, voxel_size, baseline: np.ndarray | None = None,
                   core_frac: float = 0.30, tmax_thr: float = 6.0,
                   vessel_cbv_max: float | None = None, vessel_cbv_rel: float = 2.5, csf_hu_max: float = 15.0,
                   reference: str = "contralateral", min_cluster_ml: float = 1.0,
                   restrict_core_to_hypo: bool = False) -> ThresholdResult:
    """RAPID-style thresholding: core = rCBF < 30 %, hypoperfusion = Tmax > 6 s.

    rCBF reference: mean CBF of the contralateral (less hypoperfused) hemisphere, excluding
    vessels, CSF and Tmax > 6 s voxels; hemispheres are split at the mask's x-centroid.
    The exact RAPID reference region is unpublished; this is a parameter to calibrate.
    """
    vox_ml = float(np.prod(voxel_size)) / 1000.0
    tissue = maps.mask.copy()
    # vessel exclusion: relative to the brain median so it is unit-agnostic (reference maps may be
    # in arbitrary units); an absolute mL/100 g cap can be given instead
    if vessel_cbv_max is not None:
        tissue &= maps.cbv <= vessel_cbv_max
    else:
        med = float(np.median(maps.cbv[maps.mask])) if maps.mask.any() else 0.0
        tissue &= maps.cbv <= vessel_cbv_rel * med
    if baseline is not None:
        tissue &= baseline >= csf_hu_max
    hypo_raw = tissue & (maps.tmax > tmax_thr)

    # ---- reference CBF ----
    xs = np.where(tissue)[0]
    xc = xs.mean() if len(xs) else tissue.shape[0] / 2
    left = np.zeros_like(tissue); left[: int(round(xc))] = True
    hemi = {"left": tissue & left, "right": tissue & ~left}
    frac = {s: (hypo_raw & h).sum() / max(h.sum(), 1) for s, h in hemi.items()}
    if reference == "contralateral":
        side = min(frac, key=frac.get)
        ref_vox = hemi[side] & ~hypo_raw
        ref = float(maps.cbf[ref_vox].mean()) if ref_vox.any() else float(np.median(maps.cbf[tissue]))
    elif reference == "global_median":
        side = "global"
        ref = float(np.median(maps.cbf[tissue & ~hypo_raw]))
    else:
        raise ValueError(reference)
    rcbf = np.zeros_like(maps.cbf); rcbf[tissue] = maps.cbf[tissue] / max(ref, 1e-6)

    core_raw = tissue & (rcbf < core_frac)
    if restrict_core_to_hypo:
        core_raw &= hypo_raw
    min_vox = int(round(min_cluster_ml / vox_ml))
    core = _remove_small(core_raw, min_vox)
    hypo = _remove_small(hypo_raw, min_vox)

    core_ml = core.sum() * vox_ml
    hypo_ml = hypo.sum() * vox_ml
    tmax_ml = {thr: float(_remove_small(tissue & (maps.tmax > thr), min_vox).sum() * vox_ml) for thr in (4, 6, 8, 10)}
    hir = tmax_ml[10] / tmax_ml[6] if tmax_ml[6] > 0 else 0.0
    ratio = hypo_ml / core_ml if core_ml > 0 else float("inf")
    mismatch_ml = hypo_ml - core_ml
    defuse3 = (core_ml < 70) and (ratio >= 1.8) and (mismatch_ml >= 15)
    return ThresholdResult(tissue=tissue, core=core, hypo=hypo, rcbf=rcbf, reference_cbf=ref,
                           reference_side=side, core_ml=float(core_ml), hypo_ml=float(hypo_ml),
                           mismatch_ml=float(mismatch_ml), mismatch_ratio=float(ratio), tmax_ml=tmax_ml,
                           hir=float(hir), defuse3_target=bool(defuse3),
                           params=dict(core_frac=core_frac, tmax_thr=tmax_thr, vessel_cbv_max=vessel_cbv_max, vessel_cbv_rel=vessel_cbv_rel,
                                       csf_hu_max=csf_hu_max, reference=reference, min_cluster_ml=min_cluster_ml,
                                       restrict_core_to_hypo=restrict_core_to_hypo))

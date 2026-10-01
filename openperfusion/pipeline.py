"""End-to-end: 4D HU volume -> masks -> AIF/VOF -> deconvolution -> maps -> thresholded volumes."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import time
import numpy as np

from .preprocess import brain_mask_ct, concentration_from_hu
from .aif import select_aif_vof, AifResult, curve_features
from .maps import perfusion_maps, threshold_maps, PerfusionMaps, ThresholdResult


@dataclass
class PipelineConfig:
    # masking
    mask_lo_hu: float = 0.0
    mask_hi_hu: float = 100.0
    # smoothing (in-plane voxels / frames)
    spatial_sigma: float = 2.0
    temporal_sigma: float = 0.0
    # AIF / VOF
    pvc: bool = True
    select_vof: bool = True
    # deconvolution
    method: str = "osvd"           # bcsvd | osvd | fourier
    lam: float = 0.15              # bcsvd truncation
    oi_threshold: float = 0.095    # osvd target
    pr: float = 0.15               # fourier regularisation
    # thresholds
    core_frac: float = 0.30
    tmax_thr: float = 6.0
    vessel_cbv_max: float | None = None   # absolute cap; None -> relative rule
    vessel_cbv_rel: float = 2.5           # x brain-median CBV
    csf_hu_max: float = 15.0
    reference: str = "contralateral"
    min_cluster_ml: float = 1.0
    restrict_core_to_hypo: bool = False


@dataclass
class PipelineResult:
    mask: np.ndarray
    baseline: np.ndarray
    n_baseline: int
    aif: AifResult
    maps: PerfusionMaps
    thresholds: ThresholdResult
    qc: dict = field(default_factory=dict)


def run_pipeline(hu4d: np.ndarray, dt: float, voxel_size, config: PipelineConfig | None = None,
                 mask: np.ndarray | None = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    t0 = time.time()
    if mask is None:
        mask = brain_mask_ct(hu4d, lo=cfg.mask_lo_hu, hi=cfg.mask_hi_hu)
    # pass 1: provisional baseline from the whole-brain curve; AIF from unsmoothed concentration
    conc_raw, n_base, baseline = concentration_from_hu(hu4d, mask)
    aif = select_aif_vof(conc_raw, mask, dt, select_vof=cfg.select_vof, pvc=cfg.pvc)
    # pass 2: baseline = frames before the AIF's own arrival (10 % of its peak), then re-select
    arr_idx = int(round(curve_features(aif.aif_raw[None], dt, frac=0.10).arrival[0] / dt))
    n_base2 = max(2, min(arr_idx - 1, hu4d.shape[-1] - 10))
    if n_base2 != n_base:
        n_base = n_base2
        conc_raw, _, baseline = concentration_from_hu(hu4d, mask, n_baseline=n_base)
        aif = select_aif_vof(conc_raw, mask, dt, select_vof=cfg.select_vof, pvc=cfg.pvc)
    conc, _, _ = concentration_from_hu(hu4d, mask, n_baseline=n_base, spatial_sigma=cfg.spatial_sigma,
                                       temporal_sigma=cfg.temporal_sigma)
    kw = {"bcsvd": dict(lam=cfg.lam), "osvd": dict(oi_threshold=cfg.oi_threshold), "fourier": dict(pr=cfg.pr)}[cfg.method]
    maps = perfusion_maps(conc, mask, aif.aif, dt, method=cfg.method, **kw)
    thr = threshold_maps(maps, voxel_size, baseline=baseline, core_frac=cfg.core_frac, tmax_thr=cfg.tmax_thr,
                         vessel_cbv_max=cfg.vessel_cbv_max, vessel_cbv_rel=cfg.vessel_cbv_rel, csf_hu_max=cfg.csf_hu_max, reference=cfg.reference,
                         min_cluster_ml=cfg.min_cluster_ml, restrict_core_to_hypo=cfg.restrict_core_to_hypo)
    qc = {"seconds": round(time.time() - t0, 2), "n_baseline_frames": n_base, "n_frames": hu4d.shape[-1],
          "dt": dt, "n_mask_voxels": int(mask.sum()), **{f"aif_{k}": v for k, v in aif.qc.items()}, **maps.qc,
          "config": asdict(cfg)}
    return PipelineResult(mask=mask, baseline=baseline, n_baseline=n_base, aif=aif, maps=maps, thresholds=thr, qc=qc)

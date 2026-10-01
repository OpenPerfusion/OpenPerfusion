"""Masking, baseline subtraction and smoothing for 4D CT perfusion."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def brain_mask_ct(hu4d: np.ndarray, lo: float = 0.0, hi: float = 100.0, open_radius: int = 2,
                  erode: int = 1, per_slice: bool = True) -> np.ndarray:
    """Soft-tissue brain mask from the time-averaged CT.

    HU window [lo, hi] keeps parenchyma and CSF, drops bone and air; binary opening removes
    thin connections to scalp; the largest connected component is kept per slice (safer than 3D
    for sparse slabs); holes are filled and the edge eroded to avoid skull partial volume.
    """
    # Pre-contrast frames only (an enhanced artery's time-average can exceed `hi`), and every
    # one of them must be soft tissue: at bone edges, table-toggle / residual motion makes a voxel
    # flicker between bone and brain from frame to frame, and such voxels wreck AIF selection.
    n0 = max(2, min(4, hu4d.shape[-1]))
    pre = hu4d[..., :n0]
    base = pre.mean(axis=-1)
    m = (pre.min(axis=-1) > lo) & (pre.max(axis=-1) < hi)
    # and never bone at any time point (contrast never takes parenchyma above ~150 HU)
    m &= hu4d.max(axis=-1) < 600
    if open_radius > 0:
        se = _disk(open_radius)
        for z in range(m.shape[2]):
            m[:, :, z] = ndi.binary_opening(m[:, :, z], structure=se)
    out = np.zeros_like(m)
    if per_slice:
        for z in range(m.shape[2]):
            out[:, :, z] = _largest_cc(m[:, :, z])
            out[:, :, z] = ndi.binary_fill_holes(out[:, :, z])
            if erode > 0:
                out[:, :, z] = ndi.binary_erosion(out[:, :, z], structure=_disk(erode))
    else:
        out = _largest_cc(m)
        out = ndi.binary_fill_holes(out)
    return out


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _largest_cc(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def bolus_arrival_index(mean_curve: np.ndarray, frac: float = 0.10, min_baseline: int = 2, sustain: int = 3) -> int:
    """First frame at which the curve rises `frac` of its range above its pre-peak minimum AND stays
    there for `sustain` consecutive frames (a single noisy frame does not count as arrival)."""
    c = np.asarray(mean_curve, float)
    ipk = int(np.argmax(c))
    lo = c[: max(ipk, 1)].min()
    hi = c.max()
    if hi - lo <= 0:
        return min_baseline
    thr = lo + frac * (hi - lo)
    above = c > thr
    for i in range(0, len(c) - sustain + 1):
        if above[i : i + sustain].all():
            return int(max(min_baseline, i))
    return min_baseline


def concentration_from_hu(hu4d: np.ndarray, mask: np.ndarray, n_baseline: int | None = None,
                          spatial_sigma: float = 0.0, temporal_sigma: float = 0.0):
    """Contrast concentration (HU above pre-bolus baseline) inside `mask`.

    Returns (conc, n_baseline, baseline_image). Iodine concentration is linear in HU, so
    HU differences are used directly as concentration units (the absolute scale cancels
    in CBF/CBV when the AIF is in the same units).
    """
    hu = hu4d.astype(np.float32)
    if n_baseline is None:
        # Baseline frames must precede the EARLIEST arrival, i.e. the arteries' — so use the
        # mean curve of the brightest 0.5 % of voxels, not the whole-brain mean (tissue arrives
        # seconds later and would leave arterial signal inside the "baseline").
        # Initial guess: whole-brain mean curve (robust), tissue arrival at 10 % of its range,
        # minus a 3-frame margin for the arteries' lead. The pipeline then refines this from the
        # selected AIF's own arrival (see pipeline.run_pipeline).
        cur = hu[mask]
        n_baseline = max(2, bolus_arrival_index(cur.mean(axis=0), frac=0.10) - 3)
    baseline = hu[..., :n_baseline].mean(axis=-1)
    conc = hu - baseline[..., None]
    conc[~mask] = 0.0
    if spatial_sigma > 0:
        # in-plane smoothing only; slices may be far apart
        conc = ndi.gaussian_filter(conc, sigma=(spatial_sigma, spatial_sigma, 0, 0))
    if temporal_sigma > 0:
        conc = ndi.gaussian_filter1d(conc, sigma=temporal_sigma, axis=-1)
    conc[~mask] = 0.0
    return conc, n_baseline, baseline

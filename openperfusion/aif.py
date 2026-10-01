"""Automatic arterial input function (AIF) and venous output function (VOF) selection.

Follows the published RAPID approach (Straka, Albers, Bammer, JMRI 2010): each candidate
voxel's concentration curve is summarised by peak height h, arrival time a and width w;
a linear cost  c = k1*h + k2*a + k3*w  (k1 = 1.0, k2 = -3.5, k3 = -1.0, on standardised
features) prefers tall, early, narrow curves; the AIF is the mean curve of the best
spatially connected cluster. The VOF is chosen the same way among late-arriving, tall,
high-area curves and is used for partial-volume correction of the AIF (k_av = AUC_v / AUC_a).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass
class CurveFeatures:
    peak: np.ndarray       # max concentration
    arrival: np.ndarray    # s, first crossing of 20 % of peak
    ttp: np.ndarray        # s, time to peak
    width: np.ndarray      # s, full width at half maximum (sample count * dt)
    auc: np.ndarray        # concentration * s


def curve_features(curves: np.ndarray, dt: float, frac: float = 0.2) -> CurveFeatures:
    """Vectorised features for an (N, T) array of curves."""
    c = np.asarray(curves, np.float32)
    peak = c.max(axis=1)
    ipk = c.argmax(axis=1)
    thr = (frac * peak)[:, None]
    # arrival = last sample at or below `frac` of the peak BEFORE the peak, plus one.
    # Searching backwards from the peak makes this robust to early noise spikes.
    T = c.shape[1]
    tt = np.arange(T)[None, :]
    below_before = (c <= thr) & (tt < ipk[:, None])
    last_below = np.where(below_before.any(axis=1), T - 1 - np.argmax(below_before[:, ::-1], axis=1), -1)
    first = last_below + 1
    half = c > (0.5 * peak)[:, None]
    width = half.sum(axis=1) * dt
    auc = np.trapezoid(np.clip(c, 0, None), dx=dt, axis=1)
    return CurveFeatures(peak=peak, arrival=first * dt, ttp=ipk * dt, width=width, auc=auc)


def bolus_arrival_time(curve: np.ndarray, dt: float, frac: float = 0.10) -> float:
    """Time (s) at which a curve first rises `frac` of its range above its pre-peak minimum."""
    c = np.asarray(curve, float)
    ipk = int(np.argmax(c)); lo = c[: max(ipk, 1)].min(); hi = c.max()
    if hi - lo <= 0:
        return 0.0
    above = np.where(c > lo + frac * (hi - lo))[0]
    return float(above[0] * dt) if len(above) else 0.0


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def _best_cluster(mask3d: np.ndarray, score3d: np.ndarray, min_size: int) -> np.ndarray:
    """Connected component (26-conn) of `mask3d` with the highest mean score and >= min_size voxels."""
    lab, n = ndi.label(mask3d, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask3d
    best, best_score = None, -np.inf
    for k in range(1, n + 1):
        comp = lab == k
        sz = int(comp.sum())
        if sz < min_size:
            continue
        s = float(score3d[comp].mean())
        if s > best_score:
            best, best_score = comp, s
    return best if best is not None else mask3d


def _grow(seed: np.ndarray, conc: np.ndarray, mask: np.ndarray, frac: float = 0.6, iters: int = 3) -> np.ndarray:
    """Grow a seed cluster to the full vessel footprint: connected neighbours (in-plane) whose peak
    enhancement is at least `frac` of the seed's mean peak. Averages more voxels, less selection bias."""
    peak = conc.max(axis=-1)
    thr = frac * float(peak[seed].mean())
    ok = mask & (peak >= thr)
    se = np.zeros((3, 3, 3), bool); se[:, :, 1] = True
    grown = seed.copy()
    for _ in range(iters):
        grown = ndi.binary_dilation(grown, structure=se) & ok
    return grown


@dataclass
class AifResult:
    aif: np.ndarray          # (T,) selected arterial curve (after optional PVC)
    aif_raw: np.ndarray      # (T,) before PVC
    vof: np.ndarray | None   # (T,)
    aif_mask: np.ndarray     # (X,Y,Z) voxels averaged for the AIF
    vof_mask: np.ndarray | None
    k_av: float              # partial-volume factor applied (1.0 if none)
    aif_arrival: float       # s
    vof_arrival: float | None
    qc: dict


def select_aif_vof(conc: np.ndarray, mask: np.ndarray, dt: float,
                   candidate_percentile: float = 95.0, top_percentile: float = 80.0,
                   min_cluster: int = 3, weights=(1.0, -3.5, -1.0),
                   select_vof: bool = True, pvc: bool = True,
                   vof_min_transit: float = 2.0, truncation_frac: float = 0.3,
                   pvc_max: float = 2.5, aif_min_peak: float = 15.0) -> AifResult:
    """Select AIF (and VOF) from a (X,Y,Z,T) concentration array within `mask`."""
    X, Y, Z, T = conc.shape
    idx = np.argwhere(mask)
    curves = conc[mask]                       # (N, T)
    # features on a 3-frame median-filtered copy: a one- or two-frame spike (motion, streak,
    # table-toggle flicker) must not be mistaken for a bolus
    med = ndi.median_filter(curves, size=(1, 3), mode="nearest")
    f = curve_features(med, dt)
    raw_peak = curves.max(axis=1)
    # bolus window anchored on the whole-brain tissue curve. Its time-to-peak is robust (the
    # arrival of a noisy mean curve is not); arteries peak before tissue does.
    mean_curve = ndi.gaussian_filter1d(curves.mean(axis=0), 1.0)
    ttp_tissue = float(np.argmax(mean_curve) * dt)
    t_tissue = bolus_arrival_time(mean_curve, dt)

    # ---- plausibility: a single clean bolus, not flicker, streak or noise ----
    cmin = med.min(axis=1)
    shape_ok = (f.peak > 0) & (cmin > -0.3 * f.peak) & (f.width >= 3 * dt) & (f.width <= 15.0) & (f.peak >= 0.7 * raw_peak)
    # primary window: anchored on the tissue arrival (validated on ISLES 2018)
    plausible = shape_ok & (f.ttp >= t_tissue - 8.0) & (f.ttp <= t_tissue + 12.0)
    # fallback window: anchored on the tissue peak, and the pre-arrival segment must be flat
    # (no bump above 30 % of the peak before the curve rises)
    T_idx = np.arange(T)[None, :]
    pre = np.where(T_idx < (f.arrival / dt)[:, None], med, 0.0).max(axis=1)
    plausible_fb = shape_ok & (pre <= 0.3 * f.peak) & (f.ttp >= ttp_tissue - 15.0) & (f.ttp <= ttp_tissue + 1.0)
    k1, k2, k3 = weights

    def _pick(cand, min_sz):
        cost = np.full(len(curves), -np.inf, np.float32)
        cost[cand] = k1 * _z(f.peak[cand]) + k2 * _z(f.arrival[cand]) + k3 * _z(f.width[cand])
        ctop = np.percentile(cost[cand], top_percentile)
        top = cost >= ctop
        top3d = np.zeros(mask.shape, bool); top3d[tuple(idx[top].T)] = True
        score3d = np.full(mask.shape, -np.inf, np.float32); score3d[tuple(idx.T)] = cost
        m = _best_cluster(top3d, score3d, min_sz)
        m = _grow(m, conc, mask, frac=0.6)
        return m

    # ---- primary: tallest plausible curves (large arteries, well above tissue enhancement) ----
    pk = np.where(plausible, f.peak, 0)
    pk_all = f.peak
    thr = max(np.percentile(pk, candidate_percentile), 0.25 * np.percentile(pk, 99.9))
    cand = plausible & (pk >= thr)
    if cand.sum() < 2 * min_cluster:
        cand = plausible & (pk >= np.percentile(pk, candidate_percentile))
    qc = {}
    if cand.sum() >= min_cluster:
        aif_mask = _pick(cand, min_cluster)
        aif_raw = conc[aif_mask].mean(axis=0)
    else:
        aif_mask = np.zeros(mask.shape, bool); aif_raw = np.zeros(T, np.float32)
    fa = curve_features(aif_raw[None], dt)
    # ---- fallback: thin slabs often hold no large artery. If the primary pick is weak or does
    # not lead the tissue peak, search a wider pool for small, early, narrow (partial-volumed) arteries,
    # weighting timing over amplitude, and accept 2-voxel clusters. ----
    ttp_primary = float(fa.ttp[0])
    weak = ((aif_raw.max() < aif_min_peak)            # no arterial amplitude
            or (ttp_primary > ttp_tissue - 1.0)       # does not lead the tissue peak: a vein or tissue
            or (ttp_primary < max(4.0, ttp_tissue - 15.0)))   # peaks before any bolus could: noise
    qc["aif_primary_weak"] = bool(weak)
    if weak:
        pool = plausible_fb & (f.width <= 12.0) & (f.peak >= max(12.0, np.percentile(pk_all, 90)))
        if pool.sum() >= 2:
            m2 = _pick(pool, 2)
            a2 = conc[m2].mean(axis=0)
            f2 = curve_features(a2[None], dt)
            # take the fallback if it is a plausible artery: leads the tissue peak, has some amplitude
            if float(f2.ttp[0]) <= ttp_tissue - 1.0 and a2.max() >= 12.0:
                qc["aif_fallback"] = True
                aif_mask, aif_raw = m2, a2
    aif_arrival = float(curve_features(aif_raw[None], dt).arrival[0])

    vof = vof_mask = vof_arrival = None
    k_av = 1.0
    qc.update({"n_aif_voxels": int(aif_mask.sum()), "aif_peak": float(aif_raw.max()),
               "aif_ttp": float(curve_features(aif_raw[None], dt).ttp[0]), "tissue_ttp": ttp_tissue})

    if select_vof:
        # ---- venous candidates: tall, high-area, arriving after the AIF ----
        aif_ttp = float(curve_features(aif_raw[None], dt).ttp[0])
        late = (f.arrival >= aif_arrival + vof_min_transit) & (f.ttp >= aif_ttp + vof_min_transit) & (f.ttp <= aif_ttp + 15.0)
        # venous curves may peak after the tissue window; relax the upper ttp bound for them
        vplaus = ((f.peak > 0) & (cmin > -0.3 * f.peak) & (f.width >= 3 * dt) & (f.width <= 20.0) & (f.peak >= 0.7 * raw_peak))
        vcand = vplaus & (pk_all >= thr) & late
        if vcand.sum() >= min_cluster:
            vcost = np.full(len(curves), -np.inf, np.float32)
            vcost[vcand] = _z(f.peak[vcand]) + 0.5 * _z(f.arrival[vcand]) - 0.5 * _z(f.width[vcand])
            vtop = vcost >= np.percentile(vcost[vcand], top_percentile)
            vtop3d = np.zeros(mask.shape, bool); vtop3d[tuple(idx[vtop].T)] = True
            vscore = np.full(mask.shape, -np.inf, np.float32); vscore[tuple(idx.T)] = vcost
            vof_mask = _best_cluster(vtop3d, vscore, min_cluster)
            vof_mask = _grow(vof_mask, conc, mask, frac=0.6)
            vof = conc[vof_mask].mean(axis=0)
            vof_arrival = float(curve_features(vof[None], dt).arrival[0])
            qc["n_vof_voxels"] = int(vof_mask.sum()); qc["vof_peak"] = float(vof.max())
            # truncation check: venous curve should have washed out by the last frame
            tail = float(vof[-3:].mean()) / max(float(vof.max()), 1e-6)
            qc["vof_tail_fraction"] = tail
            qc["truncated"] = bool(tail > truncation_frac)
            if pvc and not qc["truncated"]:
                auc_a = np.trapezoid(aif_raw, dx=dt)
                auc_v = np.trapezoid(vof, dx=dt)
                if auc_a > 0 and auc_v > auc_a:
                    k_av = float(auc_v / auc_a)
                # a partial-volume factor beyond ~2.5 means the "artery" is not an artery (or the
                # "vein" is not a vein); do not scale by it, flag it instead
                if k_av > pvc_max:
                    qc["k_av_rejected"] = k_av
                    k_av = 1.0
        else:
            qc["vof"] = "no late-arriving candidates"

    # QC: a weak arterial peak means the slab holds no usable artery; maps are unreliable
    qc["aif_weak"] = bool(aif_raw.max() < aif_min_peak)
    aif = aif_raw * k_av
    qc["k_av"] = k_av
    return AifResult(aif=aif, aif_raw=aif_raw, vof=vof, aif_mask=aif_mask, vof_mask=vof_mask,
                     k_av=k_av, aif_arrival=aif_arrival, vof_arrival=vof_arrival, qc=qc)

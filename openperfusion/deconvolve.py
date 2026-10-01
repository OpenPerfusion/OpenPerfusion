"""Delay-insensitive deconvolution of tissue concentration curves by the AIF.

Three engines, all circular (zero-padded to 2N so arrival before the AIF wraps rather
than being clipped — Wu et al., MRM 2003):

* bcSVD  — block-circulant SVD with a fixed truncation threshold (fraction of the largest
           singular value). Ostergaard 1996 discretisation, Wu 2003 circulant form.
* oSVD   — bcSVD with the truncation chosen per voxel so the oscillation index of the
           residue function falls below a target (Wu 2003).
* fourier — frequency-domain division with Wiener-style regularisation, the formulation
           described for RAPID (Straka, Albers, Bammer, JMRI 2010): G = |Ca|^2/(|Ca|^2+N^2),
           N = (pr/2) * max|Ca(f)|.

All return the scaled residue function  k(t) = F * R(t)  sampled at dt, its maximum and the
time of that maximum (Tmax, seconds; negative when the tissue leads the AIF).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DeconvResult:
    kmax: np.ndarray      # (N_voxels,) max of the scaled residue function [conc/conc/s]
    tmax: np.ndarray      # (N_voxels,) seconds
    residue: np.ndarray | None = None   # (N_voxels, L) optional
    lam: np.ndarray | None = None       # (N_voxels,) truncation used (oSVD)


def _pad(x: np.ndarray, L: int, axis: int = -1) -> np.ndarray:
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, L - x.shape[axis])
    return np.pad(x, pad)


def circulant_matrix(aif: np.ndarray, dt: float, L: int | None = None, discretisation: str = "simpson") -> np.ndarray:
    """L x L circulant convolution matrix D such that (D @ k) approximates dt * (aif (*) k)."""
    N = len(aif)
    L = L or 2 * N
    a = _pad(np.asarray(aif, float), L)
    if discretisation == "simpson":
        # Ostergaard 1996 eq. 6: a_ij = dt*(C(t_{i-j-1}) + 4C(t_{i-j}) + C(t_{i-j+1}))/6, circular
        col = dt * (np.roll(a, 1) + 4 * a + np.roll(a, -1)) / 6.0
    elif discretisation == "rect":
        col = dt * a
    else:
        raise ValueError(discretisation)
    i = np.arange(L)
    D = col[(i[:, None] - i[None, :]) % L]
    return D


def _tmax_from_residue(k: np.ndarray, dt: float, L: int, interp: bool = True) -> np.ndarray:
    idx = k.argmax(axis=1)
    t = idx.astype(float)
    if interp:
        # parabolic sub-sample refinement around the peak
        im = (idx - 1) % L; ip = (idx + 1) % L
        rows = np.arange(len(k))
        y0, y1, y2 = k[rows, im], k[rows, idx], k[rows, ip]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-12, den, 1), 0.0)
        t = t + np.clip(off, -0.5, 0.5)
    t = np.where(t > L / 2, t - L, t)
    return t * dt


def deconvolve_bcsvd(curves: np.ndarray, aif: np.ndarray, dt: float, lam: float = 0.15,
                     discretisation: str = "simpson", keep_residue: bool = False,
                     chunk: int = 50000) -> DeconvResult:
    """Block-circulant SVD deconvolution of (N_voxels, T) curves with a fixed truncation `lam`."""
    curves = np.asarray(curves, np.float32)
    N = curves.shape[1]
    L = 2 * N
    D = circulant_matrix(aif, dt, L, discretisation)
    U, S, Vt = np.linalg.svd(D)
    Sinv = np.where(S >= lam * S.max(), 1.0 / np.where(S > 0, S, 1), 0.0)
    P = (Vt.T * Sinv) @ U.T                      # pseudo-inverse, L x L
    P = P.astype(np.float32)
    kmax = np.empty(len(curves), np.float32); tmax = np.empty(len(curves), np.float32)
    res = np.empty((len(curves), L), np.float32) if keep_residue else None
    for s in range(0, len(curves), chunk):
        c = _pad(curves[s:s + chunk], L)
        k = c @ P.T                               # (n, L)
        kmax[s:s + chunk] = k.max(axis=1)
        tmax[s:s + chunk] = _tmax_from_residue(k, dt, L)
        if keep_residue:
            res[s:s + chunk] = k
    return DeconvResult(kmax=kmax, tmax=tmax, residue=res)


def oscillation_index(k: np.ndarray) -> np.ndarray:
    """Wu 2003: OI = (1/L) * (1/max|k|) * sum |k_i - 2k_{i-1} + k_{i-2}|."""
    L = k.shape[1]
    d2 = np.abs(k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]).sum(axis=1)
    m = np.abs(k).max(axis=1)
    return d2 / (L * np.where(m > 0, m, 1))


def deconvolve_osvd(curves: np.ndarray, aif: np.ndarray, dt: float, oi_threshold: float = 0.095,
                    lams=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
                    discretisation: str = "simpson", keep_residue: bool = False,
                    chunk: int = 50000) -> DeconvResult:
    """Oscillation-index regularised SVD: per voxel, the smallest truncation whose residue
    function has OI <= oi_threshold (Wu 2003 used 0.035 for MR; CT is noisier)."""
    curves = np.asarray(curves, np.float32)
    N = curves.shape[1]
    L = 2 * N
    D = circulant_matrix(aif, dt, L, discretisation)
    U, S, Vt = np.linalg.svd(D)
    Ps = []
    for lam in lams:
        Sinv = np.where(S >= lam * S.max(), 1.0 / np.where(S > 0, S, 1), 0.0)
        Ps.append(((Vt.T * Sinv) @ U.T).astype(np.float32))
    n = len(curves)
    kmax = np.empty(n, np.float32); tmax = np.empty(n, np.float32); lam_used = np.empty(n, np.float32)
    res = np.empty((n, L), np.float32) if keep_residue else None
    for s in range(0, n, chunk):
        c = _pad(curves[s:s + chunk], L)
        best_k = None
        done = np.zeros(len(c), bool)
        kk = np.zeros((len(c), L), np.float32)
        lu = np.full(len(c), lams[-1], np.float32)
        for lam, P in zip(lams, Ps):
            k = c @ P.T
            oi = oscillation_index(k)
            take = (~done) & (oi <= oi_threshold)
            kk[take] = k[take]; lu[take] = lam; done |= take
            if done.all():
                break
        if not done.all():                      # fall back to the strongest regularisation
            kk[~done] = k[~done]
        kmax[s:s + chunk] = kk.max(axis=1)
        tmax[s:s + chunk] = _tmax_from_residue(kk, dt, L)
        lam_used[s:s + chunk] = lu
        if keep_residue:
            res[s:s + chunk] = kk
    return DeconvResult(kmax=kmax, tmax=tmax, residue=res, lam=lam_used)


def deconvolve_fourier(curves: np.ndarray, aif: np.ndarray, dt: float, pr: float = 0.15,
                       keep_residue: bool = False, chunk: int = 50000) -> DeconvResult:
    """Circular frequency-domain deconvolution with Wiener-style regularisation (Straka 2010)."""
    curves = np.asarray(curves, np.float32)
    N = curves.shape[1]
    L = 2 * N
    Ca = np.fft.rfft(_pad(np.asarray(aif, float), L))
    Nn = (pr / 2.0) * np.abs(Ca).max()
    G = np.abs(Ca) ** 2 / (np.abs(Ca) ** 2 + Nn ** 2)
    H = (G / np.where(np.abs(Ca) > 0, Ca, 1)) / dt          # applied to Ct(f)
    n = len(curves)
    kmax = np.empty(n, np.float32); tmax = np.empty(n, np.float32)
    res = np.empty((n, L), np.float32) if keep_residue else None
    for s in range(0, n, chunk):
        c = _pad(curves[s:s + chunk], L)
        k = np.fft.irfft(np.fft.rfft(c, axis=1) * H[None, :], n=L, axis=1).astype(np.float32)
        kmax[s:s + chunk] = k.max(axis=1)
        tmax[s:s + chunk] = _tmax_from_residue(k, dt, L)
        if keep_residue:
            res[s:s + chunk] = k
    return DeconvResult(kmax=kmax, tmax=tmax, residue=res)

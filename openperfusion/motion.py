# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Rigid in-plane motion correction for 4D CT perfusion.

Head motion during a 45–60 s CTP acquisition is mostly in-plane (translation plus a small
rotation about the z axis); through-plane motion cannot be recovered from a thin slab anyway.
Each frame is registered to a reference image — the mean of the pre-contrast frames — slice by
slice:

1. translation by phase correlation of bone-and-brain edge maps (contrast enhancement changes
   the parenchyma's intensity between frames; gradients of a skull-dominated image do not), with
   sub-pixel refinement;
2. rotation by a small bracketed search (±max_rot degrees) on the normalised cross-correlation of
   the translated frame against the reference, then one more translation step at that angle.

Frames are resampled with linear interpolation (`order=1`). The per-frame parameters are returned
so a QC step can flag large motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass
class MotionResult:
    hu: np.ndarray            # corrected (X, Y, Z, T)
    dx: np.ndarray            # (Z, T) voxels, applied shift along x
    dy: np.ndarray            # (Z, T) voxels
    theta: np.ndarray         # (Z, T) degrees
    max_shift_mm: float
    reference: np.ndarray     # (X, Y, Z)


def _edges(img: np.ndarray) -> np.ndarray:
    """Edge magnitude of a skull-weighted image: robust to the parenchymal enhancement between frames."""
    g = np.clip(img, -200, 400)
    gx = ndi.sobel(g, axis=0); gy = ndi.sobel(g, axis=1)
    e = np.hypot(gx, gy)
    e -= e.mean()
    return e


def _phase_shift(ref_e: np.ndarray, mov_e: np.ndarray, max_shift: float) -> tuple[float, float]:
    """Translation (dx, dy) that maps `mov` onto `ref`, by phase correlation with parabolic sub-pixel fit."""
    F = np.fft.fft2(ref_e) * np.conj(np.fft.fft2(mov_e))
    F /= np.abs(F) + 1e-9
    r = np.real(np.fft.ifft2(F))
    r = np.fft.fftshift(r)
    cx, cy = r.shape[0] // 2, r.shape[1] // 2
    w = int(np.ceil(max_shift))
    win = r[cx - w: cx + w + 1, cy - w: cy + w + 1]
    i, j = np.unravel_index(np.argmax(win), win.shape)
    dx, dy = float(i - w), float(j - w)
    # parabolic refinement
    def refine(a, b, c):
        den = a - 2 * b + c
        return 0.0 if abs(den) < 1e-12 else 0.5 * (a - c) / den
    if 0 < i < win.shape[0] - 1:
        dx += refine(win[i - 1, j], win[i, j], win[i + 1, j])
    if 0 < j < win.shape[1] - 1:
        dy += refine(win[i, j - 1], win[i, j], win[i, j + 1])
    return dx, dy


def _apply(img: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    """Rotate by theta (deg) about the image centre then shift by (dx, dy); linear interpolation; air fill."""
    out = img
    if theta != 0.0:
        out = ndi.rotate(out, theta, reshape=False, order=1, mode="constant", cval=-1000.0)
    if dx != 0.0 or dy != 0.0:
        out = ndi.shift(out, (dx, dy), order=1, mode="constant", cval=-1000.0)
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation on a brain window (−20..150 HU): the skull ring is nearly
    rotationally symmetric, so rotation must be judged on parenchymal structure as well."""
    a = np.clip(a, -20, 150); b = np.clip(b, -20, 150)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def motion_correct(hu4d: np.ndarray, n_ref: int = 3, max_shift: float = 12.0, max_rot: float = 6.0,
                   voxel_size=(1.0, 1.0, 5.0), rot_step: float = 0.5, est_size: int = 256,
                   ncc_skip: float = 0.995) -> MotionResult:
    """Register every frame of every slice to the mean of the first `n_ref` (pre-contrast) frames.

    Parameters are estimated on a copy downsampled to about `est_size` pixels across (a 512 x 512
    vendor export is searched at 256 x 256, which is 4x faster and loses nothing at the sub-voxel
    precision the parabolic fit gives) and applied at full resolution. The rotation search is skipped
    for a frame whose translation-only fit already reaches `ncc_skip` normalised cross-correlation,
    which is most frames of a still patient.
    """
    X, Y, Z, T = hu4d.shape
    ref = hu4d[..., :n_ref].mean(axis=-1)
    out = np.empty_like(hu4d)
    dxs = np.zeros((Z, T)); dys = np.zeros((Z, T)); ths = np.zeros((Z, T))
    f = max(1, int(round(min(X, Y) / est_size)))          # integer downsampling factor for the search
    def small(img):
        return img if f == 1 else img.reshape(X // f, f, Y // f, f).mean(axis=(1, 3)) if (X % f == 0 and Y % f == 0) else ndi.zoom(img, 1.0 / f, order=1)
    for z in range(Z):
        ref_img = ref[:, :, z]
        ref_s = small(ref_img)
        ref_e = _edges(ref_s)
        for t in range(T):
            mov = hu4d[:, :, z, t]
            mov_s = small(mov)
            # translation first (on the small image; shifts scaled back to full-resolution voxels)
            dx, dy = _phase_shift(ref_e, _edges(mov_s), max_shift / f)
            best = (0.0, dx, dy, _ncc(ref_s, _apply(mov_s, dx, dy, 0.0)))
            if max_rot > 0 and best[3] < ncc_skip:
                # coarse-to-fine rotation search; translation re-estimated at each angle
                def score(th):
                    rot = ndi.rotate(mov_s, th, reshape=False, order=1, mode="constant", cval=-1000.0)
                    ddx, ddy = _phase_shift(ref_e, _edges(rot), max_shift / f)
                    return (th, ddx, ddy, _ncc(ref_s, _apply(rot, ddx, ddy, 0.0)))
                for th in np.arange(-max_rot, max_rot + 1e-9, 1.0):
                    if th == 0.0:
                        continue
                    c = score(th)
                    if c[3] > best[3]:
                        best = c
                for th in (best[0] - 0.5, best[0] + 0.5, best[0] - 0.25, best[0] + 0.25):
                    c = score(th)
                    if c[3] > best[3]:
                        best = c
            th, dx, dy, _ = best
            dx *= f; dy *= f
            out[:, :, z, t] = _apply(mov, dx, dy, th)
            dxs[z, t], dys[z, t], ths[z, t] = dx, dy, th
    max_mm = float(np.hypot(dxs * voxel_size[0], dys * voxel_size[1]).max())
    return MotionResult(hu=out, dx=dxs, dy=dys, theta=ths, max_shift_mm=max_mm, reference=ref)

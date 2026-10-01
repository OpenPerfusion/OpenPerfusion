# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Digital CT perfusion phantom with known ground truth.

Modelled on the ASIST-Japan / Kudo (Radiology 2013) digital phantom: a grid of tissue
tiles that span CBF x MTT, one slice per tracer delay, with an arterial input region,
a venous output region, "normal" reference tissue, a CSF-like block, skull and air.

Indicator-dilution model (Meier & Zierler; Ostergaard 1996):

    C_t(t) = (rho / k_H) * F * (C_a (*) R)(t - delay)

    F   = CBF / 6000            [mL/g/s]   (CBF in mL/100 g/min)
    R   = exp(-t / MTT)  or  box(t < MTT)
    CBV = CBF * MTT / 60        [mL/100 g]

Curves are generated on a fine time grid and then sampled at the acquisition interval,
so the estimator sees realistic discretisation error, not a matched discrete model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

RHO = 1.04     # g/mL brain tissue density
K_H = 0.73     # (1 - Hct_large) / (1 - Hct_small) = (1-0.45)/(1-0.25)


def gamma_variate(t: np.ndarray, t0: float, alpha: float, beta: float, peak: float) -> np.ndarray:
    """Gamma-variate bolus, normalised so its maximum equals `peak`."""
    tt = np.clip(t - t0, 0, None)
    y = tt ** alpha * np.exp(-tt / beta)
    y = np.where(t > t0, y, 0.0)
    m = y.max()
    return peak * y / m if m > 0 else y


@dataclass
class PhantomTruth:
    t: np.ndarray            # (T,) seconds
    dt: float
    aif: np.ndarray          # (T,) true arterial concentration (HU above baseline)
    vof: np.ndarray          # (T,) true venous concentration
    cbf: np.ndarray          # (X,Y,Z) mL/100g/min (0 where no tissue)
    cbv: np.ndarray          # (X,Y,Z) mL/100g
    mtt: np.ndarray          # (X,Y,Z) s
    delay: np.ndarray        # (X,Y,Z) s
    tmax: np.ndarray         # (X,Y,Z) s  (argmax of residue == delay for exp/box residues)
    tissue_mask: np.ndarray  # tiles + normal tissue
    tile_mask: np.ndarray    # the CBF x MTT tiles only
    normal_mask: np.ndarray  # reference tissue (CBF 50, MTT 4, delay 0)
    aif_mask: np.ndarray
    vof_mask: np.ndarray
    csf_mask: np.ndarray
    brain_mask: np.ndarray
    voxel_size: tuple
    baseline_hu: float


def make_phantom(
    cbf_values=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
    mtt_values=(4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0),
    delays=(0.0, 1.0, 2.0, 3.0),
    residue: str = "exp",
    tile: int = 10,
    size: int = 128,
    dt: float = 1.0,
    duration: float = 60.0,
    noise_sd: float = 3.0,
    baseline_hu: float = 35.0,
    aif_peak: float = 400.0,
    aif_partial_volume: float = 0.7,
    voxel_size=(1.0, 1.0, 5.0),
    dt_fine: float = 0.05,
    seed: int | None = 0,
):
    """Return (hu_4d, truth). hu_4d has shape (X, Y, Z, T) in Hounsfield units."""
    rng = np.random.default_rng(seed)
    cbf_values = np.asarray(cbf_values, float)
    mtt_values = np.asarray(mtt_values, float)
    nz = len(delays)
    ny_t, nx_t = len(mtt_values), len(cbf_values)

    t_fine = np.arange(0.0, duration, dt_fine)
    t = np.arange(0.0, duration, dt)
    idx = np.round(t / dt_fine).astype(int)

    # --- input functions (fine grid) ---
    aif_f = gamma_variate(t_fine, t0=5.0, alpha=3.0, beta=1.5, peak=aif_peak)
    # venous: delayed and dispersed copy of the AIF with the same area (mass conservation)
    disp = gamma_variate(t_fine, t0=0.0, alpha=2.0, beta=1.0, peak=1.0)
    disp /= disp.sum()
    vof_f = np.convolve(aif_f, disp)[: len(t_fine)]
    vof_f = np.roll(vof_f, int(round(4.0 / dt_fine)))
    vof_f *= aif_f.sum() / vof_f.sum()

    def residue_fn(mtt):
        if residue == "exp":
            return np.exp(-t_fine / mtt)
        if residue == "box":
            return (t_fine < mtt).astype(float)
        raise ValueError(residue)

    def tissue_curve(cbf, mtt, delay):
        F = cbf / 6000.0
        R = residue_fn(mtt)
        c = (RHO / K_H) * F * np.convolve(aif_f, R)[: len(t_fine)] * dt_fine
        return np.roll(c, int(round(delay / dt_fine)))

    X = Y = size
    shape = (X, Y, nz)
    cbf = np.zeros(shape); cbv = np.zeros(shape); mtt = np.zeros(shape); dly = np.zeros(shape)
    tile_mask = np.zeros(shape, bool); normal_mask = np.zeros(shape, bool)
    aif_mask = np.zeros(shape, bool); vof_mask = np.zeros(shape, bool); csf_mask = np.zeros(shape, bool)
    brain_mask = np.zeros(shape, bool)
    hu = np.zeros((X, Y, nz, len(t)), np.float32)

    yy, xx = np.mgrid[0:Y, 0:X]
    cx, cy = X / 2 - 0.5, Y / 2 - 0.5
    r = np.hypot(xx - cx, yy - cy)
    brain_r = size * 0.42
    disk = r <= brain_r
    skull = (r > brain_r) & (r <= brain_r + 4)

    grid_w, grid_h = nx_t * tile, ny_t * tile
    gx0, gy0 = int(cx - grid_w / 2), int(cy - grid_h / 2)

    normal_curve = tissue_curve(50.0, 4.0, 0.0)[idx]
    for z, delay in enumerate(delays):
        sl_cbf = np.zeros((Y, X)); sl_mtt = np.zeros((Y, X)); sl_dly = np.zeros((Y, X))
        sl_tile = np.zeros((Y, X), bool)
        curves = np.zeros((Y, X, len(t)), np.float32)
        # normal reference tissue fills the brain disk
        curves[disk] = normal_curve
        sl_cbf[disk] = 50.0; sl_mtt[disk] = 4.0; sl_dly[disk] = 0.0
        # CBF x MTT tiles
        for j, m in enumerate(mtt_values):
            for i, f in enumerate(cbf_values):
                ys = slice(gy0 + j * tile, gy0 + (j + 1) * tile)
                xs = slice(gx0 + i * tile, gx0 + (i + 1) * tile)
                curves[ys, xs] = tissue_curve(f, m, delay)[idx]
                sl_cbf[ys, xs] = f; sl_mtt[ys, xs] = m; sl_dly[ys, xs] = delay
                sl_tile[ys, xs] = True
        # arterial region (partial-volumed), venous region, CSF block
        a_sl = (slice(gy0 - 10, gy0 - 5), slice(int(cx) - 3, int(cx) + 3))
        v_sl = (slice(gy0 + grid_h + 5, gy0 + grid_h + 10), slice(int(cx) - 3, int(cx) + 3))
        c_sl = (slice(gy0 - 10, gy0 - 5), slice(int(cx) + 10, int(cx) + 18))
        curves[a_sl] = (aif_partial_volume * aif_f)[idx]
        curves[v_sl] = vof_f[idx]
        curves[c_sl] = 0.0
        sl_cbf[a_sl] = 0; sl_cbf[v_sl] = 0; sl_cbf[c_sl] = 0
        sl_mtt[a_sl] = 0; sl_mtt[v_sl] = 0; sl_mtt[c_sl] = 0

        base = np.full((Y, X), baseline_hu, np.float32)
        base[c_sl] = 5.0
        base[~disk] = -1000.0
        base[skull] = 1000.0
        vol = base[..., None] + curves
        vol[~disk] = base[~disk][..., None]
        vol += rng.normal(0.0, noise_sd, vol.shape).astype(np.float32)

        # store as (X, Y, Z, T) — transpose the (Y, X) image axes
        hu[:, :, z, :] = vol.transpose(1, 0, 2)
        cbf[:, :, z] = sl_cbf.T; mtt[:, :, z] = sl_mtt.T; dly[:, :, z] = sl_dly.T
        tile_mask[:, :, z] = sl_tile.T
        nm = disk & ~sl_tile
        nm[a_sl] = False; nm[v_sl] = False; nm[c_sl] = False
        normal_mask[:, :, z] = nm.T
        am = np.zeros((Y, X), bool); am[a_sl] = True; aif_mask[:, :, z] = am.T
        vm = np.zeros((Y, X), bool); vm[v_sl] = True; vof_mask[:, :, z] = vm.T
        cm = np.zeros((Y, X), bool); cm[c_sl] = True; csf_mask[:, :, z] = cm.T
        brain_mask[:, :, z] = disk.T

    cbv = cbf * mtt / 60.0
    tissue_mask = tile_mask | normal_mask
    truth = PhantomTruth(
        t=t, dt=dt, aif=aif_f[idx], vof=vof_f[idx],
        cbf=cbf, cbv=cbv, mtt=mtt, delay=dly, tmax=dly.copy(),
        tissue_mask=tissue_mask, tile_mask=tile_mask, normal_mask=normal_mask,
        aif_mask=aif_mask, vof_mask=vof_mask, csf_mask=csf_mask, brain_mask=brain_mask,
        voxel_size=tuple(voxel_size), baseline_hu=baseline_hu,
    )
    return hu, truth

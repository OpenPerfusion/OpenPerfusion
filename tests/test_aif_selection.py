# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""The arterial pick on synthetic 4D data with the three traps found on ISLES'24: an early, dim cluster beside
bright arteries (lead rule), an artery far smaller than the vein (vein rule), and a slow bolus whose arteries
are wider than the fixed width bound (width rule). The whole-brain tissue curve peaks at 19 s in each."""
import numpy as np
import pytest

from openperfusion.phantom import gamma_variate
from openperfusion.aif import select_aif_vof

T, DT = 50, 1.0
t = np.arange(T) * DT


def _curve(peak, t0, width_beta=1.5, alpha=3.0):
    return gamma_variate(t, t0=t0, alpha=alpha, beta=width_beta, peak=peak)


def _scene(early_peak=100.0, artery_peak=300.0, vein_peak=260.0, artery_t0=6.0, artery_beta=1.5, noise=1.5, seed=0):
    """64 x 64 x 2 brain disk of tissue (the artery convolved with a 4 s exponential residue, 20 HU, arriving
    ~1 s after the artery and peaking ~3 s after it) with three small blocks: an early dim cluster (peaks 5 s),
    bright arteries (t0 `artery_t0`, peak ~10.5 s) and a vein (peaks ~18 s)."""
    rng = np.random.default_rng(seed)
    X = Y = 64; Z = 2
    conc = np.zeros((X, Y, Z, T), np.float32)
    yy, xx = np.mgrid[0:Y, 0:X]; disk = np.hypot(xx - 31.5, yy - 31.5) <= 28
    mask = np.zeros((X, Y, Z), bool); mask[..., :] = disk[..., None]
    artery = _curve(artery_peak, artery_t0, width_beta=artery_beta)
    tissue = np.convolve(artery, np.exp(-t / 4.0))[:T]; tissue *= 20.0 / tissue.max()
    conc[mask] = tissue
    blocks = {"early": ((14, 26, 14, 26), _curve(early_peak, 3.0, width_beta=1.0, alpha=2.0)),   # peaks 5 s; a large population, as on ISLES'24
              "artery": ((40, 44, 36, 40), artery),
              "vein": ((24, 28, 42, 46), _curve(vein_peak, 12.0, width_beta=2.0))}
    for (x0, x1, y0, y1), c in blocks.values():
        conc[x0:x1, y0:y1, :, :] = c
    conc += rng.normal(0, noise, conc.shape).astype(np.float32)
    return conc, mask


def test_bright_artery_beats_early_dim_cluster():
    conc, mask = _scene()
    r = select_aif_vof(conc, mask, DT, snr_min=3.0)
    assert r.qc["aif_peak"] > 200, r.qc
    assert 1.0 <= r.qc["aif_lead"] <= 8.0, r.qc


def test_lead_rule_is_what_does_it():
    conc, mask = _scene()
    r0 = select_aif_vof(conc, mask, DT, max_lead=0.0, pvc=False, snr_min=3.0)       # rule off: the early cluster wins on arrival
    r1 = select_aif_vof(conc, mask, DT, pvc=False, snr_min=3.0)
    assert r0.qc["aif_peak"] < 100 and r1.qc["aif_peak"] > 200
    assert r1.qc.get("aif_reselected_lead")


def test_vein_rule_rescues_a_small_artery_pick():
    # lead rule off: the early 100 HU population wins on arrival; the vein (260 HU, broad) then has > 2.5 x its
    # area, so the pick is repeated among curves with at least vein/2.5 of enhancement and lands on the arteries
    conc, mask = _scene()
    r = select_aif_vof(conc, mask, DT, max_lead=0.0, snr_min=3.0)
    assert r.qc.get("aif_reselected_vein") and r.qc["aif_peak"] > 200 and r.k_av < 2.5, r.qc


def test_wide_arteries_accepted_when_the_bolus_is_slow():
    # a slow injection: tissue 2x wider, arteries 20 s wide; a fixed 15 s bound would exclude every artery
    rng = np.random.default_rng(1)
    X = Y = 64; Z = 2
    conc = np.zeros((X, Y, Z, T), np.float32)
    yy, xx = np.mgrid[0:Y, 0:X]; disk = np.hypot(xx - 31.5, yy - 31.5) <= 28
    mask = np.zeros((X, Y, Z), bool); mask[..., :] = disk[..., None]
    conc[mask] = _curve(15.0, 10.0, width_beta=5.0, alpha=3.0)         # broad tissue curve
    conc[40:44, 36:40] = _curve(250.0, 8.0, width_beta=4.5)             # broad arteries, FWHM ~20 s
    conc[24:28, 42:46] = _curve(220.0, 16.0, width_beta=5.0)
    conc += rng.normal(0, 1.5, conc.shape).astype(np.float32)
    r = select_aif_vof(conc, mask, DT)
    assert r.qc["aif_peak"] > 150, r.qc

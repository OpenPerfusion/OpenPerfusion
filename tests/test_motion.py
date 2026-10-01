# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Motion correction recovers imposed in-plane shifts and rotations on the phantom."""
import numpy as np
from scipy import ndimage as ndi

from openperfusion.phantom import make_phantom
from openperfusion.motion import motion_correct
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.validate import phantom_report


def _impose(hu, rng, max_shift=6.0, max_rot=4.0, from_frame=5):
    X, Y, Z, T = hu.shape
    moved = hu.copy()
    truth = np.zeros((T, 3))
    for t in range(from_frame, T):
        dx, dy = rng.uniform(-max_shift, max_shift, 2); th = rng.uniform(-max_rot, max_rot)
        truth[t] = (dx, dy, th)
        for z in range(Z):
            img = ndi.rotate(hu[:, :, z, t], -th, reshape=False, order=1, mode="constant", cval=-1000.0)
            moved[:, :, z, t] = ndi.shift(img, (-dx, -dy), order=1, mode="constant", cval=-1000.0)
    return moved, truth


def _add_structure(hu, truth, rng, amplitude=15.0):
    """The phantom brain is a featureless disk before contrast arrives, so rotation is unobservable
    on it; real parenchyma has 10–15 HU grey/white structure. Add a fixed smooth texture inside the brain."""
    X, Y, Z, T = hu.shape
    out = hu.copy()
    for z in range(Z):
        tex = ndi.gaussian_filter(rng.normal(0, 1, (X, Y)), 3.0)
        tex *= amplitude / tex.std()
        tex[~truth.brain_mask[:, :, z]] = 0.0
        out[:, :, z, :] += tex[:, :, None]
    return out


def test_recovers_shift_and_rotation():
    hu, tr = make_phantom(delays=(0.0, 2.0), noise_sd=3.0, size=128, tile=10, duration=50.0)
    rng = np.random.default_rng(1)
    hu = _add_structure(hu, tr, rng)
    moved, truth = _impose(hu, rng)
    res = motion_correct(moved, n_ref=3, max_shift=10, max_rot=6, rot_step=0.5)
    # estimated parameters should undo the imposed ones (same sign convention: we shift back by +dx)
    for t in range(5, hu.shape[3]):
        for z in range(hu.shape[2]):
            assert abs(res.dx[z, t] - truth[t, 0]) < 0.6, (t, z, res.dx[z, t], truth[t, 0])
            assert abs(res.dy[z, t] - truth[t, 1]) < 0.6
            assert abs(res.theta[z, t] - truth[t, 2]) < 1.0
    # and the corrected data give the right maps again
    before = phantom_report(run_pipeline(moved, tr.dt, tr.voxel_size, PipelineConfig(method="bcsvd", lam=0.1)).maps, tr)["summary"]
    after = phantom_report(run_pipeline(res.hu, tr.dt, tr.voxel_size, PipelineConfig(method="bcsvd", lam=0.1)).maps, tr)["summary"]
    assert after["pearson_cbv"] > 0.95
    assert after["pearson_cbv"] > before["pearson_cbv"]

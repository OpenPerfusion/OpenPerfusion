# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""QC verdict: a clean phantom is `ok`; a multi-peak arterial curve, a last-resort pick or a weak AIF is `reject`;
a fallback AIF or a truncated venous curve is `review`."""
import copy
import numpy as np
import pytest

from openperfusion.phantom import make_phantom
from openperfusion.pipeline import run_pipeline
from openperfusion.qc import qc_verdict


@pytest.fixture(scope="module")
def res():
    hu, truth = make_phantom(noise_sd=2.0, seed=2)
    return run_pipeline(hu, truth.dt, truth.voxel_size)


def test_clean_phantom_ok(res):
    v = qc_verdict(res)
    assert v["status"] == "ok" and v["warnings"] == [] and v["aif_peak_hu"] > 100


def test_flags_drive_verdict(res):
    r = copy.copy(res); r.aif = copy.copy(res.aif); r.aif.qc = dict(res.aif.qc)
    r.aif.qc["aif_fallback"] = True
    assert qc_verdict(r)["status"] == "review"
    r.aif.qc["truncated"] = True; r.aif.qc["vof_tail_fraction"] = 0.4
    v = qc_verdict(r); assert v["status"] == "review" and len(v["warnings"]) == 2
    r.aif.qc["aif_last_resort"] = True
    assert qc_verdict(r)["status"] == "reject"


def test_multipeak_curve_rejected(res):
    r = copy.copy(res); r.aif = copy.copy(res.aif)
    a = res.aif.aif_raw.copy()
    # add a second bolus-sized bump late in the curve
    k = int(np.argmax(a)); a2 = a.copy(); a2[min(len(a) - 1, k + 20):] += 0.8 * a.max()
    r.aif.aif_raw = a2
    v = qc_verdict(r)
    assert v["status"] == "reject" and any("multi-peak" in w for w in v["warnings"])


def test_motion_and_timing_warnings(res):
    class M: max_shift_mm = 7.0; theta = np.array([1.0, -6.5])
    v = qc_verdict(res, meta={"time_source": "instance_order"}, motion=M())
    assert v["status"] == "review" and len(v["warnings"]) == 2

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Phantom recovery tests: the pipeline must find the input functions, recover CBV and CBF
to within known SVD tolerances, and respond to delay one-for-one (delay-insensitivity)."""
import numpy as np
import pytest

from openperfusion.phantom import make_phantom
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.validate import phantom_report


@pytest.fixture(scope="module")
def phantom_run():
    hu, truth = make_phantom(noise_sd=3.0, seed=0)
    res = run_pipeline(hu, truth.dt, truth.voxel_size, PipelineConfig(method="osvd"))
    return hu, truth, res, phantom_report(res.maps, truth)["summary"]


def test_input_functions_found(phantom_run):
    _, truth, res, _ = phantom_run
    assert truth.aif_mask[res.aif.aif_mask].mean() > 0.95
    assert truth.vof_mask[res.aif.vof_mask].mean() > 0.95
    assert abs(res.aif.k_av - 1 / 0.7) / (1 / 0.7) < 0.05


def test_cbv_recovery(phantom_run):
    *_, s = phantom_run
    assert abs(s["cbv_rel_err_mean"]) < 0.10
    assert s["pearson_cbv"] > 0.98


def test_cbf_recovery_within_svd_tolerance(phantom_run):
    *_, s = phantom_run
    # truncated SVD underestimates CBF, more so at short MTT; document rather than hide it
    assert -0.25 < s["cbf_rel_err_mean_cbf30plus"] < 0.10
    assert s["pearson_cbf"] > 0.95


def test_delay_insensitive(phantom_run):
    *_, s = phantom_run
    resp = s["delay_response"]
    for d, r in resp.items():
        assert abs(r - float(d)) < 0.35, f"delay {d}: response {r}"
    for d, e in s["cbf_rel_err_by_delay"].items():
        assert abs(e - s["cbf_rel_err_by_delay"][0.0]) < 0.03


def test_tmax_bias_bounded(phantom_run):
    *_, s = phantom_run
    assert 0 <= s["tmax_bias_s"] < 3.0
    assert s["tmax_sd_s"] < 1.0


def test_vessels_excluded_from_tissue(phantom_run):
    _, truth, res, _ = phantom_run
    assert not res.thresholds.tissue[truth.aif_mask].any()
    assert not res.thresholds.tissue[truth.vof_mask].any()
    assert not res.thresholds.tissue[truth.csf_mask].any()


@pytest.mark.parametrize("method", ["bcsvd", "fourier"])
def test_other_engines_run(method):
    hu, truth = make_phantom(delays=(0.0, 2.0), noise_sd=3.0)
    res = run_pipeline(hu, truth.dt, truth.voxel_size, PipelineConfig(method=method))
    s = phantom_report(res.maps, truth)["summary"]
    assert s["pearson_cbv"] > 0.98
    assert abs(s["delay_response"][2.0] - 2.0) < 0.5

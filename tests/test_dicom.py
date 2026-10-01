"""DICOM round trip: phantom -> DICOM files (uniform and shuttle-mode timing) -> loader -> pipeline."""
import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from openperfusion.phantom import make_phantom
from openperfusion.io_dicom import write_ctp_dicom, read_ctp_dicom
from openperfusion.pipeline import run_pipeline, PipelineConfig
from openperfusion.validate import phantom_report


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(delays=(0.0, 2.0), noise_sd=2.0, size=96, tile=8, duration=50.0)


def test_uniform_round_trip(tmp_path, phantom):
    hu, truth = phantom
    n = write_ctp_dicom(hu, tmp_path / "uniform", dt=truth.dt, voxel_size=truth.voxel_size)
    assert n == hu.shape[2] * hu.shape[3]
    c = read_ctp_dicom(tmp_path / "uniform", dt=truth.dt)
    assert c.hu.shape == hu.shape
    assert c.voxel_size == pytest.approx(truth.voxel_size)
    assert np.abs(c.hu - hu).max() <= 0.5 + 1e-3             # 16-bit integer storage only
    assert c.meta["shuttle_mode"] is False


def test_shuttle_mode_resampling(tmp_path, phantom):
    """Two table positions sampled alternately every 2 s (offset 1 s): the loader must rebuild a 1 s grid."""
    hu, truth = phantom
    T = hu.shape[3]
    times = [np.arange(0, T, 2.0), np.arange(1, T, 2.0)]        # slice 0 at even seconds, slice 1 at odd
    write_ctp_dicom(hu, tmp_path / "shuttle", dt=truth.dt, voxel_size=truth.voxel_size, times_per_slice=times)
    c = read_ctp_dicom(tmp_path / "shuttle", dt=1.0)
    assert c.meta["shuttle_mode"] is True
    assert c.hu.shape[:3] == hu.shape[:3]
    # resampled curves track the true curves inside the brain (linear interpolation of a 1 s-sampled bolus)
    m = truth.tissue_mask
    T2 = min(c.hu.shape[3], T)
    err = np.abs(c.hu[..., :T2][m] - hu[..., :T2][m]).mean()
    assert err < 4.0, err
    # and the pipeline still recovers the phantom from the resampled data
    res = run_pipeline(c.hu, c.dt, c.voxel_size, PipelineConfig(method="bcsvd", lam=0.10))
    s = phantom_report(res.maps, truth)["summary"]
    assert s["pearson_cbv"] > 0.95
    assert abs(s["delay_response"][2.0] - 2.0) < 0.6

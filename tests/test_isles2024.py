# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""ISLES'24 loader: resampling through world coordinates, parquet round trip, reference-map masking, and the
rigid registration refinement (SimpleITK, skipped when not installed)."""
import gzip, io
import numpy as np, nibabel as nib, pytest

from openperfusion.phantom import make_phantom
from openperfusion.io_isles2024 import resample_to, _nifti_from_bytes, _build, load_subject_from_parquet, find_shards


@pytest.fixture(scope="module")
def phantom():
    hu, truth = make_phantom(delays=(0.0, 1.0, 2.0, 3.0), noise_sd=2.0, seed=1)
    return hu, truth


def _nifti(arr, affine):
    return nib.Nifti1Image(np.asarray(arr, np.float32), affine)


def test_resample_identity_and_shift():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(20, 18, 6)).astype(np.float32)
    A = np.diag([1.0, 1.0, 5.0, 1.0])
    img = _nifti(vol, A)
    same = resample_to(img, vol.shape, A, order=1)
    assert np.allclose(same, vol, atol=1e-5)
    # a target grid shifted by exactly one voxel in x should return the volume shifted by one voxel
    B = A.copy(); B[0, 3] = 1.0
    shifted = resample_to(img, vol.shape, B, order=0)
    assert np.allclose(shifted[:-1], vol[1:])


def test_nifti_bytes_round_trip():
    vol = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    img = _nifti(vol, np.eye(4))
    raw = img.to_bytes()
    assert np.allclose(np.asarray(_nifti_from_bytes(raw).dataobj), vol)
    assert np.allclose(np.asarray(_nifti_from_bytes(gzip.compress(raw)).dataobj), vol)


def test_build_masks_reference_background(phantom):
    hu, truth = phantom
    A = np.diag([1.0, 1.0, 5.0, 1.0])
    ctp = nib.Nifti1Image(hu, A); ctp.header.set_zooms((1.0, 1.0, 5.0, 1.0))
    # reference maps on a coarser, shifted grid with icobrain's conventions (-30 Tmax outside, 0 CBF outside)
    B = np.diag([2.0, 2.0, 5.0, 1.0]); B[0, 3] = 3.0
    shape_ref = (hu.shape[0] // 2, hu.shape[1] // 2, hu.shape[2])
    cbf = np.zeros(shape_ref, np.float32); cbf[5:-5, 5:-5] = 40.0
    tmax = np.full(shape_ref, -30.0, np.float32); tmax[5:-5, 5:-5] = 3.0
    imgs = {"ctp": ctp, "cbf": _nifti(cbf, B), "cbv": _nifti(cbf / 10, B), "mtt": _nifti(cbf / 8, B), "tmax": _nifti(tmax, B),
            "lesion_mask": _nifti((tmax > 0).astype(np.float32), B)}
    d = _build("synthetic", imgs, load_4d=True, register=False)
    assert d["dt"] == 1.0 and d["ctp4d"].shape == hu.shape and d["registration"]["method"] == "header"
    inside = np.isfinite(d["tmax"])
    assert inside.any() and (d["tmax"][inside] > 0).all(), "background (-30) must be NaN, not a value"
    assert np.isnan(d["cbf"][~inside]).all()
    # the lesion is pulled with nearest-neighbour, the maps with linear interpolation, so the two disagree only
    # along the boundary of the reference's field of view
    assert d["lesion"].dtype == bool and d["lesion"].sum() > 0 and (d["lesion"] & ~inside).sum() < 0.05 * d["lesion"].sum()


def test_parquet_round_trip(tmp_path, phantom):
    pq = pytest.importorskip("pyarrow.parquet"); pa = pytest.importorskip("pyarrow")
    hu, truth = phantom
    A = np.diag([1.0, 1.0, 5.0, 1.0])
    ctp = nib.Nifti1Image(hu, A); ctp.header.set_zooms((1.0, 1.0, 5.0, 1.0))
    tmax = _nifti(np.full(hu.shape[:3], 4.0), A); cbf = _nifti(np.full(hu.shape[:3], 30.0), A)
    row = {"subject_id": "sub-stroke9999", "age": 71, "sex": "F", "nihss_admission": 12, "mrs_admission": 0, "mrs_3month": 2,
           "ctp": {"bytes": gzip.compress(ctp.to_bytes()), "path": "ctp.nii.gz"},
           "cbf": {"bytes": cbf.to_bytes(), "path": "cbf.nii.gz"}, "cbv": {"bytes": cbf.to_bytes(), "path": "cbv.nii.gz"},
           "mtt": {"bytes": cbf.to_bytes(), "path": "mtt.nii.gz"}, "tmax": {"bytes": tmax.to_bytes(), "path": "tmax.nii.gz"}}
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "train-00000-of-00001.parquet")
    assert [p.name for p in find_shards(tmp_path)] == ["train-00000-of-00001.parquet"]
    d = load_subject_from_parquet(tmp_path / "train-00000-of-00001.parquet", load_4d=True, register=False)
    assert d["case"] == "sub-stroke9999" and d["clinical"]["nihss_admission"] == 12
    assert np.allclose(d["ctp4d"], hu) and np.nanmax(d["tmax"]) == 4.0 and d["lesion"] is None


def test_rigid_registration_recovers_shift(phantom):
    sitk = pytest.importorskip("SimpleITK")
    from openperfusion.register import register_ctp_to_ref, resample_ref_to_ctp, transform_summary
    hu, truth = phantom
    A = np.diag([1.0, 1.0, 5.0, 1.0])
    ctp = nib.Nifti1Image(hu, A)
    mean = hu[..., :3].mean(-1)
    # the "NCCT" is the same anatomy whose header claims it sits 4 mm away in x: a 4-voxel shift of the data
    ncct_data = np.roll(mean, 4, axis=0)
    ncct = _nifti(ncct_data, A)
    tx = register_ctp_to_ref(ctp, mean, ncct)
    s = transform_summary(tx)
    assert abs(abs(s["shift_mm"][0]) - 4.0) < 1.0, s
    back = resample_ref_to_ctp(ncct, ctp, tx, order=1)
    brain = (mean > 0) & (mean < 100)
    c_hdr = np.corrcoef(np.clip(mean[brain], 0, 100), np.clip(ncct_data[brain], 0, 100))[0, 1]
    c_reg = np.corrcoef(np.clip(mean[brain], 0, 100), np.clip(back[brain], 0, 100))[0, 1]
    assert c_reg > c_hdr + 0.05

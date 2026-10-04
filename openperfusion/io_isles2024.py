# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Loader for ISLES'24 (Riedel et al., Radiology: AI 2026): 149 training subjects from two centres on
Siemens (Somatom Force / Xcite / AS+) and Philips (Brilliance 64 / Ingenuity) scanners, with 4D CTP
(co-registered, resampled to 1 s), perfusion maps from icobrain cva (CBF, CBV, MTT, Tmax) in NCCT space,
and a follow-up DWI lesion mask.

Expected layout (what `unpack_isles24.py` writes from the Hugging Face parquet mirror, or the BIDS tree
from Zenodo flattened per subject):

    <root>/sub-strokeNNNN/sub-strokeNNNN_ses-01_ctp.nii.gz                 (X, Y, Z, T) HU, native CTP grid
    <root>/sub-strokeNNNN/sub-strokeNNNN_ses-01_space-ncct_{cbf,cbv,mtt,tmax}.nii.gz
    <root>/sub-strokeNNNN/sub-strokeNNNN_ses-02_space-ncct_lesion-msk.nii.gz

The maps and masks live on the NCCT grid (0.47 x 0.47 x 2 mm, tilted); the CTP on its own grid
(1 x 1 x 5 mm here). Both carry world affines, so the maps are resampled into the CTP grid through
world coordinates (linear for maps, nearest for masks). `load_subject` returns the same dict as
`io_isles2018.load_case`, so the ISLES 2018 harness (`scripts/sweep_isles2018.run_config`) runs unchanged.
"""
from __future__ import annotations

from pathlib import Path
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


def find_subjects(root) -> list[Path]:
    return sorted(p for p in Path(root).glob("sub-stroke*") if p.is_dir() and list(p.glob("*_ctp.nii.gz")))


def _first(d: Path, pattern: str):
    f = sorted(d.glob(pattern))
    return f[0] if f else None


def resample_to(src_img, target_shape, target_affine, order=1, cval=0.0) -> np.ndarray:
    """Resample a 3D NIfTI image onto another grid via world coordinates."""
    src = np.asarray(src_img.dataobj, np.float32)
    # target voxel -> world -> source voxel
    M = np.linalg.inv(src_img.affine) @ target_affine
    X, Y, Z = target_shape
    ii, jj, kk = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], axis=0)
    sv = (M @ coords)[:3]
    out = ndi.map_coordinates(src, sv, order=order, mode="constant", cval=cval)
    return out.reshape(X, Y, Z).astype(np.float32)


def load_subject(sub: Path, load_4d: bool = True, register: bool = True) -> dict:
    sub = Path(sub)
    ctp_f = _first(sub, "*_ses-01_ctp.nii*")
    if ctp_f is None:
        raise FileNotFoundError(f"no CTP in {sub}")
    ctp = nib.load(str(ctp_f))
    zooms = ctp.header.get_zooms()
    dt = float(zooms[3]) if len(zooms) > 3 and zooms[3] > 0 else 1.0
    if dt > 20:               # some headers store ms
        dt /= 1000.0
    shape3 = ctp.shape[:3]
    voxel_size = tuple(float(z) for z in zooms[:3])
    ctp_affine = ctp.affine
    out = dict(case=sub.name, dt=dt, voxel_size=voxel_size, affine=ctp_affine, paths={"ctp": str(ctp_f)}, ct=None)
    hu = np.asarray(ctp.dataobj, np.float32)
    out["ctp4d"] = hu if load_4d else None
    # alignment of the NCCT-space maps to the CTP grid: header first, refined by rigid registration of the
    # CTP pre-contrast mean to the NCCT when the NCCT is present and SimpleITK is installed
    ncct_f = _first(sub, "*_ses-01_ncct.nii*")
    tx = None
    out["registration"] = {"method": "header"}
    if ncct_f is not None and register:
        try:
            from .register import register_ctp_to_ref, resample_ref_to_ctp, transform_summary
            ncct = nib.load(str(ncct_f))
            mean = hu[..., :max(3, min(8, hu.shape[-1] // 6))].mean(-1)
            brain = (mean > 0) & (mean < 100)
            def ncc(a, b):
                a = np.clip(a[brain], 0, 100); b = np.clip(b[brain], 0, 100)
                return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
            n_hdr = resample_to(ncct, shape3, ctp_affine, order=1)
            tx_try = register_ctp_to_ref(ctp, mean, ncct)
            n_reg = resample_ref_to_ctp(ncct, ctp, tx_try)
            c0, c1 = ncc(mean, n_hdr), ncc(mean, n_reg)
            out["registration"] = {"method": "header", "ncc_header": round(c0, 3), "ncc_registered": round(c1, 3), **transform_summary(tx_try)}
            if c1 > c0 + 0.02:
                tx = tx_try; out["registration"]["method"] = "rigid"
        except ImportError:
            pass
    def pull(img, order):
        if tx is None:
            return resample_to(img, shape3, ctp_affine, order=order)
        from .register import resample_ref_to_ctp
        return resample_ref_to_ctp(img, ctp, tx, order=order)
    for key in ("cbf", "cbv", "mtt", "tmax"):
        f = _first(sub, f"*_ses-01_space-ncct_{key}.nii*")
        out[key] = pull(nib.load(str(f)), 1) if f else None
        if f: out["paths"][key] = str(f)
    les = _first(sub, "*_ses-02_space-ncct_lesion-msk.nii*")
    out["lesion"] = (pull(nib.load(str(les)), 0) > 0.5) if les else None
    if les: out["paths"]["lesion"] = str(les)
    for key in ("lvo", "cow"):
        f = _first(sub, f"*_space-ncct_{key}-msk.nii*")
        if f: out["paths"][key] = str(f)
    # icobrain fills non-brain voxels with -30 in Tmax and 0 in the other maps; mark those NaN so the
    # harness (which keeps voxels with finite reference and CBF > 0) skips them
    if out["tmax"] is not None:
        bad = ~(np.isfinite(out["cbf"]) & (out["cbf"] > 0) & (out["tmax"] > -1.0))
        for key in ("cbf", "cbv", "mtt", "tmax"):
            if out[key] is not None:
                out[key] = out[key].copy(); out[key][bad] = np.nan
    return out

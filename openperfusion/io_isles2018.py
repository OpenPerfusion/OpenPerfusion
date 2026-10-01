# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Loader for the ISLES 2018 CT perfusion challenge data (Hakim et al., Stroke 2021).

Expected layout (SMIR export, re-hosted on Zenodo record 17736412):

    TRAINING/case_1/
        SMIR.Brain.XX.O.CT.<id>/SMIR.Brain.XX.O.CT.<id>.nii          baseline CT (3D)
        SMIR.Brain.XX.O.CT_4DPWI.<id>/...nii                          4D CTP source (X,Y,Z[,1],T)
        SMIR.Brain.XX.O.CT_CBF.<id>/...nii                            RAPID CBF
        SMIR.Brain.XX.O.CT_CBV.<id>/...nii                            RAPID CBV
        SMIR.Brain.XX.O.CT_MTT.<id>/...nii                            RAPID MTT
        SMIR.Brain.XX.O.CT_Tmax.<id>/...nii                           RAPID Tmax
        SMIR.Brain.XX.O.OT.<id>/...nii                                infarct ground truth (DWI-based)

File names are matched by substring, so minor variations in the export survive.
Temporal spacing is read from the NIfTI header (pixdim[4]) and falls back to 1.0 s, which
is what the challenge resampled to. Always check `dt` in the returned dict.
"""
from __future__ import annotations

from pathlib import Path
import glob
import numpy as np
import nibabel as nib

KEYS = {
    "ct": ".CT.", "ctp4d": "CT_4DPWI", "cbf": "CT_CBF", "cbv": "CT_CBV", "mtt": "CT_MTT",
    "tmax": "CT_Tmax", "lesion": ".OT.",
}


def find_cases(root: str | Path) -> list[Path]:
    root = Path(root)
    cases = sorted([p for p in root.glob("**/case_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[-1]))
    return cases


def _find(case: Path, key: str) -> Path | None:
    hits = [Path(p) for p in glob.glob(str(case / "**" / "*.nii*"), recursive=True) if key in Path(p).name]
    return hits[0] if hits else None


def load_case(case: str | Path, load_4d: bool = True) -> dict:
    """Return dict with arrays (X,Y,Z[,T]), voxel_size, dt, paths. Missing items are None."""
    case = Path(case)
    out = {"case": case.name, "paths": {}}
    for k, key in KEYS.items():
        p = _find(case, key)
        out["paths"][k] = str(p) if p else None
        if p is None or (k == "ctp4d" and not load_4d):
            out[k] = None
            continue
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj)
        if k == "ctp4d":
            arr = np.squeeze(arr)                      # drop a singleton 4th dim if present
            if arr.ndim != 4:
                raise ValueError(f"{p}: expected 4D CTP, got shape {arr.shape}")
            zooms = img.header.get_zooms()
            out["voxel_size"] = tuple(float(z) for z in zooms[:3])
            dt = float(zooms[4]) if len(zooms) > 4 and zooms[4] > 0 else (float(zooms[3]) if len(zooms) > 3 and 0 < zooms[3] < 10 else 1.0)
            out["dt"] = dt if 0 < dt < 10 else 1.0
            out["affine"] = img.affine
            arr = arr.astype(np.float32)
        elif k in ("lesion",):
            arr = arr.astype(bool)
        else:
            arr = arr.astype(np.float32)
        out[k] = arr
    if "voxel_size" not in out and out.get("cbf") is not None:
        img = nib.load(out["paths"]["cbf"])
        out["voxel_size"] = tuple(float(z) for z in img.header.get_zooms()[:3])
        out["affine"] = img.affine
    return out


def save_nifti(arr: np.ndarray, affine, path: str | Path):
    nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), affine), str(path))

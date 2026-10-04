# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Rigid 3D registration between a CTP slab and a reference CT (NCCT) in another space, so that maps
and masks defined on the reference grid can be compared voxel by voxel with ours.

ISLES'24 stores icobrain's maps and the follow-up lesion on the NCCT grid and the 4D CTP on its own
grid. The two headers put both in scanner coordinates, but the patient moves between acquisitions,
and for some subjects the header-implied alignment is off by several millimetres and a slice or two,
which wrecks a voxelwise comparison while leaving volumes almost untouched. This module refines the
header alignment with an intensity-based rigid registration (SimpleITK, Mattes mutual information,
multi-resolution), then provides a resampler that takes a reference-space image into the CTP grid
through the refined transform.

SimpleITK is an optional dependency (`pip install SimpleITK`); without it, `register_ctp_to_ref`
raises ImportError and callers fall back to the header alignment.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib

try:
    import SimpleITK as sitk
except ImportError:                      # pragma: no cover
    sitk = None


def _to_sitk(img: nib.Nifti1Image, data: np.ndarray | None = None) -> "sitk.Image":
    """nibabel (RAS affine, x-fastest array) -> SimpleITK (LPS)."""
    arr = np.asarray(img.dataobj if data is None else data, np.float32)
    out = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))     # sitk wants z,y,x
    A = img.affine.copy()
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])                                          # RAS -> LPS
    A = flip @ A
    spacing = np.linalg.norm(A[:3, :3], axis=0)
    direction = A[:3, :3] / spacing
    out.SetSpacing(tuple(float(s) for s in spacing))
    out.SetOrigin(tuple(float(o) for o in A[:3, 3]))
    out.SetDirection(tuple(float(v) for v in direction.flatten(order="C")))
    return out


def register_ctp_to_ref(ctp_img: nib.Nifti1Image, ctp_mean: np.ndarray, ref_img: nib.Nifti1Image,
                        window=(0.0, 100.0), iterations: int = 200) -> "sitk.Transform":
    """Rigid transform (fixed = reference CT, moving = CTP mean), initialised from the headers.

    Both images are windowed to brain soft tissue so skull and air do not dominate; the metric is
    Mattes mutual information on a 3-level pyramid. Returns a SimpleITK Euler3DTransform mapping
    fixed (reference) physical points to moving (CTP) physical points.
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for registration: pip install SimpleITK")
    fixed = _to_sitk(ref_img, np.clip(np.asarray(ref_img.dataobj, np.float32), *window))
    moving = _to_sitk(ctp_img, np.clip(ctp_mean, *window))
    fixed = sitk.Cast(fixed, sitk.sitkFloat32); moving = sitk.Cast(moving, sitk.sitkFloat32)
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # headers already roughly agree; start from identity-in-physical-space rather than centred geometry
    tx = sitk.Euler3DTransform()
    tx.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2.0 for s in fixed.GetSize()]))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.2, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=iterations, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(tx, inPlace=True)
    reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0]); reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.Execute(fixed, moving)
    return tx


def transform_summary(tx: "sitk.Transform") -> dict:
    p = tx.GetParameters()
    return {"rot_deg": [round(float(np.degrees(v)), 2) for v in p[:3]], "shift_mm": [round(float(v), 2) for v in p[3:6]]}


def resample_ref_to_ctp(ref_img: nib.Nifti1Image, ctp_img: nib.Nifti1Image, tx: "sitk.Transform",
                        order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Resample an image on the reference grid into the CTP grid through the refined rigid transform
    (header alignment composed with the registration). Returns an (X, Y, Z) array in CTP voxel order."""
    moving = _to_sitk(ref_img)
    target = _to_sitk(ctp_img, np.zeros(ctp_img.shape[:3], np.float32))
    interp = sitk.sitkLinear if order == 1 else sitk.sitkNearestNeighbor
    # tx maps fixed(ref) -> moving(ctp) physical points; to pull ref values onto the CTP grid we need its inverse
    inv = tx.GetInverse()
    out = sitk.Resample(moving, target, inv, interp, float(cval), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out).transpose(2, 1, 0).astype(np.float32)

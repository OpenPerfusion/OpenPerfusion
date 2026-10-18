# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Write OpenPerfusion maps back out as DICOM so they can sit beside the source study in a PACS or Orthanc.

Two kinds of object:

* one Secondary Capture (RGB) series per map: CBF, CBV, MTT, Tmax as colour images with a fixed window and the
  thresholded regions (core, Tmax > 6 s) as colour overlays on the mean pre-contrast image, one instance per slice;
  every viewer displays Secondary Capture, which is why RAPID's own output arrives the same way;
* one summary image: the volumes, mismatch, DEFUSE-3 flag, QC verdict and warnings rendered as text, so the
  result can be read without any software.

All new objects copy the patient and study attributes from a source instance so they file under the same study,
get new Series/SOP Instance UIDs, SeriesDescription "OpenPerfusion <map> (research)" and a SeriesNumber of
9000 + n so they sort after the clinical series. Nothing here claims to be a diagnostic image: ImageType is
DERIVED\\SECONDARY, and the word "research" is in every series description.
"""
from __future__ import annotations

from pathlib import Path
import datetime as _dt
import numpy as np

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:                      # pragma: no cover
    pydicom = None

WINDOWS = {"cbf": (0.0, 100.0), "cbv": (0.0, 8.0), "mtt": (0.0, 16.0), "tmax": (0.0, 12.0)}
UNITS = {"cbf": "mL/100g/min", "cbv": "mL/100g", "mtt": "s", "tmax": "s"}
SERIES_NUMBER = {"cbf": 9001, "cbv": 9002, "mtt": 9003, "tmax": 9004, "regions": 9005, "summary": 9009}


def _jet(x: np.ndarray) -> np.ndarray:
    """(…) in [0, 1] -> (…, 3) uint8, matplotlib-free jet."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def _grey(img: np.ndarray, lo=0.0, hi=80.0) -> np.ndarray:
    g = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def map_rgb(values: np.ndarray, mask: np.ndarray, background: np.ndarray, window) -> np.ndarray:
    """Colour map inside the brain mask over a grey background (one slice, (X, Y) -> (rows, cols, 3))."""
    rgb = _grey(background.T)
    col = _jet((values.T - window[0]) / (window[1] - window[0]))
    m = mask.T
    rgb[m] = col[m]
    return rgb


def regions_rgb(core: np.ndarray, hypo: np.ndarray, background: np.ndarray) -> np.ndarray:
    rgb = _grey(background.T).astype(np.int16)
    h = hypo.T & ~core.T; c = core.T
    rgb[h] = (0.5 * rgb[h] + 0.5 * np.array([255, 40, 40])).astype(np.int16)
    rgb[c] = (0.5 * rgb[c] + 0.5 * np.array([40, 80, 255])).astype(np.int16)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _text_image(lines: list[str], width=512, height=512) -> np.ndarray:
    """Render text with PIL when available, else a blank image (the JSON summary still carries the numbers)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new("RGB", (width, height), (0, 0, 0)); d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
        except Exception:
            font = ImageFont.load_default()
        y = 12
        for ln in lines:
            d.text((12, y), ln[:60], fill=(255, 255, 255), font=font); y += 20
            if y > height - 20:
                break
        return np.asarray(im)
    except ImportError:
        return np.zeros((height, width, 3), np.uint8)


def _wrap(text: str, width: int = 58) -> list[str]:
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    return out + ([line] if line else [])


def _new_sc(src, series_uid: str, series_number: int, description: str, instance_number: int, rgb: np.ndarray, z_mm: float | None = None):
    fm = FileMetaDataset(); fm.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid(); fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
    for kw in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyID", "StudyDate", "StudyTime",
               "AccessionNumber", "ReferringPhysicianName", "StudyDescription", "FrameOfReferenceUID", "PixelSpacing", "SliceThickness",
               "ImageOrientationPatient"):
        if src is not None and kw in src:
            setattr(ds, kw, src.get(kw))
    now = _dt.datetime.now()
    ds.SOPClassUID = fm.MediaStorageSOPClassUID; ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "OT"; ds.ConversionType = "WSD"; ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.SeriesInstanceUID = series_uid; ds.SeriesNumber = series_number; ds.SeriesDescription = description
    ds.SeriesDate = now.strftime("%Y%m%d"); ds.SeriesTime = now.strftime("%H%M%S"); ds.ContentDate = ds.SeriesDate; ds.ContentTime = ds.SeriesTime
    ds.Manufacturer = "OpenPerfusion (open source, research use only)"; ds.InstanceNumber = instance_number
    if z_mm is not None and src is not None and "ImagePositionPatient" in src:
        ipp = list(src.ImagePositionPatient); ipp[2] = float(z_mm); ds.ImagePositionPatient = ipp
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel = 3; ds.PhotometricInterpretation = "RGB"; ds.PlanarConfiguration = 0
    ds.BitsAllocated = 8; ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
    ds.PixelData = np.ascontiguousarray(rgb, np.uint8).tobytes()
    return ds


def write_result_dicom(res, out_dir: str | Path, source_ds=None, z_positions=None, summary: dict | None = None, verdict: dict | None = None) -> list[Path]:
    """Write map, region and summary Secondary Capture series for a PipelineResult. Returns the files written."""
    if pydicom is None:
        raise ImportError("pydicom is required: pip install pydicom")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    written = []
    bg = res.baseline if res.baseline is not None else np.zeros(res.mask.shape)
    Z = res.mask.shape[2]
    for key in ("cbf", "cbv", "mtt", "tmax"):
        suid = generate_uid(); vals = getattr(res.maps, key)
        for z in range(Z):
            rgb = map_rgb(vals[:, :, z], res.mask[:, :, z], bg[:, :, z], WINDOWS[key])
            ds = _new_sc(source_ds, suid, SERIES_NUMBER[key], f"OpenPerfusion {key.upper()} {WINDOWS[key][0]:.0f}-{WINDOWS[key][1]:.0f} {UNITS[key]} (research)", z + 1, rgb,
                         z_positions[z] if z_positions is not None else None)
            p = out / f"{key}_{z:03d}.dcm"; ds.save_as(str(p), enforce_file_format=True); written.append(p)
    suid = generate_uid(); t = res.thresholds
    for z in range(Z):
        rgb = regions_rgb(t.core[:, :, z], t.hypo[:, :, z], bg[:, :, z])
        ds = _new_sc(source_ds, suid, SERIES_NUMBER["regions"], "OpenPerfusion core rCBF<30% blue, Tmax>6s red (research)", z + 1, rgb,
                     z_positions[z] if z_positions is not None else None)
        p = out / f"regions_{z:03d}.dcm"; ds.save_as(str(p), enforce_file_format=True); written.append(p)
    lines = ["OpenPerfusion  (open source; RESEARCH USE ONLY)", "not for clinical decision-making", ""]
    if verdict:
        lines += [f"QC: {verdict['status'].upper()}"] + sum((_wrap("- " + w) for w in verdict.get("warnings", [])), []) + [""]
    lines += [f"core (rCBF<30%)      {t.core_ml:7.1f} mL", f"Tmax > 6 s           {t.hypo_ml:7.1f} mL", f"mismatch             {t.mismatch_ml:7.1f} mL",
              f"mismatch ratio       {t.mismatch_ratio:7.1f}" if np.isfinite(t.mismatch_ratio) else "mismatch ratio          inf",
              f"Tmax > 10 s          {t.tmax_ml[10]:7.1f} mL", f"HIR                  {t.hir:7.2f}", f"DEFUSE-3 profile     {'yes' if t.defuse3_target else 'no'}"]
    if summary:
        lines += [""] + [f"{k}: {v}" for k, v in summary.items() if isinstance(v, (str, int, float))][:8]
    ds = _new_sc(source_ds, generate_uid(), SERIES_NUMBER["summary"], "OpenPerfusion summary (research)", 1, _text_image(lines))
    p = out / "summary.dcm"; ds.save_as(str(p), enforce_file_format=True); written.append(p)
    return written

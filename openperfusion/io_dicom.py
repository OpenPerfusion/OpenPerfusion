"""4D CT perfusion from DICOM.

Vendors export CTP as one or several series of 2D slices, each slice tagged with a z position
(ImagePositionPatient) and an acquisition time (AcquisitionTime / ContentTime, or a frame
counter). Shuttle / jog-mode scanners alternate table positions, so a given slice is sampled at
irregular, position-dependent times. This loader:

1. reads every DICOM file under a directory (pydicom), keeps CT images, groups by series;
2. rescales to Hounsfield units (RescaleSlope / RescaleIntercept);
3. assigns each image to a slice (by rounded z) and a time (seconds from the first frame);
4. resamples each slice's irregular time series to a common uniform grid of `dt` seconds by
   linear interpolation (the ISLES convention is 1 s), and returns a (X, Y, Z, T) array.

Enhanced multi-frame CT (one file, many frames) is handled when the per-frame functional
groups carry position and time; otherwise each frame is taken as a time point of one slice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import pydicom
except ImportError:                     # pragma: no cover
    pydicom = None


@dataclass
class Ctp4D:
    hu: np.ndarray            # (X, Y, Z, T) Hounsfield units on the uniform time grid
    dt: float                 # s
    times: np.ndarray         # (T,) s, uniform
    voxel_size: tuple         # (dx, dy, dz) mm
    z_positions: np.ndarray   # (Z,) mm
    raw_times: list           # per slice, the original acquisition times (s) — irregular for shuttle mode
    series_uid: str
    n_files: int
    meta: dict


def _time_seconds(ds) -> float | None:
    """Acquisition time in seconds of day from the first tag available."""
    for tag in ("AcquisitionTime", "ContentTime", "TriggerTime"):
        v = getattr(ds, tag, None)
        if v in (None, ""):
            continue
        if tag == "TriggerTime":
            return float(v) / 1000.0
        s = str(v)
        hh, mm, ss = int(s[0:2]), int(s[2:4]), float(s[4:]) if len(s) > 4 else 0.0
        return hh * 3600 + mm * 60 + ss
    return None


def _z_of(ds) -> float | None:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is not None and len(iop) == 6:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            n = np.cross(r, c)
            return float(np.dot(n, np.array(ipp, float)))
        return float(ipp[2])
    sl = getattr(ds, "SliceLocation", None)
    return float(sl) if sl is not None else None


def read_ctp_dicom(directory: str | Path, dt: float = 1.0, series_uid: str | None = None,
                   z_tolerance: float = 0.5) -> Ctp4D:
    """Load a 4D CTP acquisition from a directory of DICOM files."""
    if pydicom is None:
        raise ImportError("pydicom is required: pip install pydicom")
    files = [p for p in Path(directory).rglob("*") if p.is_file()]
    frames = []     # (series, z, t, image HU)
    meta = {}
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "CT") != "CT" or not hasattr(ds, "PixelData"):
            continue
        suid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        if series_uid and suid != series_uid:
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0)); icpt = float(getattr(ds, "RescaleIntercept", 0.0))
        nfr = int(getattr(ds, "NumberOfFrames", 1))
        arr = ds.pixel_array.astype(np.float32) * slope + icpt
        if not meta:
            ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
            meta = dict(pixel_spacing=(float(ps[0]), float(ps[1])), slice_thickness=float(getattr(ds, "SliceThickness", 0) or 0),
                        manufacturer=str(getattr(ds, "Manufacturer", "")), model=str(getattr(ds, "ManufacturerModelName", "")),
                        kvp=getattr(ds, "KVP", None), rows=int(ds.Rows), cols=int(ds.Columns))
        if nfr > 1 and arr.ndim == 3:
            # enhanced multi-frame: per-frame position/time if present, else frames are time points of one slice
            pfg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
            for i in range(nfr):
                z, t = None, None
                if pfg is not None:
                    fg = pfg[i]
                    pp = getattr(fg, "PlanePositionSequence", None)
                    if pp: z = float(pp[0].ImagePositionPatient[2])
                    fc = getattr(fg, "FrameContentSequence", None)
                    if fc and hasattr(fc[0], "FrameAcquisitionDateTime"):
                        s = str(fc[0].FrameAcquisitionDateTime)[8:]
                        t = int(s[0:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:])
                z = _z_of(ds) if z is None else z
                t = float(i) * dt if t is None else t
                frames.append((suid, z, t, arr[i]))
        else:
            frames.append((suid, _z_of(ds), _time_seconds(ds), arr))
    if not frames:
        raise ValueError(f"no CT images found under {directory}")
    # choose the series with the most frames if several
    series = {}
    for s, z, t, a in frames:
        series.setdefault(s, []).append((z, t, a))
    suid = max(series, key=lambda k: len(series[k]))
    fr = series[suid]
    if any(z is None for z, _, _ in fr):
        raise ValueError("DICOM images lack ImagePositionPatient / SliceLocation")
    if any(t is None for _, t, _ in fr):
        raise ValueError("DICOM images lack AcquisitionTime / ContentTime / TriggerTime")
    zs = np.array([z for z, _, _ in fr]); ts = np.array([t for _, t, _ in fr])
    # slice bins
    zu = np.unique(np.round(zs / z_tolerance) * z_tolerance)
    zu = np.array(sorted(zu))
    # merge bins closer than tolerance
    merged = [zu[0]]
    for z in zu[1:]:
        if z - merged[-1] > z_tolerance:
            merged.append(z)
    zbins = np.array(merged)
    zidx = np.array([int(np.argmin(np.abs(zbins - z))) for z in zs])
    t0 = ts.min()
    rel = ts - t0
    t_end = rel.max()
    times = np.arange(0.0, t_end + 1e-6, dt)
    rows, cols = fr[0][2].shape
    hu = np.zeros((cols, rows, len(zbins), len(times)), np.float32)   # (X, Y, Z, T): x = column index
    raw_times = []
    for k in range(len(zbins)):
        sel = np.where(zidx == k)[0]
        order = sel[np.argsort(rel[sel])]
        tk = rel[order]
        imgs = np.stack([fr[i][2] for i in order], axis=0)          # (n_k, rows, cols)
        raw_times.append(tk)
        # resample each pixel's time series onto the uniform grid (linear, clamped at the ends)
        if len(tk) == 1:
            res = np.repeat(imgs, len(times), axis=0)
        else:
            res = np.empty((len(times), rows, cols), np.float32)
            for j, tt in enumerate(times):
                if tt <= tk[0]:
                    res[j] = imgs[0]
                elif tt >= tk[-1]:
                    res[j] = imgs[-1]
                else:
                    i1 = int(np.searchsorted(tk, tt)); i0 = i1 - 1
                    w = (tt - tk[i0]) / max(tk[i1] - tk[i0], 1e-9)
                    res[j] = (1 - w) * imgs[i0] + w * imgs[i1]
        hu[:, :, k, :] = res.transpose(2, 1, 0)                    # (cols, rows, T)
    dz = float(np.median(np.diff(zbins))) if len(zbins) > 1 else (meta.get("slice_thickness") or 1.0)
    ps = meta.get("pixel_spacing", (1.0, 1.0))
    meta["n_slices"] = len(zbins); meta["n_times"] = len(times)
    meta["shuttle_mode"] = bool(max(len(t) for t in raw_times) < 0.8 * len(times))   # fewer native samples than grid points
    return Ctp4D(hu=hu, dt=dt, times=times, voxel_size=(ps[1], ps[0], dz), z_positions=zbins, raw_times=raw_times,
                 series_uid=suid, n_files=len(fr), meta=meta)


def write_ctp_dicom(hu4d: np.ndarray, directory: str | Path, dt: float = 1.0, voxel_size=(1.0, 1.0, 5.0),
                    times_per_slice: list | None = None, series_uid: str | None = None):
    """Write a (X, Y, Z, T) HU array as one DICOM file per slice per time point (test helper;
    `times_per_slice` lets a shuttle-mode acquisition with irregular, slice-dependent sampling be simulated)."""
    if pydicom is None:
        raise ImportError("pydicom is required")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    X, Y, Z, T = hu4d.shape
    suid = series_uid or generate_uid(); study = generate_uid()
    n = 0
    for z in range(Z):
        ts = times_per_slice[z] if times_per_slice is not None else np.arange(T) * dt
        for j, t in enumerate(ts):
            if times_per_slice is not None:
                # sample the true 4D at the nearest grid frame (the simulated acquisition instant)
                frame = int(round(t / dt)); frame = min(max(frame, 0), T - 1)
            else:
                frame = j
            img = hu4d[:, :, z, frame].T                            # (rows, cols)
            fm = FileMetaDataset(); fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            fm.MediaStorageSOPInstanceUID = generate_uid(); fm.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
            ds.SOPClassUID = fm.MediaStorageSOPClassUID; ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
            ds.Modality = "CT"; ds.SeriesInstanceUID = suid; ds.StudyInstanceUID = study
            ds.Manufacturer = "OpenPerfusion synthetic"; ds.PatientName = "PHANTOM"; ds.PatientID = "0"
            ds.Rows, ds.Columns = img.shape
            ds.PixelSpacing = [float(voxel_size[1]), float(voxel_size[0])]; ds.SliceThickness = float(voxel_size[2])
            ds.ImagePositionPatient = [0.0, 0.0, float(z * voxel_size[2])]; ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            sec = 8 * 3600 + float(t)
            ds.AcquisitionTime = f"{int(sec // 3600):02d}{int(sec % 3600 // 60):02d}{sec % 60:09.6f}"
            ds.RescaleIntercept = -1024.0; ds.RescaleSlope = 1.0
            ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelData = np.clip(np.round(img + 1024), 0, 65535).astype(np.uint16).tobytes()
            ds.save_as(str(directory / f"z{z:03d}_t{j:03d}.dcm"), enforce_file_format=True)
            n += 1
    return n

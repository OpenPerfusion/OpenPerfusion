# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Shadow-mode DICOM node: a Storage SCP that receives CT perfusion series, processes each complete series with
the pipeline, writes NIfTI maps, summary.json and DICOM Secondary Capture series, and (optionally) sends the
DICOM results to a destination such as a research Orthanc.

    openperfusion serve --aet OPENPERF --port 11112 --out /data/openperfusion [--push-to HOST:PORT:AET] [--motion]

How a series is recognised as complete: instances are written under <out>/incoming/<StudyUID>/<SeriesUID>/ as
they arrive; a series is processed once no new instance has arrived for `idle` seconds (default 30) and it holds
at least `min_files` images (default 60). Series that fail to parse as 4D CTP (a plain CT, a scout) are moved to
<out>/skipped/ with the reason. Results go to <out>/results/<StudyUID>/<SeriesUID>/.

In shadow mode the destination is never the clinical PACS: RAPID keeps doing its job, this node receives a copy
of the same series (an Orthanc forwarding rule, or a second destination on the scanner), and the two outputs are
compared afterwards. The push-back exists so a reader can open both side by side in a research viewer.

Needs `pip install pynetdicom` (and pydicom). Research use only.
"""
from __future__ import annotations

import json, threading, time, shutil, logging
from pathlib import Path
import numpy as np

log = logging.getLogger("openperfusion.node")


def process_series(series_dir: Path, out_dir: Path, dt: float = 1.0, motion: bool = False, config=None) -> dict:
    """Run the full chain on one received series directory; write maps, DICOM and summary.json; return the summary."""
    from .io_dicom import read_ctp_dicom
    from .io_isles2018 import save_nifti
    from .pipeline import run_pipeline, PipelineConfig
    from .qc import qc_verdict
    from .dicom_out import write_result_dicom
    import pydicom
    t0 = time.time()
    c = read_ctp_dicom(series_dir, dt=dt)
    hu, mr = c.hu, None
    if motion:
        from .motion import motion_correct
        mr = motion_correct(hu, voxel_size=c.voxel_size); hu = mr.hu
    res = run_pipeline(hu, c.dt, c.voxel_size, config or PipelineConfig())
    verdict = qc_verdict(res, meta=c.meta, motion=mr)
    out_dir.mkdir(parents=True, exist_ok=True)
    aff = np.diag([c.voxel_size[0], c.voxel_size[1], c.voxel_size[2], 1.0])
    for k in ("cbf", "cbv", "mtt", "tmax"):
        save_nifti(getattr(res.maps, k), aff, out_dir / f"{k}.nii.gz")
    t = res.thresholds
    save_nifti(t.core.astype(np.uint8), aff, out_dir / "core_rcbf30.nii.gz"); save_nifti(t.hypo.astype(np.uint8), aff, out_dir / "hypoperfusion_tmax6.nii.gz")
    src = next((pydicom.dcmread(str(p), stop_before_pixels=True) for p in sorted(series_dir.iterdir()) if p.is_file()), None)
    summary = {"status": verdict["status"], "warnings": verdict["warnings"], "core_ml": t.core_ml, "hypoperfusion_ml": t.hypo_ml,
               "mismatch_ml": t.mismatch_ml, "mismatch_ratio": t.mismatch_ratio, "tmax_ml": t.tmax_ml, "hir": t.hir,
               "defuse3_target_profile": t.defuse3_target, "motion_mm": float(getattr(mr, "max_shift_mm", 0.0)) if mr else None,
               "seconds": round(time.time() - t0, 1), "qc": {**res.qc, **verdict}, "dicom": c.meta,
               "study_uid": getattr(src, "StudyInstanceUID", None), "series_uid": c.series_uid}
    files = write_result_dicom(res, out_dir / "dicom", source_ds=src, z_positions=c.z_positions, verdict=verdict,
                               summary={"n_files": c.n_files, "dt_s": c.dt, "shuttle": c.meta.get("shuttle_mode")})
    summary["dicom_files"] = [str(p) for p in files]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def send_dicom(files, host: str, port: int, aet: str, our_aet: str = "OPENPERF") -> int:
    """C-STORE a list of files; returns the number accepted."""
    from pynetdicom import AE
    from pynetdicom.sop_class import SecondaryCaptureImageStorage
    import pydicom
    ae = AE(ae_title=our_aet); ae.add_requested_context(SecondaryCaptureImageStorage)
    assoc = ae.associate(host, port, ae_title=aet)
    if not assoc.is_established:
        raise ConnectionError(f"could not associate with {aet}@{host}:{port}")
    n = 0
    try:
        for f in files:
            st = assoc.send_c_store(pydicom.dcmread(str(f)))
            n += int(bool(st) and st.Status == 0)
    finally:
        assoc.release()
    return n


class Node:
    def __init__(self, out: str | Path, aet: str = "OPENPERF", port: int = 11112, idle: float = 30.0, min_files: int = 60,
                 dt: float = 1.0, motion: bool = False, push_to: str | None = None, config=None, push_rejects: bool = False):
        self.out = Path(out); self.aet = aet; self.port = port; self.idle = idle; self.min_files = min_files
        self.dt = dt; self.motion = motion; self.config = config; self.push_rejects = push_rejects
        self.push = None
        if push_to:
            h, p, a = push_to.split(":"); self.push = (h, int(p), a)
        self.last_seen: dict[Path, float] = {}
        self.lock = threading.Lock(); self.stop = threading.Event(); self.processed: list[dict] = []
        (self.out / "incoming").mkdir(parents=True, exist_ok=True)

    # --- receiving ---
    def _on_store(self, event):
        ds = event.dataset; ds.file_meta = event.file_meta
        study = getattr(ds, "StudyInstanceUID", "unknown"); series = getattr(ds, "SeriesInstanceUID", "unknown")
        d = self.out / "incoming" / study / series; d.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(d / f"{ds.SOPInstanceUID}.dcm"), enforce_file_format=True)
        with self.lock:
            self.last_seen[d] = time.time()
        return 0x0000

    def start(self):
        from pynetdicom import AE, evt, AllStoragePresentationContexts, ALL_TRANSFER_SYNTAXES
        self.ae = AE(ae_title=self.aet)
        # accept every storage SOP class in every transfer syntax (vendor exports are often JPEG-lossless);
        # the files are stored as received and pydicom decodes them when the series is read
        for cx in AllStoragePresentationContexts:
            self.ae.add_supported_context(cx.abstract_syntax, ALL_TRANSFER_SYNTAXES)
        self.ae.maximum_pdu_size = 0
        self.server = self.ae.start_server(("0.0.0.0", self.port), block=False, evt_handlers=[(evt.EVT_C_STORE, self._on_store)])
        self.worker = threading.Thread(target=self._loop, daemon=True); self.worker.start()
        log.info("OpenPerfusion node %s listening on %d, results under %s", self.aet, self.port, self.out)
        return self

    def shutdown(self):
        self.stop.set(); self.server.shutdown(); self.worker.join(timeout=5)

    # --- processing ---
    def _ready(self):
        now = time.time()
        with self.lock:
            due = [d for d, t in self.last_seen.items() if now - t >= self.idle]
            for d in due:
                del self.last_seen[d]
        return due

    def _loop(self):
        while not self.stop.is_set():
            for d in self._ready():
                self.handle(d)
            self.stop.wait(1.0)

    def handle(self, series_dir: Path) -> dict | None:
        n = sum(1 for p in series_dir.iterdir() if p.is_file())
        rel = series_dir.relative_to(self.out / "incoming")
        if n < self.min_files:
            self._skip(series_dir, f"{n} images, fewer than {self.min_files}: not a perfusion series"); return None
        out_dir = self.out / "results" / rel
        try:
            summary = process_series(series_dir, out_dir, dt=self.dt, motion=self.motion, config=self.config)
        except Exception as e:                         # a plain CT, a scout, a broken series
            self._skip(series_dir, f"not processed: {type(e).__name__}: {e}"); return None
        log.info("%s: %s, core %.0f mL, Tmax>6 %.0f mL, %.0f s", rel, summary["status"], summary["core_ml"], summary["hypoperfusion_ml"], summary["seconds"])
        if self.push and (summary["status"] != "reject" or self.push_rejects):
            try:
                summary["pushed"] = send_dicom(summary["dicom_files"], *self.push, our_aet=self.aet)
            except Exception as e:
                summary["pushed"] = f"failed: {e}"; log.warning("push failed: %s", e)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self._remove_incoming(series_dir)                        # the received copy is not kept
        self.processed.append(summary)
        return summary

    def _remove_incoming(self, series_dir: Path):
        shutil.rmtree(series_dir, ignore_errors=True)
        study_dir = series_dir.parent
        if study_dir != self.out / "incoming" and study_dir.exists() and not any(study_dir.iterdir()):
            study_dir.rmdir()

    def _skip(self, series_dir: Path, reason: str):
        dest = self.out / "skipped" / series_dir.relative_to(self.out / "incoming"); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(dest, ignore_errors=True); shutil.move(str(series_dir), str(dest))
        (dest / "SKIPPED.txt").write_text(reason + "\n"); log.info("%s: skipped (%s)", series_dir.name, reason)
        self._remove_incoming(series_dir)


def serve(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .pipeline import PipelineConfig
    node = Node(args.out, aet=args.aet, port=args.port, idle=args.idle, min_files=args.min_files, dt=args.dt, motion=args.motion,
                push_to=args.push_to, config=PipelineConfig()).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.shutdown()

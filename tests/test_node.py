# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Shadow-mode node end to end: a phantom series is C-STOREd to the node, the node processes it once the series
goes quiet, writes NIfTI maps, summary.json and DICOM Secondary Capture series under the same StudyInstanceUID,
and pushes the DICOM results to a second (receiving) SCP standing in for a research Orthanc. A short series is
skipped, not processed."""
import json, time, socket
from pathlib import Path
import numpy as np, pytest

pynetdicom = pytest.importorskip("pynetdicom")
pydicom = pytest.importorskip("pydicom")
from openperfusion.phantom import make_phantom
from openperfusion.io_dicom import write_ctp_dicom
from openperfusion.node import Node


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _send(directory: Path, port: int, aet: str):
    from pynetdicom import AE
    from pynetdicom.sop_class import CTImageStorage
    ae = AE(ae_title="SCANNER"); ae.add_requested_context(CTImageStorage)
    assoc = ae.associate("127.0.0.1", port, ae_title=aet); assert assoc.is_established
    n = 0
    for f in sorted(directory.iterdir()):
        st = assoc.send_c_store(pydicom.dcmread(str(f))); n += int(st.Status == 0)
    assoc.release(); return n


class Receiver:
    """Stand-in for the research Orthanc: accepts whatever is pushed and keeps it."""
    def __init__(self, port, out: Path):
        from pynetdicom import AE, evt, AllStoragePresentationContexts
        self.out = out; out.mkdir(parents=True, exist_ok=True); self.n = 0
        ae = AE(ae_title="ORTHANC"); ae.supported_contexts = AllStoragePresentationContexts
        self.server = ae.start_server(("127.0.0.1", port), block=False, evt_handlers=[(evt.EVT_C_STORE, self._store)])
    def _store(self, event):
        ds = event.dataset; ds.file_meta = event.file_meta; ds.save_as(str(self.out / f"{ds.SOPInstanceUID}.dcm"), enforce_file_format=True); self.n += 1; return 0x0000
    def shutdown(self): self.server.shutdown()


def test_node_end_to_end(tmp_path):
    hu, truth = make_phantom(delays=(0.0, 2.0, 4.0), noise_sd=2.0, seed=3)
    src = tmp_path / "src"; n_src = write_ctp_dicom(hu, src, dt=truth.dt, voxel_size=truth.voxel_size)
    study_uid = pydicom.dcmread(str(next(src.iterdir())), stop_before_pixels=True).StudyInstanceUID
    p_node, p_rx = _free_port(), _free_port()
    rx = Receiver(p_rx, tmp_path / "orthanc")
    node = Node(tmp_path / "node", aet="OPENPERF", port=p_node, idle=2.0, min_files=20, push_to=f"127.0.0.1:{p_rx}:ORTHANC").start()
    try:
        assert _send(src, p_node, "OPENPERF") == n_src
        # a 10-image "scout" series must be skipped
        scout = tmp_path / "scout"; write_ctp_dicom(hu[..., :1, :10], scout, dt=1.0, voxel_size=truth.voxel_size)
        _send(scout, p_node, "OPENPERF")
        for _ in range(120):
            if len(node.processed) >= 1 and (tmp_path / "node" / "skipped").exists():
                break
            time.sleep(0.5)
        assert len(node.processed) == 1, "the phantom series was not processed"
        s = node.processed[0]
        assert s["status"] == "ok" and s["study_uid"] == study_uid and s["core_ml"] >= 0 and s["hypoperfusion_ml"] >= 0
        res_dir = tmp_path / "node" / "results" / study_uid
        out = next(res_dir.iterdir())
        assert (out / "tmax.nii.gz").exists() and (out / "summary.json").exists()
        js = json.loads((out / "summary.json").read_text()); assert js["status"] == "ok" and js["pushed"] == len(js["dicom_files"])
        # DICOM output: 4 maps x Z + regions x Z + 1 summary, filed under the source study, research-labelled
        dcm = sorted((out / "dicom").glob("*.dcm")); Z = hu.shape[2]
        assert len(dcm) == 5 * Z + 1
        ds = pydicom.dcmread(str(dcm[0]))
        assert ds.StudyInstanceUID == study_uid and ds.Modality == "OT" and "research" in ds.SeriesDescription and ds.PhotometricInterpretation == "RGB"
        assert ds.pixel_array.shape == (hu.shape[1], hu.shape[0], 3)
        assert len({pydicom.dcmread(str(f), stop_before_pixels=True).SeriesInstanceUID for f in dcm}) == 6
        # the push arrived
        time.sleep(0.5); assert rx.n == len(dcm)
        # the incoming copy was removed; the scout went to skipped/
        assert not (tmp_path / "node" / "incoming" / study_uid).exists()
        skipped = list((tmp_path / "node" / "skipped").rglob("SKIPPED.txt")); assert len(skipped) == 1 and "not a perfusion series" in skipped[0].read_text()
    finally:
        node.shutdown(); rx.shutdown()

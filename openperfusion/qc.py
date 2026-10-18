# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Turn the pipeline's QC flags into a verdict and human-readable warnings.

The flags come from the AIF/VOF selection (`AifResult.qc`), the motion correction and the acquisition itself.
Three verdicts:

    ok      - nothing to report
    review  - the maps were produced but something about the study deserves a look before they are trusted
              (fallback arterial input, arterial input peaking >= 8 s before the tissue, short acquisition,
              large motion, assumed frame timing)
    reject  - the pipeline could not find a usable arterial input; the maps should not be reported

On ISLES'24 the one subject the `reject` rule catches (sub-stroke0028, a slab-wide periodic artefact) is the
difference between a Tmax > 6 s ICC of 0.45 and 0.89 against icobrain cva; the `review` rule catches a fallback
AIF that happened to work (sub-stroke0016). A deployed system should refuse `reject` studies with the warning
rather than report through them.
"""
from __future__ import annotations

import numpy as np
from .aif import second_peak_fraction


def qc_verdict(res, meta: dict | None = None, motion=None, max_second_peak: float = 0.5, max_lead: float = 8.0) -> dict:
    """`res`: PipelineResult; `meta`: DICOM reader metadata (optional); `motion`: MotionResult (optional)."""
    q = res.aif.qc
    warnings, status = [], "ok"
    order = {"ok": 0, "review": 1, "reject": 2}
    def raise_to(s):
        nonlocal status
        if order[s] > order[status]:
            status = s
    aif_peak = float(np.max(res.aif.aif_raw)) if res.aif.aif_raw is not None else 0.0
    second = float(second_peak_fraction(res.aif.aif_raw[None])[0]) if res.aif.aif_raw is not None else 0.0
    if q.get("aif_last_resort"):
        warnings.append("no plausible arterial input function was found; a last-resort pick was used. Maps are unreliable."); raise_to("reject")
    if q.get("aif_weak") or aif_peak < 15.0:
        warnings.append(f"arterial input function is weak ({aif_peak:.0f} HU); maps are unreliable."); raise_to("reject")
    if second > max_second_peak:
        warnings.append(f"the arterial curve rises again to {second:.0%} of its peak after the bolus (multi-peak); the acquisition may be "
                        "corrupted by motion, table movement or a resampling artefact. Maps are unreliable."); raise_to("reject")
    elif q.get("aif_fallback"):
        warnings.append(f"the primary arterial search failed its plausibility checks; a fallback curve ({aif_peak:.0f} HU) was used. "
                        "Check the AIF location before trusting the maps."); raise_to("review")
    # the two venous conditions only switch off the partial-volume correction: absolute CBF/CBV move, the relative
    # core and Tmax volumes do not, so they are reported but do not change the verdict
    lead = q.get("aif_lead")
    if lead is not None and lead >= max_lead:
        warnings.append(f"the arterial input peaks {lead:.0f} s before the whole-brain tissue curve (normal 2-5 s): it may be an early "
                        "skull-base vessel, a test bolus, or the first slug of a split bolus, and every Tmax is shifted by the "
                        "difference. Check the AIF location before trusting the maps."); raise_to("review")
    for key, what in (("aif_reselected_lead", "it peaked too far ahead of the tissue"), ("aif_reselected_vein", "it was far smaller than the venous curve")):
        if q.get(key):
            was = q[key]
            warnings.append(f"the first arterial pick ({was.get('was_peak', 0):.0f} HU) was replaced by a brighter artery because {what}.")
    if q.get("truncated"):
        warnings.append(f"venous curve truncated (tail {q.get('vof_tail_fraction', float('nan')):.0%} of peak at the last frame): "
                        "partial-volume correction skipped; absolute CBF/CBV scale uncertain, relative core (rCBF) and Tmax unaffected.")
    if q.get("vof") == "no late-arriving candidates":
        warnings.append("no venous output function found; partial-volume correction skipped; absolute CBF/CBV scale uncertain.")
    n_frames = res.qc.get("n_frames"); dt = res.qc.get("dt")
    if n_frames and dt and n_frames * dt < 40:
        warnings.append(f"short acquisition ({n_frames * dt:.0f} s); late Tmax values may be truncated."); raise_to("review")
    if motion is not None:
        ms = float(getattr(motion, "max_shift_mm", 0.0)); mr = float(np.abs(getattr(motion, "theta", np.zeros(1))).max())
        if ms > 5.0 or mr > 5.0:
            warnings.append(f"large head motion (up to {ms:.1f} mm, {mr:.1f} deg) was corrected; check the maps for residual artefact."); raise_to("review")
    if meta:
        if meta.get("time_source") == "instance_order":
            warnings.append("frame timing was taken from instance order, not from the DICOM times; Tmax is in units of the assumed interval."); raise_to("review")
        if meta.get("shuttle_mode"):
            warnings.append(f"shuttle-mode acquisition (native interval {meta.get('native_dt', '?')} s) resampled to a uniform grid; "
                            "absolute CBF may read ~5-7 % low (phantom); relative core unaffected.")
    return {"status": status, "warnings": warnings, "aif_peak_hu": aif_peak, "aif_second_peak_fraction": second,
            "aif_lead_s": lead,
            "flags": {k: bool(q.get(k, False)) for k in ("aif_primary_weak", "aif_fallback", "aif_last_resort", "aif_weak", "truncated",
                                                          "aif_reselected_lead", "aif_reselected_vein")}}

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Check that each ISLES'24 subject's icobrain maps sit on the anatomy of the NCCT whose grid they are declared on.

sub-stroke0026 showed voxelwise r = 0.04 against icobrain for CBF, CBV and Tmax alike, although our CTP registers
to its NCCT correctly (skull edges coincide). The fault is in the reference: in native NCCT space the icobrain maps
are displaced from the NCCT by ~19 mm in-plane, and the map reads a lower CBF in grey matter than in white matter.
A reference that does not sit on its own anatomy cannot be compared voxel by voxel, and its volume changes with
how it is cut by the resampling. This script scores that for every subject from the NCCT and the icobrain CBF
alone (no 4D volume, no pipeline), in a few seconds per shard:

    python scripts/isles2024_refcheck.py /path/to/isles24/data results/isles24_149 [--budget SECONDS]

Per subject, slice by slice over the slab icobrain's map covers (the NCCT reaches further than the CTP slab, so a
whole-volume overlap would only measure coverage): `ref_inplane_dice` = median Dice between icobrain's map
(CBF > 5) and the NCCT brain mask (10-60 HU, opened, largest component) and `ref_inplane_shift_mm` = median
in-plane centroid offset between the two; plus `ref_gm_wm`, median icobrain CBF in NCCT-defined grey matter over
white matter inside the map (a map on its own anatomy gives > 1). Aligned subjects sit at Dice 0.90-0.94 and
1-5 mm; `scripts/isles2024_summary.py` merges the columns into cases.csv when refcheck.csv is present.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.io_isles2024 import find_shards, _nifti_from_bytes


def brain_mask(ncct: np.ndarray) -> np.ndarray:
    b = (ncct > 10) & (ncct < 60)
    b = ndimage.binary_opening(b, iterations=3)
    lab, k = ndimage.label(b)
    if k == 0:
        return b
    sizes = ndimage.sum(b, lab, range(1, k + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def refcheck(ncct_img, cbf_img) -> dict:
    n = np.asarray(ncct_img.dataobj, np.float32); c = np.asarray(cbf_img.dataobj, np.float32)
    vox = np.array(ncct_img.header.get_zooms()[:3], float)
    brain = brain_mask(n); valid = c > 5
    per_slice = valid.sum(axis=(0, 1)); slab = [z for z in range(n.shape[2]) if per_slice[z] > 0.5 * per_slice.max()]
    dice, off = [], []
    for z in slab:
        b = brain[..., z]; v = valid[..., z]
        if b.sum() < 1000 or v.sum() < 1000:
            continue
        dice.append(2.0 * (b & v).sum() / (b.sum() + v.sum()))
        off.append(float(np.linalg.norm((np.argwhere(v).mean(0) - np.argwhere(b).mean(0)) * vox[:2])))
    ns = ndimage.gaussian_filter(n, 2.0)                       # icobrain's maps are smooth; match the scale
    inside = valid & brain
    gm = inside & (ns > 32) & (ns < 45); wm = inside & (ns >= 22) & (ns <= 32)
    med = lambda m: float(np.median(c[m])) if m.sum() > 100 else np.nan
    gm_c, wm_c = med(gm), med(wm)
    return {"ref_inplane_dice": round(float(np.median(dice)), 3) if dice else np.nan, "ref_inplane_dice_min": round(float(np.min(dice)), 3) if dice else np.nan,
            "ref_inplane_shift_mm": round(float(np.median(off)), 1) if off else np.nan, "ref_inplane_shift_max_mm": round(float(np.max(off)), 1) if off else np.nan,
            "ref_slab_slices": len(slab), "ref_cbf_gm": round(gm_c, 1), "ref_cbf_wm": round(wm_c, 1),
            "ref_gm_wm": round(gm_c / wm_c, 2) if wm_c and np.isfinite(wm_c) and wm_c > 0 else np.nan}


def main(data: Path, out: Path, budget: float):
    out.mkdir(parents=True, exist_ok=True); f = out / "refcheck.csv"; t0 = time.time()
    done = set(pd.read_csv(f).case) if f.exists() else set()
    for sh in find_shards(data):
        if time.time() - t0 > budget:
            break
        try:
            sid = pq.read_table(str(sh), columns=["subject_id"]).column("subject_id")[0].as_py()
        except Exception:
            continue
        if sid in done:
            continue
        try:
            t = pq.read_table(str(sh), columns=["ncct", "cbf"]).to_pylist()[0]
            row = {"case": sid, **refcheck(_nifti_from_bytes(t["ncct"]["bytes"]), _nifti_from_bytes(t["cbf"]["bytes"]))}
        except Exception as e:
            row = {"case": sid, "error": str(e)[:80]}
        pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(sid, {k: v for k, v in row.items() if k != "case"}, flush=True)
    print(f"elapsed {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), float(a[a.index("--budget") + 1]) if "--budget" in a else 1e9)

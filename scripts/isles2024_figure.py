# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Montage for ISLES'24 subjects: baseline CT with our mask, our Tmax, icobrain Tmax (registered into the CTP
grid), and Tmax > 6 s (ours red, icobrain yellow outline) with the follow-up lesion (cyan).

    python scripts/isles2024_figure.py /path/to/isles24_unpacked results/isles24/montage.png [--max N]
"""
import sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from openperfusion.io_isles2024 import find_subjects, load_subject
from openperfusion.pipeline import run_pipeline
from openperfusion.maps import threshold_maps

root, out = sys.argv[1], sys.argv[2]
subs = find_subjects(root)
if "--max" in sys.argv: subs = subs[: int(sys.argv[sys.argv.index("--max") + 1])]
fig, ax = plt.subplots(len(subs), 4, figsize=(16, 3.9 * len(subs)))
ax = np.atleast_2d(ax)
for row, sub in enumerate(subs):
    d = load_subject(sub); hu = d["ctp4d"]; r = run_pipeline(hu, d["dt"], d["voxel_size"])
    z = hu.shape[2] // 2; base = hu[..., :5].mean(-1)[:, :, z]; m = r.mask[:, :, z]
    ref = {k: d[k] for k in ("cbf", "cbv", "mtt", "tmax")}
    from openperfusion.maps import PerfusionMaps
    valid = r.mask & np.isfinite(ref["tmax"]) & (ref["cbf"] > 0)
    ref_maps = PerfusionMaps(cbf=ref["cbf"], cbv=ref["cbv"], mtt=ref["mtt"], tmax=ref["tmax"], ttp=None, mask=valid, dt=d["dt"], method="reference")
    t_ref = threshold_maps(ref_maps, d["voxel_size"], baseline=r.baseline)
    ax[row, 0].imshow(base.T, cmap="gray", vmin=0, vmax=80, origin="lower"); ax[row, 0].contour(m.T, levels=[0.5], colors="lime", linewidths=0.8)
    ax[row, 0].set_title(f"{d['case']} z{z}  registration: {d['registration'].get('method')}")
    ax[row, 1].imshow(np.where(m, r.maps.tmax[:, :, z], np.nan).T, cmap="jet", vmin=0, vmax=12, origin="lower"); ax[row, 1].set_title(f"OpenPerfusion Tmax, >6 s {r.thresholds.hypo_ml:.0f} mL")
    ax[row, 2].imshow(np.where(valid[:, :, z], ref["tmax"][:, :, z], np.nan).T, cmap="jet", vmin=0, vmax=12, origin="lower"); ax[row, 2].set_title(f"icobrain Tmax, >6 s {t_ref.hypo_ml:.0f} mL")
    ax[row, 3].imshow(base.T, cmap="gray", vmin=0, vmax=80, origin="lower")
    ov = np.zeros(m.shape + (4,)); ov[r.thresholds.hypo[:, :, z]] = (1, 0, 0, 0.45)
    if d["lesion"] is not None: ov[d["lesion"][:, :, z]] = (0, 1, 1, 0.55)
    ax[row, 3].imshow(ov.transpose(1, 0, 2), origin="lower")
    if t_ref.hypo[:, :, z].any(): ax[row, 3].contour(t_ref.hypo[:, :, z].T, levels=[0.5], colors="yellow", linewidths=0.9)
    ax[row, 3].set_title("ours red, icobrain yellow, follow-up lesion cyan")
    for a in ax[row]: a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=55); print("wrote", out)

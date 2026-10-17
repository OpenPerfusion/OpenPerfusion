# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Scatter and Bland-Altman plots of OpenPerfusion vs reference volumes from a per-case results CSV.

    python scripts/agreement_figure.py results/split_half/heldout_pooled_cases.csv results/split_half/agreement_heldout_94.png
    python scripts/agreement_figure.py results/isles24_50/cases.csv results/isles24_50/agreement_50.png \
        --ref "icobrain cva" --flags aif_fallback,aif_last_resort

With `--flags`, cases where any of those columns is true are drawn in red and left out of the bias and limits
(the title says how many); the point of the picture is then that the pipeline's own flags find the outliers.
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

args = sys.argv[1:]
d = pd.read_csv(args[0]); out = args[1]
ref_name = args[args.index("--ref") + 1] if "--ref" in args else "RAPID (ISLES 2018 maps)"
flag_cols = args[args.index("--flags") + 1].split(",") if "--flags" in args else []
flagged = d[flag_cols].fillna(False).astype(bool).any(axis=1).values if flag_cols else np.zeros(len(d), bool)
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
for col, (ours, ref, title) in enumerate((("hypo_ml_ours", "hypo_ml_ref", "Tmax > 6 s volume (mL)"), ("core_ml_ours", "core_ml_ref", "rCBF < 30 % core volume (mL)"))):
    x, y = d[ref].values, d[ours].values
    k = ~flagged
    lim = max(x[k].max(), y[k].max()) * 1.08
    a = ax[0, col]; a.scatter(x[k], y[k], s=18, alpha=0.7); a.plot([0, lim], [0, lim], "k--", lw=1)
    if flagged.any():
        xf, yf = x[flagged], y[flagged]
        for xi, yi, name in zip(xf, yf, d.case.values[flagged]):
            a.scatter([xi], [min(yi, lim * 0.98)], s=30, color="tab:red", marker="^" if yi > lim else "o")
            a.annotate(f"{name.replace('sub-stroke', '')}: {yi:.0f} mL" + (" (off scale)" if yi > lim else ""), (xi, min(yi, lim * 0.98)),
                       xytext=(6, -4), textcoords="offset points", fontsize=8, color="tab:red")
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel(ref_name); a.set_ylabel("OpenPerfusion")
    a.set_title(title + f", n = {k.sum()}" + (f" (+{flagged.sum()} flagged, red)" if flagged.any() else ""))
    diff = (y - x)[k]; mean = ((x + y) / 2)[k]; bias = diff.mean(); sd = diff.std(ddof=1)
    b = ax[1, col]; b.scatter(mean, diff, s=18, alpha=0.7)
    for v, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")): b.axhline(v, color="k", ls=ls, lw=1)
    b.set_xlabel("mean of the two (mL)"); b.set_ylabel(f"OpenPerfusion − {ref_name.split(' (')[0]} (mL)")
    b.set_title(f"bias {bias:+.1f} mL, limits {bias - 1.96 * sd:.0f} to {bias + 1.96 * sd:+.0f} mL" + (" (unflagged)" if flagged.any() else ""))
plt.tight_layout(); plt.savefig(out, dpi=110); print("wrote", out)

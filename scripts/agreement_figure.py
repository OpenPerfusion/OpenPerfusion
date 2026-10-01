# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Scatter and Bland-Altman plots of OpenPerfusion vs RAPID volumes from a per-case results CSV.

    python scripts/agreement_figure.py results/split_half/heldout_pooled_cases.csv results/split_half/agreement_heldout_94.png
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
for col, (ours, ref, title) in enumerate((("hypo_ml_ours", "hypo_ml_ref", "Tmax > 6 s volume (mL)"), ("core_ml_ours", "core_ml_ref", "rCBF < 30 % core volume (mL)"))):
    x, y = d[ref].values, d[ours].values
    lim = max(x.max(), y.max()) * 1.05
    a = ax[0, col]; a.scatter(x, y, s=18, alpha=0.7); a.plot([0, lim], [0, lim], "k--", lw=1)
    a.set_xlabel("RAPID (ISLES 2018 maps)"); a.set_ylabel("OpenPerfusion"); a.set_title(title + f", n = {len(d)}")
    diff = y - x; mean = (x + y) / 2; bias = diff.mean(); sd = diff.std(ddof=1)
    b = ax[1, col]; b.scatter(mean, diff, s=18, alpha=0.7)
    for v, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")): b.axhline(v, color="k", ls=ls, lw=1)
    b.set_xlabel("mean of the two (mL)"); b.set_ylabel("OpenPerfusion − RAPID (mL)")
    b.set_title(f"bias {bias:+.1f} mL, limits {bias - 1.96 * sd:.0f} to {bias + 1.96 * sd:+.0f} mL")
plt.tight_layout(); plt.savefig(out, dpi=110); print("wrote", out)

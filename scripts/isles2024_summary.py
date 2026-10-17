# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Summarise an ISLES'24 run two ways: all subjects, and excluding those the pipeline itself flagged
(AIF fallback or last resort). Both numbers belong in any report; the gap between them is the case for
honouring the flags.

    python scripts/isles2024_summary.py results/isles24_50            # reads cases.csv (+ qc.csv if present)

`qc.csv` (from `scripts/isles2024_qc.py`) carries the per-subject AIF flags; without it, nothing is excluded.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openperfusion.validate import summarize_cases

FLAGS = ("aif_fallback", "aif_last_resort")


def block(df: pd.DataFrame) -> dict:
    s = summarize_cases(df.to_dict("records"))
    out = {"n": int(len(df)),
           "hypo_icc": s["hypo_icc"], "hypo_bias": s["hypo_ba"]["bias"], "hypo_loa": [s["hypo_ba"]["loa_low"], s["hypo_ba"]["loa_high"]],
           "hypo_dice_mean": s["hypo_dice_mean"], "hypo_dice_median": float(df.dice_hypo.median()),
           "core_icc": s["core_icc"], "core_bias": s["core_ba"]["bias"], "core_loa": [s["core_ba"]["loa_low"], s["core_ba"]["loa_high"]],
           "core_dice_mean": s["core_dice_mean"], "core_dice_median": float(df.dice_core.median()),
           "r_cbf": s.get("r_cbf_mean"), "r_cbv": s.get("r_cbv_mean"), "r_tmax": s.get("r_tmax_mean"),
           "defuse3_agreement": s.get("defuse3_agreement"), "defuse3_kappa": s.get("defuse3_kappa")}
    for t in (4, 5, 6, 7, 8):
        if f"dice_hypo_t{t}" in df:
            out[f"hypo_dice_t{t}"] = float(df[f"dice_hypo_t{t}"].mean())
    for k in ("dice_hypo_lesion_ours", "dice_hypo_lesion_ref", "dice_core_lesion_ours", "dice_core_lesion_ref"):
        if k in df:
            out[k] = float(df[k].mean())
    if "reg_method" in df:
        out["registered"] = int((df.reg_method == "rigid").sum())
    return out


def main(d: Path):
    cases = pd.read_csv(d / "cases.csv").drop_duplicates("case").sort_values("case")
    flagged = []
    if (d / "qc.csv").exists():
        qc = pd.read_csv(d / "qc.csv").drop_duplicates("case")
        flagged = sorted(qc.loc[qc[list(FLAGS)].any(axis=1), "case"].tolist())
        cases = cases.merge(qc[["case", *FLAGS, "aif_primary_weak", "aif_weak", "truncated", "aif_second_peak"]], on="case", how="left")
        cases.to_csv(d / "cases.csv", index=False)
    kept = cases[~cases.case.isin(flagged)]
    summ = {"all": block(cases), "excluding_flagged": block(kept), "flagged": flagged, "flag_rule": " or ".join(FLAGS)}
    json.dump(summ, open(d / "summary.json", "w"), indent=1, default=float)
    for name in ("all", "excluding_flagged"):
        b = summ[name]
        print(f"{name:18s} n={b['n']:3d}  Tmax>6: ICC {b['hypo_icc']:.2f} bias {b['hypo_bias']:+.1f} LoA {b['hypo_loa'][0]:.0f}/{b['hypo_loa'][1]:+.0f} "
              f"Dice {b['hypo_dice_mean']:.2f} ({b['hypo_dice_median']:.2f})  core: ICC {b['core_icc']:.2f} bias {b['core_bias']:+.1f} "
              f"LoA {b['core_loa'][0]:.0f}/{b['core_loa'][1]:+.0f} Dice {b['core_dice_mean']:.2f}  r {b['r_cbf']:.2f}/{b['r_cbv']:.2f}/{b['r_tmax']:.2f}  "
              f"DEFUSE-3 {b['defuse3_agreement']:.2f}")
    print("flagged:", flagged)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

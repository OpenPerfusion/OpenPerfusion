# Enhancement-SNR gate on ISLES 2018 (94 in-sample, bcSVD 10 %, 1.5 mm)

| snr_min_tissue | Tmax>6 ICC | Tmax>6 Dice | core ICC | core Dice | DEFUSE-3 kappa |
|---|---|---|---|---|---|
| 0 | 0.953 | 0.846 | 0.982 | 0.772 | 0.777 |
| 2 | 0.955 | 0.831 | 0.926 | 0.685 | 0.679 |
| 3 | 0.947 | 0.785 | 0.754 | 0.491 | 0.644 |
| 4 | 0.904 | 0.715 | 0.519 | 0.306 | 0.632 |

Rejected: every k > 0 lowers Dice and kappa; the low-enhancement tissue the gate removes is tissue RAPID also calls.

# Arterial selection variants on ISLES 2018 (94 in-sample), tried for the ISLES'24 over-calls

| variant | Tmax>6 ICC | Tmax>6 Dice | core ICC | core Dice | DEFUSE-3 kappa |
|---|---|---|---|---|---|
| baseline weights (1, -3.5, -1), amp 0.25 | 0.953 | 0.846 | 0.982 | 0.772 | 0.777 |
| weights (2, -1, -1) | 0.943 | 0.827 | 0.978 | 0.750 | 0.721 |
| weights (1.5, -2, -1) + amp 0.5 | 0.945 | 0.838 | 0.969 | 0.761 | 0.777 |
| weights (2, -1, -1) + amp 0.5 | 0.943 | 0.832 | 0.977 | 0.752 | 0.752 |
| baseline + lead / vein / width rules (adopted) | 0.955 | 0.844 | 0.984 | 0.757 | 0.881 |

Re-weighting the cost towards amplitude fixes the ISLES'24 traps but costs 1-2 Dice points and up to 6 kappa
points against RAPID, where the arrival-weighted pick is right for thin slabs. The three targeted rules (re-select
amplitude-first only when the pick peaks >= 8 s before the tissue or is < 1/2.5 of the vein; width bound follows
the bolus) leave ISLES 2018 unchanged or better (kappa 0.78 -> 0.88) and fix four of the five ISLES'24 traps; the
fifth, a split bolus, is flagged for review by the lead rule.

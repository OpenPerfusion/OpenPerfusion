# Enhancement-SNR gate on ISLES 2018 (94 in-sample, bcSVD 10 %, 1.5 mm)

| snr_min_tissue | Tmax>6 ICC | Tmax>6 Dice | core ICC | core Dice | DEFUSE-3 kappa |
|---|---|---|---|---|---|
| 0 | 0.953 | 0.846 | 0.982 | 0.772 | 0.777 |
| 2 | 0.955 | 0.831 | 0.926 | 0.685 | 0.679 |
| 3 | 0.947 | 0.785 | 0.754 | 0.491 | 0.644 |
| 4 | 0.904 | 0.715 | 0.519 | 0.306 | 0.632 |

Rejected: every k > 0 lowers Dice and kappa; the low-enhancement tissue the gate removes is tissue RAPID also calls.

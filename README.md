# OpenPerfusion — open-source CT perfusion for acute stroke

**Status: research prototype (v0.1.0). Research use only. Not a medical device. Not for clinical decision-making.**
The name is free on PyPI. "RAPID" is a registered trademark of iSchemaView; OpenPerfusion is validated against RAPID output but is not affiliated with it.

## What it does

4D CT perfusion (Hounsfield units, `(X, Y, Z, T)`) in; RAPID-style outputs out:

1. **Masking** — soft-tissue brain mask from the pre-contrast frames (HU window, opening, largest component per slice, hole fill, edge erosion).
2. **Concentration** — pre-bolus baseline chosen from the *arterial* arrival (brightest 0.5 % of voxels), subtracted per voxel; optional in-plane and temporal Gaussian smoothing.
3. **AIF / VOF** — automatic, after Straka, Albers & Bammer (JMRI 2010): per-voxel peak height, arrival (searched backwards from the peak, robust to noise), FWHM; cost `1.0·z(h) − 3.5·z(a) − 1.0·z(w)`; best spatially connected cluster, grown to the vessel footprint. VOF = tall, late cluster; partial-volume correction `k_av = AUC_v / AUC_a`, skipped (and flagged) if the venous curve is truncated.
4. **Deconvolution** — three delay-insensitive (circular, zero-padded) engines:
   `bcsvd` (block-circulant SVD, fixed truncation; Wu 2003), `osvd` (per-voxel truncation by oscillation index; Wu 2003; default), `fourier` (Wiener-regularised frequency division, the RAPID formulation in Straka 2010).
5. **Maps** — CBF (mL/100 g/min), CBV (mL/100 g), MTT (s, central volume), Tmax (s, parabolic sub-sample peak), TTP.
6. **Thresholds** — vessel (CBV) and CSF (baseline HU) exclusion; rCBF relative to the contralateral, less-hypoperfused hemisphere; core = rCBF < 30 %; hypoperfusion = Tmax > 6 s; Tmax > 4/8/10 s volumes; HIR; cluster filter; mismatch volume and ratio; DEFUSE-3 target-profile flag.
7. **Validation harness** — digital phantom recovery (tile-wise errors, delay response) and reference-software comparison (voxel correlations, Dice of thresholded regions, volume bias / limits of agreement / ICC, DEFUSE-3 kappa), with figures.

## Install and run

```bash
git clone https://github.com/OpenPerfusion/OpenPerfusion.git && cd OpenPerfusion
pip install -e ".[dicom,dev]"   # numpy, scipy, nibabel, scikit-image, pydicom, pytest (pandas, matplotlib for the scripts)
pytest tests                # phantom recovery, DICOM round trip, motion correction (~15 s)

openperfusion phantom --out results/phantom                 # digital phantom → CSV, JSON, PNG
openperfusion isles2018 /path/to/ISLES2018/TRAINING --figures --out results/isles2018
openperfusion dicom /path/to/ctp_dicom_dir --motion --out results/case   # vendor export → NIfTI maps + summary.json
```

Defaults are the configuration validated against RAPID on ISLES 2018: block-circulant SVD with 10 % truncation,
2-voxel in-plane smoothing, contralateral-hemisphere rCBF reference, 1 mL cluster filter.

## Input

`openperfusion dicom` reads a directory of DICOM CT images (one series, one file per slice per time point, or an
enhanced multi-frame file), rescales to HU, assigns slices by position and frames by acquisition time, and resamples
each slice's time series to a uniform grid (default 1 s). Shuttle / jog-mode acquisitions, where alternating table
positions are sampled at offset, irregular times, are handled by that resampling and flagged in `summary.json`.
`--motion` applies per-frame in-plane rigid registration (phase-correlation translation on skull-edge maps, then a
coarse-to-fine rotation search on brain-windowed normalised cross-correlation) to the mean pre-contrast frame.
Both are tested by writing the digital phantom out as DICOM and reading it back, and by imposing known shifts and
rotations and recovering them (to < 0.6 voxel and < 1°).

## Digital phantom

`openperfusion.phantom.make_phantom()` builds a Kudo-style phantom: a 7 × 7 grid of tiles spanning CBF 10–70 mL/100 g/min × MTT 4–16 s, one slice per tracer delay (0–3 s), exponential or box residue functions, an arterial block (partial-volumed to 70 %), a venous block (delayed, dispersed, same area), normal reference tissue (CBF 50 / MTT 4), a CSF block, skull and air. Curves are generated on a 0.05 s grid and sampled at 1 s, so the estimator sees realistic discretisation error. Noise is Gaussian in HU.

Current recovery (oSVD, 3 HU noise, `openperfusion phantom`):

| quantity | result |
|---|---|
| AIF / VOF voxels found inside the true vessels | 100 % / 100 % |
| partial-volume factor k_av | 1.427 (true 1.429) |
| CBV tile error | −2 % mean, r = 0.995 |
| CBF tile error (CBF ≥ 30) | −10 % mean, r = 0.97; −30 % at MTT = 4 s (SVD truncation, MTT-dependent) |
| Tmax bias (well-perfused tiles) | +2.25 s ± 0.36 s (regularisation smoothing of a one-sided residue) |
| delay response for imposed 0 / 1 / 2 / 3 s | 0 / 0.95 / 1.94 / 2.83 s; CBF unchanged with delay |

The CBF underestimation at short MTT and the ~2 s positive Tmax offset are properties of truncated-SVD deconvolution at 1 s sampling, not bugs; commercial packages carry their own versions of both, which is why thresholds are software-specific and why calibration against RAPID output is the next step. Matching RAPID's Tmax > 6 s boundary may mean adjusting our threshold, our regularisation, or both — the harness is built to measure that.

## ISLES 2018: agreement with RAPID (94 training scans)

`openperfusion isles2018 TRAINING` with the defaults (block-circulant SVD, 10 % truncation, 2-voxel in-plane
smoothing, contralateral rCBF reference, 1 mL cluster filter). RAPID maps are the ISLES 2018 reference maps;
both sets of maps go through the same thresholding code so the comparison isolates the map computation.

| metric | result |
|---|---|
| voxelwise Pearson r, CBF / CBV / MTT / Tmax | 0.97 / 0.98 / 0.5 / 0.76 (Spearman 0.84 for Tmax) |
| Tmax > 6 s volume | ICC 0.86, bias −2 mL, LoA −55 to +51 mL, median Dice 0.88 (IQR 0.74–0.93) |
| rCBF < 30 % core volume | ICC 0.96, bias +0.6 mL, LoA −10 to +11 mL, median Dice 0.81 (83 cases with core in both) |
| DEFUSE-3 target profile | 83/94 agree (kappa 0.72); all 11 discordant are RAPID-yes / openperfusion-no |
| core vs follow-up infarct (Dice) | openperfusion 0.29, RAPID 0.29 |
| runtime | median 0.9 s per case (2–8 slices), 22 s for a 22-slice case |

Which of our thresholds best reproduces RAPID's boundary: Tmax > 6 s (Dice 0.77 vs 0.74 at 7 s), rCBF < 30 %
(Dice 0.71 vs 0.59 at 35 %). The +2 s Tmax offset seen on the phantom does not appear against RAPID, i.e.
RAPID carries the same regularisation bias, as expected for a truncated-SVD family.

Fallback input function: when the selected arterial curve peaks below 15 HU (thin 2–4 slice slabs that contain no
large artery), a second search accepts small, early, narrow 2-voxel clusters (partial-volumed cortical arteries). On
the 94 cases this lifts the worst cases (Tmax > 6 Dice 0.1–0.4 → 0.5–0.9) and the pooled Tmax > 6 ICC from 0.86 to
0.87, mean Dice from 0.77 to 0.79, DEFUSE-3 kappa from 0.72 to 0.74, with limits of agreement unchanged. A
timing-based trigger (primary AIF not leading the tissue peak) was tried and rejected: it replaced good AIFs in many
cases and widened the limits of agreement to ±73 mL. Cases that still have a weak AIF are flagged
(`aif.qc["aif_weak"]`) and `openperfusion dicom` prints a warning.

## Held-out numbers (split-half, the ones to quote)

`scripts/split_half.py` splits the 94 scans at random into halves A and B, scores a 12-configuration grid
(bcSVD truncation 5 / 10 / 15 %, oSVD target 0.05; smoothing 1.5 / 2.0 / 2.5 voxels) on one half, and evaluates
the best configuration on the other half, both ways. Both halves independently selected the same configuration
(bcSVD 10 %, 2.0-voxel smoothing), so the held-out and in-sample numbers coincide:

| held-out metric (each case scored once, by a configuration that did not see it) | result |
|---|---|
| Tmax > 6 s volume | ICC 0.87, bias −0.6 mL, LoA −52 to +50 mL, mean Dice 0.79, median 0.88 |
| rCBF < 30 % core volume | ICC 0.97, bias +1.2 mL, LoA −7 to +10 mL, mean Dice 0.72, median 0.81 |
| DEFUSE-3 target profile | agreement 0.89, kappa 0.74 |
| voxelwise r, CBF / CBV / Tmax | 0.98 / 0.98 / 0.77 |
| half A → test B / half B → test A | Tmax > 6 ICC 0.90 / 0.86; core ICC 0.98 / 0.97; kappa 0.73 / 0.75 |

Per-case results: `results/split_half/heldout_pooled_cases.csv`; figure: `results/split_half/agreement_heldout_94.png`.

## Next steps

1. Fallback input function (VOF-as-AIF or global-brightest) for slabs without an artery; re-run the 14 flagged cases.
2. Hold-out check: the sweep was tuned on cases 1–11 and the defaults then applied to all 94; repeat the sweep on a
   random half and test on the other half before quoting the numbers in a paper.
3. DICOM 4D reader (`pydicom`) with shuttle-mode resampling; rigid motion correction for vendor exports.
4. Request the ASIST-Japan phantom; reproduce the Kudo 2013 delay test.
5. UHN paired cohort (REB) for multi-vendor, current-RAPID-version calibration.

## Layout

```
openperfusion/phantom.py       digital phantom + ground truth
openperfusion/preprocess.py    brain mask, baseline, concentration, smoothing
openperfusion/aif.py           AIF / VOF selection and partial-volume correction
openperfusion/deconvolve.py    bcSVD, oSVD, Fourier-Wiener engines
openperfusion/maps.py          physiological scaling; thresholds, volumes, mismatch
openperfusion/pipeline.py      end-to-end run + config
openperfusion/validate.py      phantom report; reference comparison; ICC, Bland-Altman, kappa
openperfusion/report.py        figures
openperfusion/io_isles2018.py  ISLES 2018 loader
openperfusion/io_dicom.py       DICOM 4D reader (+ synthetic DICOM writer for tests)
openperfusion/motion.py         rigid in-plane motion correction
openperfusion/cli.py           `openperfusion phantom`, `openperfusion isles2018`, `openperfusion dicom`
tests/                  phantom recovery tests
```

## Citing and contributing

If you use OpenPerfusion in research, cite it via the `CITATION.cff` in this repository (GitHub's
"Cite this repository" button renders it). The methods it implements are Wu et al. (MRM 2003) for the
block-circulant deconvolution, Straka et al. (JMRI 2010) for the AIF/VOF cost function and Fourier
engine, and Hakim et al. (Stroke 2021) for the ISLES 2018 dataset; please cite those too.

Contributions are welcome; see `CONTRIBUTING.md`. The most useful thing right now is an anonymised
real CTP DICOM export per scanner vendor, and reports of cases where OpenPerfusion and a commercial
package disagree. Never attach patient data to an issue.

## Licence

Apache-2.0. Deconvolution mathematics after Østergaard et al. (MRM 1996) and Wu et al. (MRM 2003; US 7,512,435, expired June 2024); AIF cost function and Fourier engine after Straka et al. (JMRI 2010). No code was taken from PyPeT (non-commercial licence) or any commercial product.

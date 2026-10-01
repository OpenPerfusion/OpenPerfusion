# OpenPerfusion — open-source CT perfusion for acute stroke

**Status: research prototype (v0.2.0). Research use only. Not a medical device. Not for clinical decision-making.**
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
rotations and recovering them (to < 0.6 voxel and < 1°). Motion parameters are estimated on a copy downsampled to
about 256 pixels across and applied at full resolution; the rotation search is skipped for frames whose
translation-only fit is already near-perfect, so a 512 × 512 × 16-slice × 50-frame export takes about 100 s.

## Real vendor data: UniToBrain (GE LightSpeed VCT, shuttle mode)

The first real export the reader saw was patient MOL-001 of [UniToBrain](https://ieee-dataport.org/open-access/unitobrain)
(258 stroke CTPs from a GE LightSpeed VCT, 80 kVp, raw DICOM, open access): 16 slices × 5 mm at 0.49 mm in-plane,
512 × 512, JPEG-lossless, acquired as two 4 cm table positions sampled alternately every 2.8 s (shuttle mode). The
reader parsed the geometry, timing and HU scaling unaided and flagged the shuttle pattern; `scripts/fetch_unitobrain.py`
pulls single patients out of the 80 GB archive by HTTP range requests, and `scripts/unitobrain_compare.py` runs the
pipeline on one patient and compares with the dataset's own NLR maps.

It also found the first real bug. At 0.49 mm and 80 kVp the per-voxel noise is high enough that a 50 HU curve
crosses 20 % of its peak inside the noise, so weak voxels looked like the earliest-arriving ones and the AIF
selector chose a 44 HU cluster over basal arteries peaking above 500 HU. Three changes fixed it, and they are now the
defaults: each voxel's arrival threshold is at least four standard deviations of its own pre-bolus noise; a candidate
cluster is scored on its bright core (voxels above half the cluster's 95th-percentile peak) rather than on all the
partial-volume voxels along the vessel; and an AIF candidate must peak at least 2 s before the whole-brain tissue
curve (later ones are venous or dispersed). On ISLES 2018 the same changes tightened the Tmax > 6 limits of
agreement from ±50 to about ±37 mL and raised the ICC from 0.87 to 0.93 (table below), so they were not a special
case for one scanner. The MOL-001 maps show a left MCA-territory Tmax prolongation with a 1 mL core, on the same side
as the dataset's delay map; the patient did not move (max shift 0.13 mm). Whole-brain median CBF matches the NLR
reference (14 vs 15 mL/100 g/min, r = 0.73); our CBV and MTT are 1.7× theirs, a scaling question that needs a
second reference before it can be attributed to either side.

One caveat this case makes concrete: absolute CBF and MTT on it depend strongly on the SVD truncation (whole-brain
median CBF 13, 20 and 32 mL/100 g/min at 10 %, 5 % and 2 % truncation; MTT 10.6, 7.2 and 4.6 s), and the Tmax > 6 s
volume moves with it (229, 329 and 242 mL). The 10 % default was calibrated against RAPID on ISLES 2018, which is
sampled every second; this scan is sampled every 2.8 s per slice and resampled to 1 s, which changes the singular
value spectrum the truncation acts on. The rCBF < 30 % core is unaffected (it is relative to the contralateral
hemisphere), but the Tmax > 6 volume on shuttle-mode data should not be trusted to RAPID-equivalence until the
regularisation has been calibrated on RAPID-processed cases with that sampling, which is what the paired cohort is
for; the ASIST phantom settles the absolute scale independently.

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
| voxelwise Pearson r, CBF / CBV / MTT / Tmax | 0.98 / 0.98 / 0.3 / 0.79 (Spearman 0.87 for Tmax) |
| Tmax > 6 s volume | ICC 0.93, bias +0.5 mL, LoA -36 to +37 mL, median Dice 0.89 (IQR 0.81–0.93) |
| rCBF < 30 % core volume | ICC 0.98, bias +0.9 mL, LoA -7 to +8 mL, median Dice 0.82 |
| DEFUSE-3 target profile | 83/94 agree (kappa 0.71) |
| core vs follow-up infarct (Dice) | openperfusion 0.29, RAPID 0.29 |
| runtime | median 0.8 s per case (2–8 slices), 19 s for a 22-slice case |

Which of our thresholds best reproduces RAPID's boundary: Tmax > 6 s (mean Dice 0.80 vs 0.79 at 7 s and 0.77 at 5 s), rCBF < 30 %
(Dice 0.74 vs 0.60 at 35 % and 0.56 at 25 %). The +2 s Tmax offset seen on the phantom does not appear against RAPID, i.e.
RAPID carries the same regularisation bias, as expected for a truncated-SVD family.

Input-function selection, after the vendor-data fix described above (noise-relative arrival threshold, cluster-core
scoring, 2 s lead over the tissue peak): on ISLES 2018 it raised the in-sample Tmax > 6 ICC from 0.87 to 0.93 and
narrowed the limits of agreement from ±50 to ±37 mL, with 9 cases improving and 4 worsening by more than 0.15 Dice.
The fallback for artery-free thin slabs (a second search for small, early, narrow 2-voxel clusters when the primary
curve peaks below 15 HU) remains; 4 cases still rely on it and are flagged (`aif.qc["aif_weak"]`), and `openperfusion
dicom` prints a warning. A timing-based fallback trigger was tried earlier and rejected (it widened the limits to ±73 mL).

## Held-out numbers (split-half, the ones to quote)

`scripts/split_half.py` splits the 94 scans at random into halves A and B, scores a 12-configuration grid
(bcSVD truncation 5 / 10 / 15 %, oSVD target 0.05; smoothing 1.5 / 2.0 / 2.5 voxels) on one half, and evaluates
the best configuration on the other half, both ways, so every case is scored once by a configuration that did not
see it. With the current input-function selection the two halves no longer pick the same truncation (tuning on A
chose 15 %, tuning on B chose 10 %; both chose 2.0-voxel smoothing), so the held-out numbers below are slightly
worse than the in-sample ones above, as they should be:

| held-out metric | result |
|---|---|
| Tmax > 6 s volume | ICC 0.92, bias +1.5 mL, LoA -41 to +44 mL, mean Dice 0.79, median 0.87 |
| rCBF < 30 % core volume | ICC 0.98, bias +0.8 mL, LoA -7 to +8 mL, mean Dice 0.71, median 0.82 |
| DEFUSE-3 target profile | agreement 0.86, kappa 0.66 (13 discordant, all in 2–8-slice slabs; 15/15 agree in the 16–22-slice cases) |
| voxelwise r, CBF / CBV / Tmax | 0.97 / 0.98 / 0.77 |
| tune A → test B (15 %) / tune B → test A (10 %) | Tmax > 6 ICC 0.87 / 0.96; core ICC 0.98 / 0.98; kappa 0.65 / 0.68 |

The previous release (v0.1.0, before the input-function change) scored Tmax > 6 ICC 0.87 with limits −52 to +50 mL
and DEFUSE-3 kappa 0.74 on the same protocol; the change buys volume agreement at a small cost in classification
agreement near the DEFUSE-3 thresholds in thin slabs. 66 of 94 cases are within 10 mL of RAPID on
Tmax > 6 and 85 within 5 mL on core. Per-case results: `results/split_half_v7/heldout_pooled_cases.csv`; figure:
`results/split_half_v7/agreement_heldout_94.png`.

## Next steps

1. More vendor exports: Siemens, Philips and Canon, and a GE case with motion; more UniToBrain patients.
2. Request the ASIST-Japan phantom; reproduce the Kudo 2013 delay test.
3. UHN paired cohort (REB) for multi-vendor, current-RAPID-version calibration.
4. A DICOM writer for the maps (secondary capture / parametric map) for PACS push-back.

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
scripts/                       ISLES sweeps and split-half validation; UniToBrain fetch and comparison
tests/                         phantom recovery, DICOM round trip, motion correction
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

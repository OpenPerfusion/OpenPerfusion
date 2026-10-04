# OpenPerfusion — open-source CT perfusion for acute stroke

**Status: research prototype (v0.3.0). Research use only. Not a medical device. Not for clinical decision-making.**
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
1.5 mm in-plane smoothing, contralateral-hemisphere rCBF reference, 1 mL cluster filter.

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

## Real vendor data: UniToBrain (GE LightSpeed VCT, 11 patients, two acquisition modes)

The first real exports the reader saw were eleven patients of [UniToBrain](https://ieee-dataport.org/open-access/unitobrain)
(258 stroke CTPs from a GE LightSpeed VCT, 80 kVp, raw DICOM, open access). `scripts/fetch_unitobrain.py` pulls
single patients out of the 80 GB archive by HTTP range requests (about 130 MB per patient); `scripts/unitobrain_batch.py`
runs every patient and compares with the dataset's own NLR maps; per-patient results are in `results/unitobrain/`.
They turned out to be two different protocols, and between them they found five real bugs, all fixed and all now defaults:

| what the data looked like | what broke | fix |
|---|---|---|
| 512 × 512 at 0.49 mm, 80 kVp (noise 5–8 HU per voxel) | a 50 HU curve crosses 20 % of its peak inside the noise, so weak voxels looked like the earliest arrivals and the AIF selector chose a 44 HU cluster over basal arteries at 500 HU | per-voxel arrival threshold ≥ 4 SD of the curve's own pre-bolus noise; candidate clusters scored on their bright core; AIF must peak ≥ 2 s before the whole-brain tissue curve |
| 0.5 s cine frames, slow boluses | noise bumps with peaks of 20–30 HU entered the candidate pool and won on "arrival" | candidate peak must be ≥ 6 × its own noise SD |
| ICA next to the cavernous sinus at 0.49 mm | the 6-voxel arterial seed grew into 12 voxels peaking 4 s later | cluster growth restricted to voxels peaking within 2 s of the seed |
| GE cine mode: AcquisitionTime identical for all 89 frames; frame time only in private (0019,1024) Mid Scan Time | all frames collapsed onto one time point | the reader chooses the timing source per series (GE mid-scan time, AcquisitionTime, TriggerTime, ContentTime, else InstanceNumber order) and records it in `meta["time_source"]` |
| 0.49 mm voxels vs ISLES's 0.8–1.0 mm | smoothing and mask erosion were in voxels, so vendor exports got half the physical smoothing and the Tmax maps were patchy | `spatial_sigma_mm`, `mask_open_mm`, `mask_erode_mm` (1.5 / 2 / 1 mm, the smoothing re-selected by split-half), converted per dataset |

Patient MOL-001 is a two-position shuttle acquisition (16 slices × 5 mm sampled alternately every 2.8 s); the other
ten are cine mode (8 slices, 89 frames at 0.5 s, 44 s). All eleven now load unaided, get an arterial input function
(peaks 100–340 HU) and produce maps a clinician would recognise: left MCA-territory Tmax lesions in MOL-001, -007 and
-012 and a right-sided one in MOL-011, each on the same side as the dataset's own delay map; a clean negative in
MOL-005; one patient (MOL-006) who moved 6.4 mm and 6.5°, the first real test of the motion correction, which
registered the frames in 66 s and still produced a lateralised map. Whole-brain median MTT is 6–9 s on the cine
cases. Correlation with the NLR reference maps is 0.6 for CBF and 0.6–0.8 for CBV on the ten cine cases (the
reference maps are from a different, registered and filtered reconstruction, so voxelwise agreement is bounded);
their absolute CBF is about half of ours and their delay map is not the same quantity as Tmax, so only sidedness
and relative patterns are compared. On ISLES 2018 the same changes were a net gain (tables below), so none of them
is a special case for one scanner.

One caveat these cases make concrete: absolute CBF and MTT, and therefore the Tmax > 6 s volume, depend on the SVD
truncation (on MOL-001, whole-brain median CBF 13, 20 and 32 mL/100 g/min at 10 %, 5 % and 2 % truncation; MTT
10.6, 7.2 and 4.6 s; Tmax > 6 229, 329 and 242 mL). The 10 % default was calibrated against RAPID on 1 s ISLES data;
shuttle and 0.5 s cine data are resampled to 1 s, which changes the singular value spectrum the truncation acts on.
The rCBF < 30 % core is unaffected (it is relative to the contralateral hemisphere), but Tmax > 6 volumes on other
sampling schemes should not be called RAPID-equivalent until the regularisation has been calibrated on
RAPID-processed cases with that sampling, which is what the paired cohort is for; the ASIST phantom settles the
absolute scale independently.

## ISLES'24: Siemens and Philips scanners, icobrain cva as the reference (first 10 subjects)

[ISLES'24](https://pubs.rsna.org/doi/10.1148/ryai.250603) has 149 training subjects from two centres on Siemens
(Somatom Force, Xcite, AS+) and Philips (Brilliance 64, Ingenuity) scanners, with the 4D CTP (co-registered,
1 s frames, 1 x 1 x 5 mm here) and CBF/CBV/MTT/Tmax maps from icobrain cva, a second FDA-cleared package, plus the
follow-up DWI lesion. `scripts/unpack_isles24.py` unpacks the Hugging Face parquet mirror; `openperfusion/io_isles2024.py`
loads a subject in the same shape as the ISLES 2018 loader so the same harness runs (`scripts/run_isles2024.py`).

The maps live on the NCCT grid and the CTP on its own, and for some subjects the header alignment between the two
is off by millimetres to centimetres (sub-stroke0010: 37 mm), so the loader refines it with a rigid registration of
the CTP pre-contrast mean to the NCCT (`openperfusion/register.py`, SimpleITK, mutual information) and keeps the
registered alignment when it improves the brain-window correlation. That raised voxelwise r for CBF from 0.57 to
0.76 and Tmax > 6 Dice from 0.60 to 0.68 on these ten.

| metric, 10 subjects, defaults as validated on ISLES 2018 (no tuning) | result |
|---|---|
| Tmax > 6 s volume | ICC 0.95, bias +1.1 mL, LoA -61 to +63 mL, mean Dice 0.68 (0.37–0.85) |
| rCBF < 30 % core volume | ICC 0.88, bias -0.2 mL, mean Dice 0.33 (cores are 0–34 mL) |
| voxelwise r, CBF / CBV / Tmax | 0.76 / 0.73 / 0.61 |
| which Tmax threshold best matches icobrain's > 6 s boundary | 6 s (Dice 0.68; 5 s 0.65, 7 s 0.65) |
| hypoperfusion vs follow-up infarct, Dice | ours 0.17, icobrain 0.18 |

Reading it: against a second commercial package, on two more vendors, with no tuning, the volumes agree as well
as they do with RAPID (ICC 0.95) and the 6 s boundary is the same; voxelwise overlap is lower than against RAPID
because icobrain's maps are much more heavily smoothed than ours (visible in `results/isles24_10/montage.png`)
and because the comparison crosses a registration. Per-subject results: `results/isles24_10/cases.csv`. The
remaining 139 subjects are the obvious next run.

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

`openperfusion isles2018 TRAINING` with the defaults (block-circulant SVD, 10 % truncation, 1.5 mm in-plane
smoothing, contralateral rCBF reference, 1 mL cluster filter). RAPID maps are the ISLES 2018 reference maps;
both sets of maps go through the same thresholding code so the comparison isolates the map computation.

| metric | result |
|---|---|
| voxelwise Pearson r, CBF / CBV / MTT / Tmax | 0.98 / 0.98 / 0.3 / 0.81 (Spearman 0.89 for Tmax) |
| Tmax > 6 s volume | ICC 0.95, bias +4.2 mL, LoA -27 to +35 mL, median Dice 0.90 (IQR 0.84–0.93) |
| rCBF < 30 % core volume | ICC 0.98, bias -0.7 mL, LoA -7 to +6 mL, median Dice 0.83 |
| DEFUSE-3 target profile | 87/94 agree (kappa 0.80) |
| core vs follow-up infarct (Dice) | openperfusion 0.27, RAPID 0.29 |
| runtime | median 1.0 s per case (2–8 slices), 25 s for a 22-slice case |

Which of our thresholds best reproduces RAPID's boundary: Tmax > 6 s (mean Dice 0.83 vs 0.82 at 7 s and 0.77 at 5 s), rCBF < 30 %
(Dice 0.76 vs 0.64 at 35 % and 0.50 at 25 %). The +2 s Tmax offset seen on the phantom does not appear against RAPID, i.e.
RAPID carries the same regularisation bias, as expected for a truncated-SVD family.

Input-function selection, after the vendor-data fixes described above (noise-relative arrival threshold and SNR floor,
cluster-core scoring, timing-constrained growth, 2 s lead over the tissue peak): on ISLES 2018 they raised the
in-sample Tmax > 6 ICC from 0.87 to 0.95 and narrowed the limits of agreement from ±50 to about ±27 mL, with no case
worse by more than 0.15 Dice than before and several better. The fallback for artery-free thin slabs (a second search
for small, early, narrow 2-voxel clusters when the primary curve peaks below 15 HU) remains; one case still relies on
it and is flagged (`aif.qc["aif_weak"]`), and `openperfusion dicom` prints a warning.

## Held-out numbers (split-half, the ones to quote)

`scripts/split_half.py` splits the 94 scans at random into halves A and B, scores a 12-configuration grid
(bcSVD truncation 5 / 10 / 15 %, oSVD target 0.05; smoothing 1.5 / 2.0 / 2.5 mm) on one half, and evaluates
the best configuration on the other half, both ways, so every case is scored once by a configuration that did not
see it. Both halves independently selected the same configuration (bcSVD 10 %, 1.5 mm smoothing), which is now the
default; the in-sample table above therefore uses the same configuration and the two agree:

| held-out metric | result |
|---|---|
| Tmax > 6 s volume | ICC 0.95, bias +4.2 mL, LoA -27 to +35 mL, mean Dice 0.83, median 0.90 |
| rCBF < 30 % core volume | ICC 0.98, bias -0.7 mL, LoA -7 to +6 mL, mean Dice 0.76, median 0.83 |
| DEFUSE-3 target profile | agreement 0.93, kappa 0.80 (7 discordant, all in 2–8-slice slabs; 15/15 agree in the 16–22-slice cases) |
| voxelwise r, CBF / CBV / Tmax | 0.98 / 0.98 / 0.81 |
| tune A → test B / tune B → test A | Tmax > 6 ICC 0.96 / 0.94; core ICC 0.99 / 0.98; kappa 0.88 / 0.73 |

For the record, the same protocol gave Tmax > 6 ICC 0.87 with limits −52 to +50 mL and DEFUSE-3 kappa 0.74 for
v0.1.0, and ICC 0.92, −41 to +44 mL, kappa 0.66 for v0.2.0; every change since was motivated by a real vendor export
and checked here. 71 of 94 cases are within 10 mL of RAPID on Tmax > 6 and 87 within 5 mL on core.
Per-case results: `results/split_half_v9/heldout_pooled_cases.csv`; figure: `results/split_half_v9/agreement_heldout_94.png`.

## Next steps

1. More vendor exports: Siemens, Philips and Canon. (GE: 11 UniToBrain patients in two acquisition modes, including one with 6 mm of motion, are done.)
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
openperfusion/io_isles2024.py  ISLES'24 loader (icobrain maps registered into the CTP grid)
openperfusion/register.py       rigid CTP-to-NCCT registration (SimpleITK, optional)
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

# Contributing to OpenPerfusion

Thanks for looking. OpenPerfusion is a research prototype: an open-source CT perfusion pipeline
validated against RAPID output on ISLES 2018. It is **not a medical device and must not be used
for clinical decision-making**. Every contribution should keep that true until there is a
regulatory pathway for it not to be.

## What helps most right now

In rough order of usefulness:

1. **Real vendor DICOM exports.** The DICOM reader and motion correction are tested on synthetic
   DICOM only. One fully anonymised CTP export per scanner (GE, Siemens, Philips, Canon/Toshiba),
   including shuttle / jog-mode acquisitions, will find more bugs than any amount of code review.
   See *Sharing imaging data* below before sending anything.
2. **Reports of disagreement with a commercial package.** If you have a case where OpenPerfusion
   and RAPID / Viz / Olea / syngo disagree substantially, the `summary.json` and the QC block
   from `openperfusion dicom` (no images needed) tell us a lot.
3. **Validation on other public data** (ISLES 2024, CENTER-TBI perfusion, any paired dataset
   with reference maps).
4. Fixes, tests and documentation.

## Sharing imaging data

Never attach patient data to an issue, pull request or discussion. Before sharing a DICOM export
privately, strip or replace every identifying tag (names, IDs, dates of birth, accession and
study dates, institution, referring physician, private tags, and burned-in text on the images),
regenerate the UIDs, and make sure your institution's research-ethics and privacy rules allow it.
`pydicom`'s anonymisation examples, `dicom-anonymizer`, or the RSNA Anonymizer all work. If in
doubt, ask first by opening a discussion that describes the scanner and acquisition without any
patient details.

## Development setup

```bash
git clone https://github.com/OpenPerfusion/OpenPerfusion.git
cd OpenPerfusion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dicom,dev]"
pytest -q            # 11 tests, about 10 s
openperfusion phantom --out results/phantom   # digital-phantom sanity check
```

The ISLES 2018 training set is needed for `openperfusion isles2018` and the scripts under
`scripts/`; it is free to download after registration (see the README).

## Pull requests

- Open an issue or discussion first for anything beyond a small fix, so the design can be agreed
  before the work is done.
- Keep the phantom tests passing and add a test for new behaviour. A change to the deconvolution,
  AIF selection or thresholding should be accompanied by its effect on the ISLES 2018 held-out
  numbers (`scripts/split_half.py`) in the PR description.
- Match the existing style: plain NumPy / SciPy, type hints, a module docstring that says what
  the method is and which paper it follows, no new heavy dependencies without discussion.
- Do not copy code from software with incompatible licences (PyPeT is non-commercial; commercial
  perfusion packages are off limits). Re-implement from the published method and cite it.
- Every file carries an `SPDX-License-Identifier: Apache-2.0` header; new files should too.

By contributing you agree that your contribution is licensed under the Apache License 2.0 that
covers the project.

## Reporting a problem

Open an issue with the OpenPerfusion version, the command you ran, the full `summary.json` or
traceback, and the scanner / acquisition details (vendor, model, slab coverage, frame interval,
shuttle or not). Please do not include images or identifiers.

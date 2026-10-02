# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""openperfusion — open-source CT perfusion for acute stroke.

Research use only. Not a medical device. Not for clinical decision-making.

Pipeline (see README):
    4D CTP  ->  preprocess (mask, baseline, concentration)
            ->  AIF / VOF selection
            ->  deconvolution (block-circulant SVD / oSVD, or Fourier-Wiener)
            ->  CBF, CBV, MTT, Tmax, TTP maps
            ->  rCBF < 30 % core, Tmax > 6 s hypoperfusion, mismatch
            ->  validation against reference maps (RAPID via ISLES 2018) or phantom truth
"""

__version__ = "0.3.0"

from .phantom import make_phantom, PhantomTruth  # noqa: F401
from .preprocess import brain_mask_ct, concentration_from_hu  # noqa: F401
from .aif import select_aif_vof, CurveFeatures  # noqa: F401
from .deconvolve import deconvolve_bcsvd, deconvolve_osvd, deconvolve_fourier  # noqa: F401
from .maps import perfusion_maps, PerfusionMaps, threshold_maps, ThresholdResult  # noqa: F401
from .pipeline import run_pipeline, PipelineConfig, PipelineResult  # noqa: F401

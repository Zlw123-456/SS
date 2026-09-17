"""Standalone band-allocation rules for ABAR speech separation.

Import BandRecalibration from abar_speech.recalibration when using PyTorch.
Allocation-only usage does not import or require PyTorch.
"""

from .allocation import band_slices, build_exp_band, build_hybrid_band

__all__ = ["band_slices", "build_exp_band", "build_hybrid_band"]
__version__ = "0.1.0"

"""Fixed-budget frequency-band allocation, independent of any backbone."""

import math
from numbers import Integral, Real

import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def build_exp_band(total_bins, num_bands, alpha=2.0):
    """Return positive integer widths using the experimental rounding rule.

    Fractional widths are proportional to exp(alpha * linspace(0, 1, K)).
    After flooring and enforcing a one-bin minimum, add any remaining bins
    from the highest band downward. Remove any excess from bands wider than
    one, starting at the lowest band. Repeat until coverage is exact.

    This is not a largest-remainder allocation. Changing the correction order
    changes the partition used in the experiments.
    """
    total_bins = _integer(total_bins, "total_bins")
    num_bands = _integer(num_bands, "num_bands")
    if num_bands > total_bins:
        raise ValueError("num_bands cannot exceed total_bins")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real number")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    if num_bands == 1:
        return [total_bins]

    # Keep the arithmetic and correction order of the research implementation.
    positions = np.linspace(0.0, 1.0, num_bands)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            weights = np.exp(alpha * positions)
            raw_width = weights / weights.sum() * total_bins
    except FloatingPointError as error:
        raise ValueError("alpha or total_bins exceeds the supported numeric range") from error
    band_width = np.maximum(np.floor(raw_width).astype(np.int64), 1)
    difference = int(total_bins - band_width.sum())

    while difference > 0:
        for index in range(num_bands - 1, -1, -1):
            if difference == 0:
                break
            band_width[index] += 1
            difference -= 1

    while difference < 0:
        changed = False
        for index in range(num_bands):
            if difference == 0:
                break
            if band_width[index] > 1:
                band_width[index] -= 1
                difference += 1
                changed = True
        if not changed:
            raise RuntimeError("Unable to correct the allocated band widths")

    widths = [int(width) for width in band_width]
    if len(widths) != num_bands or sum(widths) != total_bins or min(widths) < 1:
        raise RuntimeError("Invalid frequency-band allocation")
    return widths


def build_hybrid_band(total_bins, low_bins=60, total_bands=67, alpha=2.0):
    """Allocate low-frequency single-bin bands and exponential high bands.

    Speech configuration: F=321, K_low=60, K=67, alpha=2.0.
    Result: [1] * 60 + [11, 15, 21, 30, 43, 59, 82].
    low_bins=0 is supported for exponential-only allocation experiments.
    """
    total_bins = _integer(total_bins, "total_bins")
    total_bands = _integer(total_bands, "total_bands")
    low_bins = _integer(low_bins, "low_bins", minimum=0)
    if low_bins >= total_bands:
        raise ValueError("low_bins must be smaller than total_bands")
    if total_bands > total_bins:
        raise ValueError("total_bands cannot exceed total_bins")
    return [1] * low_bins + build_exp_band(
        total_bins=total_bins - low_bins,
        num_bands=total_bands - low_bins,
        alpha=alpha,
    )


def band_slices(widths):
    """Convert ordered positive widths into contiguous, half-open slices."""
    widths = list(widths)
    if not widths:
        raise ValueError("widths cannot be empty")
    result = []
    start = 0
    for value in widths:
        width = _integer(value, "band width")
        result.append(slice(start, start + width))
        start += width
    return result

"""Neutral numerical primitives shared by coverage post-processing backends.

This module intentionally contains no segmentation, viewer, or legacy
homogeneity workflow.  Keeping these small primitives here lets production
local-heterogeneity code remain independent of the optional legacy diagnostic
comparison implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CoverageMetricComponents:
    """Numerator, denominator, and ratio for one coverage metric."""

    numerator: float
    denominator: float
    value: float | None


def coordinate_maps(
    shape: tuple[int, int], center_rc: tuple[float, float], radius_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized radial and ``[0, 2*pi)`` angular coordinate maps."""

    if not (math.isfinite(radius_px) and radius_px > 0):
        raise ValueError("Sphere radius must be a positive finite value.")
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    radius = np.hypot(rows - center_rc[0], cols - center_rc[1]) / radius_px
    angle = np.mod(
        np.arctan2(rows - center_rc[0], cols - center_rc[1]),
        2.0 * math.pi,
    )
    return radius, angle


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the lower deterministic median for non-negative weights.

    When cumulative weight reaches exactly 50 percent, the lower value is
    selected (``searchsorted(..., side="left")``).  This convention is stable
    for tied values and matches the historic coverage implementation.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        raise ValueError(
            "Weighted median requires at least one positive finite weight."
        )
    order = np.argsort(values[keep], kind="stable")
    sorted_values = values[keep][order]
    cumulative = np.cumsum(weights[keep][order])
    index = np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left")
    return float(sorted_values[index])

"""Lightweight central spherical-cap coverage post-processing.

This module intentionally operates only on already segmented bead and Ag
masks.  It does not contain, or invoke, any bead or Ag segmentation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np


SPHERE_DIAMETER_METRICS = ("mean_xy_diameter", "equivalent_diameter")
CAP_COVERAGE_METRICS = (
    "projected_fraction",
    "surface_weighted_fraction",
    "projected_over_cap_surface",
)


def normalize_sphere_diameter_metric(metric: str) -> str:
    """Validate the scalar used solely for spherical-cap geometry."""

    value = str(metric).strip().casefold()
    if value not in SPHERE_DIAMETER_METRICS:
        raise ValueError(
            f"Unsupported sphere_diameter_metric {metric!r}. Supported values: "
            "mean_xy_diameter, equivalent_diameter."
        )
    return value


def normalize_cap_coverage_metric(metric: str) -> str:
    """Validate one explicit central-cap coverage metric name."""

    value = str(metric).strip().casefold()
    if value not in CAP_COVERAGE_METRICS:
        raise ValueError(
            f"Unsupported selected_cap_coverage_metric {metric!r}. Supported values: "
            "projected_fraction, surface_weighted_fraction, projected_over_cap_surface."
        )
    return value


def sphere_diameter_px(
    equivalent_diameter_px: float, x_diameter_px: float, y_diameter_px: float, metric: str
) -> float:
    """Return the cap-geometry sphere diameter from already measured ROI values."""

    if normalize_sphere_diameter_metric(metric) == "equivalent_diameter":
        return float(equivalent_diameter_px)
    return float((x_diameter_px + y_diameter_px) / 2.0)


@dataclass(frozen=True)
class CoverageCapGeometry:
    """Geometric properties of one central projected spherical cap."""

    center_rc: tuple[float, float]
    radius_fraction: float
    sphere_radius_px: float
    cap_radius_px: float
    half_angle_rad: float
    half_angle_deg: float
    height_px: float
    theoretical_circle_mask: np.ndarray
    valid_mask: np.ndarray
    completeness: float


@dataclass(frozen=True)
class CoverageCapMetrics:
    """Central-cap coverage values and calibrated geometric quantities."""

    geometry: CoverageCapGeometry
    valid: bool
    invalid_reason: str | None
    projected_fraction: float | None
    surface_weighted_fraction: float | None
    projected_over_cap_surface: float | None
    bead_pixel_count: int
    ag_pixel_count: int
    theoretical_circle_pixel_count: int
    # Backward-compatible aliases retained for existing JSON/viewer callers.
    projected_coverage: float | None
    surface_weighted_coverage: float | None
    projected_area_px2: float
    projected_area_m2: float | None
    surface_area_px2: float
    surface_area_m2: float | None
    cap_radius_m: float | None
    height_m: float | None

    def selected_value(self, metric: str) -> float | None:
        """Return one explicitly selected cap metric without changing geometry."""

        return getattr(self, normalize_cap_coverage_metric(metric))


def nearest_mask_pixel(mask: np.ndarray, center_rc: tuple[float, float]) -> tuple[int, int]:
    """Return a valid in-mask pixel nearest to ``center_rc``."""

    candidate = np.asarray(mask, dtype=bool)
    if candidate.ndim != 2 or not candidate.any():
        raise ValueError("A non-empty two-dimensional bead mask is required.")
    row = int(np.clip(round(center_rc[0]), 0, candidate.shape[0] - 1))
    col = int(np.clip(round(center_rc[1]), 0, candidate.shape[1] - 1))
    if candidate[row, col]:
        return row, col
    rows, cols = np.nonzero(candidate)
    nearest = int(np.argmin((rows - row) ** 2 + (cols - col) ** 2))
    return int(rows[nearest]), int(cols[nearest])


def compute_coverage_cap_metrics(
    bead_mask: np.ndarray,
    ag_mask: np.ndarray,
    center_rc: tuple[float, float],
    sphere_radius_px: float,
    radius_fraction: float,
    pixel_size_m: Optional[float],
    *,
    compute_surface_weighted: bool = True,
    min_completeness: float,
) -> CoverageCapMetrics:
    """Compute central-cap metrics from existing full-size segmentation masks.

    The denominator is the bead-mask intersection with the theoretical circle.
    Completeness independently describes whether the circle itself was retained
    by the segmented bead ROI.
    """

    bead = np.asarray(bead_mask, dtype=bool)
    ag = np.asarray(ag_mask, dtype=bool)
    if bead.ndim != 2 or ag.ndim != 2 or bead.shape != ag.shape or not bead.any():
        raise ValueError("Bead and Ag masks must be matching non-empty 2D arrays.")
    if not (math.isfinite(sphere_radius_px) and sphere_radius_px > 0):
        raise ValueError("Sphere radius must be a positive finite pixel value.")
    if not (0.0 < radius_fraction <= 1.0):
        raise ValueError("Cap radius fraction must satisfy 0 < f <= 1.")
    if not (0.0 <= min_completeness <= 1.0):
        raise ValueError("Minimum cap completeness must be between 0 and 1.")

    anchor_row, anchor_col = nearest_mask_pixel(bead, center_rc)
    cap_radius_px = float(radius_fraction * sphere_radius_px)
    rows, cols = np.ogrid[: bead.shape[0], : bead.shape[1]]
    distance_sq = (rows - anchor_row) ** 2 + (cols - anchor_col) ** 2
    theoretical = distance_sq <= cap_radius_px * cap_radius_px
    theoretical_count = int(theoretical.sum())
    valid_mask = theoretical & bead
    completeness = float(valid_mask.sum() / theoretical_count) if theoretical_count else 0.0
    half_angle_rad = float(math.asin(radius_fraction))
    height_px = float(sphere_radius_px * (1.0 - math.sqrt(max(0.0, 1.0 - radius_fraction**2))))
    projected_area_px2 = float(math.pi * cap_radius_px * cap_radius_px)
    surface_area_px2 = float(2.0 * math.pi * sphere_radius_px * height_px)
    valid = completeness >= min_completeness and bool(valid_mask.any())
    invalid_reason = None if valid else (
        f"cap completeness {completeness:.3f} < minimum {min_completeness:.3f}"
    )

    projected: float | None = None
    weighted: float | None = None
    projected_over_surface: float | None = None
    bead_count = int(valid_mask.sum())
    ag_count = int((ag & valid_mask).sum())
    if valid:
        projected = float(ag_count / bead_count) if bead_count else None
        # This inexpensive post-processing value is always calculated.  The
        # argument remains for compatibility with older callers/configs.
        if bead_count:
            # The cap fraction can equal one, so keep the denominator finite at
            # the limiting circle edge without changing ordinary interior values.
            radial_sq = distance_sq.astype(np.float64)
            denom = np.sqrt(np.maximum(sphere_radius_px**2 - radial_sq, 1e-12))
            weights = sphere_radius_px / denom
            valid_weights = weights[valid_mask]
            weighted = float(weights[ag & valid_mask].sum() / valid_weights.sum())
        projected_over_surface = float(ag_count / surface_area_px2) if surface_area_px2 > 0 else None

    scale2 = pixel_size_m * pixel_size_m if pixel_size_m and pixel_size_m > 0 else None
    return CoverageCapMetrics(
        geometry=CoverageCapGeometry(
            center_rc=(float(anchor_row), float(anchor_col)),
            radius_fraction=float(radius_fraction),
            sphere_radius_px=float(sphere_radius_px),
            cap_radius_px=cap_radius_px,
            half_angle_rad=half_angle_rad,
            half_angle_deg=float(math.degrees(half_angle_rad)),
            height_px=height_px,
            theoretical_circle_mask=theoretical,
            valid_mask=valid_mask,
            completeness=completeness,
        ),
        valid=valid,
        invalid_reason=invalid_reason,
        projected_fraction=projected,
        surface_weighted_fraction=weighted,
        projected_over_cap_surface=projected_over_surface,
        bead_pixel_count=bead_count,
        ag_pixel_count=ag_count,
        theoretical_circle_pixel_count=theoretical_count,
        projected_coverage=projected,
        surface_weighted_coverage=weighted,
        projected_area_px2=projected_area_px2,
        projected_area_m2=float(projected_area_px2 * scale2) if scale2 else None,
        surface_area_px2=surface_area_px2,
        surface_area_m2=float(surface_area_px2 * scale2) if scale2 else None,
        cap_radius_m=float(cap_radius_px * pixel_size_m) if pixel_size_m else None,
        height_m=float(height_px * pixel_size_m) if pixel_size_m else None,
    )


def cumulative_cap_sweep(
    bead_mask: np.ndarray,
    ag_mask: np.ndarray,
    center_rc: tuple[float, float],
    sphere_radius_px: float,
    fractions: Sequence[float],
    pixel_size_m: Optional[float],
    *,
    min_completeness: float,
) -> list[CoverageCapMetrics]:
    """Evaluate many cap fractions from existing masks without segmentation."""

    return [
        compute_coverage_cap_metrics(
            bead_mask, ag_mask, center_rc, sphere_radius_px, float(fraction), pixel_size_m,
            compute_surface_weighted=True, min_completeness=min_completeness,
        )
        for fraction in fractions
    ]


def annular_cap_profile(
    bead_mask: np.ndarray,
    ag_mask: np.ndarray,
    center_rc: tuple[float, float],
    sphere_radius_px: float,
    *,
    bins: int = 16,
) -> list[dict[str, float | int | None]]:
    """Compute non-overlapping local radial cap metrics from existing masks."""

    bead = np.asarray(bead_mask, dtype=bool)
    ag = np.asarray(ag_mask, dtype=bool)
    if bead.ndim != 2 or ag.shape != bead.shape or not bead.any() or sphere_radius_px <= 0:
        raise ValueError("Matching non-empty masks and a positive sphere radius are required.")
    if bins < 1:
        raise ValueError("Profile bins must be positive.")
    anchor_row, anchor_col = nearest_mask_pixel(bead, center_rc)
    rows, cols = np.ogrid[: bead.shape[0], : bead.shape[1]]
    radial_sq = (rows - anchor_row) ** 2 + (cols - anchor_col) ** 2
    normalized = np.sqrt(radial_sq) / sphere_radius_px
    weights = sphere_radius_px / np.sqrt(np.maximum(sphere_radius_px**2 - radial_sq, 1e-12))
    output: list[dict[str, float | int | None]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        annulus = (normalized >= low) & ((normalized < high) if index < bins - 1 else (normalized <= high))
        bead_region = annulus & bead
        ag_region = bead_region & ag
        bead_count = int(bead_region.sum())
        ag_count = int(ag_region.sum())
        # The spherical-zone area for theta_low..theta_high is
        # 2*pi*R^2*(cos(theta_low)-cos(theta_high)).
        surface_area = float(2.0 * math.pi * sphere_radius_px**2 * (
            math.sqrt(max(0.0, 1.0 - low**2)) - math.sqrt(max(0.0, 1.0 - high**2))
        ))
        output.append({
            "r_over_R_inner": low,
            "r_over_R_outer": high,
            "r_over_R_center": (low + high) / 2.0,
            "bead_pixel_count": bead_count,
            "ag_pixel_count": ag_count,
            "projected_fraction": float(ag_count / bead_count) if bead_count else None,
            "surface_weighted_fraction": (
                float(weights[ag_region].sum() / weights[bead_region].sum())
                if bead_count and float(weights[bead_region].sum()) > 0 else None
            ),
            "projected_over_cap_surface": float(ag_count / surface_area) if surface_area > 0 else None,
        })
    return output

"""Lightweight central spherical-cap coverage post-processing.

This module intentionally operates only on already segmented bead and Ag
masks.  It does not contain, or invoke, any bead or Ag segmentation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage as ndi


SPHERE_DIAMETER_METRICS = (
    "mean_xy_diameter",
    "equivalent_diameter",
    "max_inscribed_circle",
)
CAP_COVERAGE_METRICS = (
    "projected_fraction",
    "surface_weighted_fraction",
    "projected_over_cap_surface",
)


def equal_width_radial_intervals(
    inner_fraction: float, outer_fraction: float, requested_width_fraction: float
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Construct an exact, gap-free equal-width normalized radial partition."""

    if not (0 <= inner_fraction < outer_fraction <= 1 and 0 < requested_width_fraction <= outer_fraction - inner_fraction):
        raise ValueError("Radial interval fractions are invalid.")
    count = max(1, int(round((outer_fraction - inner_fraction) / requested_width_fraction)))
    effective = (outer_fraction - inner_fraction) / count
    return tuple(
        (inner_fraction + index * effective, inner_fraction + (index + 1) * effective)
        for index in range(count)
    ), float(effective)


def normalize_sphere_diameter_metric(metric: str) -> str:
    """Validate the scalar used solely for spherical-cap geometry."""

    value = str(metric).strip().casefold()
    if value not in SPHERE_DIAMETER_METRICS:
        raise ValueError(
            f"Unsupported sphere_diameter_metric {metric!r}. Supported values: "
            "mean_xy_diameter, equivalent_diameter, max_inscribed_circle."
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

    selected = normalize_sphere_diameter_metric(metric)
    if selected == "equivalent_diameter":
        return float(equivalent_diameter_px)
    if selected == "max_inscribed_circle":
        raise ValueError("max_inscribed_circle requires the bead mask geometry.")
    return float((x_diameter_px + y_diameter_px) / 2.0)


@dataclass(frozen=True)
class SphereGeometry:
    """Sphere reference center/radius selected from an accepted bead mask."""

    center_rc: tuple[float, float]
    radius_px: float
    mode: str
    plateau_pixel_count: int = 1


def maximum_inscribed_circle(bead_mask: np.ndarray) -> SphereGeometry:
    """Find a deterministic maximum inscribed circle of a non-empty bead mask.

    The Euclidean distance transform gives the distance to the nearest outside
    pixel.  A tied maximum plateau is represented by its mean row/column so
    repeated calls produce a stable center rather than choosing an arbitrary
    first pixel.
    """

    mask = np.asarray(bead_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("A non-empty two-dimensional bead mask is required.")
    distance = ndi.distance_transform_edt(mask)
    maximum = float(distance.max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("The bead mask has no positive inscribed-circle radius.")
    plateau = np.isclose(distance, maximum, rtol=0.0, atol=1e-12) & mask
    rows, cols = np.nonzero(plateau)
    return SphereGeometry(
        center_rc=(float(rows.mean()), float(cols.mean())),
        radius_px=maximum,
        mode="max_inscribed_circle",
        plateau_pixel_count=int(rows.size),
    )


def sphere_geometry_from_mask(
    bead_mask: np.ndarray,
    *,
    centroid_rc: tuple[float, float],
    equivalent_diameter_px: float,
    x_diameter_px: float,
    y_diameter_px: float,
    metric: str,
) -> SphereGeometry:
    """Resolve one diagnostic sphere geometry without invoking segmentation."""

    selected = normalize_sphere_diameter_metric(metric)
    if selected == "max_inscribed_circle":
        return maximum_inscribed_circle(bead_mask)
    diameter = sphere_diameter_px(
        equivalent_diameter_px, x_diameter_px, y_diameter_px, selected
    )
    return SphereGeometry(
        center_rc=(float(centroid_rc[0]), float(centroid_rc[1])),
        radius_px=float(diameter / 2.0),
        mode=selected,
    )


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


@dataclass(frozen=True)
class CapRadiusSensitivity:
    """Methodological sensitivity of one cap metric to radius selection."""

    metric: str
    point_count: int
    interval_low_fraction: float | None
    interval_high_fraction: float | None
    median_percent: float | None
    q10_percent: float | None
    q90_percent: float | None
    q10_q90_half_width_pp: float | None
    half_range_pp: float | None
    slope_pp_per_R: float | None


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


def cap_sensitivity_fractions(
    radius_fraction: float, half_width: float, step_fraction: float
) -> tuple[float, ...]:
    """Return a clipped sensitivity grid that always contains the fixed radius."""

    if not (0 < radius_fraction <= 1 and 0 <= half_width < 1 and 0 < step_fraction <= 1):
        raise ValueError("Cap sensitivity fractions are invalid.")
    epsilon = np.finfo(float).eps
    low = max(epsilon, radius_fraction - half_width)
    high = min(1.0, radius_fraction + half_width)
    values = list(np.arange(low, high + step_fraction * 0.5, step_fraction, dtype=float))
    values.extend((low, high, radius_fraction))
    return tuple(sorted({min(1.0, max(epsilon, float(value))) for value in values}))


def summarize_cap_sensitivity(
    metrics: Sequence[CoverageCapMetrics], metric: str
) -> CapRadiusSensitivity:
    """Summarize valid cumulative-cap points without calling their spread SD."""

    metric = normalize_cap_coverage_metric(metric)
    pairs = [
        (item.geometry.radius_fraction, item.selected_value(metric))
        for item in metrics
        if item.valid and item.selected_value(metric) is not None
    ]
    if len(pairs) < 3:
        return CapRadiusSensitivity(metric, len(pairs), None, None, None, None, None, None, None, None)
    fractions = np.asarray([item[0] for item in pairs], dtype=float)
    values = np.asarray([item[1] for item in pairs], dtype=float)
    q10, median, q90 = (float(np.quantile(values, q)) * 100.0 for q in (.1, .5, .9))
    slope = float(np.polyfit(fractions, values * 100.0, 1)[0]) if len(pairs) > 1 else None
    return CapRadiusSensitivity(
        metric=metric,
        point_count=len(pairs),
        interval_low_fraction=float(fractions.min()),
        interval_high_fraction=float(fractions.max()),
        median_percent=median,
        q10_percent=q10,
        q90_percent=q90,
        q10_q90_half_width_pp=(q90 - q10) / 2.0,
        half_range_pp=float((values.max() - values.min()) * 50.0),
        slope_pp_per_R=slope,
    )


def annular_cap_profile(
    bead_mask: np.ndarray,
    ag_mask: np.ndarray,
    center_rc: tuple[float, float],
    sphere_radius_px: float,
    *, width_fraction: float = 0.05, inner_fraction: float = 0.0, outer_fraction: float = 1.0, bins: int | None = None,
) -> list[dict[str, float | int | None]]:
    """Compute non-overlapping local radial cap metrics from existing masks."""

    bead = np.asarray(bead_mask, dtype=bool)
    ag = np.asarray(ag_mask, dtype=bool)
    if bead.ndim != 2 or ag.shape != bead.shape or not bead.any() or sphere_radius_px <= 0:
        raise ValueError("Matching non-empty masks and a positive sphere radius are required.")
    if bins is not None:
        width_fraction = (outer_fraction-inner_fraction) / bins
    if not (0 < width_fraction <= outer_fraction-inner_fraction and 0 <= inner_fraction < outer_fraction <= 1):
        raise ValueError("Annulus fractions are invalid.")
    anchor_row, anchor_col = nearest_mask_pixel(bead, center_rc)
    rows, cols = np.ogrid[: bead.shape[0], : bead.shape[1]]
    radial_sq = (rows - anchor_row) ** 2 + (cols - anchor_col) ** 2
    normalized = np.sqrt(radial_sq) / sphere_radius_px
    weights = sphere_radius_px / np.sqrt(np.maximum(sphere_radius_px**2 - radial_sq, 1e-12))
    output: list[dict[str, float | int | None]] = []
    intervals, effective_width = equal_width_radial_intervals(inner_fraction, outer_fraction, width_fraction)
    for index, (low, high) in enumerate(intervals):
        annulus = (normalized >= low) & ((normalized < high) if index < len(intervals) - 1 else (normalized <= high))
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
            "effective_width_fraction": effective_width,
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

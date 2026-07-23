"""Shared, mask-only radial and polar coverage-homogeneity post-processing.

The functions in this module deliberately receive already segmented masks.  They
do not import a viewer and never invoke bead or Ag segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from coverage_cap import (
    CAP_COVERAGE_METRICS,
    equal_width_radial_intervals,
    normalize_cap_coverage_metric,
)
from coverage_metric_utils import (
    CoverageMetricComponents,
    coordinate_maps,
    weighted_median,
)


@dataclass(frozen=True)
class SegmentCoverage:
    """One non-overlapping radial ring or annular polar sector."""

    index: int
    inner: float
    outer: float
    center: float
    bead_pixel_count: int
    ag_pixel_count: int
    completeness: float
    valid: bool
    projected_fraction_components: CoverageMetricComponents
    surface_weighted_fraction_components: CoverageMetricComponents
    projected_over_cap_surface_components: CoverageMetricComponents
    surface_area_px2: float
    start_angle_deg: float | None = None
    end_angle_deg: float | None = None

    def components(self, metric: str) -> CoverageMetricComponents:
        return getattr(self, f"{normalize_cap_coverage_metric(metric)}_components")

    # Compatibility properties for existing diagnostic plotting code.
    @property
    def projected_fraction(self) -> float | None:
        return self.projected_fraction_components.value

    @property
    def surface_weighted_fraction(self) -> float | None:
        return self.surface_weighted_fraction_components.value

    @property
    def projected_over_cap_surface(self) -> float | None:
        return self.projected_over_cap_surface_components.value


@dataclass(frozen=True)
class MetricHomogeneitySummary:
    """Denominator-weighted summary of one partition and one metric."""

    metric: str
    valid_count: int
    total_denominator: float
    reconstructed_coverage: float | None
    weighted_mean: float | None
    weighted_sd_pp: float | None
    weighted_median: float | None
    weighted_mad_pp: float | None
    minimum: float | None
    maximum: float | None
    range_pp: float | None
    slope_pp_per_R: float | None = None

    # Existing diagnostic code uses these concise names.
    @property
    def sd_pp(self) -> float | None:
        return self.weighted_sd_pp

    @property
    def mad_pp(self) -> float | None:
        return self.weighted_mad_pp


@dataclass(frozen=True)
class PolarRotationSummary:
    """Sector details and metric summaries for one grid rotation."""

    rotation_index: int
    rotation_offset_deg: float
    sectors: tuple[SegmentCoverage, ...]
    summaries_by_metric: dict[str, MetricHomogeneitySummary]


@dataclass(frozen=True)
class PolarRotationAggregate:
    """Robust distribution of polar-sector statistics over grid rotations."""

    metric: str
    sd_median_pp: float | None
    sd_q10_pp: float | None
    sd_q90_pp: float | None
    sd_iqr_pp: float | None
    sd_range_pp: float | None
    mad_median_pp: float | None
    mad_q10_pp: float | None
    mad_q90_pp: float | None
    mad_iqr_pp: float | None
    partition_delta_max_pp: float | None


@dataclass(frozen=True)
class CoverageHomogeneityResult:
    """Complete reusable homogeneity result for one accepted coverage ROI."""

    rings: tuple[SegmentCoverage, ...]
    sectors: tuple[SegmentCoverage, ...]
    radial_summaries_by_metric: dict[str, MetricHomogeneitySummary]
    polar_summaries_by_metric: dict[str, MetricHomogeneitySummary]
    direct_domain_components: dict[str, CoverageMetricComponents]
    radial_reconstructed_components: dict[str, CoverageMetricComponents]
    polar_reconstructed_components: dict[str, CoverageMetricComponents]
    radial_partition_delta_pp: dict[str, float | None]
    polar_partition_delta_pp: dict[str, float | None]
    radial_polar_partition_delta_pp: dict[str, float | None]
    requested_radial_ring_width_fraction: float
    effective_radial_ring_width_fraction: float
    polar_rotation_summaries: tuple[PolarRotationSummary, ...]
    polar_rotation_aggregates: dict[str, PolarRotationAggregate]
    r_over_R: np.ndarray
    phi_rad: np.ndarray
    display_rotation_deg: float
    selected_metric: str

    @property
    def radial_summary(self) -> MetricHomogeneitySummary:
        return self.radial_summaries_by_metric[self.selected_metric]

    @property
    def polar_summary(self) -> MetricHomogeneitySummary:
        return self.polar_summaries_by_metric[self.selected_metric]


def equal_width_intervals(
    inner: float, outer: float, requested_width: float
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Return an exact, gap-free equal-width partition of ``[inner, outer]``.

    The requested width selects a rounded number of intervals; the stored
    effective width then exactly covers the configured domain.
    """

    return equal_width_radial_intervals(inner, outer, requested_width)


def _components(numerator: float, denominator: float) -> CoverageMetricComponents:
    return CoverageMetricComponents(
        numerator=float(numerator),
        denominator=float(denominator),
        value=float(numerator / denominator) if denominator > 0 else None,
    )


def _make_segment(
    mask: np.ndarray,
    bead: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    *,
    index: int,
    inner: float,
    outer: float,
    surface_area_px2: float,
    min_completeness: float,
    theoretical_reference_count: int | None = None,
    start_angle_deg: float | None = None,
    end_angle_deg: float | None = None,
) -> SegmentCoverage:
    reference = mask & bead
    ag_reference = reference & ag
    bead_count = int(reference.sum())
    ag_count = int(ag_reference.sum())
    theoretical = int(mask.sum()) if theoretical_reference_count is None else theoretical_reference_count
    completeness = float(bead_count / theoretical) if theoretical else 0.0
    valid = bool(bead_count and completeness >= min_completeness)
    projected = _components(ag_count, bead_count)
    weighted = _components(float(weights[ag_reference].sum()), float(weights[reference].sum()))
    over_surface = _components(ag_count, surface_area_px2)
    return SegmentCoverage(
        index=index,
        inner=float(inner),
        outer=float(outer),
        center=float((inner + outer) / 2.0),
        bead_pixel_count=bead_count,
        ag_pixel_count=ag_count,
        completeness=completeness,
        valid=valid,
        projected_fraction_components=projected,
        surface_weighted_fraction_components=weighted,
        projected_over_cap_surface_components=over_surface,
        surface_area_px2=float(surface_area_px2),
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
    )


def summarize_segments(
    items: Iterable[SegmentCoverage], metric: str, *, slope: bool = False
) -> MetricHomogeneitySummary:
    """Calculate denominator-weighted union coverage and profile variability."""

    metric = normalize_cap_coverage_metric(metric)
    valid = [item for item in items if item.valid and item.components(metric).value is not None]
    if not valid:
        return MetricHomogeneitySummary(metric, 0, 0.0, None, None, None, None, None, None, None, None)
    values = np.asarray([item.components(metric).value for item in valid], dtype=float)
    denominators = np.asarray([item.components(metric).denominator for item in valid], dtype=float)
    numerators = np.asarray([item.components(metric).numerator for item in valid], dtype=float)
    total = float(denominators.sum())
    mean = float(numerators.sum() / total) if total > 0 else None
    if mean is None:
        return MetricHomogeneitySummary(metric, len(valid), total, None, None, None, None, None, None, None, None)
    variance = float(np.sum(denominators * (values - mean) ** 2) / total)
    median = weighted_median(values, denominators)
    mad = weighted_median(np.abs(values - median), denominators)
    slope_value: float | None = None
    if slope and len(valid) > 1:
        x = np.asarray([item.center for item in valid], dtype=float)
        x_bar = float(np.average(x, weights=denominators))
        denom = float(np.sum(denominators * (x - x_bar) ** 2))
        if denom > 0:
            slope_value = float(np.sum(denominators * (x - x_bar) * (values * 100.0 - mean * 100.0)) / denom)
    return MetricHomogeneitySummary(
        metric, len(valid), total, mean, mean, math.sqrt(max(variance, 0.0)) * 100.0,
        median, mad * 100.0, float(values.min()), float(values.max()),
        float((values.max() - values.min()) * 100.0), slope_value,
    )


def _reconstruct(items: Iterable[SegmentCoverage], metric: str) -> CoverageMetricComponents:
    components = [item.components(metric) for item in items if item.components(metric).denominator > 0]
    return _components(sum(value.numerator for value in components), sum(value.denominator for value in components))


def _sector_segments(
    domain: np.ndarray,
    phi: np.ndarray,
    bead: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    *,
    inner: float,
    outer: float,
    sectors: int,
    radius_px: float,
    min_completeness: float,
    offset_rad: float,
) -> tuple[SegmentCoverage, ...]:
    delta = 2.0 * math.pi / sectors
    zone = radius_px**2 * (math.sqrt(max(0.0, 1.0 - inner**2)) - math.sqrt(max(0.0, 1.0 - outer**2)))
    shifted = np.mod(phi - offset_rad, 2.0 * math.pi)
    output: list[SegmentCoverage] = []
    for index in range(sectors):
        start = index * delta
        end = (index + 1) * delta
        sector_mask = domain & (shifted >= start) & ((shifted < end) if index < sectors - 1 else (shifted <= end))
        output.append(_make_segment(
            sector_mask, bead, ag, weights, index=index, inner=inner, outer=outer,
            surface_area_px2=delta * zone, min_completeness=min_completeness,
            start_angle_deg=float(math.degrees((start + offset_rad) % (2 * math.pi))),
            end_angle_deg=float(math.degrees((end + offset_rad) % (2 * math.pi))),
        ))
    return tuple(output)


def _rotation_aggregate(
    metric: str, rotations: tuple[PolarRotationSummary, ...], direct: CoverageMetricComponents) -> PolarRotationAggregate:
    summaries = [item.summaries_by_metric[metric] for item in rotations]
    sd = np.asarray([item.weighted_sd_pp for item in summaries if item.weighted_sd_pp is not None], dtype=float)
    mad = np.asarray([item.weighted_mad_pp for item in summaries if item.weighted_mad_pp is not None], dtype=float)
    deltas = [
        abs(item.reconstructed_coverage - direct.value) * 100.0
        for item in summaries
        if item.reconstructed_coverage is not None and direct.value is not None
    ]
    def quant(values: np.ndarray, q: float) -> float | None:
        return float(np.quantile(values, q)) if values.size else None
    return PolarRotationAggregate(
        metric=metric, sd_median_pp=quant(sd, .5), sd_q10_pp=quant(sd, .1), sd_q90_pp=quant(sd, .9),
        sd_iqr_pp=(quant(sd, .75) - quant(sd, .25)) if sd.size else None,
        sd_range_pp=(float(sd.max() - sd.min()) if sd.size else None),
        mad_median_pp=quant(mad, .5), mad_q10_pp=quant(mad, .1), mad_q90_pp=quant(mad, .9),
        mad_iqr_pp=(quant(mad, .75) - quant(mad, .25)) if mad.size else None,
        partition_delta_max_pp=max(deltas) if deltas else None,
    )


def compute_homogeneity(
    bead: np.ndarray,
    ag: np.ndarray,
    center_rc: tuple[float, float],
    radius_px: float,
    *,
    inner: float,
    outer: float,
    width: float,
    sectors: int,
    min_completeness: float,
    metric: str,
    polar_rotation_samples: int = 1,
    polar_display_rotation_deg: float = 0.0,
) -> CoverageHomogeneityResult:
    """Compute exact radial and polar partitions from accepted masks.

    Both partitions cover the same complete radial domain.  Segment validity is
    used only for variability summaries; reconstruction uses all non-empty
    segments so it is a direct implementation quality-control check.
    """

    if not (0 <= inner < outer <= 1 and 0 < width <= outer - inner and sectors >= 2 and 0 <= min_completeness <= 1):
        raise ValueError("Invalid homogeneity geometry.")
    if polar_rotation_samples < 1 or not math.isfinite(polar_display_rotation_deg):
        raise ValueError("Polar rotation settings are invalid.")
    selected = normalize_cap_coverage_metric(metric)
    bead = np.asarray(bead, dtype=bool)
    ag = np.asarray(ag, dtype=bool)
    if bead.ndim != 2 or ag.shape != bead.shape or not bead.any():
        raise ValueError("Matching non-empty bead and Ag masks are required.")
    r, phi = coordinate_maps(bead.shape, center_rc, radius_px)
    radial_sq = (r * radius_px) ** 2
    weights = radius_px / np.sqrt(np.maximum(radius_px**2 - radial_sq, 1e-12))
    domain = (r >= inner) & (r <= outer)
    domain_surface = float(2 * math.pi * radius_px**2 * (math.sqrt(1 - inner**2) - math.sqrt(1 - outer**2)))
    direct_segment = _make_segment(domain, bead, ag, weights, index=0, inner=inner, outer=outer,
                                   surface_area_px2=domain_surface, min_completeness=0.0)
    direct = {name: direct_segment.components(name) for name in CAP_COVERAGE_METRICS}

    intervals, effective_width = equal_width_intervals(inner, outer, width)
    rings: list[SegmentCoverage] = []
    for index, (low, high) in enumerate(intervals):
        ring = (r >= low) & ((r < high) if index < len(intervals) - 1 else (r <= high))
        surface = 2 * math.pi * radius_px**2 * (math.sqrt(max(0.0, 1 - low**2)) - math.sqrt(max(0.0, 1 - high**2)))
        rings.append(_make_segment(ring, bead, ag, weights, index=index, inner=low, outer=high,
                                   surface_area_px2=surface, min_completeness=min_completeness))
    ring_tuple = tuple(rings)
    radial_summaries = {name: summarize_segments(ring_tuple, name, slope=True) for name in CAP_COVERAGE_METRICS}
    radial_reconstructed = {name: _reconstruct(ring_tuple, name) for name in CAP_COVERAGE_METRICS}

    display_offset = math.radians(polar_display_rotation_deg) % (2 * math.pi)
    display_sectors = _sector_segments(domain, phi, bead, ag, weights, inner=inner, outer=outer,
                                       sectors=sectors, radius_px=radius_px, min_completeness=min_completeness,
                                       offset_rad=display_offset)
    polar_summaries = {name: summarize_segments(display_sectors, name) for name in CAP_COVERAGE_METRICS}
    polar_reconstructed = {name: _reconstruct(display_sectors, name) for name in CAP_COVERAGE_METRICS}

    delta = 2 * math.pi / sectors
    rotations: list[PolarRotationSummary] = []
    for rotation_index in range(polar_rotation_samples):
        offset = rotation_index * delta / polar_rotation_samples
        sector_values = _sector_segments(domain, phi, bead, ag, weights, inner=inner, outer=outer,
                                         sectors=sectors, radius_px=radius_px, min_completeness=min_completeness,
                                         offset_rad=offset)
        rotations.append(PolarRotationSummary(
            rotation_index, float(math.degrees(offset)), sector_values,
            {name: summarize_segments(sector_values, name) for name in CAP_COVERAGE_METRICS},
        ))
    rotation_tuple = tuple(rotations)
    aggregates = {name: _rotation_aggregate(name, rotation_tuple, direct[name]) for name in CAP_COVERAGE_METRICS}
    def delta_pp(left: CoverageMetricComponents, right: CoverageMetricComponents) -> float | None:
        return abs(left.value - right.value) * 100.0 if left.value is not None and right.value is not None else None
    return CoverageHomogeneityResult(
        rings=ring_tuple, sectors=display_sectors, radial_summaries_by_metric=radial_summaries,
        polar_summaries_by_metric=polar_summaries, direct_domain_components=direct,
        radial_reconstructed_components=radial_reconstructed, polar_reconstructed_components=polar_reconstructed,
        radial_partition_delta_pp={name: delta_pp(radial_reconstructed[name], direct[name]) for name in CAP_COVERAGE_METRICS},
        polar_partition_delta_pp={name: delta_pp(polar_reconstructed[name], direct[name]) for name in CAP_COVERAGE_METRICS},
        radial_polar_partition_delta_pp={name: delta_pp(radial_reconstructed[name], polar_reconstructed[name]) for name in CAP_COVERAGE_METRICS},
        requested_radial_ring_width_fraction=float(width), effective_radial_ring_width_fraction=effective_width,
        polar_rotation_summaries=rotation_tuple, polar_rotation_aggregates=aggregates,
        r_over_R=r, phi_rad=phi, display_rotation_deg=float(polar_display_rotation_deg), selected_metric=selected,
    )

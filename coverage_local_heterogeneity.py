"""Shared mask-only local coverage-domain and joint-grid post-processing.

This module deliberately contains no segmentation or Matplotlib code.  It is
used by production summaries, batch export and the diagnostic viewer, ensuring
that display preferences never alter the numerical result.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from time import perf_counter
import tracemalloc

import numpy as np

from coverage_cap import CAP_COVERAGE_METRICS, normalize_cap_coverage_metric
from coverage_metric_utils import CoverageMetricComponents, coordinate_maps, weighted_median

LOGGER = logging.getLogger(__name__)


def _components(numerator: float, denominator: float) -> CoverageMetricComponents:
    return CoverageMetricComponents(
        float(numerator), float(denominator),
        float(numerator / denominator) if denominator > 0 else None,
    )


@dataclass(frozen=True)
class CoverageDomainResult:
    """Direct coverage over one complete annular homogeneity domain."""

    inner_radius_fraction: float
    outer_radius_fraction: float
    center_rc: tuple[float, float]
    sphere_radius_px: float
    completeness: float
    valid: bool
    projected_fraction: CoverageMetricComponents
    surface_weighted_fraction: CoverageMetricComponents
    projected_over_surface: CoverageMetricComponents

    def components(self, metric: str) -> CoverageMetricComponents:
        name = normalize_cap_coverage_metric(metric)
        if name == "projected_over_cap_surface":
            name = "projected_over_surface"
        return getattr(self, name)


@dataclass(frozen=True)
class LocalDirectRadialProfile:
    """Rotation-independent direct coverage for fixed annular bands."""

    values: np.ndarray
    denominators: np.ndarray
    valid: np.ndarray
    completeness: np.ndarray
    centers: np.ndarray
    weighted_sd_pp: float | None
    slope_pp_per_R: float | None


@dataclass(frozen=True)
class LocalGridWorkspace:
    """Invariant full-image geometry shared by display and robust rotations.

    The workspace is deliberately local to one calculation.  It is never put
    in a module-level cache and is not retained by compact robust rotations.
    """

    r_over_R: np.ndarray
    phi_rad: np.ndarray
    surface_weights: np.ndarray
    bead_mask: np.ndarray
    ag_mask: np.ndarray
    domain_mask: np.ndarray
    reference_mask: np.ndarray
    ag_reference_mask: np.ndarray
    radial_index_map: np.ndarray
    radial_edges: np.ndarray
    domain: CoverageDomainResult
    direct_radial_profiles: dict[str, LocalDirectRadialProfile]
    center_rc: tuple[float, float]
    sphere_radius_px: float
    inner_fraction: float
    outer_fraction: float
    radial_band_count: int
    min_segment_completeness: float


@dataclass(frozen=True)
class LocalGridCell:
    radial_index: int
    sector_index: int
    inner_fraction: float
    outer_fraction: float
    start_angle_deg: float
    end_angle_deg: float
    completeness: float
    theoretical_pixel_count: int
    reference_pixel_count: int
    ag_pixel_count: int
    valid: bool
    projected_fraction: CoverageMetricComponents
    surface_weighted_fraction: CoverageMetricComponents
    projected_over_surface: CoverageMetricComponents

    def components(self, metric: str) -> CoverageMetricComponents:
        name = normalize_cap_coverage_metric(metric)
        if name == "projected_over_cap_surface":
            name = "projected_over_surface"
        return getattr(self, name)


@dataclass(frozen=True)
class LocalMetricResult:
    """One metric's grid values, profiles and weighted patchiness summaries."""

    values: np.ndarray
    radial_profile: np.ndarray
    polar_profile: np.ndarray
    total_weighted_sd_pp: float | None
    total_weighted_mad_pp: float | None
    residual_weighted_sd_pp: float | None
    residual_weighted_mad_pp: float | None
    reconstructed: CoverageMetricComponents
    reconstruction_delta_pp: float | None
    radial_weighted_sd_pp: float | None = None
    radial_slope_pp_per_R: float | None = None
    polar_weighted_sd_pp: float | None = None
    polar_weighted_mad_pp: float | None = None
    polar_denominators: np.ndarray | None = None
    polar_valid: np.ndarray | None = None
    radial_denominators: np.ndarray | None = None
    radial_valid: np.ndarray | None = None
    radial_completeness: np.ndarray | None = None
    radial_centers: np.ndarray | None = None

    @property
    def scientifically_valid(self) -> bool:
        """Whether the standard joint-grid statistics are numerically usable."""

        return bool(
            self.reconstructed.denominator > 0
            and self.radial_weighted_sd_pp is not None
            and self.polar_weighted_sd_pp is not None
            and self.polar_weighted_mad_pp is not None
            and self.total_weighted_sd_pp is not None
            and self.total_weighted_mad_pp is not None
            and self.residual_weighted_sd_pp is not None
            and self.residual_weighted_mad_pp is not None
        )


@dataclass(frozen=True)
class LocalGridRotationResult:
    """One compact angular partition with no full-image invariant arrays."""

    cells: tuple[LocalGridCell, ...]
    valid: np.ndarray
    metrics: dict[str, LocalMetricResult]
    radial_edges: np.ndarray
    sector_edges_deg: np.ndarray
    metric: str
    display_rotation_deg: float

    @property
    def values(self) -> np.ndarray:
        return self.metrics[self.metric].values

    @property
    def radial_profile(self) -> np.ndarray:
        return self.metrics[self.metric].radial_profile

    @property
    def polar_profile(self) -> np.ndarray:
        return self.metrics[self.metric].polar_profile

    @property
    def total_weighted_sd_pp(self) -> float | None:
        return self.metrics[self.metric].total_weighted_sd_pp

    @property
    def total_weighted_mad_pp(self) -> float | None:
        return self.metrics[self.metric].total_weighted_mad_pp

    @property
    def residual_weighted_sd_pp(self) -> float | None:
        return self.metrics[self.metric].residual_weighted_sd_pp

    @property
    def residual_weighted_mad_pp(self) -> float | None:
        return self.metrics[self.metric].residual_weighted_mad_pp


@dataclass(frozen=True)
class LocalHeterogeneityResult:
    """All three local-grid metrics plus selected-metric compatibility views."""

    cells: tuple[LocalGridCell, ...]
    valid: np.ndarray
    metrics: dict[str, LocalMetricResult]
    domain: CoverageDomainResult
    radial_edges: np.ndarray
    sector_edges_deg: np.ndarray
    r_over_R: np.ndarray
    phi_rad: np.ndarray
    metric: str
    display_rotation_deg: float = 0.0
    polar_rotation_samples: int = 1
    rotation_results: tuple[LocalGridRotationResult, ...] = ()
    rotation_aggregates: dict[str, "LocalRotationAggregate"] | None = None

    @property
    def values(self) -> np.ndarray:
        return self.metrics[self.metric].values

    @property
    def radial_profile(self) -> np.ndarray:
        return self.metrics[self.metric].radial_profile

    @property
    def polar_profile(self) -> np.ndarray:
        return self.metrics[self.metric].polar_profile

    @property
    def total_weighted_sd_pp(self) -> float | None:
        return self.metrics[self.metric].total_weighted_sd_pp

    @property
    def total_weighted_mad_pp(self) -> float | None:
        return self.metrics[self.metric].total_weighted_mad_pp

    @property
    def residual_weighted_sd_pp(self) -> float | None:
        return self.metrics[self.metric].residual_weighted_sd_pp

    @property
    def residual_weighted_mad_pp(self) -> float | None:
        return self.metrics[self.metric].residual_weighted_mad_pp

    @property
    def metric_validity(self) -> dict[str, bool]:
        """Scientific usability by metric under the existing validity rules."""

        has_valid_cell = bool(np.any(self.valid))
        output: dict[str, bool] = {}
        for name, summary in self.metrics.items():
            valid = bool(self.domain.valid and has_valid_cell and summary.scientifically_valid)
            if self.rotation_results:
                aggregate = (self.rotation_aggregates or {}).get(name)
                valid = bool(
                    valid
                    and aggregate is not None
                    and aggregate.polar_sd_median_pp is not None
                    and aggregate.total_local_sd_median_pp is not None
                    and aggregate.residual_sd_median_pp is not None
                )
            output[name] = valid
        return output

    @property
    def scientifically_valid(self) -> bool:
        """Conservative overall validity across all three coverage metrics."""

        validity = self.metric_validity
        return bool(validity and all(validity.values()))


@dataclass(frozen=True)
class LocalRotationAggregate:
    """Orientation-robust local-grid summaries; spreads are methodological."""
    polar_sd_median_pp: float | None
    polar_sd_q10_pp: float | None
    polar_sd_q90_pp: float | None
    polar_sd_iqr_pp: float | None
    polar_sd_range_pp: float | None
    polar_mad_median_pp: float | None
    polar_mad_iqr_pp: float | None
    total_local_sd_median_pp: float | None
    total_local_sd_q10_pp: float | None
    total_local_sd_q90_pp: float | None
    total_local_sd_iqr_pp: float | None
    total_local_mad_median_pp: float | None
    total_local_mad_iqr_pp: float | None
    residual_sd_median_pp: float | None
    residual_sd_q10_pp: float | None
    residual_sd_q90_pp: float | None
    residual_sd_iqr_pp: float | None
    residual_mad_median_pp: float | None
    residual_mad_iqr_pp: float | None
    rotation_reconstruction_delta_max_pp: float | None



def _weighted_profile(values: np.ndarray, denominators: np.ndarray, axis: int) -> np.ndarray:
    numerator = np.nansum(values * denominators, axis=axis)
    denominator = np.nansum(np.where(np.isfinite(values), denominators, 0.0), axis=axis)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)


def _metric_result(values: np.ndarray, valid: np.ndarray, denominators: np.ndarray, components: list[CoverageMetricComponents], domain: CoverageMetricComponents, radial_centers: np.ndarray | None = None) -> LocalMetricResult:
    radial_profile = _weighted_profile(values, denominators, axis=1)
    polar_profile = _weighted_profile(values, denominators, axis=0)
    usable = valid & np.isfinite(values) & (denominators > 0)
    total_sd = total_mad = residual_sd = residual_mad = None
    if usable.any():
        data, weight = values[usable], denominators[usable]
        mean = float(np.sum(data * weight) / weight.sum())
        total_sd = float(np.sqrt(np.sum(weight * (data - mean) ** 2) / weight.sum()) * 100.0)
        median = weighted_median(data, weight)
        total_mad = float(weighted_median(np.abs(data - median), weight) * 100.0)
        # Residual patchiness is what is left after additive radial/polar
        # structure; it is not assumed to be pure measurement noise.
        residual = values - radial_profile[:, None] - polar_profile[None, :] + mean
        residual_data = residual[usable]
        residual_mean = float(np.sum(residual_data * weight) / weight.sum())
        residual_sd = float(np.sqrt(np.sum(weight * (residual_data - residual_mean) ** 2) / weight.sum()) * 100.0)
        residual_median = weighted_median(residual_data, weight)
        residual_mad = float(weighted_median(np.abs(residual_data - residual_median), weight) * 100.0)
    non_empty = [item for item in components if item.denominator > 0]
    reconstructed = _components(sum(item.numerator for item in non_empty), sum(item.denominator for item in non_empty))
    delta = (
        abs(reconstructed.value - domain.value) * 100.0
        if reconstructed.value is not None and domain.value is not None else None
    )
    radial_sd = radial_slope = polar_sd = polar_mad = None
    radial_weights = np.nansum(np.where(np.isfinite(values), denominators, 0.0), axis=1)
    radial_ok = np.isfinite(radial_profile) & (radial_weights > 0)
    if radial_ok.any():
        rv, rw = radial_profile[radial_ok], radial_weights[radial_ok]
        rmean = float(np.sum(rv * rw) / rw.sum())
        radial_sd = float(np.sqrt(np.sum(rw * (rv - rmean) ** 2) / rw.sum()) * 100.0)
        x = (radial_centers if radial_centers is not None else np.arange(radial_profile.size, dtype=float))[radial_ok]
        if x.size >= 2 and not np.allclose(x, x[0]):
            xmean = float(np.sum(x * rw) / rw.sum())
            denom = float(np.sum(rw * (x - xmean) ** 2))
            if denom > 0:
                radial_slope = float(np.sum(rw * (x - xmean) * ((rv * 100.0) - rmean * 100.0)) / denom)
    polar_weights = np.nansum(np.where(np.isfinite(values), denominators, 0.0), axis=0)
    polar_ok = np.isfinite(polar_profile) & (polar_weights > 0)
    if polar_ok.any():
        pv, pw = polar_profile[polar_ok], polar_weights[polar_ok]
        pmean = float(np.sum(pv * pw) / pw.sum())
        polar_sd = float(np.sqrt(np.sum(pw * (pv - pmean) ** 2) / pw.sum()) * 100.0)
        pmedian = weighted_median(pv, pw)
        polar_mad = float(weighted_median(np.abs(pv - pmedian), pw) * 100.0)
    return LocalMetricResult(
        values,
        radial_profile,
        polar_profile,
        total_sd,
        total_mad,
        residual_sd,
        residual_mad,
        reconstructed,
        delta,
        radial_sd,
        radial_slope,
        polar_sd,
        polar_mad,
        polar_weights,
        polar_ok,
        radial_weights,
        radial_ok,
    )


def local_grid_indices(result: LocalHeterogeneityResult, row: int, col: int) -> tuple[int, int] | None:
    """Return the shared half-open local-grid cell index for one image pixel."""
    if not (0 <= row < result.r_over_R.shape[0] and 0 <= col < result.r_over_R.shape[1]):
        return None
    radius = float(result.r_over_R[row, col])
    if not (result.radial_edges[0] <= radius <= result.radial_edges[-1]):
        return None
    radial = min(int(np.searchsorted(result.radial_edges, radius, side="right") - 1), len(result.radial_edges) - 2)
    angle = float((np.degrees(result.phi_rad[row, col]) - result.display_rotation_deg) % 360.0)
    sector = min(int(np.searchsorted(result.sector_edges_deg, angle, side="right") - 1), len(result.sector_edges_deg) - 2)
    return radial, sector


def local_grid_index_maps(result: LocalHeterogeneityResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized grid indices and domain membership for image overlays."""
    radial = np.minimum(np.searchsorted(result.radial_edges, result.r_over_R, side="right") - 1, len(result.radial_edges) - 2)
    angle = (np.degrees(result.phi_rad) - result.display_rotation_deg) % 360.0
    sector = np.minimum(np.searchsorted(result.sector_edges_deg, angle, side="right") - 1, len(result.sector_edges_deg) - 2)
    domain = (result.r_over_R >= result.radial_edges[0]) & (result.r_over_R <= result.radial_edges[-1]) & (radial >= 0) & (sector >= 0)
    return radial.astype(int), sector.astype(int), domain


def _debug_memory_values() -> tuple[int, int] | None:
    return tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else None


def _direct_radial_profile(
    numerator: np.ndarray,
    denominator: np.ndarray,
    band_valid: np.ndarray,
    band_completeness: np.ndarray,
    radial_centers: np.ndarray,
) -> LocalDirectRadialProfile:
    profile = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.size, np.nan),
        where=denominator > 0,
    )
    use = band_valid & np.isfinite(profile) & (denominator > 0)
    weighted_sd = slope = None
    if use.any():
        values_1d = profile[use]
        weights_1d = denominator[use]
        mean = float(np.sum(values_1d * weights_1d) / weights_1d.sum())
        weighted_sd = float(
            np.sqrt(np.sum(weights_1d * (values_1d - mean) ** 2) / weights_1d.sum()) * 100.0
        )
        x = radial_centers[use]
        if x.size >= 2:
            x_mean = float(np.sum(x * weights_1d) / weights_1d.sum())
            fit_denominator = float(np.sum(weights_1d * (x - x_mean) ** 2))
            if fit_denominator > 0:
                slope = float(
                    np.sum(
                        weights_1d
                        * (x - x_mean)
                        * ((values_1d * 100.0) - mean * 100.0)
                    )
                    / fit_denominator
                )
    return LocalDirectRadialProfile(
        profile,
        denominator,
        use,
        band_completeness,
        radial_centers,
        weighted_sd,
        slope,
    )


def build_local_grid_workspace(
    bead_mask: np.ndarray,
    ag_mask: np.ndarray,
    center_rc: tuple[float, float],
    sphere_radius_px: float,
    *,
    inner_fraction: float,
    outer_fraction: float,
    radial_band_count: int,
    min_segment_completeness: float,
) -> LocalGridWorkspace:
    """Build all rotation-invariant geometry and direct radial results once."""

    started = perf_counter()
    if not (0 <= inner_fraction < outer_fraction <= 1):
        raise ValueError("Local heterogeneity radius fractions are invalid.")
    if radial_band_count < 1:
        raise ValueError("Local heterogeneity radial band count is invalid.")
    if not 0 <= min_segment_completeness <= 1:
        raise ValueError("Local heterogeneity completeness is invalid.")
    bead = np.asarray(bead_mask, dtype=bool)
    ag = np.asarray(ag_mask, dtype=bool)
    if bead.ndim != 2 or bead.shape != ag.shape or not bead.any():
        raise ValueError("Matching non-empty bead and Ag masks are required.")
    r, phi = coordinate_maps(bead.shape, center_rc, sphere_radius_px)
    radial_edges = np.linspace(inner_fraction, outer_fraction, radial_band_count + 1)
    domain_mask = (r >= inner_fraction) & (r <= outer_fraction)
    radial_sq = (r * sphere_radius_px) ** 2
    weights = sphere_radius_px / np.sqrt(np.maximum(sphere_radius_px**2 - radial_sq, 1e-12))
    zone_surface = 2.0 * math.pi * sphere_radius_px**2 * (
        math.sqrt(1.0 - inner_fraction**2) - math.sqrt(1.0 - outer_fraction**2)
    )
    reference = domain_mask & bead
    ag_reference = reference & ag
    completeness = float(reference.sum() / domain_mask.sum()) if domain_mask.any() else 0.0
    domain = CoverageDomainResult(
        float(inner_fraction), float(outer_fraction), (float(center_rc[0]), float(center_rc[1])), float(sphere_radius_px), completeness,
        bool(reference.any() and completeness >= min_segment_completeness),
        _components(float(ag_reference.sum()), float(reference.sum())),
        _components(float(weights[ag_reference].sum()), float(weights[reference].sum())),
        _components(float(ag_reference.sum()), zone_surface),
    )
    radial_index = np.searchsorted(radial_edges, r, side="right") - 1
    radial_index = np.clip(radial_index, 0, radial_band_count - 1)
    radial_domain_ids = radial_index[domain_mask]
    radial_ref_ids = radial_index[reference]
    radial_ag_ids = radial_index[ag_reference]
    band_theoretical = np.bincount(radial_domain_ids, minlength=radial_band_count).astype(float)
    band_ref = np.bincount(radial_ref_ids, minlength=radial_band_count).astype(float)
    band_ag = np.bincount(radial_ag_ids, minlength=radial_band_count).astype(float)
    band_wref = np.bincount(
        radial_ref_ids,
        weights=weights[reference],
        minlength=radial_band_count,
    )
    band_wag = np.bincount(
        radial_ag_ids,
        weights=weights[ag_reference],
        minlength=radial_band_count,
    )
    band_surface = 2.0 * math.pi * sphere_radius_px**2 * (
        np.sqrt(1.0 - radial_edges[:-1] ** 2)
        - np.sqrt(1.0 - radial_edges[1:] ** 2)
    )
    band_completeness = np.divide(
        band_ref,
        band_theoretical,
        out=np.zeros_like(band_ref),
        where=band_theoretical > 0,
    )
    band_valid = (band_ref > 0) & (band_completeness >= min_segment_completeness)
    radial_midpoints = (radial_edges[:-1] + radial_edges[1:]) / 2.0

    def weighted_radial_centers(
        pixel_mask: np.ndarray,
        pixel_weights: np.ndarray,
    ) -> np.ndarray:
        """Return metric-denominator-weighted mean ``r/R`` per annulus."""

        ids = radial_index[pixel_mask]
        weights_1d = np.asarray(pixel_weights, dtype=float)
        denominator = np.bincount(
            ids,
            weights=weights_1d,
            minlength=radial_band_count,
        )
        numerator = np.bincount(
            ids,
            weights=weights_1d * r[pixel_mask],
            minlength=radial_band_count,
        )
        return np.divide(
            numerator,
            denominator,
            out=radial_midpoints.copy(),
            where=denominator > 0,
        )

    projected_centers = weighted_radial_centers(
        reference,
        np.ones(int(reference.sum()), dtype=float),
    )
    surface_weighted_centers = weighted_radial_centers(
        reference,
        weights[reference],
    )
    projected_over_surface_centers = weighted_radial_centers(
        domain_mask,
        weights[domain_mask],
    )
    direct_radial = {
        "projected_fraction": _direct_radial_profile(
            band_ag, band_ref, band_valid, band_completeness, projected_centers
        ),
        "surface_weighted_fraction": _direct_radial_profile(
            band_wag,
            band_wref,
            band_valid,
            band_completeness,
            surface_weighted_centers,
        ),
        "projected_over_cap_surface": _direct_radial_profile(
            band_ag,
            band_surface,
            band_valid,
            band_completeness,
            projected_over_surface_centers,
        ),
    }
    workspace = LocalGridWorkspace(
        r,
        phi,
        weights,
        bead,
        ag,
        domain_mask,
        reference,
        ag_reference,
        radial_index,
        radial_edges,
        domain,
        direct_radial,
        (float(center_rc[0]), float(center_rc[1])),
        float(sphere_radius_px),
        float(inner_fraction),
        float(outer_fraction),
        int(radial_band_count),
        float(min_segment_completeness),
    )
    memory = _debug_memory_values()
    if memory is None:
        LOGGER.debug("Local grid workspace construction: %.3f s", perf_counter() - started)
    else:
        LOGGER.debug(
            "Local grid workspace construction: %.3f s; traced current=%d bytes, peak=%d bytes",
            perf_counter() - started,
            memory[0],
            memory[1],
        )
    return workspace


def compute_local_grid_from_workspace(
    workspace: LocalGridWorkspace,
    *,
    polar_sector_count: int,
    metric: str,
    rotation_offset_deg: float = 0.0,
) -> LocalGridRotationResult:
    """Compute only the angularly sensitive portion of one joint grid."""

    if polar_sector_count < 2:
        raise ValueError("Local heterogeneity polar sector count is invalid.")
    metric = normalize_cap_coverage_metric(metric)
    if not math.isfinite(rotation_offset_deg):
        raise ValueError("Local heterogeneity display rotation is invalid.")
    rotation_offset_deg = float(rotation_offset_deg % 360.0)
    radial_band_count = workspace.radial_band_count
    sector_edges = np.linspace(0.0, 2.0 * math.pi, polar_sector_count + 1)
    shifted_phi = np.mod(
        workspace.phi_rad - math.radians(rotation_offset_deg),
        2.0 * math.pi,
    )
    cell_count = radial_band_count * polar_sector_count
    sector_index = np.floor(shifted_phi / (2.0 * math.pi / polar_sector_count)).astype(int)
    sector_index = np.clip(sector_index, 0, polar_sector_count - 1)
    cell_ids = workspace.radial_index_map * polar_sector_count + sector_index
    domain_ids = cell_ids[workspace.domain_mask]
    theoretical_counts = np.bincount(domain_ids, minlength=cell_count).astype(float)
    reference_ids = cell_ids[workspace.reference_mask]
    ag_ids = cell_ids[workspace.ag_reference_mask]
    reference_counts = np.bincount(reference_ids, minlength=cell_count).astype(float)
    ag_counts = np.bincount(ag_ids, minlength=cell_count).astype(float)
    weighted_reference = np.bincount(
        reference_ids,
        weights=workspace.surface_weights[workspace.reference_mask],
        minlength=cell_count,
    )
    weighted_ag = np.bincount(
        ag_ids,
        weights=workspace.surface_weights[workspace.ag_reference_mask],
        minlength=cell_count,
    )
    delta_phi = 2.0 * math.pi / polar_sector_count
    zone_factors = np.asarray([
        delta_phi * workspace.sphere_radius_px**2 * (
            math.sqrt(1.0 - low**2) - math.sqrt(1.0 - high**2)
        )
        for low, high in zip(workspace.radial_edges[:-1], workspace.radial_edges[1:])
    ])
    surface_denominators = np.repeat(zone_factors, polar_sector_count)
    values = {name: np.full((radial_band_count, polar_sector_count), np.nan, dtype=float) for name in CAP_COVERAGE_METRICS}
    denominators = {name: np.zeros((radial_band_count, polar_sector_count), dtype=float) for name in CAP_COVERAGE_METRICS}
    valid = (
        (reference_counts > 0)
        & (
            np.divide(
                reference_counts,
                theoretical_counts,
                out=np.zeros_like(reference_counts),
                where=theoretical_counts > 0,
            )
            >= workspace.min_segment_completeness
        )
    ).reshape(radial_band_count, polar_sector_count)
    components_by_metric = {name: [] for name in CAP_COVERAGE_METRICS}
    cells: list[LocalGridCell] = []
    for cell_id in range(cell_count):
        radial_i, sector_i = divmod(cell_id, polar_sector_count)
        low, high = workspace.radial_edges[radial_i], workspace.radial_edges[radial_i + 1]
        start, end = sector_edges[sector_i], sector_edges[sector_i + 1]
        count, ag_count = reference_counts[cell_id], ag_counts[cell_id]
        completeness_cell = float(count / theoretical_counts[cell_id]) if theoretical_counts[cell_id] else 0.0
        projected = _components(ag_count, count)
        weighted = _components(weighted_ag[cell_id], weighted_reference[cell_id])
        over_surface = _components(ag_count, surface_denominators[cell_id])
        cell_valid = bool(valid[radial_i, sector_i])
        start_deg = math.degrees(start) + rotation_offset_deg
        end_deg = math.degrees(end) + rotation_offset_deg
        cell = LocalGridCell(
            radial_i,
            sector_i,
            float(low),
            float(high),
            float(start_deg),
            float(end_deg),
            completeness_cell,
            int(theoretical_counts[cell_id]),
            int(count),
            int(ag_count),
            cell_valid,
            projected,
            weighted,
            over_surface,
        )
        cells.append(cell)
        for name, component in (("projected_fraction", projected), ("surface_weighted_fraction", weighted), ("projected_over_cap_surface", over_surface)):
            components_by_metric[name].append(component)
            if cell_valid and component.value is not None:
                values[name][radial_i, sector_i] = component.value
                denominators[name][radial_i, sector_i] = component.denominator
    metric_results = {
        name: _metric_result(
            values[name],
            valid,
            denominators[name],
            components_by_metric[name],
            workspace.domain.components(name),
            workspace.direct_radial_profiles[name].centers,
        )
        for name in CAP_COVERAGE_METRICS
    }
    for name, direct in workspace.direct_radial_profiles.items():
        summary = metric_results[name]
        metric_results[name] = replace(
            summary,
            radial_profile=direct.values,
            radial_weighted_sd_pp=direct.weighted_sd_pp,
            radial_slope_pp_per_R=direct.slope_pp_per_R,
            radial_denominators=direct.denominators,
            radial_valid=direct.valid,
            radial_completeness=direct.completeness,
            radial_centers=direct.centers,
        )
    return LocalGridRotationResult(
        tuple(cells),
        valid,
        metric_results,
        workspace.radial_edges,
        np.degrees(sector_edges),
        metric,
        rotation_offset_deg,
    )


def compute_display_grid_from_workspace(
    workspace: LocalGridWorkspace,
    *,
    polar_sector_count: int,
    metric: str,
    rotation_offset_deg: float = 0.0,
) -> LocalHeterogeneityResult:
    """Build the display result while retaining only its required maps."""

    started = perf_counter()
    rotation = compute_local_grid_from_workspace(
        workspace,
        polar_sector_count=polar_sector_count,
        metric=metric,
        rotation_offset_deg=rotation_offset_deg,
    )
    result = LocalHeterogeneityResult(
        rotation.cells,
        rotation.valid,
        rotation.metrics,
        workspace.domain,
        rotation.radial_edges,
        rotation.sector_edges_deg,
        workspace.r_over_R,
        workspace.phi_rad,
        rotation.metric,
        rotation.display_rotation_deg,
    )
    LOGGER.debug(
        "Local display-grid calculation: %.3f s (%d cells)",
        perf_counter() - started,
        len(rotation.cells),
    )
    return result


def _compute_single_local_grid(
    bead_mask: np.ndarray, ag_mask: np.ndarray, center_rc: tuple[float, float], sphere_radius_px: float,
    *, inner_fraction: float, outer_fraction: float, radial_band_count: int,
    polar_sector_count: int, min_segment_completeness: float, metric: str,
    polar_rotation_samples: int = 1, display_rotation_deg: float = 0.0,
) -> LocalHeterogeneityResult:
    """Backward-compatible one-display-grid wrapper around one workspace."""

    if polar_rotation_samples < 1:
        raise ValueError("Local heterogeneity polar_rotation_samples must be at least 1.")
    workspace = build_local_grid_workspace(
        bead_mask,
        ag_mask,
        center_rc,
        sphere_radius_px,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
        radial_band_count=radial_band_count,
        min_segment_completeness=min_segment_completeness,
    )
    display = compute_display_grid_from_workspace(
        workspace,
        polar_sector_count=polar_sector_count,
        metric=metric,
        rotation_offset_deg=display_rotation_deg,
    )
    return replace(display, polar_rotation_samples=polar_rotation_samples)


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def compute_local_grid_at_rotation(
    bead_mask: np.ndarray, ag_mask: np.ndarray, center_rc: tuple[float, float], sphere_radius_px: float,
    *, inner_fraction: float, outer_fraction: float, radial_band_count: int,
    polar_sector_count: int, min_segment_completeness: float, metric: str,
    rotation_offset_deg: float = 0.0,
) -> LocalHeterogeneityResult:
    """Compute one display-ready local grid at a specified angular offset."""
    return _compute_single_local_grid(
        bead_mask, ag_mask, center_rc, sphere_radius_px, inner_fraction=inner_fraction,
        outer_fraction=outer_fraction, radial_band_count=radial_band_count,
        polar_sector_count=polar_sector_count, min_segment_completeness=min_segment_completeness,
        metric=metric, polar_rotation_samples=1, display_rotation_deg=rotation_offset_deg,
    )


def compute_local_rotation_series(
    bead_mask: np.ndarray, ag_mask: np.ndarray, center_rc: tuple[float, float], sphere_radius_px: float,
    *, inner_fraction: float, outer_fraction: float, radial_band_count: int,
    polar_sector_count: int, min_segment_completeness: float, metric: str,
    polar_rotation_samples: int,
) -> tuple[LocalGridRotationResult, ...]:
    """Compute the orientation-robust rotations across exactly one sector width."""
    workspace = build_local_grid_workspace(
        bead_mask,
        ag_mask,
        center_rc,
        sphere_radius_px,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
        radial_band_count=radial_band_count,
        min_segment_completeness=min_segment_completeness,
    )
    return compute_local_rotation_series_from_workspace(
        workspace,
        polar_sector_count=polar_sector_count,
        metric=metric,
        polar_rotation_samples=polar_rotation_samples,
    )


def compute_local_rotation_series_from_workspace(
    workspace: LocalGridWorkspace,
    *,
    polar_sector_count: int,
    metric: str,
    polar_rotation_samples: int,
) -> tuple[LocalGridRotationResult, ...]:
    """Incrementally compute and retain only compact robust rotations."""

    started = perf_counter()
    if polar_rotation_samples < 1:
        raise ValueError("Local heterogeneity polar_rotation_samples must be at least 1.")
    if polar_sector_count < 2:
        raise ValueError("Local heterogeneity polar sector count is invalid.")
    delta = 360.0 / polar_sector_count
    compact_rotations: list[LocalGridRotationResult] = []
    for index in range(polar_rotation_samples):
        compact_rotations.append(
            compute_local_grid_from_workspace(
                workspace,
                polar_sector_count=polar_sector_count,
                metric=metric,
                rotation_offset_deg=index * delta / polar_rotation_samples,
            )
        )
    result = tuple(compact_rotations)
    memory = _debug_memory_values()
    if memory is None:
        LOGGER.debug(
            "Local robust rotations: %.3f s (%d compact rotations)",
            perf_counter() - started,
            polar_rotation_samples,
        )
    else:
        LOGGER.debug(
            "Local robust rotations: %.3f s (%d compact rotations); traced current=%d bytes, peak=%d bytes",
            perf_counter() - started,
            polar_rotation_samples,
            memory[0],
            memory[1],
        )
    return result


def aggregate_local_rotation_results(rotations: tuple[LocalGridRotationResult, ...]) -> dict[str, LocalRotationAggregate]:
    """Aggregate rotation sensitivity; rotation spread is methodological only."""
    aggregates: dict[str, LocalRotationAggregate] = {}
    for name in CAP_COVERAGE_METRICS:
        polar_sd = [
            item.metrics[name].polar_weighted_sd_pp
            for item in rotations
            if item.metrics[name].polar_weighted_sd_pp is not None
        ]
        polar_mad = [
            item.metrics[name].polar_weighted_mad_pp
            for item in rotations
            if item.metrics[name].polar_weighted_mad_pp is not None
        ]
        total_sd = [item.metrics[name].total_weighted_sd_pp for item in rotations if item.metrics[name].total_weighted_sd_pp is not None]
        total_mad = [item.metrics[name].total_weighted_mad_pp for item in rotations if item.metrics[name].total_weighted_mad_pp is not None]
        residual_sd = [item.metrics[name].residual_weighted_sd_pp for item in rotations if item.metrics[name].residual_weighted_sd_pp is not None]
        residual_mad = [item.metrics[name].residual_weighted_mad_pp for item in rotations if item.metrics[name].residual_weighted_mad_pp is not None]
        deltas = [item.metrics[name].reconstruction_delta_pp for item in rotations if item.metrics[name].reconstruction_delta_pp is not None]
        aggregates[name] = LocalRotationAggregate(
            _quantile(polar_sd,.5), _quantile(polar_sd,.1), _quantile(polar_sd,.9), (_quantile(polar_sd,.75)-_quantile(polar_sd,.25)) if polar_sd else None, (max(polar_sd)-min(polar_sd)) if polar_sd else None,
            _quantile(polar_mad,.5), (_quantile(polar_mad,.75)-_quantile(polar_mad,.25)) if polar_mad else None,
            _quantile(total_sd,.5), _quantile(total_sd,.1), _quantile(total_sd,.9), (_quantile(total_sd,.75)-_quantile(total_sd,.25)) if total_sd else None,
            _quantile(total_mad,.5), (_quantile(total_mad,.75)-_quantile(total_mad,.25)) if total_mad else None,
            _quantile(residual_sd,.5), _quantile(residual_sd,.1), _quantile(residual_sd,.9), (_quantile(residual_sd,.75)-_quantile(residual_sd,.25)) if residual_sd else None,
            _quantile(residual_mad,.5), (_quantile(residual_mad,.75)-_quantile(residual_mad,.25)) if residual_mad else None,
            max(deltas) if deltas else None,
        )
    return aggregates


def compute_local_heterogeneity(
    bead_mask: np.ndarray, ag_mask: np.ndarray, center_rc: tuple[float, float], sphere_radius_px: float,
    *, inner_fraction: float, outer_fraction: float, radial_band_count: int,
    polar_sector_count: int, min_segment_completeness: float, metric: str,
    polar_rotation_samples: int = 1, display_rotation_deg: float = 0.0,
) -> LocalHeterogeneityResult:
    """Return a display grid plus robust rotations spanning one sector width."""
    total_started = perf_counter()
    workspace = build_local_grid_workspace(
        bead_mask,
        ag_mask,
        center_rc,
        sphere_radius_px,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
        radial_band_count=radial_band_count,
        min_segment_completeness=min_segment_completeness,
    )
    display = compute_display_grid_from_workspace(
        workspace,
        polar_sector_count=polar_sector_count,
        metric=metric,
        rotation_offset_deg=display_rotation_deg,
    )
    rotations = compute_local_rotation_series_from_workspace(
        workspace,
        polar_sector_count=polar_sector_count,
        metric=metric,
        polar_rotation_samples=polar_rotation_samples,
    )
    aggregates = aggregate_local_rotation_results(rotations)
    result = replace(
        display,
        polar_rotation_samples=polar_rotation_samples,
        rotation_results=rotations,
        rotation_aggregates=aggregates,
    )
    memory = _debug_memory_values()
    if memory is None:
        LOGGER.debug(
            "Local heterogeneity total: %.3f s (one workspace, %d rotations)",
            perf_counter() - total_started,
            polar_rotation_samples,
        )
    else:
        LOGGER.debug(
            "Local heterogeneity total: %.3f s (one workspace, %d rotations); traced current=%d bytes, peak=%d bytes",
            perf_counter() - total_started,
            polar_rotation_samples,
            memory[0],
            memory[1],
        )
    return result

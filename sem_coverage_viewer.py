from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import configparser
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from skimage.filters import gaussian, sobel, threshold_otsu
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import closing, dilation, disk, erosion, opening, remove_small_holes, remove_small_objects, white_tophat
from skimage.segmentation import clear_border, find_boundaries, watershed
from skimage.transform import resize, rescale
from tifffile import imread

from path_utils import (
    expand_user_path,
    path_to_config_text,
    resolve_existing_input_path,
    resolve_optional_file_in_folder,
)
from sem_coverage import (
    AnalyzerConfig,
    SEMCoverageAnalyzer,
    SegmentationError,
    validate_analyzer_config,
)
from tabular_export import sort_paths
from coverage_cap import (
    CAP_COVERAGE_METRICS,
    CapRadiusSensitivity,
    CoverageCapMetrics,
    cap_sensitivity_fractions,
    cumulative_cap_sweep,
    compute_coverage_cap_metrics,
    normalize_cap_coverage_metric,
    normalize_sphere_diameter_metric,
    maximum_inscribed_circle,
    sphere_geometry_from_mask,
    summarize_cap_sensitivity,
)
from coverage_local_heterogeneity import LocalHeterogeneityResult, compute_local_heterogeneity

if TYPE_CHECKING:
    from coverage_homogeneity import CoverageHomogeneityResult


@dataclass(frozen=True)
class CoverageViewerConfig:
    analyzer: AnalyzerConfig = AnalyzerConfig()
    detector_choice_index: int = 0
    min_bead_area_px: int = 500
    min_roi_eq_diameter_px: float = 140.0
    min_roi_solidity: float = 0.82
    max_roi_anisotropy_ratio: float = 1.65
    sphere_anisotropy_check: bool = False
    max_global_sphere_anisotropy_ratio: float = 1.25
    sphere_solidity_check: bool = False
    min_global_sphere_solidity: float = 0.90
    salvage_open_radius_px: int = 7
    bead_morph_fallback: bool = True
    bead_morph_downscale: float = 0.25
    bead_morph_blur_sigma: float = 4.0
    bead_morph_gradient_percentile: float = 80.0
    bead_morph_close_radius: int = 2
    bead_morph_dilate_radius: int = 2
    bead_morph_erode_radius_px: int = 20
    bead_morph_min_object_area_ratio: float = 0.08
    split_touching_beads: bool = True
    split_trigger_eq_diameter_px: float = 430.0
    split_trigger_anisotropy_ratio: float = 1.45
    split_trigger_solidity_below: float = 0.90
    split_min_distance_px: int = 70
    split_peak_threshold_rel: float = 0.55
    split_max_peaks: int = 4
    split_min_child_area_ratio: float = 0.18
    ag_enable_secondary_coverage: bool = False
    ag_coverage_tophat_radius: int = 15
    ag_coverage_tophat_radii: Optional[list[int]] = None
    ag_coverage_threshold_rel: float = 0.8
    ag_coverage_adaptive_threshold: bool = True
    ag_coverage_adaptive_block_size: int = 151
    ag_coverage_adaptive_k_std: float = 2.0
    ag_coverage_min_object_size: int = 9
    ag_coverage_closing_radius: int = 0
    ag_coverage_use_union_with_count: bool = True
    # Central-cap post-processing affects only final coverage reporting; bead
    # and Ag segmentation masks are always produced by the existing pipeline.
    coverage_cap_enabled: bool = True
    coverage_cap_radius_fraction: float = 0.25
    coverage_cap_min_completeness: float = 0.98
    sphere_diameter_metric: str = "mean_xy_diameter"
    selected_cap_coverage_metric: str = "projected_over_cap_surface"
    coverage_cap_graph_mode: str = "cumulative"
    coverage_cap_annulus_width_fraction: float = 0.05
    coverage_cap_sensitivity_enabled: bool = True
    coverage_cap_sensitivity_half_width: float = 0.05
    coverage_cap_sensitivity_step_fraction: float = 0.01
    coverage_homogeneity_enabled: bool = True
    homogeneity_inner_radius_fraction: float = 0.10
    homogeneity_outer_radius_fraction: float = 0.75
    radial_ring_width_fraction: float = 0.05
    polar_sector_count: int = 12
    homogeneity_min_segment_completeness: float = 0.95
    homogeneity_view_mode: str = "radial"
    polar_rotation_samples: int = 12
    polar_display_rotation_deg: float = 0.0
    # Diagnostic-only shared local polar-grid settings.  They operate on
    # existing masks and never affect production segmentation.
    local_heterogeneity_enabled: bool = True
    local_heterogeneity_inner_radius_fraction: float = 0.10
    local_heterogeneity_outer_radius_fraction: float = 0.75
    local_heterogeneity_radial_band_count: int = 8
    local_heterogeneity_polar_sector_count: int = 12
    local_heterogeneity_polar_rotation_samples: int = 12
    local_heterogeneity_display_rotation_deg: float = 0.0
    local_heterogeneity_min_segment_completeness: float = 0.95
    local_heterogeneity_show_heatmap: bool = True
    local_heterogeneity_show_1d_profiles: bool = True
    # Deprecated compatibility field.  All explicit cap metrics are now
    # available; it no longer selects the primary reported cap metric.
    coverage_cap_surface_weighting_enabled: bool = False
    default_show_scale: bool = True
    default_show_bead_boundary: bool = True
    default_show_diameter_lines: bool = True
    default_show_ag_boundary: bool = True
    default_show_ag_count_boundary: bool = False
    default_show_ag_peaks: bool = True


@dataclass(frozen=True)
class CoverageAppConfig:
    folder: str = ""
    file: Optional[str] = None
    viewer: CoverageViewerConfig = CoverageViewerConfig()
    summary_json_path: Optional[str] = None


@dataclass(frozen=True)
class SEMMetadata:
    pixel_size_x_m: Optional[float]
    pixel_size_y_m: Optional[float]
    magnification: Optional[float]
    image_strip_size_px: Optional[int]
    view_fields_count_x: Optional[int]
    view_fields_count_y: Optional[int]
    note: str
    device: str
    date: str
    time: str

    @property
    def mean_pixel_size_m(self) -> Optional[float]:
        vals = [v for v in (self.pixel_size_x_m, self.pixel_size_y_m) if v and v > 0]
        if not vals:
            return None
        return float(sum(vals) / len(vals))


@dataclass(frozen=True)
class BeadMetrics:
    centroid_rc: tuple[float, float]
    equivalent_diameter_px: float
    equivalent_diameter_m: Optional[float]
    x_diameter_px: float
    y_diameter_px: float
    x_diameter_m: Optional[float]
    y_diameter_m: Optional[float]
    major_axis_px: float
    minor_axis_px: float
    major_axis_m: Optional[float]
    minor_axis_m: Optional[float]
    anisotropy_ratio: float
    solidity: float
    sphere_surface_area_m2: Optional[float]
    sphere_diameter_px: float
    sphere_diameter_m: Optional[float]
    sphere_radius_px: float
    sphere_radius_m: Optional[float]
    sphere_volume_m3: Optional[float]
    # The measurement centroid remains available for geometric overlays; this
    # optional center is the sphere/cap reference selected by the active mode.
    sphere_center_rc: tuple[float, float] | None = None


@dataclass(frozen=True)
class BeadCoverageResult:
    roi_index: int
    bead_mask: np.ndarray
    ag_mask: np.ndarray
    ag_count_mask: np.ndarray
    count_feature: np.ndarray
    coverage_feature: np.ndarray
    ag_peak_coords: np.ndarray
    ag_threshold: float
    ag_count_threshold: float
    ag_peak_threshold: float
    ag_coverage_threshold: float
    coverage: float
    coverage_percent: float
    projected_ag_count: int
    sphere_ag_count_est: float
    sphere_np_density_per_um2: Optional[float]
    bead_area_px: int
    ag_area_px: int
    bead_metrics: BeadMetrics
    legacy_full_projected_coverage: float = 0.0
    legacy_full_projected_coverage_percent: float = 0.0
    cap_metrics: CoverageCapMetrics | None = None
    cap_projected_coverage: float | None = None
    cap_projected_coverage_percent: float | None = None
    cap_surface_weighted_coverage: float | None = None
    cap_surface_weighted_coverage_percent: float | None = None
    cap_projected_over_surface_coverage: float | None = None
    cap_projected_over_surface_coverage_percent: float | None = None
    selected_coverage_method: str = "legacy_full_projected"
    cap_sensitivity: dict[str, CapRadiusSensitivity] = field(default_factory=dict)
    homogeneity: CoverageHomogeneityResult | None = None
    local_heterogeneity: LocalHeterogeneityResult | None = None


@dataclass(frozen=True)
class CoverageImageResult:
    image_path: Path
    raw: np.ndarray
    cropped: np.ndarray
    norm: np.ndarray
    display: np.ndarray
    metadata: SEMMetadata
    roi_results: list[BeadCoverageResult]
    bead_raw_union: np.ndarray
    bead_refined_union: np.ndarray
    ag_count_feature_union: np.ndarray
    ag_coverage_feature_union: np.ndarray
    crop_row: int
    config: CoverageViewerConfig
    diagnostics: "CoverageSegmentationDiagnostics | None" = None


@dataclass(frozen=True)
class FailedImagePreview:
    image_path: Path
    cropped: np.ndarray
    norm: np.ndarray
    display: np.ndarray
    metadata: SEMMetadata
    crop_row: int


@dataclass(frozen=True)
class RejectedBeadCandidate:
    """One bead candidate rejected by the shared production ROI evaluator."""

    candidate_index: int
    mask: np.ndarray
    source: str
    area_px: int
    equivalent_diameter_px: float
    solidity: float
    anisotropy_ratio: float
    centroid_rc: tuple[float, float]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CoverageSegmentationDiagnostics:
    """Structured candidate data retained for expected tuning failures."""

    stage: str
    raw_candidate_union: np.ndarray
    accepted_candidate_union: np.ndarray
    rejected_candidate_union: np.ndarray
    rejected_candidates: tuple[RejectedBeadCandidate, ...]
    primary_error: str | None = None
    fallback_error: str | None = None


class CoverageSegmentationFailure(SegmentationError):
    """Expected no-ROI outcome with a preview and structured diagnostics."""

    def __init__(self, message: str, preview: FailedImagePreview, diagnostics: CoverageSegmentationDiagnostics):
        super().__init__(message)
        self.preview = preview
        self.diagnostics = diagnostics


class _ROIRefinementFailure(SegmentationError):
    """Internal production-filter failure retaining candidates for diagnostics."""

    def __init__(self, message: str, rejected: list[RejectedBeadCandidate]):
        super().__init__(message)
        self.rejected = tuple(rejected)


def _dataclass_from_dict(cls, data: dict):
    kwargs = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        if hasattr(field.type, "__dataclass_fields__") and isinstance(value, dict):
            value = _dataclass_from_dict(field.type, value)
        kwargs[field.name] = value
    return cls(**kwargs)


def load_app_config(config_path: str | Path) -> CoverageAppConfig:
    config_path = Path(config_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    viewer_data = dict(data.get("viewer", {}))
    analyzer_data = dict(viewer_data.get("analyzer", {}))
    for legacy_key in (
        "bead_hough_fallback",
        "bead_hough_downscale",
        "bead_hough_blur_sigma",
        "bead_hough_canny_low_threshold",
        "bead_hough_canny_high_threshold",
        "bead_hough_min_radius_ratio",
        "bead_hough_max_radius_ratio",
        "bead_hough_min_score",
    ):
        viewer_data.pop(legacy_key, None)
    if "norm_percentiles" in analyzer_data:
        analyzer_data["norm_percentiles"] = tuple(analyzer_data["norm_percentiles"])
    if "display_percentiles" in analyzer_data:
        analyzer_data["display_percentiles"] = tuple(analyzer_data["display_percentiles"])
    analyzer = AnalyzerConfig(**analyzer_data) if analyzer_data else AnalyzerConfig()
    validate_analyzer_config(analyzer)
    viewer_data["analyzer"] = analyzer
    viewer = CoverageViewerConfig(**viewer_data)
    viewer = replace(
        viewer,
        sphere_diameter_metric=normalize_sphere_diameter_metric(viewer.sphere_diameter_metric),
        selected_cap_coverage_metric=normalize_cap_coverage_metric(
            viewer.selected_cap_coverage_metric
        ),
    )
    _validate_postprocessing_config(viewer)
    return CoverageAppConfig(
        folder=str(data.get("folder") or ""),
        file=data.get("file") or None,
        viewer=viewer,
        summary_json_path=data.get("summary_json_path"),
    )


def _validate_postprocessing_config(config: CoverageViewerConfig) -> None:
    """Validate additive cap/homogeneity settings without affecting segmentation."""

    if not (0.0 < config.coverage_cap_radius_fraction <= 1.0):
        raise ValueError("coverage_cap_radius_fraction must satisfy 0 < value <= 1.")
    if not (0.0 <= config.coverage_cap_min_completeness <= 1.0):
        raise ValueError("coverage_cap_min_completeness must be between 0 and 1.")
    if not (0.0 <= config.coverage_cap_sensitivity_half_width < 1.0):
        raise ValueError("coverage_cap_sensitivity_half_width must satisfy 0 <= value < 1.")
    if not (0.0 < config.coverage_cap_sensitivity_step_fraction <= 1.0):
        raise ValueError("coverage_cap_sensitivity_step_fraction must satisfy 0 < value <= 1.")
    if config.polar_rotation_samples < 1 or not math.isfinite(config.polar_display_rotation_deg):
        raise ValueError("Polar rotation settings are invalid.")
    if not (0 <= config.homogeneity_inner_radius_fraction < config.homogeneity_outer_radius_fraction <= 1):
        raise ValueError("Homogeneity inner/outer radius fractions are invalid.")
    if not (0 < config.radial_ring_width_fraction <= config.homogeneity_outer_radius_fraction - config.homogeneity_inner_radius_fraction):
        raise ValueError("radial_ring_width_fraction is invalid.")
    if config.polar_sector_count < 2 or not (0 <= config.homogeneity_min_segment_completeness <= 1):
        raise ValueError("Homogeneity sector/completeness settings are invalid.")
    if not (0 <= config.local_heterogeneity_inner_radius_fraction < config.local_heterogeneity_outer_radius_fraction <= 1):
        raise ValueError("Local heterogeneity inner/outer radius fractions are invalid.")
    if config.local_heterogeneity_radial_band_count < 1 or config.local_heterogeneity_polar_sector_count < 2:
        raise ValueError("Local heterogeneity band/sector counts are invalid.")
    if not (0 <= config.local_heterogeneity_min_segment_completeness <= 1):
        raise ValueError("Local heterogeneity completeness is invalid.")
    if config.local_heterogeneity_polar_rotation_samples < 1 or not math.isfinite(config.local_heterogeneity_display_rotation_deg):
        raise ValueError("Local heterogeneity rotation settings are invalid.")


def save_default_config(config_path: str | Path, folder: str | Path) -> None:
    config = CoverageAppConfig(
        folder=path_to_config_text(folder),
        file=None,
        summary_json_path=path_to_config_text(
            Path(folder).resolve() / "sem_coverage_viewer_summary.json"
        ),
    )
    Path(config_path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def _read_hdr_metadata(hdr_path: Path) -> SEMMetadata:
    cfg = configparser.ConfigParser()
    cfg.read(hdr_path, encoding="utf-8")
    main = cfg["MAIN"] if "MAIN" in cfg else {}

    def _maybe_float(key: str) -> Optional[float]:
        value = main.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _maybe_int(key: str) -> Optional[int]:
        value = main.get(key)
        if value in (None, ""):
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    return SEMMetadata(
        pixel_size_x_m=_maybe_float("PixelSizeX"),
        pixel_size_y_m=_maybe_float("PixelSizeY"),
        magnification=_maybe_float("Magnification"),
        image_strip_size_px=_maybe_int("ImageStripSize"),
        view_fields_count_x=_maybe_int("ViewFieldsCountX"),
        view_fields_count_y=_maybe_int("ViewFieldsCountY"),
        note=main.get("Note", ""),
        device=main.get("Device", ""),
        date=main.get("Date", ""),
        time=main.get("Time", ""),
    )


def _paired_hdr_path(image_path: Path) -> Optional[Path]:
    hdr_name = f"{image_path.stem}-tif.hdr"
    hdr_path = image_path.with_name(hdr_name)
    return hdr_path if hdr_path.exists() else None


def _crop_infobar(img: np.ndarray, analyzer: SEMCoverageAnalyzer, strip_rows: Optional[int]) -> tuple[np.ndarray, int]:
    if strip_rows is not None and 10 <= strip_rows < img.shape[0]:
        crop_row = img.shape[0] - int(strip_rows)
        return img[:crop_row, :], crop_row
    return analyzer._crop_infobar(img)


def _select_detector_view(img: np.ndarray, metadata: SEMMetadata, detector_choice_index: int) -> np.ndarray:
    count_x = metadata.view_fields_count_x or 1
    count_y = metadata.view_fields_count_y or 1
    detector_count = int(count_x * count_y)
    if detector_count <= 1:
        return img

    index = int(detector_choice_index)
    if index < 0 or index >= detector_count:
        raise ValueError(
            f"detector_choice_index={index} is outside available detector views "
            f"0..{detector_count - 1} ({count_x}x{count_y})."
        )
    if img.shape[1] % count_x != 0 or img.shape[0] % count_y != 0:
        raise ValueError(
            f"Cannot split image shape {img.shape} into detector view grid "
            f"{count_x}x{count_y}."
        )

    row = index // count_x
    col = index % count_x
    tile_h = img.shape[0] // count_y
    tile_w = img.shape[1] // count_x
    return img[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w]


def _segment_bead_components(analyzer: SEMCoverageAnalyzer, norm: np.ndarray, min_bead_area_px: int) -> list[np.ndarray]:
    cfg = analyzer.config
    # Same processing steps as SEMCoverageAnalyzer._segment_bead, except we keep all components.
    blur = analyzer._segment_bead.__globals__["gaussian"](norm, sigma=cfg.bead_blur_sigma, preserve_range=True)
    t = analyzer._segment_bead.__globals__["threshold_otsu"](blur)
    mask = blur > t
    mask = closing(mask, analyzer._fp_bead_close)
    mask = opening(mask, analyzer._fp_bead_open)
    lab = label(mask)
    if lab.max() == 0:
        raise SegmentationError("Bead segmentation failed: no components found.")

    components: list[np.ndarray] = []
    for region in regionprops(lab):
        bead = lab == region.label
        bead = remove_small_holes(bead, max_size=cfg.bead_hole_area - 1)
        if int(bead.sum()) < min_bead_area_px:
            continue
        components.append(bead)
    if not components:
        raise SegmentationError("Bead segmentation failed: no components above area threshold.")
    return components


def _largest_region(mask: np.ndarray):
    regions = regionprops(label(mask.astype(np.uint8)))
    if not regions:
        return None
    return max(regions, key=lambda r: r.area)


def _region_stats(mask: np.ndarray) -> Optional[dict[str, float]]:
    region = _largest_region(mask)
    if region is None:
        return None
    major = float(region.axis_major_length)
    minor = float(region.axis_minor_length)
    return {
        "area": float(region.area),
        "equivalent_diameter_px": float(region.equivalent_diameter_area),
        "major_axis_px": major,
        "minor_axis_px": minor,
        "anisotropy_ratio": major / max(minor, 1e-6),
        "solidity": float(region.solidity),
    }


def _is_valid_roi(mask: np.ndarray, config: CoverageViewerConfig) -> bool:
    return not _roi_rejection_reasons(mask, config)[0]


def _roi_rejection_reasons(mask: np.ndarray, config: CoverageViewerConfig) -> tuple[tuple[str, ...], dict[str, float] | None, tuple[float, float]]:
    """Evaluate the one production ROI rule set and expose its failed clauses."""

    stats = _region_stats(mask)
    if stats is None:
        return ("empty component",), None, (float("nan"), float("nan"))
    region = _largest_region(mask)
    centroid = (float(region.centroid[0]), float(region.centroid[1])) if region else (float("nan"), float("nan"))
    reasons: list[str] = []
    if stats["area"] < float(config.min_bead_area_px):
        reasons.append(f"area {stats['area']:.0f} px < min_bead_area_px {config.min_bead_area_px:.0f} px")
    if stats["equivalent_diameter_px"] < float(config.min_roi_eq_diameter_px):
        reasons.append(
            f"equivalent diameter {stats['equivalent_diameter_px']:.1f} px < "
            f"min_roi_eq_diameter_px {config.min_roi_eq_diameter_px:.1f} px"
        )
    if stats["solidity"] < float(config.min_roi_solidity):
        reasons.append(f"solidity {stats['solidity']:.3f} < min_roi_solidity {config.min_roi_solidity:.3f}")
    if stats["anisotropy_ratio"] > float(config.max_roi_anisotropy_ratio):
        reasons.append(
            f"anisotropy {stats['anisotropy_ratio']:.3f} > "
            f"max_roi_anisotropy_ratio {config.max_roi_anisotropy_ratio:.3f}"
        )
    return tuple(reasons), stats, centroid


def _evaluate_bead_components(
    components: list[np.ndarray], config: CoverageViewerConfig, source: str, *, start_index: int = 1
) -> tuple[list[np.ndarray], list[RejectedBeadCandidate]]:
    """Apply the shared production ROI filters and retain structured rejections."""

    accepted: list[np.ndarray] = []
    rejected: list[RejectedBeadCandidate] = []
    for offset, component in enumerate(components):
        mask = np.asarray(component, dtype=bool)
        reasons, stats, centroid = _roi_rejection_reasons(mask, config)
        if not reasons:
            accepted.append(mask)
            continue
        stats = stats or {"area": 0.0, "equivalent_diameter_px": 0.0, "solidity": 0.0, "anisotropy_ratio": float("inf")}
        rejected.append(
            RejectedBeadCandidate(
                candidate_index=start_index + offset,
                mask=mask,
                source=source,
                area_px=int(stats["area"]),
                equivalent_diameter_px=float(stats["equivalent_diameter_px"]),
                solidity=float(stats["solidity"]),
                anisotropy_ratio=float(stats["anisotropy_ratio"]),
                centroid_rc=centroid,
                rejection_reasons=reasons,
            )
        )
    return accepted, rejected


def _should_try_split(mask: np.ndarray, config: CoverageViewerConfig) -> bool:
    if not config.split_touching_beads:
        return False
    stats = _region_stats(mask)
    if stats is None:
        return False
    return (
        stats["equivalent_diameter_px"] >= float(config.split_trigger_eq_diameter_px)
        or stats["anisotropy_ratio"] >= float(config.split_trigger_anisotropy_ratio)
        or stats["solidity"] <= float(config.split_trigger_solidity_below)
    )


def _split_touching_beads(mask: np.ndarray, config: CoverageViewerConfig) -> list[np.ndarray]:
    distance = ndi.distance_transform_edt(mask)
    max_dist = float(distance.max())
    if max_dist <= 0:
        return [mask]

    peak_coords = peak_local_max(
        distance,
        labels=mask.astype(np.uint8),
        min_distance=max(1, int(config.split_min_distance_px)),
        threshold_abs=max_dist * float(config.split_peak_threshold_rel),
        exclude_border=False,
    )
    if peak_coords.shape[0] < 2 or peak_coords.shape[0] > int(config.split_max_peaks):
        return [mask]

    markers = np.zeros(mask.shape, dtype=np.int32)
    for idx, (row, col) in enumerate(peak_coords, start=1):
        markers[int(row), int(col)] = idx
    markers = ndi.label(markers > 0)[0]
    split_labels = watershed(-distance, markers=markers, mask=mask)
    if int(split_labels.max()) < 2:
        return [mask]

    parent_area = float(mask.sum())
    min_child_area = parent_area * float(config.split_min_child_area_ratio)
    children: list[np.ndarray] = []
    for child_label in range(1, int(split_labels.max()) + 1):
        child = split_labels == child_label
        if float(child.sum()) < min_child_area:
            return [mask]
        if not _is_valid_roi(child, config):
            return [mask]
        children.append(child)
    return children if len(children) >= 2 else [mask]


def _salvage_roi_by_opening(mask: np.ndarray, config: CoverageViewerConfig, hole_area: int) -> Optional[np.ndarray]:
    radius = int(config.salvage_open_radius_px)
    if radius <= 0:
        return None
    opened = opening(mask, disk(radius))
    lab = label(opened)
    if lab.max() == 0:
        return None
    largest = max(regionprops(lab), key=lambda r: r.area)
    core = lab == largest.label
    restored = dilation(core, disk(radius)) & mask
    restored = remove_small_holes(restored, max_size=max(int(hole_area) - 1, 0))
    return restored if _is_valid_roi(restored, config) else None


def _segment_bead_by_morphology(
    analyzer: SEMCoverageAnalyzer,
    cropped: np.ndarray,
    config: CoverageViewerConfig,
    *,
    collect_diagnostics: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[RejectedBeadCandidate], np.ndarray]:
    scale = float(config.bead_morph_downscale)
    if not 0.05 <= scale <= 1.0:
        raise SegmentationError("Morphology bead fallback failed: bead_morph_downscale must be between 0.05 and 1.0.")

    display = analyzer._scale_for_display(cropped)
    small = rescale(display, scale, anti_aliasing=True, preserve_range=True).astype(np.float32)
    smooth = gaussian(small, sigma=float(config.bead_morph_blur_sigma), preserve_range=True)
    gradient = sobel(smooth)
    threshold = float(np.percentile(gradient, float(config.bead_morph_gradient_percentile)))
    edges = gradient > threshold
    if not np.any(edges):
        raise SegmentationError("Morphology bead fallback failed: no edge pixels after gradient thresholding.")

    close_radius = max(1, int(config.bead_morph_close_radius))
    dilate_radius = max(0, int(config.bead_morph_dilate_radius))
    mask = closing(edges, disk(close_radius))
    if dilate_radius > 0:
        mask = dilation(mask, disk(dilate_radius))
    mask = ndi.binary_fill_holes(mask)
    mask = clear_border(mask)
    min_size = max(1, int(cropped.shape[0] * cropped.shape[1] * float(config.bead_morph_min_object_area_ratio) * scale * scale))
    lab = label(mask.astype(bool))
    if lab.max() > 0:
        keep = np.zeros(mask.shape, dtype=bool)
        for region in regionprops(lab):
            if int(region.area) >= min_size:
                keep[lab == region.label] = True
        mask = keep
    lab = label(mask)
    if lab.max() == 0:
        raise SegmentationError("Morphology bead fallback failed: no enclosed component found.")

    raw_candidates: list[np.ndarray] = []
    for region in sorted(regionprops(lab), key=lambda item: item.area, reverse=True):
        small_component = lab == region.label
        bead = resize(small_component, cropped.shape, order=0, preserve_range=True, anti_aliasing=False).astype(bool)
        erode_radius = int(config.bead_morph_erode_radius_px)
        if erode_radius > 0:
            bead = erosion(bead, disk(erode_radius))
        raw_candidates.append(bead)
    candidates, rejected = _evaluate_bead_components(raw_candidates, config, "morphology fallback")
    if candidates:
        # Preserve historical selection of the largest valid fallback ROI.
        selected = [candidates[0]]
        if collect_diagnostics:
            raw_union = np.zeros(cropped.shape, dtype=bool)
            for candidate in raw_candidates:
                raw_union |= candidate
            return selected, rejected, raw_union
        return selected
    raise SegmentationError("Morphology bead fallback failed: no enclosed component passed ROI filters.")


def _refine_bead_components(
    components: list[np.ndarray], config: CoverageViewerConfig, *, collect_diagnostics: bool = False
) -> list[np.ndarray] | tuple[list[np.ndarray], list[RejectedBeadCandidate]]:
    refined: list[np.ndarray] = []
    rejected: list[RejectedBeadCandidate] = []
    candidate_number = 1
    for component in components:
        candidates = _split_touching_beads(component, config) if _should_try_split(component, config) else [component]
        accepted_candidates, rejected_candidates = _evaluate_bead_components(
            candidates, config, "primary", start_index=candidate_number
        )
        candidate_number += len(candidates)
        accepted = bool(accepted_candidates)
        refined.extend(accepted_candidates)
        rejected.extend(rejected_candidates)
        if accepted:
            continue
        salvaged = _salvage_roi_by_opening(component, config, config.analyzer.bead_hole_area)
        if salvaged is not None:
            refined.append(salvaged)
    if not refined:
        component_stats = []
        for component in components:
            stats = _region_stats(component)
            if stats is None:
                continue
            component_stats.append(stats)
        component_stats.sort(key=lambda item: item["area"], reverse=True)
        details = []
        for stats in component_stats[:3]:
            details.append(
                "area={area:.0f}, eq_diam={equivalent_diameter_px:.1f}px, "
                "solidity={solidity:.3f}, anisotropy={anisotropy_ratio:.3f}".format(**stats)
            )
        suffix = f" Largest rejected components: {'; '.join(details)}." if details else ""
        message = f"Bead segmentation failed: no valid bead-like ROI remained after filtering.{suffix}"
        if collect_diagnostics:
            raise _ROIRefinementFailure(message, rejected)
        raise SegmentationError(message)
    return (refined, rejected) if collect_diagnostics else refined


def _segment_ag_coverage(
    analyzer: SEMCoverageAnalyzer,
    img: np.ndarray,
    bead_mask: np.ndarray,
    count_mask: np.ndarray,
    count_feat: np.ndarray,
    count_thr: float,
    config: CoverageViewerConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not config.ag_enable_secondary_coverage:
        return count_mask.copy(), count_feat.copy(), float(count_thr)

    cfg = analyzer.config
    roi = bead_mask.copy()
    if analyzer._fp_ag_roi_erode is not None:
        roi = erosion(roi, analyzer._fp_ag_roi_erode)
    if roi.sum() < 500:
        raise SegmentationError("Ag coverage segmentation failed: eroded ROI too small.")

    img_f = img.astype(np.float32)
    if cfg.ag_use_log:
        img_f = np.log1p(img_f)

    viewer_radius = max(1, int(analyzer.config.ag_tophat_radius))
    if hasattr(analyzer, "_viewer_coverage_tophat_radius"):
        viewer_radius = int(analyzer._viewer_coverage_tophat_radius)
    viewer_radii = getattr(analyzer, "_viewer_coverage_tophat_radii", None)
    radii = [int(r) for r in viewer_radii] if viewer_radii else [viewer_radius]
    radii = sorted({max(1, int(r)) for r in radii})
    feat = np.zeros(img_f.shape, dtype=np.float32)
    for radius in radii:
        radius_feat = white_tophat(img_f, footprint=disk(radius)).astype(np.float32)
        feat = np.maximum(feat, radius_feat)

    vals = feat[roi]
    if vals.size < 500:
        raise SegmentationError("Ag coverage segmentation failed: insufficient ROI pixels for thresholding.")
    if float(vals.max() - vals.min()) < 1e-12:
        t = float(vals.max()) + 1e-6
    else:
        t = float(threshold_otsu(vals))

    thr_rel = getattr(analyzer, "_viewer_coverage_threshold_rel", 1.0)
    mask = (feat > (t * float(thr_rel))) & roi
    if bool(getattr(analyzer, "_viewer_coverage_adaptive_threshold", False)):
        block_size = int(getattr(analyzer, "_viewer_coverage_adaptive_block_size", 151))
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(15, block_size)
        local_mean = ndi.uniform_filter(feat, size=block_size, mode="reflect")
        local_sq_mean = ndi.uniform_filter(feat * feat, size=block_size, mode="reflect")
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
        k_std = float(getattr(analyzer, "_viewer_coverage_adaptive_k_std", 1.8))
        adaptive_mask = (feat > (local_mean + k_std * local_std)) & roi
        mask |= adaptive_mask
    min_size = int(getattr(analyzer, "_viewer_coverage_min_object_size", cfg.ag_min_object_size))
    if min_size > 1:
        mask = remove_small_objects(mask, max_size=min_size - 1)
    closing_radius = int(getattr(analyzer, "_viewer_coverage_closing_radius", 0))
    if closing_radius > 0:
        mask = closing(mask, disk(closing_radius))
    if bool(getattr(analyzer, "_viewer_coverage_use_union_with_count", True)):
        mask |= count_mask
    mask = opening(mask, analyzer._fp_open1)
    return mask, feat, float(t * float(thr_rel))


def _measure_bead(
    bead_mask: np.ndarray,
    pixel_size_m: Optional[float],
    sphere_diameter_metric: str = "mean_xy_diameter",
) -> BeadMetrics:
    region = max(regionprops(label(bead_mask.astype(np.uint8))), key=lambda r: r.area)
    rows, cols = region.coords[:, 0], region.coords[:, 1]
    x_px = float(cols.max() - cols.min() + 1)
    y_px = float(rows.max() - rows.min() + 1)
    major_px = float(region.axis_major_length)
    minor_px = float(region.axis_minor_length)
    anisotropy = major_px / max(minor_px, 1e-6)

    def _scaled(value_px: float) -> Optional[float]:
        return float(value_px * pixel_size_m) if pixel_size_m else None

    eqd_px = float(region.equivalent_diameter_area)
    eqd_m = _scaled(eqd_px)
    # Coverage uses the same X/Y bounding-box diameters shown by the viewer.
    # This is deliberately separate from equivalent diameter and major/minor.
    geometry = sphere_geometry_from_mask(
        bead_mask,
        centroid_rc=(float(region.centroid[0]), float(region.centroid[1])),
        equivalent_diameter_px=eqd_px,
        x_diameter_px=x_px,
        y_diameter_px=y_px,
        metric=sphere_diameter_metric,
    )
    sphere_radius_px = float(geometry.radius_px)
    sphere_diameter_px = float(2.0 * sphere_radius_px)
    sphere_diameter_m = _scaled(sphere_diameter_px)
    sphere_radius_m = _scaled(sphere_radius_px)
    sphere_area = None
    sphere_volume = None
    if sphere_radius_m is not None:
        sphere_area = float(4.0 * math.pi * sphere_radius_m * sphere_radius_m)
        sphere_volume = float((4.0 / 3.0) * math.pi * sphere_radius_m**3)

    return BeadMetrics(
        centroid_rc=(float(region.centroid[0]), float(region.centroid[1])),
        equivalent_diameter_px=eqd_px,
        equivalent_diameter_m=eqd_m,
        x_diameter_px=x_px,
        y_diameter_px=y_px,
        x_diameter_m=_scaled(x_px),
        y_diameter_m=_scaled(y_px),
        major_axis_px=major_px,
        minor_axis_px=minor_px,
        major_axis_m=_scaled(major_px),
        minor_axis_m=_scaled(minor_px),
        anisotropy_ratio=float(anisotropy),
        solidity=float(region.solidity),
        sphere_surface_area_m2=sphere_area,
        sphere_diameter_px=sphere_diameter_px,
        sphere_diameter_m=sphere_diameter_m,
        sphere_radius_px=sphere_radius_px,
        sphere_radius_m=sphere_radius_m,
        sphere_volume_m3=sphere_volume,
        sphere_center_rc=geometry.center_rc,
    )


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _include_roi_in_global_summary(roi: BeadCoverageResult, config: CoverageViewerConfig) -> bool:
    if config.sphere_anisotropy_check and float(roi.bead_metrics.anisotropy_ratio) > float(config.max_global_sphere_anisotropy_ratio):
        return False
    if config.sphere_solidity_check and float(roi.bead_metrics.solidity) < float(config.min_global_sphere_solidity):
        return False
    return True


def _format_length_m(value_m: Optional[float]) -> str:
    if value_m is None or not math.isfinite(value_m):
        return "n/a"
    abs_val = abs(value_m)
    if abs_val < 1e-6:
        return f"{value_m * 1e9:.0f} nm"
    if abs_val < 1e-3:
        return f"{value_m * 1e6:.2f} um"
    return f"{value_m * 1e3:.3f} mm"


def _format_px_or_length(value_m: Optional[float], value_px: float) -> str:
    return _format_length_m(value_m) if value_m is not None else f"{value_px:.1f} px"


def _nice_scale_length_m(target_m: float) -> float:
    if target_m <= 0:
        return 0.0
    exponent = math.floor(math.log10(target_m))
    fraction = target_m / (10 ** exponent)
    for base in (1.0, 2.0, 5.0, 10.0):
        if fraction <= base:
            return base * (10 ** exponent)
    return 10 ** (exponent + 1)


def _build_roi_result(
    analyzer: SEMCoverageAnalyzer,
    cropped: np.ndarray,
    bead_mask: np.ndarray,
    pixel_size_m: Optional[float],
    roi_index: int,
    config: CoverageViewerConfig,
) -> BeadCoverageResult:
    count_mask, count_feat, count_thr = analyzer._segment_ag(cropped, bead_mask)
    analyzer._viewer_coverage_tophat_radius = int(config.ag_coverage_tophat_radius)
    analyzer._viewer_coverage_tophat_radii = config.ag_coverage_tophat_radii
    analyzer._viewer_coverage_threshold_rel = float(config.ag_coverage_threshold_rel)
    analyzer._viewer_coverage_adaptive_threshold = bool(config.ag_coverage_adaptive_threshold)
    analyzer._viewer_coverage_adaptive_block_size = int(config.ag_coverage_adaptive_block_size)
    analyzer._viewer_coverage_adaptive_k_std = float(config.ag_coverage_adaptive_k_std)
    analyzer._viewer_coverage_min_object_size = int(config.ag_coverage_min_object_size)
    analyzer._viewer_coverage_closing_radius = int(config.ag_coverage_closing_radius)
    analyzer._viewer_coverage_use_union_with_count = bool(config.ag_coverage_use_union_with_count)
    ag_mask, coverage_feat, coverage_thr = _segment_ag_coverage(analyzer, cropped, bead_mask, count_mask, count_feat, count_thr, config)
    legacy_coverage = analyzer._compute_coverage(bead_mask, ag_mask)
    projected_ag_count = analyzer._count_ag_peaks(count_feat, count_mask, count_thr)
    peak_thr = float(count_thr) * float(analyzer.config.count_thr_rel)
    ag_peak_coords = peak_local_max(
        count_feat,
        labels=count_mask.astype(np.uint8),
        min_distance=int(analyzer.config.count_min_distance),
        threshold_abs=peak_thr,
        exclude_border=False,
    )
    bead_metrics = _measure_bead(
        bead_mask, pixel_size_m, config.sphere_diameter_metric
    )
    cap_metrics = compute_coverage_cap_metrics(
        bead_mask,
        ag_mask,
        bead_metrics.sphere_center_rc or bead_metrics.centroid_rc,
        bead_metrics.sphere_radius_px,
        config.coverage_cap_radius_fraction,
        pixel_size_m,
        compute_surface_weighted=True,
        min_completeness=config.coverage_cap_min_completeness,
    )
    selected_cap_metric = normalize_cap_coverage_metric(config.selected_cap_coverage_metric)
    selected_cap_value = cap_metrics.selected_value(selected_cap_metric)
    cap_sensitivity: dict[str, CapRadiusSensitivity] = {}
    if config.coverage_cap_sensitivity_enabled:
        fractions = cap_sensitivity_fractions(
            config.coverage_cap_radius_fraction,
            config.coverage_cap_sensitivity_half_width,
            config.coverage_cap_sensitivity_step_fraction,
        )
        sweep = cumulative_cap_sweep(
            bead_mask, ag_mask, bead_metrics.sphere_center_rc or bead_metrics.centroid_rc, bead_metrics.sphere_radius_px,
            fractions, pixel_size_m, min_completeness=config.coverage_cap_min_completeness,
        )
        cap_sensitivity = {
            name: summarize_cap_sensitivity(sweep, name) for name in CAP_COVERAGE_METRICS
        }
    sphere_center = bead_metrics.sphere_center_rc or bead_metrics.centroid_rc
    # Legacy radial/polar homogeneity remains available through the diagnostic
    # adapter, but is not calculated by production/batch scientific analysis.
    homogeneity: CoverageHomogeneityResult | None = None
    local_heterogeneity: LocalHeterogeneityResult | None = None
    if config.local_heterogeneity_enabled:
        local_heterogeneity = compute_local_heterogeneity(
            bead_mask, ag_mask, sphere_center, bead_metrics.sphere_radius_px,
            inner_fraction=config.local_heterogeneity_inner_radius_fraction,
            outer_fraction=config.local_heterogeneity_outer_radius_fraction,
            radial_band_count=config.local_heterogeneity_radial_band_count,
            polar_sector_count=config.local_heterogeneity_polar_sector_count,
            min_segment_completeness=config.local_heterogeneity_min_segment_completeness,
            metric=selected_cap_metric,
            polar_rotation_samples=config.local_heterogeneity_polar_rotation_samples,
            display_rotation_deg=config.local_heterogeneity_display_rotation_deg,
        )
    if config.coverage_cap_enabled and cap_metrics.valid and selected_cap_value is not None:
        coverage = float(selected_cap_value)
        selected_method = f"cap_{selected_cap_metric}"
    else:
        coverage = float(legacy_coverage)
        selected_method = "legacy_full_projected"
    sphere_count_est = float(projected_ag_count * 2.0)
    density_per_um2 = None
    if bead_metrics.sphere_surface_area_m2 and bead_metrics.sphere_surface_area_m2 > 0:
        density_per_um2 = sphere_count_est / (bead_metrics.sphere_surface_area_m2 * 1e12)

    return BeadCoverageResult(
        roi_index=roi_index,
        bead_mask=bead_mask,
        ag_mask=ag_mask,
        ag_count_mask=count_mask,
        count_feature=count_feat,
        coverage_feature=coverage_feat,
        ag_peak_coords=ag_peak_coords,
        # Backward-compatible alias: ag_threshold remains the effective
        # threshold of the mask used for coverage.
        ag_threshold=float(coverage_thr),
        # Effective primary Ag/count-mask threshold: Otsu multiplied by
        # analyzer.ag_mask_threshold_rel.
        ag_count_threshold=float(count_thr),
        # Effective local-maximum threshold. This does not segment either mask.
        ag_peak_threshold=peak_thr,
        # Effective independent secondary threshold when enabled; otherwise
        # equal to the primary threshold because the primary mask is coverage.
        ag_coverage_threshold=float(coverage_thr),
        coverage=float(coverage),
        coverage_percent=float(coverage * 100.0),
        projected_ag_count=int(projected_ag_count),
        sphere_ag_count_est=float(sphere_count_est),
        sphere_np_density_per_um2=_safe_float(density_per_um2),
        bead_area_px=int(bead_mask.sum()),
        ag_area_px=int(ag_mask.sum()),
        bead_metrics=bead_metrics,
        legacy_full_projected_coverage=float(legacy_coverage),
        legacy_full_projected_coverage_percent=float(legacy_coverage * 100.0),
        cap_metrics=cap_metrics,
        cap_projected_coverage=cap_metrics.projected_coverage,
        cap_projected_coverage_percent=(
            float(cap_metrics.projected_coverage * 100.0)
            if cap_metrics.projected_coverage is not None
            else None
        ),
        cap_surface_weighted_coverage=cap_metrics.surface_weighted_coverage,
        cap_surface_weighted_coverage_percent=(
            float(cap_metrics.surface_weighted_coverage * 100.0)
            if cap_metrics.surface_weighted_coverage is not None
            else None
        ),
        cap_projected_over_surface_coverage=cap_metrics.projected_over_cap_surface,
        cap_projected_over_surface_coverage_percent=(
            float(cap_metrics.projected_over_cap_surface * 100.0)
            if cap_metrics.projected_over_cap_surface is not None
            else None
        ),
        selected_coverage_method=selected_method,
        cap_sensitivity=cap_sensitivity,
        homogeneity=homogeneity,
        local_heterogeneity=local_heterogeneity,
    )


def analyze_coverage_image(
    image_path: str | Path,
    config: CoverageViewerConfig,
    *,
    collect_diagnostics: bool = False,
) -> CoverageImageResult:
    """Run the production coverage pipeline, optionally retaining ROI failures.

    ``collect_diagnostics`` changes failure reporting only.  The segmentation,
    filters, and accepted result construction are shared with regular analysis.
    """
    image_path = Path(image_path)
    analyzer = SEMCoverageAnalyzer(config.analyzer)
    raw, cropped, norm, display, metadata, crop_row = _load_preprocessed_image(image_path, analyzer, config)
    primary_error: str | None = None
    fallback_error: str | None = None
    rejected: list[RejectedBeadCandidate] = []
    bead_raw_union = np.zeros(cropped.shape, dtype=bool)
    accepted_union = np.zeros(cropped.shape, dtype=bool)
    try:
        raw_components = _segment_bead_components(analyzer, norm, config.min_bead_area_px)
        for component in raw_components:
            bead_raw_union |= component
        try:
            if collect_diagnostics:
                bead_components, rejected = _refine_bead_components(
                    raw_components, config, collect_diagnostics=True
                )
            else:
                bead_components = _refine_bead_components(raw_components, config)
            for component in bead_components:
                accepted_union |= component
        except SegmentationError as exc:
            if isinstance(exc, _ROIRefinementFailure):
                rejected = list(exc.rejected)
            primary_error = str(exc)
            if not config.bead_morph_fallback:
                raise
            try:
                fallback_result = _segment_bead_by_morphology(analyzer, cropped, config, collect_diagnostics=collect_diagnostics)
                if collect_diagnostics:
                    bead_components, fallback_rejected, fallback_raw = fallback_result
                    rejected.extend(fallback_rejected)
                    bead_raw_union = fallback_raw
                else:
                    bead_components = fallback_result
                accepted_union = np.zeros(cropped.shape, dtype=bool)
                for component in bead_components:
                    accepted_union |= component
            except SegmentationError as fallback_exc:
                fallback_error = str(fallback_exc)
                raise fallback_exc
    except SegmentationError as exc:
        primary_error = primary_error or str(exc)
        if not config.bead_morph_fallback:
            if collect_diagnostics:
                rejected_union = np.zeros(cropped.shape, dtype=bool)
                for candidate in rejected:
                    rejected_union |= candidate.mask
                diagnostics = CoverageSegmentationDiagnostics(
                    stage="primary", raw_candidate_union=bead_raw_union,
                    accepted_candidate_union=accepted_union,
                    rejected_candidate_union=rejected_union,
                    rejected_candidates=tuple(rejected), primary_error=primary_error,
                )
                preview = FailedImagePreview(image_path, cropped, norm, display, metadata, int(crop_row))
                raise CoverageSegmentationFailure(str(exc), preview, diagnostics) from exc
            raise
        try:
            fallback_result = _segment_bead_by_morphology(analyzer, cropped, config, collect_diagnostics=collect_diagnostics)
            if collect_diagnostics:
                bead_components, fallback_rejected, fallback_raw = fallback_result
                rejected.extend(fallback_rejected)
                bead_raw_union = fallback_raw
            else:
                bead_components = fallback_result
            if not collect_diagnostics:
                bead_raw_union = np.zeros(cropped.shape, dtype=bool)
            accepted_union = np.zeros(cropped.shape, dtype=bool)
            for component in bead_components:
                if not collect_diagnostics:
                    bead_raw_union |= component
                accepted_union |= component
        except SegmentationError as fallback_exc:
            fallback_error = str(fallback_exc)
            if collect_diagnostics:
                rejected_union = np.zeros(cropped.shape, dtype=bool)
                for candidate in rejected:
                    rejected_union |= candidate.mask
                diagnostics = CoverageSegmentationDiagnostics(
                    stage="fallback", raw_candidate_union=bead_raw_union,
                    accepted_candidate_union=accepted_union,
                    rejected_candidate_union=rejected_union,
                    rejected_candidates=tuple(rejected), primary_error=primary_error,
                    fallback_error=fallback_error,
                )
                preview = FailedImagePreview(image_path, cropped, norm, display, metadata, int(crop_row))
                raise CoverageSegmentationFailure(
                    "No valid bead ROI survived primary segmentation or morphology fallback.", preview, diagnostics
                ) from fallback_exc
            raise
    bead_refined_union = np.zeros(cropped.shape, dtype=bool)
    for component in bead_components:
        bead_refined_union |= component
    roi_results: list[BeadCoverageResult] = []
    ag_count_feature_union = np.zeros(cropped.shape, dtype=np.float32)
    ag_coverage_feature_union = np.zeros(cropped.shape, dtype=np.float32)
    for idx, bead_mask in enumerate(bead_components):
        try:
            roi = _build_roi_result(analyzer, cropped, bead_mask, metadata.mean_pixel_size_m, idx + 1, config)
            roi_results.append(roi)
            ag_count_feature_union = np.maximum(ag_count_feature_union, roi.count_feature.astype(np.float32))
            ag_coverage_feature_union = np.maximum(ag_coverage_feature_union, roi.coverage_feature.astype(np.float32))
        except SegmentationError:
            continue
    rejected_union = np.zeros(cropped.shape, dtype=bool)
    for candidate in rejected:
        rejected_union |= candidate.mask
    diagnostics = CoverageSegmentationDiagnostics(
        stage="accepted",
        raw_candidate_union=bead_raw_union,
        accepted_candidate_union=bead_refined_union,
        rejected_candidate_union=rejected_union,
        rejected_candidates=tuple(rejected),
        primary_error=primary_error,
        fallback_error=fallback_error,
    ) if collect_diagnostics else None
    return CoverageImageResult(
        image_path=image_path,
        raw=raw,
        cropped=cropped,
        norm=norm,
        display=display,
        metadata=metadata,
        roi_results=roi_results,
        bead_raw_union=bead_raw_union,
        bead_refined_union=bead_refined_union,
        ag_count_feature_union=ag_count_feature_union,
        ag_coverage_feature_union=ag_coverage_feature_union,
        crop_row=int(crop_row),
        config=config,
        diagnostics=diagnostics,
    )


def _empty_metadata() -> SEMMetadata:
    return SEMMetadata(
        pixel_size_x_m=None,
        pixel_size_y_m=None,
        magnification=None,
        image_strip_size_px=None,
        view_fields_count_x=None,
        view_fields_count_y=None,
        note="",
        device="",
        date="",
        time="",
    )


def _load_preprocessed_image(
    image_path: str | Path,
    analyzer: SEMCoverageAnalyzer,
    config: CoverageViewerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SEMMetadata, int]:
    image_path = Path(image_path)
    hdr_path = _paired_hdr_path(image_path)
    metadata = _read_hdr_metadata(hdr_path) if hdr_path else _empty_metadata()
    raw = imread(str(image_path))
    cropped, crop_row = _crop_infobar(raw, analyzer, metadata.image_strip_size_px)
    cropped = _select_detector_view(cropped, metadata, config.detector_choice_index)
    norm = analyzer._normalize(cropped)
    display = analyzer._scale_for_display(cropped)
    return raw, cropped, norm, display, metadata, int(crop_row)


def load_failed_image_preview(image_path: str | Path, config: CoverageViewerConfig) -> FailedImagePreview:
    image_path = Path(image_path)
    analyzer = SEMCoverageAnalyzer(config.analyzer)
    _, cropped, norm, display, metadata, crop_row = _load_preprocessed_image(image_path, analyzer, config)
    return FailedImagePreview(
        image_path=image_path,
        cropped=cropped,
        norm=norm,
        display=display,
        metadata=metadata,
        crop_row=crop_row,
    )


def _resolve_image_paths(folder: str | Path, file: Optional[str] = None) -> list[Path]:
    folder = Path(folder)
    if file:
        image_path = resolve_optional_file_in_folder(folder, file)
        if not image_path.exists():
            raise FileNotFoundError(f"Configured TIFF file not found: '{image_path}'.")
        if not image_path.is_file():
            raise FileNotFoundError(f"Configured TIFF path is not a file: '{image_path}'.")
        return [image_path]
    return sort_paths(list(folder.glob("*.tif")))


def _metric_components_record(components) -> dict[str, float | None]:
    """Serialize one shared post-processing component triple for JSON output."""

    return {
        "numerator": _safe_float(components.numerator),
        "denominator": _safe_float(components.denominator),
        "fraction": _safe_float(components.value),
        "percent": _safe_float(components.value * 100.0) if components.value is not None else None,
    }


def _segment_record(segment) -> dict:
    values = {
        "index": segment.index,
        "inner_radius_fraction": _safe_float(segment.inner),
        "outer_radius_fraction": _safe_float(segment.outer),
        "center_radius_fraction": _safe_float(segment.center),
        "valid": bool(segment.valid),
        "completeness": _safe_float(segment.completeness),
        "bead_pixel_count": int(segment.bead_pixel_count),
        "ag_pixel_count": int(segment.ag_pixel_count),
        "surface_area_px2": _safe_float(segment.surface_area_px2),
    }
    if segment.start_angle_deg is not None:
        values["start_angle_deg"] = _safe_float(segment.start_angle_deg)
        values["end_angle_deg"] = _safe_float(segment.end_angle_deg)
        values["center_angle_deg"] = _safe_float((segment.start_angle_deg + (segment.end_angle_deg - segment.start_angle_deg) % 360.0 / 2.0) % 360.0)
    for metric in CAP_COVERAGE_METRICS:
        component = segment.components(metric)
        prefix = metric
        values.update({
            f"{prefix}_numerator": _safe_float(component.numerator),
            f"{prefix}_denominator": _safe_float(component.denominator),
            f"{prefix}_fraction": _safe_float(component.value),
            f"{prefix}_percent": _safe_float(component.value * 100.0) if component.value is not None else None,
        })
    return values


def _homogeneity_record(result: CoverageHomogeneityResult | None, selected_metric: str) -> dict:
    """Serialize shared homogeneity results without including masks/maps."""

    if result is None:
        return {"homogeneity_enabled": False, "radial_ring_results": [], "polar_display_rotation_sector_results": [], "polar_rotation_summaries": []}
    record: dict = {
        "homogeneity_enabled": True,
        "homogeneity_inner_radius_fraction": result.rings[0].inner if result.rings else None,
        "homogeneity_outer_radius_fraction": result.rings[-1].outer if result.rings else None,
        "requested_radial_ring_width_fraction": result.requested_radial_ring_width_fraction,
        "effective_radial_ring_width_fraction": result.effective_radial_ring_width_fraction,
        "polar_display_rotation_deg": result.display_rotation_deg,
        "radial_ring_results": [_segment_record(item) for item in result.rings],
        "polar_display_rotation_sector_results": [_segment_record(item) for item in result.sectors],
        "polar_rotation_summaries": [],
    }
    for metric in CAP_COVERAGE_METRICS:
        direct = result.direct_domain_components[metric]
        radial = result.radial_reconstructed_components[metric]
        polar = result.polar_reconstructed_components[metric]
        for label, component in (("direct_domain", direct), ("radial_reconstructed", radial), ("polar_reconstructed", polar)):
            record[f"{metric}_{label}_fraction"] = _safe_float(component.value)
            record[f"{metric}_{label}_percent"] = _safe_float(component.value * 100.0) if component.value is not None else None
        record[f"{metric}_radial_partition_delta_pp"] = _safe_float(result.radial_partition_delta_pp[metric])
        record[f"{metric}_polar_partition_delta_pp"] = _safe_float(result.polar_partition_delta_pp[metric])
        record[f"{metric}_radial_polar_partition_delta_pp"] = _safe_float(result.radial_polar_partition_delta_pp[metric])
        radial_summary = result.radial_summaries_by_metric[metric]
        for prefix, summary in ((f"{metric}_radial", radial_summary),):
            record.update({
                f"{prefix}_weighted_mean_percent": _safe_float(summary.weighted_mean * 100.0) if summary.weighted_mean is not None else None,
                f"{prefix}_weighted_sd_pp": _safe_float(summary.weighted_sd_pp),
                f"{prefix}_weighted_median_percent": _safe_float(summary.weighted_median * 100.0) if summary.weighted_median is not None else None,
                f"{prefix}_weighted_mad_pp": _safe_float(summary.weighted_mad_pp),
                f"{prefix}_range_pp": _safe_float(summary.range_pp),
                f"{prefix}_slope_pp_per_R": _safe_float(summary.slope_pp_per_R),
                f"{prefix}_valid_ring_count": summary.valid_count,
            })
        aggregate = result.polar_rotation_aggregates[metric]
        for field_name in ("sd_median_pp", "sd_q10_pp", "sd_q90_pp", "sd_iqr_pp", "sd_range_pp", "mad_median_pp", "mad_q10_pp", "mad_q90_pp", "mad_iqr_pp", "partition_delta_max_pp"):
            record[f"{metric}_polar_rotation_{field_name}"] = _safe_float(getattr(aggregate, field_name))
    # Generic aliases always identify their selected metric in the same record.
    record.update({
        "radial_weighted_sd_pp": record.get(f"{selected_metric}_radial_weighted_sd_pp"),
        "radial_weighted_mad_pp": record.get(f"{selected_metric}_radial_weighted_mad_pp"),
        "radial_coverage_slope_pp_per_R": record.get(f"{selected_metric}_radial_slope_pp_per_R"),
        "polar_rotation_sd_median_pp": record.get(f"{selected_metric}_polar_rotation_sd_median_pp"),
        "polar_rotation_sd_iqr_pp": record.get(f"{selected_metric}_polar_rotation_sd_iqr_pp"),
        "radial_partition_delta_pp": record.get(f"{selected_metric}_radial_partition_delta_pp"),
        "polar_partition_delta_pp": record.get(f"{selected_metric}_polar_partition_delta_pp"),
        "radial_polar_partition_delta_pp": record.get(f"{selected_metric}_radial_polar_partition_delta_pp"),
    })
    for rotation in result.polar_rotation_summaries:
        item = {"rotation_index": rotation.rotation_index, "rotation_offset_deg": rotation.rotation_offset_deg}
        for metric, summary in rotation.summaries_by_metric.items():
            item.update({
                f"{metric}_reconstructed_coverage_percent": _safe_float(summary.reconstructed_coverage * 100.0) if summary.reconstructed_coverage is not None else None,
                f"{metric}_weighted_sd_pp": _safe_float(summary.weighted_sd_pp),
                f"{metric}_weighted_mad_pp": _safe_float(summary.weighted_mad_pp),
                f"{metric}_range_pp": _safe_float(summary.range_pp),
            })
        record["polar_rotation_summaries"].append(item)
    return record


def _local_heterogeneity_record(result: LocalHeterogeneityResult | None, selected_metric: str) -> dict:
    """Serialize shared direct-domain/local-grid data without full-size maps."""

    if result is None:
        return {
            "homogeneity_domain": None,
            "local_heterogeneity": None,
            "local_grid_cells": [],
        }
    domain = result.domain
    record: dict[str, object] = {
        "homogeneity_domain": {},
        "local_heterogeneity": {},
        "local_grid_cells": [],
    }
    domain_record = record["homogeneity_domain"]
    assert isinstance(domain_record, dict)
    domain_record.update({
        "inner_radius_fraction": _safe_float(domain.inner_radius_fraction),
        "outer_radius_fraction": _safe_float(domain.outer_radius_fraction),
        "completeness": _safe_float(domain.completeness),
        "valid": bool(domain.valid),
    })
    for metric in CAP_COVERAGE_METRICS:
        component = domain.components(metric)
        domain_record.update({
            f"{metric}_numerator": _safe_float(component.numerator),
            f"{metric}_denominator": _safe_float(component.denominator),
            f"{metric}_fraction": _safe_float(component.value),
            f"{metric}_percent": _safe_float(component.value * 100.0) if component.value is not None else None,
        })
    local_record = record["local_heterogeneity"]
    assert isinstance(local_record, dict)
    local_record.update({
        "radial_band_count": int(result.values.shape[0]),
        "polar_sector_count": int(result.values.shape[1]),
        "display_rotation_deg": _safe_float(result.display_rotation_deg),
        "valid": bool(result.scientifically_valid),
    })
    for metric, summary in result.metrics.items():
        local_record.update({
            f"{metric}_valid": bool(result.metric_validity.get(metric, False)),
            f"{metric}_reconstructed_fraction": _safe_float(summary.reconstructed.value),
            f"{metric}_reconstructed_percent": _safe_float(summary.reconstructed.value * 100.0) if summary.reconstructed.value is not None else None,
            f"{metric}_reconstruction_delta_pp": _safe_float(summary.reconstruction_delta_pp),
            f"{metric}_total_weighted_sd_pp": _safe_float(summary.total_weighted_sd_pp),
            f"{metric}_total_weighted_mad_pp": _safe_float(summary.total_weighted_mad_pp),
            f"{metric}_residual_weighted_sd_pp": _safe_float(summary.residual_weighted_sd_pp),
            f"{metric}_residual_weighted_mad_pp": _safe_float(summary.residual_weighted_mad_pp),
            f"{metric}_radial_weighted_sd_pp": _safe_float(summary.radial_weighted_sd_pp),
            f"{metric}_radial_slope_pp_per_R": _safe_float(summary.radial_slope_pp_per_R),
            f"{metric}_radial_profile": [_safe_float(value) for value in summary.radial_profile],
            f"{metric}_polar_profile": [_safe_float(value) for value in summary.polar_profile],
        })
    # The direct annular profile is intentionally serialized separately from
    # display-grid cells.  It is fixed by radial bands and does not inherit any
    # polar-sector completeness or display-rotation dependence.
    local_record["radial_profile_details"] = []
    radial_band_count = len(result.radial_edges) - 1
    for radial_index in range(radial_band_count):
        completeness_values = [
            summary.radial_completeness[radial_index]
            for summary in result.metrics.values()
            if summary.radial_completeness is not None
            and radial_index < len(summary.radial_completeness)
        ]
        valid_values = [
            bool(summary.radial_valid[radial_index])
            for summary in result.metrics.values()
            if summary.radial_valid is not None
            and radial_index < len(summary.radial_valid)
        ]
        profile_item: dict[str, object] = {
            "radial_band_index": radial_index,
            "radial_inner_fraction": _safe_float(result.radial_edges[radial_index]),
            "radial_outer_fraction": _safe_float(result.radial_edges[radial_index + 1]),
            "valid": bool(any(valid_values)),
            "completeness": _safe_float(completeness_values[0]) if completeness_values else None,
        }
        midpoint = (
            result.radial_edges[radial_index]
            + result.radial_edges[radial_index + 1]
        ) / 2.0
        for metric, field_name in (
            ("projected_fraction", "radial_center_proj_fraction"),
            ("projected_over_cap_surface", "radial_center_caps_fraction"),
            ("surface_weighted_fraction", "radial_center_surfw_fraction"),
        ):
            summary = result.metrics[metric]
            center = (
                summary.radial_centers[radial_index]
                if summary.radial_centers is not None
                and radial_index < len(summary.radial_centers)
                and np.isfinite(summary.radial_centers[radial_index])
                else midpoint
            )
            profile_item[field_name] = _safe_float(center)
        for metric, summary in result.metrics.items():
            value = summary.radial_profile[radial_index] if radial_index < len(summary.radial_profile) else float("nan")
            profile_item[f"{metric}_percent"] = _safe_float(value * 100.0) if np.isfinite(value) else None
        local_record["radial_profile_details"].append(profile_item)
    for metric, aggregate in (result.rotation_aggregates or {}).items():
        for field_name in (
            "polar_sd_median_pp", "total_local_sd_median_pp",
            "residual_sd_median_pp",
        ):
            local_record[f"{metric}_{field_name}"] = _safe_float(
                getattr(aggregate, field_name)
            )
    record["local_rotation_summaries"] = []
    record["local_rotation_sector_profiles"] = []
    for rotation_index, rotation in enumerate(result.rotation_results):
        sector_cells_by_index = {
            sector_index: [
                cell
                for cell in rotation.cells
                if cell.sector_index == sector_index
            ]
            for sector_index in range(len(rotation.sector_edges_deg) - 1)
        }
        sector_validity = {
            sector_index: any(cell.valid for cell in sector_cells)
            for sector_index, sector_cells in sector_cells_by_index.items()
        }
        for metric, summary in rotation.metrics.items():
            valid_cells = [cell for cell in rotation.cells if cell.valid]
            record["local_rotation_summaries"].append({
                "rotation_index": rotation_index, "rotation_offset_deg": _safe_float(rotation.display_rotation_deg), "metric": metric,
                "valid_sector_count": int(sum(sector_validity.values())), "valid_cell_count": len(valid_cells),
                "polar_sd_pp": _safe_float(summary.polar_weighted_sd_pp), "polar_mad_pp": _safe_float(summary.polar_weighted_mad_pp),
                "local_total_sd_pp": _safe_float(summary.total_weighted_sd_pp), "local_total_mad_pp": _safe_float(summary.total_weighted_mad_pp),
                "local_residual_sd_pp": _safe_float(summary.residual_weighted_sd_pp), "local_residual_mad_pp": _safe_float(summary.residual_weighted_mad_pp),
                "reconstructed_coverage_pct": _safe_float(summary.reconstructed.value * 100.0) if summary.reconstructed.value is not None else None, "reconstruction_delta_pp": _safe_float(summary.reconstruction_delta_pp),
            })
        # One compact sector profile per robust rotation.  Its metric values
        # are sums of every cell's numerators/denominators, never arithmetic
        # means of cell percentages.  Cell validity remains separate from the
        # all-cell population used for reconstruction.
        for sector_index in range(len(rotation.sector_edges_deg) - 1):
            sector_cells = sector_cells_by_index[sector_index]
            reference_pixels = sum(cell.reference_pixel_count for cell in sector_cells)
            theoretical_pixels = sum(
                cell.theoretical_pixel_count for cell in sector_cells
            )
            item = {
                "rotation_index": rotation_index,
                "rotation_offset_deg": _safe_float(rotation.display_rotation_deg),
                "polar_sector_index": sector_index,
                "polar_start_deg": _safe_float(rotation.display_rotation_deg + rotation.sector_edges_deg[sector_index]),
                "polar_end_deg": _safe_float(rotation.display_rotation_deg + rotation.sector_edges_deg[sector_index + 1]),
                "valid": bool(sector_validity[sector_index]),
                "theoretical_pixel_count": int(theoretical_pixels),
                "reference_pixel_count": int(reference_pixels),
                "completeness": _safe_float(reference_pixels / theoretical_pixels) if theoretical_pixels > 0.0 else None,
            }
            for metric in CAP_COVERAGE_METRICS:
                # Reconstruction uses every cell with a defined metric
                # component, including cells that fail the local completeness
                # threshold.  This is the same population used by the backend
                # rotation reconstruction.
                numerator = sum(
                    cell.components(metric).numerator for cell in sector_cells
                )
                denominator = sum(
                    cell.components(metric).denominator for cell in sector_cells
                )
                item[f"{metric}_numerator"] = _safe_float(numerator)
                item[f"{metric}_denominator"] = _safe_float(denominator)
                item[f"{metric}_percent"] = _safe_float(numerator / denominator * 100.0) if denominator > 0.0 else None
            record["local_rotation_sector_profiles"].append(item)
    for cell in result.cells:
        item = {
            "rotation_offset_deg": _safe_float(result.display_rotation_deg),
            "radial_index": cell.radial_index,
            "sector_index": cell.sector_index,
            "inner_radius_fraction": _safe_float(cell.inner_fraction),
            "outer_radius_fraction": _safe_float(cell.outer_fraction),
            "center_radius_fraction": _safe_float((cell.inner_fraction + cell.outer_fraction) / 2.0),
            "start_angle_deg": _safe_float(cell.start_angle_deg),
            "end_angle_deg": _safe_float(cell.end_angle_deg),
            "center_angle_deg": _safe_float((cell.start_angle_deg + cell.end_angle_deg) / 2.0),
            "valid": bool(cell.valid),
            "completeness": _safe_float(cell.completeness),
            "theoretical_pixel_count": int(cell.theoretical_pixel_count),
            "reference_pixel_count": int(cell.reference_pixel_count),
            "ag_pixel_count": int(cell.ag_pixel_count),
        }
        for metric in CAP_COVERAGE_METRICS:
            component = cell.components(metric)
            item.update({
                f"{metric}_numerator": _safe_float(component.numerator),
                f"{metric}_denominator": _safe_float(component.denominator),
                f"{metric}_fraction": _safe_float(component.value),
                f"{metric}_percent": _safe_float(component.value * 100.0) if component.value is not None else None,
            })
        record["local_grid_cells"].append(item)
    # Scalar aliases are deliberately explicit about the selected metric.
    selected = normalize_cap_coverage_metric(selected_metric)
    selected_domain = domain.components(selected)
    selected_grid = result.metrics[selected]
    record.update({
        "homogeneity_domain_inner_radius_fraction": _safe_float(domain.inner_radius_fraction),
        "homogeneity_domain_outer_radius_fraction": _safe_float(domain.outer_radius_fraction),
        "homogeneity_domain_completeness": _safe_float(domain.completeness),
        "homogeneity_domain_valid": bool(domain.valid),
        "primary_homogeneity_domain_coverage_fraction": _safe_float(selected_domain.value),
        "primary_homogeneity_domain_coverage_percent": _safe_float(selected_domain.value * 100.0) if selected_domain.value is not None else None,
        "primary_local_grid_reconstruction_delta_pp": _safe_float(selected_grid.reconstruction_delta_pp),
        "local_total_weighted_sd_pp": _safe_float(selected_grid.total_weighted_sd_pp),
        "local_total_weighted_mad_pp": _safe_float(selected_grid.total_weighted_mad_pp),
        "local_residual_weighted_sd_pp": _safe_float(selected_grid.residual_weighted_sd_pp),
        "local_residual_weighted_mad_pp": _safe_float(selected_grid.residual_weighted_mad_pp),
    })
    for metric in CAP_COVERAGE_METRICS:
        component = domain.components(metric)
        record[f"homogeneity_domain_{metric}"] = _safe_float(component.value)
        record[f"homogeneity_domain_{metric}_percent"] = _safe_float(component.value * 100.0) if component.value is not None else None
        record[f"local_grid_reconstructed_{metric}_percent"] = local_record.get(f"{metric}_reconstructed_percent")
        record[f"local_grid_{metric}_delta_pp"] = local_record.get(f"{metric}_reconstruction_delta_pp")
    return record


def _cap_sensitivity_record(values: dict[str, CapRadiusSensitivity], selected_metric: str) -> dict:
    record: dict = {}
    for metric, summary in values.items():
        prefix = f"{metric}_sensitivity"
        for field_name in ("point_count", "interval_low_fraction", "interval_high_fraction", "median_percent", "q10_percent", "q90_percent", "q10_q90_half_width_pp", "half_range_pp", "slope_pp_per_R"):
            record[f"{prefix}_{field_name}"] = _safe_float(getattr(summary, field_name))
    for field_name in ("q10_q90_half_width_pp", "half_range_pp", "slope_pp_per_R"):
        record[f"primary_cap_radius_sensitivity_{field_name}"] = record.get(f"{selected_metric}_sensitivity_{field_name}")
    return record


def build_coverage_summary(
    folder: str | Path,
    config: CoverageViewerConfig,
    file: Optional[str] = None,
    *,
    results: dict[Path, CoverageImageResult] | None = None,
    failures: list[dict] | None = None,
) -> dict:
    """Build a coverage summary, optionally reusing already analysed images.

    ``results`` is used by the batch path so JSON, PNG and tables share the
    single branch analysis rather than invoking the pipeline a second time.
    """
    folder = Path(folder)
    image_paths = _resolve_image_paths(folder, file)
    images = []
    failed_images = list(failures or [])
    coverage_vals = []
    coverage_pct_vals = []
    projected_counts = []
    sphere_counts = []
    sphere_densities = []
    bead_diameters = []
    primary_cap_coverages_pct: list[float] = []
    radial_sd_values: list[float] = []
    polar_sd_values: list[float] = []
    sensitivity_values: list[float] = []
    partition_deltas: list[float] = []
    cap_metric_values_pct: dict[str, list[float]] = {metric: [] for metric in CAP_COVERAGE_METRICS}
    domain_metric_values_pct: dict[str, list[float]] = {metric: [] for metric in CAP_COVERAGE_METRICS}
    primary_domain_values_pct: list[float] = []
    local_total_sd_values: list[float] = []
    local_residual_sd_values: list[float] = []
    local_delta_values: list[float] = []

    def _between_roi_stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean_percent": None, "median_percent": None, "sd_pp": None, "sem_pp": None, "min_percent": None, "max_percent": None}
        array = np.asarray(values, dtype=float)
        sd = float(np.std(array, ddof=1)) if array.size >= 2 else None
        return {
            "count": int(array.size), "mean_percent": _safe_float(float(array.mean())),
            "median_percent": _safe_float(float(np.median(array))), "sd_pp": _safe_float(sd),
            "sem_pp": _safe_float(sd / math.sqrt(array.size)) if sd is not None else None,
            "min_percent": _safe_float(float(array.min())), "max_percent": _safe_float(float(array.max())),
        }

    for image_path in image_paths:
        if results is not None:
            res = results.get(image_path)
            if res is None:
                if not any(item.get("file") == image_path.name for item in failed_images):
                    failed_images.append({"file": image_path.name, "sample": image_path.parent.name, "error": "No result available"})
                continue
        else:
            try:
                res = analyze_coverage_image(image_path, config)
            except Exception as exc:
                failed_images.append(
                    {
                        "file": image_path.name,
                        "sample": image_path.parent.name,
                        "error": str(exc),
                    }
                )
                continue
        rois = []
        included_roi_count = 0
        for roi in res.roi_results:
            include_in_global = _include_roi_in_global_summary(roi, config)
            inscribed = maximum_inscribed_circle(roi.bead_mask)
            pixel_size_m = res.metadata.mean_pixel_size_m
            rois.append(
                {
                    "roi_index": roi.roi_index,
                    "included_in_global_summary": bool(include_in_global),
                    "coverage": _safe_float(roi.coverage),
                    "coverage_percent": _safe_float(roi.coverage_percent),
                    "legacy_full_projected_coverage_percent": _safe_float(roi.legacy_full_projected_coverage_percent),
                    "cap_projected_coverage_percent": _safe_float(roi.cap_projected_coverage_percent),
                    "cap_projected_fraction": _safe_float(roi.cap_projected_coverage),
                    "cap_projected_fraction_percent": _safe_float(roi.cap_projected_coverage_percent),
                    "cap_surface_weighted_coverage_percent": _safe_float(roi.cap_surface_weighted_coverage_percent),
                    "cap_surface_weighted_fraction": _safe_float(roi.cap_surface_weighted_coverage),
                    "cap_surface_weighted_fraction_percent": _safe_float(roi.cap_surface_weighted_coverage_percent),
                    "cap_projected_over_surface_coverage_percent": _safe_float(roi.cap_projected_over_surface_coverage_percent),
                    "cap_projected_over_cap_surface": _safe_float(roi.cap_projected_over_surface_coverage),
                    "cap_projected_over_cap_surface_percent": _safe_float(roi.cap_projected_over_surface_coverage_percent),
                    "selected_coverage_percent": _safe_float(roi.coverage_percent),
                    "selected_coverage_method": roi.selected_coverage_method,
                    "projected_ag_count": int(roi.projected_ag_count),
                    "sphere_ag_count_est": _safe_float(roi.sphere_ag_count_est),
                    "sphere_np_density_per_um2": _safe_float(roi.sphere_np_density_per_um2),
                    "ag_threshold": _safe_float(roi.ag_threshold),
                    "ag_count_threshold": _safe_float(roi.ag_count_threshold),
                    "ag_peak_threshold": _safe_float(roi.ag_peak_threshold),
                    "ag_coverage_threshold": _safe_float(roi.ag_coverage_threshold),
                    "bead_area_px": int(roi.bead_area_px),
                    "ag_area_px": int(roi.ag_area_px),
                    "bead_eq_diameter_m": _safe_float(roi.bead_metrics.equivalent_diameter_m),
                    "diam_xy_mean_um": _safe_float(((roi.bead_metrics.x_diameter_m + roi.bead_metrics.y_diameter_m) / 2.0 * 1e6) if roi.bead_metrics.x_diameter_m is not None and roi.bead_metrics.y_diameter_m is not None else None),
                    "diam_eq_um": _safe_float(roi.bead_metrics.equivalent_diameter_m * 1e6) if roi.bead_metrics.equivalent_diameter_m is not None else None,
                    "diam_inscribed_um": _safe_float(2.0 * inscribed.radius_px * pixel_size_m * 1e6) if pixel_size_m is not None else None,
                    "selected_sphere_diameter_um": _safe_float(roi.bead_metrics.sphere_diameter_m * 1e6) if roi.bead_metrics.sphere_diameter_m is not None else None,
                    "sphere_diameter_px": _safe_float(roi.bead_metrics.sphere_diameter_px),
                    "sphere_diameter_m": _safe_float(roi.bead_metrics.sphere_diameter_m),
                    "sphere_radius_px": _safe_float(roi.bead_metrics.sphere_radius_px),
                    "sphere_radius_m": _safe_float(roi.bead_metrics.sphere_radius_m),
                    "sphere_surface_area_m2": _safe_float(roi.bead_metrics.sphere_surface_area_m2),
                    "sphere_volume_m3": _safe_float(roi.bead_metrics.sphere_volume_m3),
                    "bead_x_diameter_m": _safe_float(roi.bead_metrics.x_diameter_m),
                    "bead_y_diameter_m": _safe_float(roi.bead_metrics.y_diameter_m),
                    "bead_anisotropy_ratio": _safe_float(roi.bead_metrics.anisotropy_ratio),
                    "bead_solidity": _safe_float(roi.bead_metrics.solidity),
                    "coverage_cap_enabled": bool(config.coverage_cap_enabled),
                    "coverage_cap_radius_fraction": float(config.coverage_cap_radius_fraction),
                    "sphere_diameter_metric": config.sphere_diameter_metric,
                    "selected_cap_coverage_metric": config.selected_cap_coverage_metric,
                    "primary_cap_coverage_fraction": _safe_float(roi.cap_metrics.selected_value(config.selected_cap_coverage_metric)) if roi.cap_metrics else None,
                    "primary_cap_coverage_percent": _safe_float(roi.cap_metrics.selected_value(config.selected_cap_coverage_metric) * 100.0) if roi.cap_metrics and roi.cap_metrics.selected_value(config.selected_cap_coverage_metric) is not None else None,
                    "coverage_cap_projected_radius_px": _safe_float(roi.cap_metrics.geometry.cap_radius_px) if roi.cap_metrics else None,
                    "coverage_cap_projected_radius_m": _safe_float(roi.cap_metrics.cap_radius_m) if roi.cap_metrics else None,
                    "coverage_cap_half_angle_deg": _safe_float(roi.cap_metrics.geometry.half_angle_deg) if roi.cap_metrics else None,
                    "coverage_cap_height_m": _safe_float(roi.cap_metrics.height_m) if roi.cap_metrics else None,
                    "coverage_cap_projected_area_m2": _safe_float(roi.cap_metrics.projected_area_m2) if roi.cap_metrics else None,
                    "coverage_cap_surface_area_m2": _safe_float(roi.cap_metrics.surface_area_m2) if roi.cap_metrics else None,
                    "coverage_cap_completeness": _safe_float(roi.cap_metrics.geometry.completeness) if roi.cap_metrics else None,
                    "coverage_cap_valid": bool(roi.cap_metrics.valid) if roi.cap_metrics else False,
                    "polar_rotation_samples": int(config.polar_rotation_samples),
                    "polar_sector_count": int(config.polar_sector_count),
                    **_cap_sensitivity_record(roi.cap_sensitivity, config.selected_cap_coverage_metric),
                    **_homogeneity_record(roi.homogeneity, config.selected_cap_coverage_metric),
                    **_local_heterogeneity_record(roi.local_heterogeneity, config.selected_cap_coverage_metric),
                }
            )
            if not include_in_global:
                continue
            included_roi_count += 1
            coverage_vals.append(float(roi.coverage))
            coverage_pct_vals.append(float(roi.coverage_percent))
            projected_counts.append(int(roi.projected_ag_count))
            sphere_counts.append(float(roi.sphere_ag_count_est))
            if roi.sphere_np_density_per_um2 is not None:
                sphere_densities.append(float(roi.sphere_np_density_per_um2))
            if roi.bead_metrics.equivalent_diameter_m is not None:
                bead_diameters.append(float(roi.bead_metrics.equivalent_diameter_m))
            if roi.cap_metrics and roi.cap_metrics.valid:
                primary = roi.cap_metrics.selected_value(config.selected_cap_coverage_metric)
                if primary is not None:
                    primary_cap_coverages_pct.append(float(primary * 100.0))
                for metric in CAP_COVERAGE_METRICS:
                    value = roi.cap_metrics.selected_value(metric)
                    if value is not None:
                        cap_metric_values_pct[metric].append(float(value * 100.0))
            if roi.local_heterogeneity and roi.local_heterogeneity.domain.valid:
                local_result = roi.local_heterogeneity
                selected_domain = local_result.domain.components(config.selected_cap_coverage_metric).value
                if selected_domain is not None:
                    primary_domain_values_pct.append(float(selected_domain * 100.0))
                for metric in CAP_COVERAGE_METRICS:
                    value = local_result.domain.components(metric).value
                    if value is not None:
                        domain_metric_values_pct[metric].append(float(value * 100.0))
                selected_local = local_result.metrics[config.selected_cap_coverage_metric]
                if selected_local.total_weighted_sd_pp is not None:
                    local_total_sd_values.append(float(selected_local.total_weighted_sd_pp))
                if selected_local.residual_weighted_sd_pp is not None:
                    local_residual_sd_values.append(float(selected_local.residual_weighted_sd_pp))
                if selected_local.reconstruction_delta_pp is not None:
                    local_delta_values.append(float(selected_local.reconstruction_delta_pp))
            hom_record = _homogeneity_record(roi.homogeneity, config.selected_cap_coverage_metric)
            for source, target in (("radial_weighted_sd_pp", radial_sd_values), ("polar_rotation_sd_median_pp", polar_sd_values), ("radial_partition_delta_pp", partition_deltas), ("polar_partition_delta_pp", partition_deltas)):
                value = hom_record.get(source)
                if value is not None:
                    target.append(float(value))
            sensitivity = roi.cap_sensitivity.get(config.selected_cap_coverage_metric)
            if sensitivity and sensitivity.q10_q90_half_width_pp is not None:
                sensitivity_values.append(float(sensitivity.q10_q90_half_width_pp))

        image_summary: dict[str, object] = {}
        for prefix, key in (("primary_cap", "primary_cap_coverage_percent"), ("primary_homogeneity_domain", "primary_homogeneity_domain_coverage_percent")):
            image_summary.update({f"{prefix}_{name}": value for name, value in _between_roi_stats([
                float(roi[key]) for roi in rois if roi.get("included_in_global_summary") and roi.get(key) is not None and (prefix != "primary_homogeneity_domain" or roi.get("homogeneity_domain_valid"))
            ]).items()})
        for metric in CAP_COVERAGE_METRICS:
            for prefix, key in (("cap", f"cap_{metric}_percent"), ("homogeneity_domain", f"homogeneity_domain_{metric}_percent")):
                image_summary.update({f"{prefix}_{metric}_{name}": value for name, value in _between_roi_stats([
                    float(roi[key]) for roi in rois if roi.get("included_in_global_summary") and roi.get(key) is not None and (prefix != "homogeneity_domain" or roi.get("homogeneity_domain_valid"))
                ]).items()})
        images.append(
            {
                "file": image_path.name,
                "sample": image_path.parent.name,
                "date": res.metadata.date,
                "time": res.metadata.time,
                "device": res.metadata.device,
                "magnification": _safe_float(res.metadata.magnification),
                "pixel_size_m": _safe_float(res.metadata.mean_pixel_size_m),
                "roi_count": len(rois),
                "included_roi_count": included_roi_count,
                "excluded_roi_count": len(rois) - included_roi_count,
                "image_summary": image_summary,
                "rois": rois,
            }
        )

    def _mean(values: list[float]) -> Optional[float]:
        return _safe_float(float(np.mean(values))) if values else None

    def _std(values: list[float]) -> Optional[float]:
        return _safe_float(float(np.std(values, ddof=1))) if len(values) >= 2 else None

    def _median(values: list[float]) -> Optional[float]:
        return _safe_float(float(np.median(values))) if values else None

    def _sem(values: list[float]) -> Optional[float]:
        return _safe_float(float(np.std(values, ddof=1) / math.sqrt(len(values)))) if len(values) >= 2 else None

    return {
        "folder": str(folder),
        "file": file,
        "viewer_config": asdict(config),
        "global_summary": {
            "image_count": len(images),
            "failed_image_count": len(failed_images),
            "input_image_count": len(image_paths),
            "total_roi_count": sum(image["roi_count"] for image in images),
            "included_roi_count": sum(image["included_roi_count"] for image in images),
            "sphere_anisotropy_check": bool(config.sphere_anisotropy_check),
            "max_global_sphere_anisotropy_ratio": float(config.max_global_sphere_anisotropy_ratio),
            "sphere_solidity_check": bool(config.sphere_solidity_check),
            "min_global_sphere_solidity": float(config.min_global_sphere_solidity),
            "mean_coverage": _mean(coverage_vals),
            "sd_coverage": _std(coverage_vals),
            "mean_coverage_percent": _mean(coverage_pct_vals),
            "sd_coverage_percent": _std(coverage_pct_vals),
            "median_coverage_percent": _median(coverage_pct_vals),
            "mean_projected_ag_count": _mean(projected_counts),
            "sd_projected_ag_count": _std(projected_counts),
            "mean_sphere_ag_count_est": _mean(sphere_counts),
            "sd_sphere_ag_count_est": _std(sphere_counts),
            "mean_sphere_np_density_per_um2": _mean(sphere_densities),
            "sd_sphere_np_density_per_um2": _std(sphere_densities),
            "mean_bead_eq_diameter_m": _mean(bead_diameters),
            "sd_bead_eq_diameter_m": _std(bead_diameters),
            "selected_cap_coverage_metric": config.selected_cap_coverage_metric,
            "coverage_cap_radius_fraction": float(config.coverage_cap_radius_fraction),
            "primary_cap_coverage_count": len(primary_cap_coverages_pct),
            "primary_cap_coverage_mean_percent": _mean(primary_cap_coverages_pct),
            "primary_cap_coverage_median_percent": _median(primary_cap_coverages_pct),
            "primary_cap_coverage_sd_pp": _std(primary_cap_coverages_pct),
            "primary_cap_coverage_sem_pp": _sem(primary_cap_coverages_pct),
            "primary_cap_coverage_min_percent": _safe_float(min(primary_cap_coverages_pct)) if primary_cap_coverages_pct else None,
            "primary_cap_coverage_max_percent": _safe_float(max(primary_cap_coverages_pct)) if primary_cap_coverages_pct else None,
            "primary_homogeneity_domain_coverage_count": len(primary_domain_values_pct),
            "primary_homogeneity_domain_coverage_mean_percent": _mean(primary_domain_values_pct),
            "primary_homogeneity_domain_coverage_median_percent": _median(primary_domain_values_pct),
            "primary_homogeneity_domain_coverage_sd_pp": _std(primary_domain_values_pct),
            "primary_homogeneity_domain_coverage_sem_pp": _sem(primary_domain_values_pct),
            "primary_homogeneity_domain_coverage_min_percent": _safe_float(min(primary_domain_values_pct)) if primary_domain_values_pct else None,
            "primary_homogeneity_domain_coverage_max_percent": _safe_float(max(primary_domain_values_pct)) if primary_domain_values_pct else None,
            "local_total_heterogeneity_mean_sd_pp": _mean(local_total_sd_values),
            "local_residual_heterogeneity_mean_sd_pp": _mean(local_residual_sd_values),
            "local_grid_reconstruction_delta_mean_pp": _mean(local_delta_values),
            "radial_homogeneity_count": len(radial_sd_values),
            "radial_homogeneity_mean_sd_pp": _mean(radial_sd_values),
            "radial_homogeneity_median_sd_pp": _median(radial_sd_values),
            "radial_homogeneity_sd_of_sd_pp": _std(radial_sd_values),
            "polar_homogeneity_count": len(polar_sd_values),
            "polar_homogeneity_mean_sd_pp": _mean(polar_sd_values),
            "polar_homogeneity_median_sd_pp": _median(polar_sd_values),
            "polar_homogeneity_sd_of_sd_pp": _std(polar_sd_values),
            "cap_radius_sensitivity_median_pp": _median(sensitivity_values),
            "cap_radius_sensitivity_max_pp": _safe_float(max(sensitivity_values)) if sensitivity_values else None,
            "partition_qc_max_delta_pp": _safe_float(max(partition_deltas)) if partition_deltas else None,
            "projected_fraction_coverage_count": len(cap_metric_values_pct["projected_fraction"]),
            "projected_fraction_coverage_mean_percent": _mean(cap_metric_values_pct["projected_fraction"]),
            "projected_fraction_coverage_median_percent": _median(cap_metric_values_pct["projected_fraction"]),
            "projected_fraction_coverage_sd_pp": _std(cap_metric_values_pct["projected_fraction"]),
            "surface_weighted_fraction_coverage_count": len(cap_metric_values_pct["surface_weighted_fraction"]),
            "surface_weighted_fraction_coverage_mean_percent": _mean(cap_metric_values_pct["surface_weighted_fraction"]),
            "surface_weighted_fraction_coverage_median_percent": _median(cap_metric_values_pct["surface_weighted_fraction"]),
            "surface_weighted_fraction_coverage_sd_pp": _std(cap_metric_values_pct["surface_weighted_fraction"]),
            "projected_over_cap_surface_coverage_count": len(cap_metric_values_pct["projected_over_cap_surface"]),
            "projected_over_cap_surface_coverage_mean_percent": _mean(cap_metric_values_pct["projected_over_cap_surface"]),
            "projected_over_cap_surface_coverage_median_percent": _median(cap_metric_values_pct["projected_over_cap_surface"]),
            "projected_over_cap_surface_coverage_sd_pp": _std(cap_metric_values_pct["projected_over_cap_surface"]),
            **{
                f"homogeneity_domain_{metric}_coverage_{name}": value
                for metric, values in domain_metric_values_pct.items()
                for name, value in {
                    "count": len(values), "mean_percent": _mean(values), "median_percent": _median(values),
                    "sd_pp": _std(values), "sem_pp": _sem(values),
                    "min_percent": _safe_float(min(values)) if values else None,
                    "max_percent": _safe_float(max(values)) if values else None,
                }.items()
            },
        },
        "images": images,
        "failed_images": failed_images,
    }


def build_coverage_image_record(
    image_path: str | Path,
    config: CoverageViewerConfig,
    result: CoverageImageResult,
) -> dict[str, object]:
    """Serialize one already-analysed image without retaining its arrays.

    This deliberately reuses the established summary serializer so the
    streaming batch path cannot drift from interactive/single-image JSON
    output.  The singleton ``results`` mapping exists only for this call and
    is discarded before the next image is analysed.
    """

    path = Path(image_path)
    summary = build_coverage_summary(
        path.parent,
        config,
        file=path.name,
        results={path: result},
    )
    images = summary.get("images", [])
    if len(images) != 1:
        raise RuntimeError(f"Could not serialize coverage result for '{path}'.")
    return images[0]


def _serializable_roi_values(
    images: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        roi
        for image in images
        for roi in image.get("rois", [])
        if isinstance(roi, dict) and roi.get("included_in_global_summary")
    ]


def build_coverage_summary_from_records(
    folder: str | Path,
    config: CoverageViewerConfig,
    *,
    image_paths: list[Path] | tuple[Path, ...],
    image_records: list[dict[str, object]] | tuple[dict[str, object], ...],
    failures: list[dict] | tuple[dict, ...] = (),
    file: Optional[str] = None,
) -> dict:
    """Finalize one sample from lightweight JSON-compatible image records.

    Sample statistics are computed from the retained ROI records themselves,
    not by averaging the image summaries.  Consequently the streaming batch
    has the same between-bead statistical unit as :func:`build_coverage_summary`
    while no :class:`CoverageImageResult` needs to survive finalization.
    """

    folder = Path(folder)
    images = list(image_records)
    failed_images = list(failures)
    included_rois = _serializable_roi_values(images)

    def values(key: str, *, predicate=lambda _roi: True) -> list[float]:
        output: list[float] = []
        for roi in included_rois:
            value = roi.get(key)
            if value is not None and predicate(roi):
                output.append(float(value))
        return output

    def mean(items: list[float]) -> Optional[float]:
        return _safe_float(float(np.mean(items))) if items else None

    def std(items: list[float]) -> Optional[float]:
        return _safe_float(float(np.std(items, ddof=1))) if len(items) >= 2 else None

    def median(items: list[float]) -> Optional[float]:
        return _safe_float(float(np.median(items))) if items else None

    def sem(items: list[float]) -> Optional[float]:
        return _safe_float(float(np.std(items, ddof=1) / math.sqrt(len(items)))) if len(items) >= 2 else None

    cap_valid = lambda roi: bool(roi.get("coverage_cap_valid"))
    domain_valid = lambda roi: bool(roi.get("homogeneity_domain_valid"))
    coverage_vals = values("coverage")
    coverage_pct_vals = values("coverage_percent")
    projected_counts = values("projected_ag_count")
    sphere_counts = values("sphere_ag_count_est")
    sphere_densities = values("sphere_np_density_per_um2")
    bead_diameters = values("bead_eq_diameter_m")
    primary_cap_coverages_pct = values("primary_cap_coverage_percent", predicate=cap_valid)
    primary_domain_values_pct = values("primary_homogeneity_domain_coverage_percent", predicate=domain_valid)
    local_total_sd_values = values("local_total_weighted_sd_pp", predicate=domain_valid)
    local_residual_sd_values = values("local_residual_weighted_sd_pp", predicate=domain_valid)
    local_delta_values = values("primary_local_grid_reconstruction_delta_pp", predicate=domain_valid)
    radial_sd_values = values("radial_weighted_sd_pp")
    polar_sd_values = values("polar_rotation_sd_median_pp")
    sensitivity_values = values("primary_cap_radius_sensitivity_q10_q90_half_width_pp")
    partition_deltas = values("radial_partition_delta_pp") + values("polar_partition_delta_pp")
    cap_metric_values_pct = {
        metric: values(f"cap_{metric}_percent", predicate=cap_valid)
        for metric in CAP_COVERAGE_METRICS
    }
    domain_metric_values_pct = {
        metric: values(f"homogeneity_domain_{metric}_percent", predicate=domain_valid)
        for metric in CAP_COVERAGE_METRICS
    }

    global_summary: dict[str, object] = {
        "image_count": len(images),
        "failed_image_count": len(failed_images),
        "input_image_count": len(image_paths),
        "total_roi_count": sum(int(image.get("roi_count", 0)) for image in images),
        "included_roi_count": sum(int(image.get("included_roi_count", 0)) for image in images),
        "sphere_anisotropy_check": bool(config.sphere_anisotropy_check),
        "max_global_sphere_anisotropy_ratio": float(config.max_global_sphere_anisotropy_ratio),
        "sphere_solidity_check": bool(config.sphere_solidity_check),
        "min_global_sphere_solidity": float(config.min_global_sphere_solidity),
        "mean_coverage": mean(coverage_vals),
        "sd_coverage": std(coverage_vals),
        "mean_coverage_percent": mean(coverage_pct_vals),
        "sd_coverage_percent": std(coverage_pct_vals),
        "median_coverage_percent": median(coverage_pct_vals),
        "mean_projected_ag_count": mean(projected_counts),
        "sd_projected_ag_count": std(projected_counts),
        "mean_sphere_ag_count_est": mean(sphere_counts),
        "sd_sphere_ag_count_est": std(sphere_counts),
        "mean_sphere_np_density_per_um2": mean(sphere_densities),
        "sd_sphere_np_density_per_um2": std(sphere_densities),
        "mean_bead_eq_diameter_m": mean(bead_diameters),
        "sd_bead_eq_diameter_m": std(bead_diameters),
        "selected_cap_coverage_metric": config.selected_cap_coverage_metric,
        "coverage_cap_radius_fraction": float(config.coverage_cap_radius_fraction),
        "primary_cap_coverage_count": len(primary_cap_coverages_pct),
        "primary_cap_coverage_mean_percent": mean(primary_cap_coverages_pct),
        "primary_cap_coverage_median_percent": median(primary_cap_coverages_pct),
        "primary_cap_coverage_sd_pp": std(primary_cap_coverages_pct),
        "primary_cap_coverage_sem_pp": sem(primary_cap_coverages_pct),
        "primary_cap_coverage_min_percent": _safe_float(min(primary_cap_coverages_pct)) if primary_cap_coverages_pct else None,
        "primary_cap_coverage_max_percent": _safe_float(max(primary_cap_coverages_pct)) if primary_cap_coverages_pct else None,
        "primary_homogeneity_domain_coverage_count": len(primary_domain_values_pct),
        "primary_homogeneity_domain_coverage_mean_percent": mean(primary_domain_values_pct),
        "primary_homogeneity_domain_coverage_median_percent": median(primary_domain_values_pct),
        "primary_homogeneity_domain_coverage_sd_pp": std(primary_domain_values_pct),
        "primary_homogeneity_domain_coverage_sem_pp": sem(primary_domain_values_pct),
        "primary_homogeneity_domain_coverage_min_percent": _safe_float(min(primary_domain_values_pct)) if primary_domain_values_pct else None,
        "primary_homogeneity_domain_coverage_max_percent": _safe_float(max(primary_domain_values_pct)) if primary_domain_values_pct else None,
        "local_total_heterogeneity_mean_sd_pp": mean(local_total_sd_values),
        "local_residual_heterogeneity_mean_sd_pp": mean(local_residual_sd_values),
        "local_grid_reconstruction_delta_mean_pp": mean(local_delta_values),
        "radial_homogeneity_count": len(radial_sd_values),
        "radial_homogeneity_mean_sd_pp": mean(radial_sd_values),
        "radial_homogeneity_median_sd_pp": median(radial_sd_values),
        "radial_homogeneity_sd_of_sd_pp": std(radial_sd_values),
        "polar_homogeneity_count": len(polar_sd_values),
        "polar_homogeneity_mean_sd_pp": mean(polar_sd_values),
        "polar_homogeneity_median_sd_pp": median(polar_sd_values),
        "polar_homogeneity_sd_of_sd_pp": std(polar_sd_values),
        "cap_radius_sensitivity_median_pp": median(sensitivity_values),
        "cap_radius_sensitivity_max_pp": _safe_float(max(sensitivity_values)) if sensitivity_values else None,
        "partition_qc_max_delta_pp": _safe_float(max(partition_deltas)) if partition_deltas else None,
    }
    for metric in CAP_COVERAGE_METRICS:
        cap_values = cap_metric_values_pct[metric]
        global_summary.update({
            f"{metric}_coverage_count": len(cap_values),
            f"{metric}_coverage_mean_percent": mean(cap_values),
            f"{metric}_coverage_median_percent": median(cap_values),
            f"{metric}_coverage_sd_pp": std(cap_values),
        })
        domain_values = domain_metric_values_pct[metric]
        global_summary.update({
            f"homogeneity_domain_{metric}_coverage_count": len(domain_values),
            f"homogeneity_domain_{metric}_coverage_mean_percent": mean(domain_values),
            f"homogeneity_domain_{metric}_coverage_median_percent": median(domain_values),
            f"homogeneity_domain_{metric}_coverage_sd_pp": std(domain_values),
            f"homogeneity_domain_{metric}_coverage_sem_pp": sem(domain_values),
            f"homogeneity_domain_{metric}_coverage_min_percent": _safe_float(min(domain_values)) if domain_values else None,
            f"homogeneity_domain_{metric}_coverage_max_percent": _safe_float(max(domain_values)) if domain_values else None,
        })

    return {
        "folder": str(folder),
        "file": file,
        "viewer_config": asdict(config),
        "global_summary": global_summary,
        "images": images,
        "failed_images": failed_images,
    }


def write_coverage_summary_json(folder: str | Path, config: CoverageViewerConfig, output_path: str | Path, file: Optional[str] = None) -> None:
    summary = build_coverage_summary(folder, config, file)
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


class CoverageDatasetViewer:
    def __init__(self, folder: str | Path, config: CoverageViewerConfig, file: Optional[str] = None):
        self.folder = Path(folder)
        self.file = file
        self.config = config
        self.image_paths = _resolve_image_paths(self.folder, self.file)
        if not self.image_paths:
            raise FileNotFoundError(f"No TIFF files found in '{self.folder}'.")

        self.index = 0
        self._cache: dict[Path, CoverageImageResult] = {}
        self._error_cache: dict[Path, str] = {}
        self._failed_preview_cache: dict[Path, FailedImagePreview] = {}
        self.show_scale = config.default_show_scale
        self.show_bead_boundary = config.default_show_bead_boundary
        self.show_diameter_lines = config.default_show_diameter_lines
        self.show_ag_boundary = config.default_show_ag_boundary
        self.show_ag_count_boundary = config.default_show_ag_count_boundary
        self.show_ag_peaks = config.default_show_ag_peaks
        self.view_modes = ["display", "norm", "bead_raw", "bead_refined", "ag_count_feature", "ag_coverage_feature"]
        self.view_mode_index = 0

        self.fig = None
        self.ax_image = None
        self.ax_info = None
        self.image_artist = None
        self.check_buttons: Optional[CheckButtons] = None
        self.overlay_artists: list[object] = []

    def _get_result(self, index: int) -> Optional[CoverageImageResult]:
        path = self.image_paths[index]
        if path in self._error_cache:
            return None
        if path not in self._cache:
            try:
                self._cache[path] = analyze_coverage_image(path, self.config)
            except Exception as exc:
                self._error_cache[path] = str(exc)
                return None
        return self._cache[path]

    def _get_error(self, index: int) -> Optional[str]:
        return self._error_cache.get(self.image_paths[index])

    def _get_failed_preview(self, index: int) -> Optional[FailedImagePreview]:
        path = self.image_paths[index]
        if path not in self._failed_preview_cache:
            try:
                self._failed_preview_cache[path] = load_failed_image_preview(path, self.config)
            except Exception:
                return None
        return self._failed_preview_cache[path]

    def _scale_feature_image(self, img: np.ndarray) -> np.ndarray:
        vals = img[np.isfinite(img)]
        if vals.size == 0 or float(vals.max()) <= float(vals.min()) + 1e-12:
            return np.zeros(img.shape, dtype=np.float32)
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.5))
        hi = max(hi, lo + 1e-6)
        out = (img.astype(np.float32) - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)

    def _base_gray(self, res: CoverageImageResult) -> np.ndarray:
        mode = self.view_modes[self.view_mode_index]
        if mode == "norm":
            return np.clip(res.norm.astype(np.float32), 0.0, 1.0)
        if mode == "bead_raw":
            return res.bead_raw_union.astype(np.float32)
        if mode == "bead_refined":
            return res.bead_refined_union.astype(np.float32)
        if mode == "ag_count_feature":
            return self._scale_feature_image(res.ag_count_feature_union)
        if mode == "ag_coverage_feature":
            return self._scale_feature_image(res.ag_coverage_feature_union)
        return res.display.astype(np.float32)

    def _failed_preview_image(self, preview: FailedImagePreview) -> np.ndarray:
        mode = self.view_modes[self.view_mode_index]
        if mode == "norm":
            base_gray = np.clip(preview.norm.astype(np.float32), 0.0, 1.0)
        else:
            base_gray = preview.display.astype(np.float32)
        return np.dstack([base_gray, base_gray, base_gray]).astype(np.float32)

    def _set_image_data(self, img: np.ndarray) -> None:
        self.image_artist.set_data(img)
        h, w = img.shape[:2]
        self.ax_image.set_xlim(-0.5, w - 0.5)
        self.ax_image.set_ylim(h - 0.5, -0.5)

    def _display_image(self, res: CoverageImageResult) -> np.ndarray:
        base_gray = self._base_gray(res)
        base = np.dstack([base_gray, base_gray, base_gray]).astype(np.float32)
        if self.show_bead_boundary:
            for roi in res.roi_results:
                include_in_global = _include_roi_in_global_summary(roi, res.config)
                color = (0.0, 1.0, 0.0) if include_in_global else (1.0, 0.0, 0.0)
                base[find_boundaries(roi.bead_mask, mode="outer")] = color
        if self.show_ag_boundary:
            ag_union = np.zeros(res.display.shape, dtype=bool)
            for roi in res.roi_results:
                ag_union |= roi.ag_mask
            base[find_boundaries(ag_union, mode="outer")] = (1.0, 0.0, 0.0)
        if self.show_ag_count_boundary:
            ag_count_union = np.zeros(res.display.shape, dtype=bool)
            for roi in res.roi_results:
                ag_count_union |= roi.ag_count_mask
            base[find_boundaries(ag_count_union, mode="outer")] = (1.0, 1.0, 0.0)
        return base

    def _clear_overlays(self) -> None:
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.overlay_artists.clear()

    def _make_scale_overlay(self, res: CoverageImageResult) -> list[object]:
        pixel_size_m = res.metadata.mean_pixel_size_m
        if pixel_size_m is None:
            return []
        h, w = res.display.shape
        scale_length_m = _nice_scale_length_m(w * pixel_size_m * 0.22)
        scale_length_px = scale_length_m / pixel_size_m
        x0 = w * 0.06
        y0 = h * 0.92
        label_artist = self.ax_image.text(x0, y0 - 11, _format_length_m(scale_length_m), color="white", fontsize=10, va="bottom", ha="left")
        return [
            Rectangle((x0 - 8, y0 - 28), scale_length_px + 16, 36, facecolor=(0.0, 0.0, 0.0, 0.35), edgecolor="none"),
            Line2D([x0, x0 + scale_length_px], [y0, y0], color="white", linewidth=3),
            Line2D([x0, x0], [y0 - 7, y0 + 7], color="white", linewidth=1.5),
            Line2D([x0 + scale_length_px, x0 + scale_length_px], [y0 - 7, y0 + 7], color="white", linewidth=1.5),
            label_artist,
        ]

    def _make_diameter_overlay(self, res: CoverageImageResult) -> list[object]:
        artists: list[object] = []
        for roi in res.roi_results:
            m = roi.bead_metrics
            include_in_global = _include_roi_in_global_summary(roi, res.config)
            color = "cyan" if include_in_global else "red"
            row, col = m.centroid_rc
            x_half = m.x_diameter_px / 2.0
            y_half = m.y_diameter_px / 2.0
            text = self.ax_image.text(
                col,
                row - y_half - 8,
                f"x={_format_px_or_length(m.x_diameter_m, m.x_diameter_px)}  y={_format_px_or_length(m.y_diameter_m, m.y_diameter_px)}",
                color=color,
                fontsize=8,
                ha="center",
                va="bottom",
                bbox={"facecolor": (0.0, 0.0, 0.0, 0.45), "edgecolor": "none", "pad": 1.5},
            )
            artists.extend(
                [
                    Line2D([col - x_half, col + x_half], [row, row], color=color, linewidth=1.2),
                    Line2D([col, col], [row - y_half, row + y_half], color=color, linewidth=1.2),
                    text,
                ]
            )
        return artists

    def _make_peak_overlay(self, res: CoverageImageResult) -> list[object]:
        artists: list[object] = []
        for roi in res.roi_results:
            if roi.ag_peak_coords.size:
                artists.append(self.ax_image.plot(roi.ag_peak_coords[:, 1], roi.ag_peak_coords[:, 0], "c.", markersize=4)[0])
        return artists

    def _update_info(self, res: Optional[CoverageImageResult], image_path: Path, error: Optional[str] = None) -> None:
        self.ax_info.clear()
        self.ax_info.axis("off")
        mode_label = self.view_modes[self.view_mode_index]
        if res is None:
            lines = [
                f"File: {image_path.name}",
                f"Sample: {image_path.parent.name}",
                f"Mode: {mode_label}",
                "Status: failed",
                "",
                error or "Unknown analysis error.",
                "",
                "Showing cropped detector preview when available.",
                "",
                "This frame was skipped during summary generation.",
                "Tune ROI filters or split thresholds if needed.",
            ]
            self.ax_info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace", transform=self.ax_info.transAxes)
            return
        lines = [
            f"File: {res.image_path.name}",
            f"Sample: {res.image_path.parent.name}",
            f"Mode: {mode_label}",
            f"ROIs: {len(res.roi_results)}",
            "",
        ]
        for roi in res.roi_results:
            m = roi.bead_metrics
            include_in_global = _include_roi_in_global_summary(roi, res.config)
            lines.extend(
                [
                    f"ROI {roi.roi_index}",
                    f"  Global summary: {'yes' if include_in_global else 'no'}",
                    f"  Coverage: {roi.coverage:.4f} ({roi.coverage_percent:.2f}%)",
                    f"  NP count image: {roi.projected_ag_count}",
                    f"  NP count sphere est: {roi.sphere_ag_count_est:.1f}",
                    f"  NP density sphere: {roi.sphere_np_density_per_um2:.2f} / um^2" if roi.sphere_np_density_per_um2 is not None else "  NP density sphere: n/a",
                    f"  Bead eq: {_format_px_or_length(m.equivalent_diameter_m, m.equivalent_diameter_px)}",
                    f"  Bead x/y: {_format_px_or_length(m.x_diameter_m, m.x_diameter_px)} / {_format_px_or_length(m.y_diameter_m, m.y_diameter_px)}",
                    f"  Bead anisotropy: {m.anisotropy_ratio:.3f}",
                    f"  Bead solidity: {m.solidity:.3f}",
                    f"  Ag primary mask thr: {roi.ag_count_threshold:.4f}",
                    f"  Ag peak thr: {roi.ag_peak_threshold:.4f}",
                    (
                        f"  Ag secondary coverage thr: {roi.ag_coverage_threshold:.4f}"
                        if self.config.ag_enable_secondary_coverage
                        else "  Ag coverage mask: primary mask"
                    ),
                    "",
                ]
            )
        if res.metadata.mean_pixel_size_m:
            lines.append(f"Pixel size: {_format_length_m(res.metadata.mean_pixel_size_m)} / px")
        if res.metadata.device:
            lines.append(f"Instrument: {res.metadata.device}")
        if res.metadata.magnification:
            lines.append(f"Magnification: {res.metadata.magnification:.0f}x")
        if res.metadata.date or res.metadata.time:
            lines.append(f"Acquired: {res.metadata.date} {res.metadata.time}".strip())
        self.ax_info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace", transform=self.ax_info.transAxes)

    def _render_current(self) -> None:
        image_path = self.image_paths[self.index]
        res = self._get_result(self.index)
        if res is None:
            preview = self._get_failed_preview(self.index)
            img = self._failed_preview_image(preview) if preview is not None else np.zeros((512, 512, 3), dtype=np.float32)
            self._set_image_data(img)
            title_suffix = "failed, showing cropped preview" if preview is not None else "failed"
            self.ax_image.set_title(f"{self.index + 1}/{len(self.image_paths)}  {image_path.name}  [{self.view_modes[self.view_mode_index]}]  [{title_suffix}]", fontsize=11)
            self._clear_overlays()
            self._update_info(None, image_path, self._get_error(self.index))
            self.fig.canvas.draw_idle()
            return
        self._set_image_data(self._display_image(res))
        self.ax_image.set_title(f"{self.index + 1}/{len(self.image_paths)}  {res.image_path.name}  [{self.view_modes[self.view_mode_index]}]", fontsize=11)
        self._clear_overlays()
        if self.show_scale:
            self.overlay_artists.extend(self._make_scale_overlay(res))
        if self.show_diameter_lines:
            self.overlay_artists.extend(self._make_diameter_overlay(res))
        if self.show_ag_peaks:
            self.overlay_artists.extend(self._make_peak_overlay(res))
        for artist in self.overlay_artists:
            if getattr(artist, "axes", None) is None:
                self.ax_image.add_artist(artist)
        self._update_info(res, image_path)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "right":
            self.index = (self.index + 1) % len(self.image_paths)
            self._render_current()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.image_paths)
            self._render_current()
        elif event.key == "up":
            self.view_mode_index = (self.view_mode_index - 1) % len(self.view_modes)
            self._render_current()
        elif event.key == "down":
            self.view_mode_index = (self.view_mode_index + 1) % len(self.view_modes)
            self._render_current()

    def _on_checks(self, _label: str) -> None:
        if self.check_buttons is None:
            return
        status = self.check_buttons.get_status()
        self.show_scale = bool(status[0])
        self.show_bead_boundary = bool(status[1])
        self.show_diameter_lines = bool(status[2])
        self.show_ag_boundary = bool(status[3])
        self.show_ag_count_boundary = bool(status[4])
        self.show_ag_peaks = bool(status[5])
        self._render_current()

    def show(self) -> None:
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_image = self.fig.add_axes([0.04, 0.10, 0.62, 0.80])
        self.ax_info = self.fig.add_axes([0.70, 0.10, 0.20, 0.72])
        ax_checks = self.fig.add_axes([0.91, 0.10, 0.07, 0.40])
        first = self._get_result(self.index)
        if first is not None:
            first_image = self._display_image(first)
        else:
            first_preview = self._get_failed_preview(self.index)
            first_image = self._failed_preview_image(first_preview) if first_preview is not None else np.zeros((512, 512, 3), dtype=np.float32)
        self.image_artist = self.ax_image.imshow(first_image, vmin=0.0, vmax=1.0)
        self.ax_image.axis("off")
        self.ax_image.set_autoscale_on(False)
        ax_checks.set_title("Overlays", fontsize=10)
        self.check_buttons = CheckButtons(
            ax_checks,
            ["Scale", "Bead", "Size", "Ag cov", "Ag count", "Ag peaks"],
            [self.show_scale, self.show_bead_boundary, self.show_diameter_lines, self.show_ag_boundary, self.show_ag_count_boundary, self.show_ag_peaks],
        )
        for text in self.check_buttons.labels:
            text.set_fontsize(10)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.check_buttons.on_clicked(self._on_checks)
        self._render_current()
        self.fig.suptitle("SEM Coverage Viewer  |  left/right = next image  |  up/down = analysis view", fontsize=12)
        plt.show()


def main(config_path: str | Path = "sem_coverage_viewer_config.json") -> None:
    run_from_config(config_path)


DEFAULT_CONFIG_PATH = Path("sem_coverage_viewer_config.json")


def run_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    folder_override: str | Path | None = None,
    file_override: str | Path | None = None,
) -> None:
    """Run the SEM coverage viewer from one config file and temporary overrides."""

    config_path = expand_user_path(config_path)
    if not config_path.exists():
        save_default_config(
            config_path,
            r"C:\Users\pavel\Desktop\AVCR\codes\sem_coverage\testData\100226\PVP 10 kDa, 10x AgNPs",
        )
    app_cfg = load_app_config(config_path)
    effective_folder = (
        expand_user_path(folder_override)
        if folder_override is not None
        else resolve_existing_input_path(
            app_cfg.folder, config_path=config_path, description="image folder"
        )
    )
    effective_file = (
        resolve_optional_file_in_folder(effective_folder, file_override)
        if file_override is not None
        else (
            resolve_optional_file_in_folder(effective_folder, app_cfg.file)
            if app_cfg.file
            else None
        )
    )
    if app_cfg.summary_json_path:
        write_coverage_summary_json(
            effective_folder,
            app_cfg.viewer,
            expand_user_path(app_cfg.summary_json_path),
            None if effective_file is None else str(effective_file),
        )
    CoverageDatasetViewer(
        effective_folder,
        app_cfg.viewer,
        None if effective_file is None else str(effective_file),
    ).show()


if __name__ == "__main__":
    main()

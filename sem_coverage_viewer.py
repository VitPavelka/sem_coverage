from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

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

from sem_coverage import AnalyzerConfig, SEMCoverageAnalyzer, SegmentationError


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
    ag_coverage_closing_radius: int = 1
    ag_coverage_use_union_with_count: bool = True
    default_show_scale: bool = True
    default_show_bead_boundary: bool = True
    default_show_diameter_lines: bool = True
    default_show_ag_boundary: bool = True
    default_show_ag_count_boundary: bool = False
    default_show_ag_peaks: bool = True


@dataclass(frozen=True)
class CoverageAppConfig:
    folder: str
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
    ag_coverage_threshold: float
    coverage: float
    coverage_percent: float
    projected_ag_count: int
    sphere_ag_count_est: float
    sphere_np_density_per_um2: Optional[float]
    bead_area_px: int
    ag_area_px: int
    bead_metrics: BeadMetrics


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


@dataclass(frozen=True)
class FailedImagePreview:
    image_path: Path
    cropped: np.ndarray
    norm: np.ndarray
    display: np.ndarray
    metadata: SEMMetadata
    crop_row: int


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
    viewer_data["analyzer"] = analyzer
    viewer = CoverageViewerConfig(**viewer_data)
    return CoverageAppConfig(
        folder=data["folder"],
        file=data.get("file") or None,
        viewer=viewer,
        summary_json_path=data.get("summary_json_path"),
    )


def save_default_config(config_path: str | Path, folder: str | Path) -> None:
    config = CoverageAppConfig(
        folder=str(folder),
        file=None,
        summary_json_path=str(Path(folder).resolve() / "sem_coverage_viewer_summary.json"),
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
    stats = _region_stats(mask)
    if stats is None:
        return False
    if stats["area"] < float(config.min_bead_area_px):
        return False
    if stats["equivalent_diameter_px"] < float(config.min_roi_eq_diameter_px):
        return False
    if stats["solidity"] < float(config.min_roi_solidity):
        return False
    if stats["anisotropy_ratio"] > float(config.max_roi_anisotropy_ratio):
        return False
    return True


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
) -> list[np.ndarray]:
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

    candidates = []
    for region in sorted(regionprops(lab), key=lambda item: item.area, reverse=True):
        small_component = lab == region.label
        bead = resize(small_component, cropped.shape, order=0, preserve_range=True, anti_aliasing=False).astype(bool)
        erode_radius = int(config.bead_morph_erode_radius_px)
        if erode_radius > 0:
            bead = erosion(bead, disk(erode_radius))
        if _is_valid_roi(bead, config):
            candidates.append(bead)
    if candidates:
        return [candidates[0]]
    raise SegmentationError("Morphology bead fallback failed: no enclosed component passed ROI filters.")


def _refine_bead_components(components: list[np.ndarray], config: CoverageViewerConfig) -> list[np.ndarray]:
    refined: list[np.ndarray] = []
    for component in components:
        candidates = _split_touching_beads(component, config) if _should_try_split(component, config) else [component]
        accepted = False
        for candidate in candidates:
            if _is_valid_roi(candidate, config):
                refined.append(candidate)
                accepted = True
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
        raise SegmentationError(f"Bead segmentation failed: no valid bead-like ROI remained after filtering.{suffix}")
    return refined


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


def _measure_bead(bead_mask: np.ndarray, pixel_size_m: Optional[float]) -> BeadMetrics:
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
    sphere_area = None
    if eqd_m is not None:
        radius_m = eqd_m / 2.0
        sphere_area = float(4.0 * math.pi * radius_m * radius_m)

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
    coverage = analyzer._compute_coverage(bead_mask, ag_mask)
    projected_ag_count = analyzer._count_ag_peaks(count_feat, count_mask, count_thr)
    ag_peak_coords = peak_local_max(
        count_feat,
        labels=count_mask.astype(np.uint8),
        min_distance=int(analyzer.config.count_min_distance),
        threshold_abs=float(count_thr) * float(analyzer.config.count_thr_rel),
        exclude_border=False,
    )
    bead_metrics = _measure_bead(bead_mask, pixel_size_m)
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
        ag_threshold=float(coverage_thr),
        ag_count_threshold=float(count_thr),
        ag_coverage_threshold=float(coverage_thr),
        coverage=float(coverage),
        coverage_percent=float(coverage * 100.0),
        projected_ag_count=int(projected_ag_count),
        sphere_ag_count_est=float(sphere_count_est),
        sphere_np_density_per_um2=_safe_float(density_per_um2),
        bead_area_px=int(bead_mask.sum()),
        ag_area_px=int(ag_mask.sum()),
        bead_metrics=bead_metrics,
    )


def analyze_coverage_image(image_path: str | Path, config: CoverageViewerConfig) -> CoverageImageResult:
    image_path = Path(image_path)
    analyzer = SEMCoverageAnalyzer(config.analyzer)
    raw, cropped, norm, display, metadata, crop_row = _load_preprocessed_image(image_path, analyzer, config)
    try:
        bead_components = _segment_bead_components(analyzer, norm, config.min_bead_area_px)
        bead_raw_union = np.zeros(cropped.shape, dtype=bool)
        for component in bead_components:
            bead_raw_union |= component
        try:
            bead_components = _refine_bead_components(bead_components, config)
        except SegmentationError:
            if not config.bead_morph_fallback:
                raise
            bead_components = _segment_bead_by_morphology(analyzer, cropped, config)
    except SegmentationError:
        if not config.bead_morph_fallback:
            raise
        bead_components = _segment_bead_by_morphology(analyzer, cropped, config)
        bead_raw_union = np.zeros(cropped.shape, dtype=bool)
        for component in bead_components:
            bead_raw_union |= component
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
        image_path = Path(file)
        if not image_path.is_absolute():
            image_path = folder / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Configured TIFF file not found: '{image_path}'.")
        if not image_path.is_file():
            raise FileNotFoundError(f"Configured TIFF path is not a file: '{image_path}'.")
        return [image_path]
    return sorted(folder.glob("*.tif"))


def build_coverage_summary(folder: str | Path, config: CoverageViewerConfig, file: Optional[str] = None) -> dict:
    folder = Path(folder)
    image_paths = _resolve_image_paths(folder, file)
    images = []
    failed_images = []
    coverage_vals = []
    coverage_pct_vals = []
    projected_counts = []
    sphere_counts = []
    sphere_densities = []
    bead_diameters = []

    for image_path in image_paths:
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
            rois.append(
                {
                    "roi_index": roi.roi_index,
                    "included_in_global_summary": bool(include_in_global),
                    "coverage": _safe_float(roi.coverage),
                    "coverage_percent": _safe_float(roi.coverage_percent),
                    "projected_ag_count": int(roi.projected_ag_count),
                    "sphere_ag_count_est": _safe_float(roi.sphere_ag_count_est),
                    "sphere_np_density_per_um2": _safe_float(roi.sphere_np_density_per_um2),
                    "ag_threshold": _safe_float(roi.ag_threshold),
                    "ag_count_threshold": _safe_float(roi.ag_count_threshold),
                    "ag_coverage_threshold": _safe_float(roi.ag_coverage_threshold),
                    "bead_area_px": int(roi.bead_area_px),
                    "ag_area_px": int(roi.ag_area_px),
                    "bead_eq_diameter_m": _safe_float(roi.bead_metrics.equivalent_diameter_m),
                    "bead_x_diameter_m": _safe_float(roi.bead_metrics.x_diameter_m),
                    "bead_y_diameter_m": _safe_float(roi.bead_metrics.y_diameter_m),
                    "bead_anisotropy_ratio": _safe_float(roi.bead_metrics.anisotropy_ratio),
                    "bead_solidity": _safe_float(roi.bead_metrics.solidity),
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
                "rois": rois,
            }
        )

    def _mean(values: list[float]) -> Optional[float]:
        return _safe_float(float(np.mean(values))) if values else None

    def _std(values: list[float]) -> Optional[float]:
        return _safe_float(float(np.std(values, ddof=1))) if len(values) >= 2 else None

    def _median(values: list[float]) -> Optional[float]:
        return _safe_float(float(np.median(values))) if values else None

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
        },
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
                    f"  Ag cov thr: {roi.ag_coverage_threshold:.4f}",
                    f"  Ag count thr: {roi.ag_count_threshold:.4f}",
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
    config_path = Path(config_path)
    if not config_path.exists():
        save_default_config(config_path, r"C:\Users\pavel\Desktop\AVCR\codes\sem_coverage\testData\100226\PVP 10 kDa, 10x AgNPs")
    app_cfg = load_app_config(config_path)
    if app_cfg.summary_json_path:
        write_coverage_summary_json(app_cfg.folder, app_cfg.viewer, app_cfg.summary_json_path, app_cfg.file)
    CoverageDatasetViewer(app_cfg.folder, app_cfg.viewer, app_cfg.file).show()


if __name__ == "__main__":
    main()

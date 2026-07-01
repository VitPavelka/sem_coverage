from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import json
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from scipy import ndimage as ndi
from skimage import io as skio
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu, threshold_sauvola
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening, remove_small_objects, white_tophat
from skimage.segmentation import find_boundaries, watershed

from path_utils import (
    expand_user_path,
    path_to_config_text,
    resolve_existing_input_path,
    resolve_optional_file_in_folder,
)
from tabular_export import sort_paths


log = logging.getLogger(__name__)


class TEMAnalysisError(RuntimeError):
    """Base TEM analysis error."""


class TEMSegmentationError(TEMAnalysisError):
    """Raised when TEM particle segmentation cannot produce usable results."""


@dataclass(frozen=True)
class ViewerConfig:
    strip_rows: Optional[int] = None
    footer_tail_rows: int = 160
    dark_footer_k_mad: float = 6.0
    dark_footer_min_run: int = 12

    display_percentiles: tuple[float, float] = (1.0, 99.5)
    feature_percentiles: tuple[float, float] = (1.0, 99.5)

    pixel_size_nm: Optional[float] = None
    fov_nm: Optional[float] = None

    detector: str = "dog"
    dog_sigma_small: float = 1.2
    dog_sigma_large: float = 6.0
    dog_foreground_percentile: float = 99.0
    intensity_percentile_dark: float = 35.0

    tophat_radius: int = 7
    sauvola_window_size: int = 41
    sauvola_k: float = 0.15

    closing_radius: int = 1
    opening_radius: int = 1
    min_area_px: int = 8
    max_area_px: Optional[int] = None
    max_anisotropy_ratio: float = 3.0
    min_solidity: float = 0.2

    split_touching: bool = True
    split_min_distance: int = 6
    split_threshold_rel: float = 0.35
    split_exclude_border: bool = False

    measurement_mode: str = "mask_chords"
    measure_step_px: float = 0.5
    histogram_metric: str = "mean_axes"
    major_axis_color: str = "cyan"
    minor_axis_color: str = "orange"

    default_show_scale: bool = True
    default_show_boundaries: bool = True
    default_show_measures: bool = True
    default_show_histogram: bool = True


@dataclass(frozen=True)
class AppConfig:
    folder: str
    file: Optional[str] = None
    summary_json_path: Optional[str] = None
    viewer: ViewerConfig = ViewerConfig()


@dataclass(frozen=True)
class TEMMetadata:
    pixel_size_nm: Optional[float]
    fov_nm: Optional[float]
    image_width_px: int
    image_height_px: int
    crop_row: int
    detector: str


@dataclass(frozen=True)
class TEMParticleMeasurement:
    label_id: int
    centroid_rc: tuple[float, float]
    area_px: int
    equivalent_diameter_px: float
    equivalent_diameter_nm: Optional[float]
    major_axis_length_px: float
    minor_axis_length_px: float
    major_axis_length_nm: Optional[float]
    minor_axis_length_nm: Optional[float]
    orientation_rad: float
    display_major_axis_length_px: float
    display_minor_axis_length_px: float
    display_major_axis_length_nm: Optional[float]
    display_minor_axis_length_nm: Optional[float]
    mean_axis_length_px: float
    mean_axis_length_nm: Optional[float]
    solidity: float
    eccentricity: float
    anisotropy_ratio: float
    rejected: bool
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.rejected


@dataclass(frozen=True)
class TEMAnalysisResult:
    image_path: Path
    raw_gray: np.ndarray
    cropped_gray: np.ndarray
    display: np.ndarray
    feature: np.ndarray
    candidate_mask: np.ndarray
    valid_mask: np.ndarray
    outlier_mask: np.ndarray
    labels: np.ndarray
    measurements: list[TEMParticleMeasurement]
    metadata: TEMMetadata


@dataclass(frozen=True)
class TEMFailedPreview:
    image_path: Path
    raw_gray: np.ndarray
    cropped_gray: np.ndarray
    display: np.ndarray
    metadata: TEMMetadata


def _dataclass_from_dict(cls, data: dict):
    kwargs = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        if field.type is tuple[float, float] and isinstance(value, list):
            value = tuple(value)
        if hasattr(field.type, "__dataclass_fields__") and isinstance(value, dict):
            value = _dataclass_from_dict(field.type, value)
        kwargs[field.name] = value
    return cls(**kwargs)


def load_app_config(config_path: str | Path) -> AppConfig:
    """Load TEM viewer application config from a JSON file."""
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    viewer = _dataclass_from_dict(ViewerConfig, data.get("viewer", {}))
    return AppConfig(
        folder=data["folder"],
        file=data.get("file"),
        summary_json_path=data.get("summary_json_path"),
        viewer=viewer,
    )


def save_default_config(config_path: str | Path, folder: str | Path) -> None:
    """Write a default TEM viewer config JSON next to the runner."""
    config = AppConfig(
        folder=path_to_config_text(folder),
        summary_json_path=path_to_config_text(
            Path(folder).resolve() / "tem_particle_summary.json"
        ),
    )
    Path(config_path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure module-level logging for TEM processing."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def _resolve_image_paths(folder: str | Path, file: Optional[str] = None, exclude_dirs: Optional[set[str]] = None) -> list[Path]:
    """Resolve TEM image paths from a folder, with recursive fallback when needed."""
    folder_path = Path(folder)
    exclude_dirs = exclude_dirs or set()
    if file:
        path = resolve_optional_file_in_folder(folder_path, file)
        if not path.exists():
            raise FileNotFoundError(f"Requested file does not exist: '{path}'.")
        return [path]
    paths = sort_paths(
        [
            *folder_path.glob("*.png"),
            *folder_path.glob("*.PNG"),
            *folder_path.glob("*.jpg"),
            *folder_path.glob("*.JPG"),
            *folder_path.glob("*.jpeg"),
            *folder_path.glob("*.JPEG"),
        ]
    )
    if not paths:
        paths = sort_paths(
            [
                *folder_path.rglob("*.png"),
                *folder_path.rglob("*.PNG"),
                *folder_path.rglob("*.jpg"),
                *folder_path.rglob("*.JPG"),
                *folder_path.rglob("*.jpeg"),
                *folder_path.rglob("*.JPEG"),
            ]
        )
        if exclude_dirs:
            paths = [path for path in paths if not any(part in exclude_dirs for part in path.parts)]
    if not paths:
        raise FileNotFoundError(f"No TEM PNG/JPG images found in '{folder_path}'.")
    return paths


def _load_tem_image(image_path: Path) -> np.ndarray:
    try:
        arr = skio.imread(str(image_path))
    except Exception as exc:
        raise TEMAnalysisError(f"Failed to read image '{image_path.name}': {exc}") from exc

    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        rgb = arr[..., :3]
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.float32) / np.iinfo(rgb.dtype).max
        else:
            rgb = rgb.astype(np.float32)
        gray = rgb2gray(rgb).astype(np.float32)
    else:
        raise TEMAnalysisError(f"Unsupported image shape for '{image_path.name}': {arr.shape}.")

    if np.issubdtype(arr.dtype, np.integer) and gray.max(initial=0.0) > 1.0:
        gray /= float(np.iinfo(arr.dtype).max)
    else:
        gray = gray.astype(np.float32)

    if gray.ndim != 2 or gray.size == 0:
        raise TEMAnalysisError(f"Image '{image_path.name}' did not produce a usable grayscale plane.")
    return gray


def _crop_dark_footer(img: np.ndarray, cfg: ViewerConfig) -> tuple[np.ndarray, int]:
    h = img.shape[0]
    if cfg.strip_rows is not None and 1 <= int(cfg.strip_rows) < h:
        crop_row = h - int(cfg.strip_rows)
        return img[:crop_row, :], int(crop_row)

    tail = min(int(cfg.footer_tail_rows), h)
    start = h - tail
    row_mean = img[start:].mean(axis=1).astype(np.float64)
    half = max(5, len(row_mean) // 2)
    baseline = float(np.median(row_mean[:half]))
    mad = float(np.median(np.abs(row_mean[:half] - baseline))) + 1e-9
    thresh = baseline - float(cfg.dark_footer_k_mad) * mad

    run = 0
    crop_row = h
    for i, val in enumerate(row_mean):
        is_dark = bool(val < thresh)
        run = run + 1 if is_dark else 0
        if run >= int(cfg.dark_footer_min_run):
            crop_row = start + (i - run + 1)
            break

    crop_row = max(10, min(int(crop_row), h))
    return img[:crop_row, :], crop_row


def _pixel_size_nm(cfg: ViewerConfig, width_px: int) -> Optional[float]:
    if cfg.pixel_size_nm is not None and cfg.pixel_size_nm > 0:
        return float(cfg.pixel_size_nm)
    if cfg.fov_nm is not None and cfg.fov_nm > 0 and width_px > 0:
        return float(cfg.fov_nm) / float(width_px)
    return None


def _scale_for_display(img: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return np.zeros(img.shape, dtype=np.float32)
    lo, hi = np.percentile(vals, percentiles)
    hi = max(float(hi), float(lo) + 1e-6)
    out = (img.astype(np.float32) - float(lo)) / (float(hi) - float(lo))
    return np.clip(out, 0.0, 1.0)


def _cleanup_mask(mask: np.ndarray, cfg: ViewerConfig) -> np.ndarray:
    out = mask.astype(bool)
    if cfg.closing_radius > 0:
        out = closing(out, disk(int(cfg.closing_radius)))
    if cfg.opening_radius > 0:
        out = opening(out, disk(int(cfg.opening_radius)))
    if cfg.min_area_px > 1:
        out = remove_small_objects(out, max_size=int(cfg.min_area_px) - 1)
    return out


def _segment_dog(img: np.ndarray, cfg: ViewerConfig) -> tuple[np.ndarray, np.ndarray, float]:
    feature = gaussian(img, sigma=cfg.dog_sigma_large, preserve_range=True) - gaussian(
        img, sigma=cfg.dog_sigma_small, preserve_range=True
    )
    thr = float(np.percentile(feature, cfg.dog_foreground_percentile))
    intensity_thr = float(np.percentile(img, cfg.intensity_percentile_dark))
    mask = (feature > thr) & (img < intensity_thr)
    log.debug(
        "DoG detector: feat_thr=%.6f intensity_thr=%.6f feat_p99=%.6f",
        thr,
        intensity_thr,
        float(np.percentile(feature, 99.0)),
    )
    return feature.astype(np.float32), _cleanup_mask(mask, cfg), thr


def _segment_invert_tophat_otsu(img: np.ndarray, cfg: ViewerConfig) -> tuple[np.ndarray, np.ndarray, float]:
    inv = float(np.max(img)) - img.astype(np.float32)
    feature = white_tophat(inv, footprint=disk(max(1, int(cfg.tophat_radius)))).astype(np.float32)
    vals = feature[feature > 0]
    thr = float(threshold_otsu(vals)) if vals.size else float(feature.max()) + 1e-6
    mask = feature > thr
    log.debug("Invert top-hat detector: thr=%.6f nonzero=%d", thr, int(vals.size))
    return feature, _cleanup_mask(mask, cfg), thr


def _segment_sauvola(img: np.ndarray, cfg: ViewerConfig) -> tuple[np.ndarray, np.ndarray, float]:
    window_size = max(15, int(cfg.sauvola_window_size))
    if window_size % 2 == 0:
        window_size += 1
    thr_img = threshold_sauvola(img, window_size=window_size, k=float(cfg.sauvola_k))
    mask = img < thr_img
    feature = (thr_img - img).astype(np.float32)
    log.debug(
        "Sauvola detector: window=%d k=%.3f feature_p99=%.6f",
        window_size,
        float(cfg.sauvola_k),
        float(np.percentile(feature, 99.0)),
    )
    return feature, _cleanup_mask(mask, cfg), float(np.median(thr_img))


def _split_labels(mask: np.ndarray, feature: np.ndarray, cfg: ViewerConfig) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    if not cfg.split_touching:
        return label(mask)

    dist = ndi.distance_transform_edt(mask)
    thr_abs = float(dist.max()) * float(cfg.split_threshold_rel)
    peaks = peak_local_max(
        dist,
        labels=mask.astype(np.uint8),
        min_distance=max(1, int(cfg.split_min_distance)),
        threshold_abs=thr_abs,
        exclude_border=bool(cfg.split_exclude_border),
    )
    log.debug("Touching split: dist_max=%.4f thr_abs=%.4f peaks=%d", float(dist.max()), thr_abs, int(peaks.shape[0]))
    if peaks.shape[0] <= 1:
        return label(mask)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for idx, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = idx
    markers = ndi.label(markers > 0)[0]
    labels = watershed(-dist, markers, mask=mask)
    if int(labels.max()) == 0:
        return label(mask)
    return labels.astype(np.int32)


def _snap_point_to_mask(point_rc: tuple[float, float], coords: np.ndarray) -> tuple[float, float]:
    """Snap a point to the nearest mask pixel when the rounded centroid leaves the component."""
    if coords.size == 0:
        return point_rc
    deltas = coords.astype(np.float64) - np.array(point_rc, dtype=np.float64)
    idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    return float(coords[idx, 0]), float(coords[idx, 1])


def _trace_mask_chord(
    mask: np.ndarray,
    center_rc: tuple[float, float],
    direction_rc: tuple[float, float],
    step_px: float,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Trace a boundary-constrained chord through a mask along a chosen axis direction."""
    step_px = max(float(step_px), 0.1)
    dir_vec = np.array(direction_rc, dtype=np.float64)
    norm = float(np.linalg.norm(dir_vec))
    if norm <= 1e-12:
        raise TEMSegmentationError("Invalid zero-length axis direction for chord tracing.")
    dir_vec /= norm
    center = np.array(center_rc, dtype=np.float64)
    h, w = mask.shape

    def _inside(point: np.ndarray) -> bool:
        row = int(round(float(point[0])))
        col = int(round(float(point[1])))
        return 0 <= row < h and 0 <= col < w and bool(mask[row, col])

    if not _inside(center):
        coords = np.argwhere(mask)
        center = np.array(_snap_point_to_mask((float(center[0]), float(center[1])), coords), dtype=np.float64)

    def _walk(sign: float) -> np.ndarray:
        last_inside = center.copy()
        point = center.copy()
        max_steps = int(math.ceil(max(h, w) * 4 / step_px))
        for _ in range(max_steps):
            point = point + sign * dir_vec * step_px
            if not _inside(point):
                break
            last_inside = point.copy()
        return last_inside

    p0 = _walk(-1.0)
    p1 = _walk(1.0)
    chord_len = float(np.linalg.norm(p1 - p0))
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), chord_len


def _compute_display_axes(region, label_mask: np.ndarray, cfg: ViewerConfig) -> dict[str, object]:
    """Compute oriented mask-constrained display chords for major/minor axes."""
    if cfg.measurement_mode != "mask_chords":
        log.warning("Unsupported measurement_mode '%s'; falling back to 'mask_chords'.", cfg.measurement_mode)
    orientation = float(region.orientation)
    center = (float(region.centroid[0]), float(region.centroid[1]))
    rounded = (int(round(center[0])), int(round(center[1])))
    if not (0 <= rounded[0] < label_mask.shape[0] and 0 <= rounded[1] < label_mask.shape[1] and label_mask[rounded]):
        center = _snap_point_to_mask(center, region.coords)

    major_dir = (math.cos(orientation), -math.sin(orientation))
    minor_dir = (math.sin(orientation), math.cos(orientation))
    major_p0, major_p1, major_len = _trace_mask_chord(label_mask, center, major_dir, cfg.measure_step_px)
    minor_p0, minor_p1, minor_len = _trace_mask_chord(label_mask, center, minor_dir, cfg.measure_step_px)
    return {
        "center_rc": center,
        "orientation_rad": orientation,
        "major_endpoints_rc": (major_p0, major_p1),
        "minor_endpoints_rc": (minor_p0, minor_p1),
        "display_major_axis_length_px": major_len,
        "display_minor_axis_length_px": minor_len,
    }


def _classify_measurements(labels: np.ndarray, pixel_size_nm: Optional[float], cfg: ViewerConfig) -> tuple[list[TEMParticleMeasurement], np.ndarray, np.ndarray]:
    """Convert labeled TEM particles into filtered measurements and masks."""
    measurements: list[TEMParticleMeasurement] = []
    valid_mask = np.zeros(labels.shape, dtype=bool)
    outlier_mask = np.zeros(labels.shape, dtype=bool)

    for region in regionprops(labels):
        label_mask = labels == int(region.label)
        area_px = int(region.area)
        eq_px = float(region.equivalent_diameter_area)
        major_px = float(region.axis_major_length)
        minor_px = float(region.axis_minor_length)
        display_axes = _compute_display_axes(region, label_mask, cfg)
        display_major_px = float(display_axes["display_major_axis_length_px"])
        display_minor_px = float(display_axes["display_minor_axis_length_px"])
        mean_axis_px = 0.5 * (display_major_px + display_minor_px)
        anisotropy = display_major_px / max(display_minor_px, 1e-6)
        reasons: list[str] = []

        if area_px < int(cfg.min_area_px):
            reasons.append("too_small")
        if cfg.max_area_px is not None and area_px > int(cfg.max_area_px):
            reasons.append("too_large")
        if anisotropy > float(cfg.max_anisotropy_ratio):
            reasons.append("anisotropic")
        if float(region.solidity) < float(cfg.min_solidity):
            reasons.append("low_solidity")

        rejected = len(reasons) > 0
        if rejected:
            outlier_mask |= label_mask
        else:
            valid_mask |= label_mask

        measurements.append(
            TEMParticleMeasurement(
                label_id=int(region.label),
                centroid_rc=(float(display_axes["center_rc"][0]), float(display_axes["center_rc"][1])),
                area_px=area_px,
                equivalent_diameter_px=eq_px,
                equivalent_diameter_nm=(eq_px * pixel_size_nm) if pixel_size_nm is not None else None,
                major_axis_length_px=major_px,
                minor_axis_length_px=minor_px,
                major_axis_length_nm=(major_px * pixel_size_nm) if pixel_size_nm is not None else None,
                minor_axis_length_nm=(minor_px * pixel_size_nm) if pixel_size_nm is not None else None,
                orientation_rad=float(display_axes["orientation_rad"]),
                display_major_axis_length_px=display_major_px,
                display_minor_axis_length_px=display_minor_px,
                display_major_axis_length_nm=(display_major_px * pixel_size_nm) if pixel_size_nm is not None else None,
                display_minor_axis_length_nm=(display_minor_px * pixel_size_nm) if pixel_size_nm is not None else None,
                mean_axis_length_px=mean_axis_px,
                mean_axis_length_nm=(mean_axis_px * pixel_size_nm) if pixel_size_nm is not None else None,
                solidity=float(region.solidity),
                eccentricity=float(region.eccentricity),
                anisotropy_ratio=float(anisotropy),
                rejected=rejected,
                reasons=tuple(reasons),
            )
        )

    return measurements, valid_mask, outlier_mask


def _detect_particles(img: np.ndarray, cfg: ViewerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the configured TEM detector and optional touching-particle split."""
    detector = cfg.detector.lower().strip()
    if detector == "dog":
        feature, mask, thr = _segment_dog(img, cfg)
    elif detector == "invert_tophat_otsu":
        feature, mask, thr = _segment_invert_tophat_otsu(img, cfg)
    elif detector == "sauvola":
        feature, mask, thr = _segment_sauvola(img, cfg)
    else:
        raise TEMSegmentationError(f"Unknown detector mode '{cfg.detector}'.")

    labels = _split_labels(mask, feature, cfg)
    if int(labels.max()) == 0:
        raise TEMSegmentationError("Particle segmentation produced no components.")
    return feature, mask, labels, float(thr)


def _make_metadata(img: np.ndarray, crop_row: int, cfg: ViewerConfig) -> TEMMetadata:
    return TEMMetadata(
        pixel_size_nm=_pixel_size_nm(cfg, img.shape[1]),
        fov_nm=float(cfg.fov_nm) if cfg.fov_nm is not None else None,
        image_width_px=int(img.shape[1]),
        image_height_px=int(img.shape[0]),
        crop_row=int(crop_row),
        detector=cfg.detector,
    )


def load_failed_image_preview(image_path: str | Path, cfg: ViewerConfig = ViewerConfig()) -> TEMFailedPreview:
    """Load and crop a TEM image for fallback preview when analysis fails."""
    path = Path(image_path)
    raw_gray = _load_tem_image(path)
    cropped_gray, crop_row = _crop_dark_footer(raw_gray, cfg)
    metadata = _make_metadata(cropped_gray, crop_row, cfg)
    display = _scale_for_display(cropped_gray, cfg.display_percentiles)
    return TEMFailedPreview(
        image_path=path,
        raw_gray=raw_gray,
        cropped_gray=cropped_gray,
        display=display,
        metadata=metadata,
    )


def analyze_tem_image(image_path: str | Path, cfg: ViewerConfig = ViewerConfig()) -> TEMAnalysisResult:
    """Analyze one TEM image and return particle masks, measurements, and metadata."""
    path = Path(image_path)
    raw_gray = _load_tem_image(path)
    cropped_gray, crop_row = _crop_dark_footer(raw_gray, cfg)
    metadata = _make_metadata(cropped_gray, crop_row, cfg)
    display = _scale_for_display(cropped_gray, cfg.display_percentiles)
    feature, candidate_mask, labels, detector_thr = _detect_particles(cropped_gray, cfg)
    measurements, valid_mask, outlier_mask = _classify_measurements(labels, metadata.pixel_size_nm, cfg)
    valid_mean_nm = [m.mean_axis_length_nm for m in measurements if m.valid and m.mean_axis_length_nm is not None]
    median_nm = float(np.median(valid_mean_nm)) if valid_mean_nm else float("nan")
    log.debug(
        "TEM feature stats for %s: thr=%.6f feature_p99=%.6f candidate_px=%d labels=%d",
        path.name,
        detector_thr,
        float(np.percentile(feature, 99.0)),
        int(candidate_mask.sum()),
        int(labels.max()),
    )
    log.info(
        "TEM image %s: valid=%d total=%d median_mean_axis_nm=%s pixel_size_nm=%s",
        path.name,
        sum(1 for m in measurements if m.valid),
        len(measurements),
        f"{median_nm:.2f}" if math.isfinite(median_nm) else "n/a",
        f"{metadata.pixel_size_nm:.4f}" if metadata.pixel_size_nm is not None else "n/a",
    )
    return TEMAnalysisResult(
        image_path=path,
        raw_gray=raw_gray,
        cropped_gray=cropped_gray,
        display=display,
        feature=feature,
        candidate_mask=candidate_mask,
        valid_mask=valid_mask,
        outlier_mask=outlier_mask,
        labels=labels,
        measurements=measurements,
        metadata=metadata,
    )


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _format_length_nm(value_nm: Optional[float]) -> str:
    if value_nm is None or not math.isfinite(value_nm):
        return "n/a"
    if abs(value_nm) >= 1000.0:
        return f"{value_nm / 1000.0:.3f} um"
    return f"{value_nm:.1f} nm"


def _nice_scale_length_nm(target_nm: float) -> float:
    if target_nm <= 0:
        return 0.0
    exponent = math.floor(math.log10(target_nm))
    fraction = target_nm / (10 ** exponent)
    for base in (1.0, 2.0, 5.0, 10.0):
        if fraction <= base:
            return base * (10 ** exponent)
    return 10 ** (exponent + 1)


def _measurements_to_dicts(measurements: list[TEMParticleMeasurement]) -> list[dict]:
    items = []
    for m in measurements:
        items.append(
            {
                "label_id": int(m.label_id),
                "centroid_rc": [float(m.centroid_rc[0]), float(m.centroid_rc[1])],
                "area_px": int(m.area_px),
                "eq_diameter_px": float(m.equivalent_diameter_px),
                "eq_diameter_nm": _safe_float(m.equivalent_diameter_nm),
                "major_axis_length_px": float(m.major_axis_length_px),
                "minor_axis_length_px": float(m.minor_axis_length_px),
                "major_axis_length_nm": _safe_float(m.major_axis_length_nm),
                "minor_axis_length_nm": _safe_float(m.minor_axis_length_nm),
                "orientation_rad": float(m.orientation_rad),
                "display_major_axis_length_px": float(m.display_major_axis_length_px),
                "display_minor_axis_length_px": float(m.display_minor_axis_length_px),
                "display_major_axis_length_nm": _safe_float(m.display_major_axis_length_nm),
                "display_minor_axis_length_nm": _safe_float(m.display_minor_axis_length_nm),
                "mean_axis_length_px": float(m.mean_axis_length_px),
                "mean_axis_length_nm": _safe_float(m.mean_axis_length_nm),
                "solidity": float(m.solidity),
                "eccentricity": float(m.eccentricity),
                "anisotropy_ratio": float(m.anisotropy_ratio),
                "valid": bool(m.valid),
                "rejected": bool(m.rejected),
                "reasons": list(m.reasons),
            }
        )
    return items


def _summarize_result(res: TEMAnalysisResult) -> dict:
    """Build one per-image TEM summary entry."""
    valid = [m for m in res.measurements if m.valid]
    eq_px = [m.equivalent_diameter_px for m in valid]
    eq_nm = [m.equivalent_diameter_nm for m in valid if m.equivalent_diameter_nm is not None]
    mean_axis_px = [m.mean_axis_length_px for m in valid]
    mean_axis_nm = [m.mean_axis_length_nm for m in valid if m.mean_axis_length_nm is not None]
    count = len(valid)
    return {
        "file": res.image_path.name,
        "sample": res.image_path.parent.name,
        "status": "ok",
        "crop_row": int(res.metadata.crop_row),
        "pixel_size_nm": _safe_float(res.metadata.pixel_size_nm),
        "fov_nm": _safe_float(res.metadata.fov_nm),
        "detector": res.metadata.detector,
        "image_width_px": int(res.metadata.image_width_px),
        "image_height_px": int(res.metadata.image_height_px),
        "particle_count": count,
        "candidate_count": len(res.measurements),
        "flagged_count": len(res.measurements) - count,
        "mean_axis_length_px": _safe_float(float(np.mean(mean_axis_px))) if mean_axis_px else None,
        "median_mean_axis_length_px": _safe_float(float(np.median(mean_axis_px))) if mean_axis_px else None,
        "sd_mean_axis_length_px": _safe_float(float(np.std(mean_axis_px, ddof=1))) if len(mean_axis_px) > 1 else 0.0 if mean_axis_px else None,
        "mean_axis_length_nm": _safe_float(float(np.mean(mean_axis_nm))) if mean_axis_nm else None,
        "median_mean_axis_length_nm": _safe_float(float(np.median(mean_axis_nm))) if mean_axis_nm else None,
        "sd_mean_axis_length_nm": _safe_float(float(np.std(mean_axis_nm, ddof=1))) if len(mean_axis_nm) > 1 else 0.0 if mean_axis_nm else None,
        "mean_eq_diameter_px": _safe_float(float(np.mean(eq_px))) if eq_px else None,
        "median_eq_diameter_px": _safe_float(float(np.median(eq_px))) if eq_px else None,
        "sd_eq_diameter_px": _safe_float(float(np.std(eq_px, ddof=1))) if len(eq_px) > 1 else 0.0 if eq_px else None,
        "mean_eq_diameter_nm": _safe_float(float(np.mean(eq_nm))) if eq_nm else None,
        "median_eq_diameter_nm": _safe_float(float(np.median(eq_nm))) if eq_nm else None,
        "sd_eq_diameter_nm": _safe_float(float(np.std(eq_nm, ddof=1))) if len(eq_nm) > 1 else 0.0 if eq_nm else None,
        "particles": _measurements_to_dicts(res.measurements),
    }


def build_tem_summary_from_paths(image_paths: list[Path], cfg: ViewerConfig, folder: str | Path) -> dict:
    """Process an explicit list of TEM image paths and build a JSON-serializable summary."""
    images: list[dict] = []
    global_nm: list[float] = []
    global_px: list[float] = []
    global_mean_axis_nm: list[float] = []
    global_mean_axis_px: list[float] = []
    failure_count = 0

    for image_path in image_paths:
        try:
            res = analyze_tem_image(image_path, cfg)
            entry = _summarize_result(res)
            images.append(entry)
            global_px.extend([float(m["eq_diameter_px"]) for m in entry["particles"] if m["valid"]])
            global_nm.extend([float(m["eq_diameter_nm"]) for m in entry["particles"] if m["valid"] and m["eq_diameter_nm"] is not None])
            global_mean_axis_px.extend([float(m["mean_axis_length_px"]) for m in entry["particles"] if m["valid"]])
            global_mean_axis_nm.extend([float(m["mean_axis_length_nm"]) for m in entry["particles"] if m["valid"] and m["mean_axis_length_nm"] is not None])
        except Exception as exc:
            failure_count += 1
            log.exception("TEM analysis failed for %s", image_path.name)
            preview = load_failed_image_preview(image_path, cfg)
            images.append(
                {
                    "file": image_path.name,
                    "sample": image_path.parent.name,
                    "status": "failed",
                    "error": str(exc),
                    "crop_row": int(preview.metadata.crop_row),
                    "pixel_size_nm": _safe_float(preview.metadata.pixel_size_nm),
                    "fov_nm": _safe_float(preview.metadata.fov_nm),
                    "detector": preview.metadata.detector,
                    "image_width_px": int(preview.metadata.image_width_px),
                    "image_height_px": int(preview.metadata.image_height_px),
                    "particle_count": 0,
                    "candidate_count": 0,
                    "flagged_count": 0,
                    "particles": [],
                }
            )

    global_summary = {
        "image_count": len(images),
        "failed_images": int(failure_count),
        "images_with_particles": sum(1 for image in images if image["status"] == "ok" and image["particle_count"] > 0),
        "total_particles": int(len(global_px)),
        "mean_axis_length_px": _safe_float(float(np.mean(global_mean_axis_px))) if global_mean_axis_px else None,
        "median_mean_axis_length_px": _safe_float(float(np.median(global_mean_axis_px))) if global_mean_axis_px else None,
        "sd_mean_axis_length_px": _safe_float(float(np.std(global_mean_axis_px, ddof=1))) if len(global_mean_axis_px) > 1 else 0.0 if global_mean_axis_px else None,
        "mean_axis_length_nm": _safe_float(float(np.mean(global_mean_axis_nm))) if global_mean_axis_nm else None,
        "median_mean_axis_length_nm": _safe_float(float(np.median(global_mean_axis_nm))) if global_mean_axis_nm else None,
        "sd_mean_axis_length_nm": _safe_float(float(np.std(global_mean_axis_nm, ddof=1))) if len(global_mean_axis_nm) > 1 else 0.0 if global_mean_axis_nm else None,
        "mean_eq_diameter_px": _safe_float(float(np.mean(global_px))) if global_px else None,
        "median_eq_diameter_px": _safe_float(float(np.median(global_px))) if global_px else None,
        "sd_eq_diameter_px": _safe_float(float(np.std(global_px, ddof=1))) if len(global_px) > 1 else 0.0 if global_px else None,
        "mean_eq_diameter_nm": _safe_float(float(np.mean(global_nm))) if global_nm else None,
        "median_eq_diameter_nm": _safe_float(float(np.median(global_nm))) if global_nm else None,
        "sd_eq_diameter_nm": _safe_float(float(np.std(global_nm, ddof=1))) if len(global_nm) > 1 else 0.0 if global_nm else None,
    }
    hist_vals = global_mean_axis_nm if global_mean_axis_nm else global_mean_axis_px
    hist_key = "histogram_mean_axis_length_nm" if global_mean_axis_nm else "histogram_mean_axis_length_px"
    if hist_vals:
        hist_counts, hist_edges = np.histogram(hist_vals, bins=min(max(5, len(hist_vals) // 4), 20))
        global_summary[hist_key] = {
            "bin_edges": [float(v) for v in hist_edges],
            "counts": [int(v) for v in hist_counts],
        }

    return {
        "folder": str(Path(folder)),
        "file": None,
        "viewer_config": asdict(cfg),
        "global_summary": global_summary,
        "images": images,
    }


def build_tem_summary(folder: str | Path, cfg: ViewerConfig, file: Optional[str] = None) -> dict:
    """Process a TEM folder or one selected file and build a JSON-serializable summary."""
    image_paths = _resolve_image_paths(folder, file)
    summary = build_tem_summary_from_paths(image_paths, cfg, folder)
    summary["file"] = file
    return summary


def write_tem_summary_json(folder: str | Path, cfg: ViewerConfig, out_path: str | Path, file: Optional[str] = None) -> None:
    """Write TEM summary JSON to disk."""
    summary = build_tem_summary(folder, cfg, file)
    Path(out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _overlay_image(res: TEMAnalysisResult, show_boundaries: bool = True) -> np.ndarray:
    base = np.dstack([res.display, res.display, res.display]).astype(np.float32)
    if not show_boundaries:
        return base
    base[find_boundaries(res.valid_mask, mode="outer")] = (0.0, 1.0, 0.0)
    base[find_boundaries(res.outlier_mask, mode="outer")] = (1.0, 0.1, 0.1)
    return base


def _feature_display(feature: np.ndarray, cfg: ViewerConfig) -> np.ndarray:
    return _scale_for_display(feature, cfg.feature_percentiles)


def _make_scale_overlay(ax, shape: tuple[int, int], pixel_size_nm: Optional[float]) -> list[object]:
    if pixel_size_nm is None or pixel_size_nm <= 0:
        return []
    h, w = shape
    scale_length_nm = _nice_scale_length_nm(w * pixel_size_nm * 0.22)
    scale_length_px = scale_length_nm / pixel_size_nm
    x0 = w * 0.06
    y0 = h * 0.92
    return [
        Rectangle((x0 - 8, y0 - 28), scale_length_px + 16, 36, facecolor=(0.0, 0.0, 0.0, 0.35), edgecolor="none"),
        Line2D([x0, x0 + scale_length_px], [y0, y0], color="white", linewidth=3),
        Line2D([x0, x0], [y0 - 7, y0 + 7], color="white", linewidth=1.5),
        Line2D([x0 + scale_length_px, x0 + scale_length_px], [y0 - 7, y0 + 7], color="white", linewidth=1.5),
        ax.text(x0, y0 - 11, _format_length_nm(scale_length_nm), color="white", fontsize=10, va="bottom", ha="left"),
    ]


def _make_measure_overlay(ax, measurements: list[TEMParticleMeasurement], cfg: ViewerConfig) -> list[object]:
    """Draw oriented major/minor axis overlays constrained by the particle mask."""
    artists: list[object] = []
    for meas in measurements:
        row, col = meas.centroid_rc
        major_color = cfg.major_axis_color if meas.valid else "#ff6b6b"
        minor_color = cfg.minor_axis_color if meas.valid else "#ff9f43"
        major_dir = np.array((math.cos(meas.orientation_rad), -math.sin(meas.orientation_rad)), dtype=np.float64)
        minor_dir = np.array((math.sin(meas.orientation_rad), math.cos(meas.orientation_rad)), dtype=np.float64)
        center = np.array((row, col), dtype=np.float64)
        major_half = 0.5 * float(meas.display_major_axis_length_px)
        minor_half = 0.5 * float(meas.display_minor_axis_length_px)
        major_p0 = center - major_dir * major_half
        major_p1 = center + major_dir * major_half
        minor_p0 = center - minor_dir * minor_half
        minor_p1 = center + minor_dir * minor_half
        major_label = _format_length_nm(meas.display_major_axis_length_nm) if meas.display_major_axis_length_nm is not None else f"{meas.display_major_axis_length_px:.1f}px"
        minor_label = _format_length_nm(meas.display_minor_axis_length_nm) if meas.display_minor_axis_length_nm is not None else f"{meas.display_minor_axis_length_px:.1f}px"
        if not meas.valid and meas.reasons:
            major_label += " !"
            minor_label += " !"
        label_offset = 4.0
        major_text_pos = major_p1 + minor_dir * label_offset
        minor_text_pos = minor_p1 + major_dir * label_offset
        artists.extend(
            [
                Line2D([major_p0[1], major_p1[1]], [major_p0[0], major_p1[0]], color=major_color, linewidth=1.2, alpha=0.95),
                Line2D([minor_p0[1], minor_p1[1]], [minor_p0[0], minor_p1[0]], color=minor_color, linewidth=1.2, alpha=0.95),
                ax.text(
                    major_text_pos[1],
                    major_text_pos[0],
                    major_label,
                    color=major_color,
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    bbox={"facecolor": (0.0, 0.0, 0.0, 0.45), "edgecolor": "none", "pad": 1.2},
                ),
                ax.text(
                    minor_text_pos[1],
                    minor_text_pos[0],
                    minor_label,
                    color=minor_color,
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    bbox={"facecolor": (0.0, 0.0, 0.0, 0.45), "edgecolor": "none", "pad": 1.2},
                ),
            ]
        )
    return artists


class TEMDatasetViewer:
    """Interactive TEM dataset viewer with cached results and overlay toggles."""

    def __init__(self, folder: str | Path, config: ViewerConfig = ViewerConfig(), file: Optional[str] = None):
        self.folder = Path(folder)
        self.config = config
        self.image_paths = _resolve_image_paths(folder, file)
        self._cache: dict[Path, TEMAnalysisResult] = {}
        self._error_cache: dict[Path, str] = {}
        self._failed_preview_cache: dict[Path, TEMFailedPreview] = {}
        self.index = 0
        self.show_scale = config.default_show_scale
        self.show_boundaries = config.default_show_boundaries
        self.show_measures = config.default_show_measures
        self.show_histogram = config.default_show_histogram

        self.fig = None
        self.ax_display = None
        self.ax_feature = None
        self.ax_overlay = None
        self.ax_hist = None
        self.display_artist = None
        self.feature_artist = None
        self.overlay_artist = None
        self.overlay_artists: list[object] = []
        self.check_buttons: Optional[CheckButtons] = None

    def _get_result(self, index: int) -> Optional[TEMAnalysisResult]:
        path = self.image_paths[index]
        if path in self._error_cache:
            return None
        if path not in self._cache:
            try:
                self._cache[path] = analyze_tem_image(path, self.config)
            except Exception as exc:
                self._error_cache[path] = str(exc)
                log.exception("TEM viewer analysis failed for %s", path.name)
                return None
        return self._cache[path]

    def _get_failed_preview(self, index: int) -> Optional[TEMFailedPreview]:
        path = self.image_paths[index]
        if path not in self._failed_preview_cache:
            try:
                self._failed_preview_cache[path] = load_failed_image_preview(path, self.config)
            except Exception:
                return None
        return self._failed_preview_cache[path]

    def _clear_overlay_artists(self) -> None:
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.overlay_artists.clear()

    def _metric_values(self, res: TEMAnalysisResult) -> tuple[np.ndarray, str]:
        metric = self.config.histogram_metric
        valid = [m for m in res.measurements if m.valid]
        if metric == "eq_diameter":
            nm_vals = [m.equivalent_diameter_nm for m in valid if m.equivalent_diameter_nm is not None]
            if nm_vals:
                return np.array(nm_vals, dtype=np.float64), "Eq diameter [nm]"
            return np.array([m.equivalent_diameter_px for m in valid], dtype=np.float64), "Eq diameter [px]"
        if metric == "major_axis":
            nm_vals = [m.display_major_axis_length_nm for m in valid if m.display_major_axis_length_nm is not None]
            if nm_vals:
                return np.array(nm_vals, dtype=np.float64), "Major axis [nm]"
            return np.array([m.display_major_axis_length_px for m in valid], dtype=np.float64), "Major axis [px]"
        if metric == "minor_axis":
            nm_vals = [m.display_minor_axis_length_nm for m in valid if m.display_minor_axis_length_nm is not None]
            if nm_vals:
                return np.array(nm_vals, dtype=np.float64), "Minor axis [nm]"
            return np.array([m.display_minor_axis_length_px for m in valid], dtype=np.float64), "Minor axis [px]"
        nm_vals = [m.mean_axis_length_nm for m in valid if m.mean_axis_length_nm is not None]
        if nm_vals:
            return np.array(nm_vals, dtype=np.float64), "Mean axes [nm]"
        return np.array([m.mean_axis_length_px for m in valid], dtype=np.float64), "Mean axes [px]"

    def _set_image_data(self, artist, image: np.ndarray, ax) -> None:
        artist.set_data(image)
        h, w = image.shape[:2]
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")

    def _update_hist_or_info(self, res: Optional[TEMAnalysisResult], image_path: Path, error: Optional[str] = None) -> None:
        self.ax_hist.clear()
        if res is None:
            self.ax_hist.axis("off")
            self.ax_hist.text(
                0.02,
                0.98,
                "\n".join(
                    [
                        f"File: {image_path.name}",
                        "Status: failed",
                        "",
                        error or "Unknown analysis error.",
                        "",
                        "Showing cropped preview only.",
                    ]
                ),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
                transform=self.ax_hist.transAxes,
            )
            return

        metric_vals, metric_label = self._metric_values(res)
        lines = [
            f"File: {res.image_path.name}",
            f"Sample: {res.image_path.parent.name}",
            f"Detector: {res.metadata.detector}",
            f"Particles: {sum(1 for m in res.measurements if m.valid)}",
            f"Flagged: {sum(1 for m in res.measurements if not m.valid)}",
            f"Pixel size: {_format_length_nm(res.metadata.pixel_size_nm)} / px",
        ]
        if metric_vals.size:
            unit = "nm" if "[nm]" in metric_label else "px"
            fmt = _format_length_nm if unit == "nm" else (lambda value: f"{value:.1f} px")
            lines.extend(
                [
                    f"Primary metric: {metric_label}",
                    f"Median size: {fmt(float(np.median(metric_vals)))}",
                    f"Mean size: {fmt(float(np.mean(metric_vals)))}",
                    f"SD size: {fmt(float(np.std(metric_vals, ddof=1))) if metric_vals.size > 1 else ('0.0 nm' if unit == 'nm' else '0.0 px')}",
                ]
            )
        eq_nm = [m.equivalent_diameter_nm for m in res.measurements if m.valid and m.equivalent_diameter_nm is not None]
        eq_px = [m.equivalent_diameter_px for m in res.measurements if m.valid]
        if eq_nm:
            lines.extend(
                [
                    f"Median eq: {_format_length_nm(float(np.median(eq_nm)))}",
                    f"Mean eq: {_format_length_nm(float(np.mean(eq_nm)))}",
                ]
            )
        elif eq_px:
            lines.extend(
                [
                    f"Median eq: {float(np.median(eq_px)):.1f} px",
                    f"Mean eq: {float(np.mean(eq_px)):.1f} px",
                ]
            )
        if self.show_histogram and metric_vals.size:
            self.ax_hist.hist(metric_vals, bins=min(max(5, metric_vals.size // 3), 15), color="#4cc9f0", edgecolor="#0b1f2a")
            self.ax_hist.set_xlabel(metric_label)
            self.ax_hist.set_ylabel("Count")
            self.ax_hist.set_title("Particle Size Distribution")
            self.ax_hist.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                transform=self.ax_hist.transAxes,
                bbox={"facecolor": (1.0, 1.0, 1.0, 0.70), "edgecolor": "none", "pad": 2.0},
            )
        else:
            self.ax_hist.axis("off")
            self.ax_hist.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
                transform=self.ax_hist.transAxes,
            )

    def _render_current(self) -> None:
        image_path = self.image_paths[self.index]
        res = self._get_result(self.index)
        self._clear_overlay_artists()

        if res is None:
            preview = self._get_failed_preview(self.index)
            if preview is not None:
                gray = preview.display.astype(np.float32)
                base = np.dstack([gray, gray, gray]).astype(np.float32)
                self._set_image_data(self.display_artist, base, self.ax_display)
                self._set_image_data(self.feature_artist, gray, self.ax_feature)
                self._set_image_data(self.overlay_artist, base, self.ax_overlay)
            self.ax_display.set_title(f"{self.index + 1}/{len(self.image_paths)}  {image_path.name}  [failed]")
            self.ax_feature.set_title("Feature unavailable")
            self.ax_overlay.set_title("Overlay fallback preview")
            self._update_hist_or_info(res, image_path, self._error_cache.get(image_path))
            self.fig.canvas.draw_idle()
            return

        display_rgb = np.dstack([res.display, res.display, res.display]).astype(np.float32)
        overlay_rgb = _overlay_image(res, show_boundaries=self.show_boundaries)
        feature_img = _feature_display(res.feature, self.config)
        self._set_image_data(self.display_artist, display_rgb, self.ax_display)
        self._set_image_data(self.feature_artist, feature_img, self.ax_feature)
        self._set_image_data(self.overlay_artist, overlay_rgb, self.ax_overlay)

        if self.show_scale:
            self.overlay_artists.extend(_make_scale_overlay(self.ax_overlay, res.display.shape, res.metadata.pixel_size_nm))
        if self.show_measures:
            self.overlay_artists.extend(_make_measure_overlay(self.ax_overlay, res.measurements, self.config))
        for artist in self.overlay_artists:
            if getattr(artist, "axes", None) is None:
                self.ax_overlay.add_artist(artist)

        self.ax_display.set_title(f"{self.index + 1}/{len(self.image_paths)}  {res.image_path.name}  [display]")
        self.ax_feature.set_title(f"Feature [{self.config.detector}]")
        self.ax_overlay.set_title("Overlay")
        self._update_hist_or_info(res, image_path)
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "right":
            self.index = (self.index + 1) % len(self.image_paths)
            self._render_current()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.image_paths)
            self._render_current()

    def _on_checks(self, _label: str) -> None:
        if self.check_buttons is None:
            return
        status = self.check_buttons.get_status()
        self.show_boundaries = bool(status[0])
        self.show_measures = bool(status[1])
        self.show_scale = bool(status[2])
        self._render_current()

    def show(self) -> None:
        self.fig = plt.figure(figsize=(14, 9))
        self.ax_display = self.fig.add_subplot(2, 2, 1)
        self.ax_feature = self.fig.add_subplot(2, 2, 2)
        self.ax_overlay = self.fig.add_subplot(2, 2, 3)
        self.ax_hist = self.fig.add_subplot(2, 2, 4)
        ax_checks = self.fig.add_axes([0.90, 0.17, 0.08, 0.18])

        first = self._get_result(self.index)
        if first is not None:
            display_rgb = np.dstack([first.display, first.display, first.display]).astype(np.float32)
            overlay_rgb = _overlay_image(first, show_boundaries=self.show_boundaries)
            feature_img = _feature_display(first.feature, self.config)
        else:
            preview = self._get_failed_preview(self.index)
            if preview is not None:
                display_rgb = np.dstack([preview.display, preview.display, preview.display]).astype(np.float32)
                overlay_rgb = display_rgb.copy()
                feature_img = preview.display.astype(np.float32)
            else:
                display_rgb = np.zeros((512, 512, 3), dtype=np.float32)
                overlay_rgb = display_rgb.copy()
                feature_img = np.zeros((512, 512), dtype=np.float32)

        self.display_artist = self.ax_display.imshow(display_rgb, vmin=0.0, vmax=1.0)
        self.feature_artist = self.ax_feature.imshow(feature_img, cmap="magma", vmin=0.0, vmax=1.0)
        self.overlay_artist = self.ax_overlay.imshow(overlay_rgb, vmin=0.0, vmax=1.0)
        for ax in (self.ax_display, self.ax_feature, self.ax_overlay):
            ax.axis("off")
            ax.set_autoscale_on(False)

        ax_checks.set_title("Overlays", fontsize=10)
        self.check_buttons = CheckButtons(
            ax_checks,
            ["Boundaries", "Measures", "Scale"],
            [self.show_boundaries, self.show_measures, self.show_scale],
        )
        for text in self.check_buttons.labels:
            text.set_fontsize(10)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.check_buttons.on_clicked(self._on_checks)
        self._render_current()
        self.fig.suptitle("TEM Particle Viewer  |  left/right = next image", fontsize=12)
        plt.show()


def _example_tem_paths() -> list[Path]:
    root = Path(__file__).resolve().parent / "testData" / "TEM"
    paths = [root / "TEM SeNPs 1.png", root / "TEM SeNPs 2.png"]
    return [path for path in paths if path.exists()]


def smoke_test_examples() -> dict[str, int]:
    """Run a small smoke test on bundled example TEM PNG files when present."""
    paths = _example_tem_paths()
    counts: dict[str, int] = {}
    for path in paths:
        res = analyze_tem_image(path)
        counts[path.name] = sum(1 for m in res.measurements if m.valid)
    return counts


def main(config_path: str | Path = "tem_particle_viewer_config.json") -> None:
    """Run TEM summary export or open the interactive TEM viewer."""
    run_from_config(config_path)


DEFAULT_CONFIG_PATH = Path("tem_particle_viewer_config.json")


def run_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    folder_override: str | Path | None = None,
    file_override: str | Path | None = None,
) -> None:
    """Run the TEM viewer from one config file and temporary CLI overrides."""

    config_path = expand_user_path(config_path)
    if not config_path.exists():
        save_default_config(config_path, Path(__file__).resolve().parent / "testData" / "TEM")
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
        write_tem_summary_json(
            effective_folder,
            app_cfg.viewer,
            expand_user_path(app_cfg.summary_json_path),
            None if effective_file is None else str(effective_file),
        )
    else:
        TEMDatasetViewer(
            effective_folder,
            app_cfg.viewer,
            None if effective_file is None else str(effective_file),
        ).show()


if __name__ == "__main__":
    setup_logging()
    main()

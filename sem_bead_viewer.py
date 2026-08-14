from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Optional, Sequence

import configparser
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening, remove_small_objects
from skimage.segmentation import find_boundaries, watershed
from tifffile import imread

from path_utils import expand_user_path, path_to_config_text, resolve_existing_input_path
from tabular_export import sort_paths


BEAD_SIZE_METRICS = ("equivalent_diameter", "mean_xy_diameter")


def normalize_size_distribution_metric(metric: str) -> str:
    """Validate and normalize the reporting-only bead size metric name."""

    value = str(metric).strip().casefold()
    if value not in BEAD_SIZE_METRICS:
        raise ValueError(
            f"Unsupported bead size_distribution_metric {metric!r}. "
            "Supported values: equivalent_diameter, mean_xy_diameter."
        )
    return value


def bead_size_metric_label(metric: str, unit: str) -> str:
    """Return the user-facing axis label for a validated size metric."""

    normalized = normalize_size_distribution_metric(metric)
    prefix = "Equivalent diameter" if normalized == "equivalent_diameter" else "Mean X/Y diameter"
    return f"{prefix} [{unit}]"


@dataclass(frozen=True)
class ViewerConfig:
    infobar_tail_rows: int = 320
    infobar_k_mad: float = 8.0
    infobar_min_run: int = 10
    display_percentiles: tuple[float, float] = (0.5, 99.5)

    dog_sigma_small: float = 1.2
    dog_sigma_large: float = 8.0
    dog_foreground_percentile: float = 80.0
    intensity_percentile: float = 97.5

    closing_radius: int = 2
    opening_radius: int = 1
    min_object_area_px: int = 50
    diameter_size_limits: bool = True
    min_diameter_px: float = 14.0
    max_diameter_px: float = 60.0
    size_distribution_metric: str = "mean_xy_diameter"

    peak_min_distance_px: int = 8
    peak_threshold_px: float = 3.5
    use_watershed_split: bool = True
    split_only_suspicious: bool = True
    split_min_distance_px: int = 10
    split_threshold_px: float = 5.0
    split_min_peak_count: int = 2
    split_max_peak_count: int = 4
    split_min_child_area_px: int = 120
    split_min_child_diameter_px: float = 18.0
    split_max_child_diameter_px: float = 45.0
    split_trigger_diameter_px: float = 34.0
    split_trigger_axis_ratio: float = 1.18
    split_trigger_solidity_below: float = 0.90

    boundary_linewidth: float = 1.0
    outlier_axis_ratio: float = 1.22
    global_size_outliers: bool = True
    outlier_mad_zscore: float = 3.5
    min_solidity: float = 0.72
    max_eccentricity: float = 0.95
    edge_touch_margin_px: int = 0
    include_edge_candidates: bool = True

    default_show_scale: bool = True
    default_show_boundaries: bool = True
    default_show_measures: bool = True


@dataclass(frozen=True)
class AppConfig:
    folder: str = ""
    viewer: ViewerConfig = ViewerConfig()
    summary_json_path: Optional[str] = None
    file: Optional[str] = None


@dataclass(frozen=True)
class SEMMetadata:
    pixel_size_x_m: Optional[float]
    pixel_size_y_m: Optional[float]
    magnification: Optional[float]
    image_strip_size_px: Optional[int]
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
class BeadMeasurement:
    label_id: int
    centroid_rc: tuple[float, float]
    equivalent_diameter_px: float
    equivalent_diameter_m: Optional[float]
    x_diameter_px: float
    y_diameter_px: float
    x_diameter_m: Optional[float]
    y_diameter_m: Optional[float]
    area_px: int
    bbox: tuple[int, int, int, int]
    solidity: float
    eccentricity: float
    edge_touched: bool
    anisotropic: bool
    global_outlier: bool
    rejected: bool
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.rejected

    @property
    def axis_ratio(self) -> float:
        denom = max(min(self.x_diameter_px, self.y_diameter_px), 1e-6)
        return float(max(self.x_diameter_px, self.y_diameter_px) / denom)

    @property
    def mean_diameter_m(self) -> Optional[float]:
        """Backward-compatible alias for :attr:`mean_xy_diameter_m`."""

        return self.mean_xy_diameter_m

    @property
    def mean_xy_diameter_px(self) -> float:
        """Arithmetic mean of full horizontal and vertical mask extents."""

        return float((self.x_diameter_px + self.y_diameter_px) / 2.0)

    @property
    def mean_xy_diameter_m(self) -> Optional[float]:
        vals = [v for v in (self.x_diameter_m, self.y_diameter_m) if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))


def bead_size_value_px(measurement: BeadMeasurement, metric: str) -> float:
    """Return one reported bead-size value in pixels without changing validity."""

    normalized = normalize_size_distribution_metric(metric)
    if normalized == "equivalent_diameter":
        return float(measurement.equivalent_diameter_px)
    return measurement.mean_xy_diameter_px


def bead_size_value_m(measurement: BeadMeasurement, metric: str) -> Optional[float]:
    """Return one reported bead-size value in metres when calibrated."""

    normalized = normalize_size_distribution_metric(metric)
    if normalized == "equivalent_diameter":
        return measurement.equivalent_diameter_m
    return measurement.mean_xy_diameter_m


def bead_size_vector_px(
    measurements: Sequence[BeadMeasurement], metric: str
) -> np.ndarray:
    """Return the selected reporting-size vector for the supplied beads."""

    normalized = normalize_size_distribution_metric(metric)
    return np.asarray(
        [bead_size_value_px(measurement, normalized) for measurement in measurements],
        dtype=np.float64,
    )


def valid_bead_size_vectors(
    measurements: Sequence[BeadMeasurement], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return calibrated and pixel vectors for valid beads only."""

    normalized = normalize_size_distribution_metric(metric)
    valid = [measurement for measurement in measurements if measurement.valid]
    values_px = bead_size_vector_px(valid, normalized)
    values_m = np.asarray(
        [value for measurement in valid if (value := bead_size_value_m(measurement, normalized)) is not None],
        dtype=np.float64,
    )
    return values_m, values_px


def robust_mad_zscores(values: np.ndarray) -> np.ndarray:
    """Return robust z-scores from the median and median absolute deviation."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros(0, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad <= 0:
        return np.zeros(array.shape, dtype=np.float64)
    return 0.6745 * np.abs(array - median) / mad


def mad_outlier_flags(values: np.ndarray, zscore_limit: float) -> np.ndarray:
    """Return robust MAD outlier flags for one already-selected size vector."""

    return robust_mad_zscores(values) > float(zscore_limit)


def global_size_outlier_flags(
    measurements: Sequence[BeadMeasurement], config: ViewerConfig
) -> np.ndarray:
    """Flag provisional candidates using the configured scalar size metric."""

    selected_sizes = bead_size_vector_px(
        measurements, config.size_distribution_metric
    )
    return mad_outlier_flags(selected_sizes, config.outlier_mad_zscore)


@dataclass(frozen=True)
class BeadAnalysisResult:
    image_path: Path
    hdr_path: Optional[Path]
    raw: np.ndarray
    cropped: np.ndarray
    display: np.ndarray
    feature: np.ndarray
    candidate_mask: np.ndarray
    valid_mask: np.ndarray
    outlier_mask: np.ndarray
    labels: np.ndarray
    measurements: list[BeadMeasurement]
    metadata: SEMMetadata
    crop_row: int


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
    config_path = Path(config_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    viewer_data = data.get("viewer", {})
    viewer = _dataclass_from_dict(ViewerConfig, viewer_data)
    viewer = replace(
        viewer,
        size_distribution_metric=normalize_size_distribution_metric(
            viewer.size_distribution_metric
        ),
    )
    return AppConfig(
        folder=str(data.get("folder") or ""),
        file=data.get("file"),
        viewer=viewer,
        summary_json_path=data.get("summary_json_path"),
    )


def save_default_config(config_path: str | Path, folder: str | Path) -> None:
    config = AppConfig(
        folder=path_to_config_text(folder),
        summary_json_path=path_to_config_text(
            Path(folder).resolve() / "sem_bead_viewer_summary.json"
        ),
    )
    Path(config_path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def _read_hdr_metadata(hdr_path: Path) -> SEMMetadata:
    cfg = configparser.ConfigParser()
    cfg.read(hdr_path, encoding="utf-8")
    main = cfg["MAIN"] if "MAIN" in cfg else {}

    def _maybe_float(section: dict, key: str) -> Optional[float]:
        value = section.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    return SEMMetadata(
        pixel_size_x_m=_maybe_float(main, "PixelSizeX"),
        pixel_size_y_m=_maybe_float(main, "PixelSizeY"),
        magnification=_maybe_float(main, "Magnification"),
        image_strip_size_px=int(float(main["ImageStripSize"])) if main.get("ImageStripSize") else None,
        note=main.get("Note", ""),
        device=main.get("Device", ""),
        date=main.get("Date", ""),
        time=main.get("Time", ""),
    )


def _paired_hdr_path(image_path: Path) -> Optional[Path]:
    hdr_name = f"{image_path.stem}-tif.hdr"
    hdr_path = image_path.with_name(hdr_name)
    return hdr_path if hdr_path.exists() else None


def _crop_infobar(img: np.ndarray, cfg: ViewerConfig, strip_rows: Optional[int] = None) -> tuple[np.ndarray, int]:
    h = img.shape[0]
    if strip_rows is not None and 10 <= strip_rows < h:
        crop_row = h - int(strip_rows)
        return img[:crop_row, :], crop_row

    tail = min(cfg.infobar_tail_rows, h)
    start = h - tail

    row_mean = img[start:].mean(axis=1).astype(np.float64)
    half = max(10, len(row_mean) // 2)

    baseline = np.median(row_mean[:half])
    mad = np.median(np.abs(row_mean[:half] - baseline)) + 1e-9
    thresh = baseline + cfg.infobar_k_mad * mad

    above = row_mean > thresh
    run = 0
    crop_row = h
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= cfg.infobar_min_run:
            crop_row = start + (i - run + 1)
            break

    crop_row = max(crop_row, 10)
    return img[:crop_row, :], crop_row


def _scale_for_display(img: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    lo_p, hi_p = percentiles
    lo, hi = np.percentile(img, (lo_p, hi_p))
    hi = max(float(hi), float(lo) + 1e-6)
    out = (img.astype(np.float32) - float(lo)) / (float(hi) - float(lo))
    return np.clip(out, 0.0, 1.0)


def _segment_candidates(img: np.ndarray, cfg: ViewerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat = gaussian(img, sigma=cfg.dog_sigma_small, preserve_range=True) - gaussian(
        img, sigma=cfg.dog_sigma_large, preserve_range=True
    )

    vals = feat[feat > np.percentile(feat, cfg.dog_foreground_percentile)]
    feat_thr = float(threshold_otsu(vals)) if vals.size else 0.0
    intensity_thr = float(np.percentile(img, cfg.intensity_percentile))

    mask = (feat > feat_thr) & (img > intensity_thr)
    mask = closing(mask, disk(cfg.closing_radius))
    mask = opening(mask, disk(cfg.opening_radius))
    mask = remove_small_objects(mask, max_size=cfg.min_object_area_px - 1)

    if not mask.any():
        return feat, mask, np.zeros(mask.shape, dtype=np.int32)

    base_labels = label(mask)
    if not cfg.use_watershed_split:
        return feat, mask, base_labels

    labels = np.zeros(mask.shape, dtype=np.int32)
    next_label = 1
    for region in regionprops(base_labels):
        region_mask = base_labels == region.label
        submasks = _split_region_if_needed(region, region_mask, cfg)
        for submask in submasks:
            labels[submask] = next_label
            next_label += 1

    return feat, mask, labels


def _should_try_split(region, cfg: ViewerConfig) -> bool:
    if not cfg.split_only_suspicious:
        return True

    x_px, y_px = _region_xy_diameters(region)
    axis_ratio = max(x_px, y_px) / max(min(x_px, y_px), 1e-6)
    eqd_px = float(region.equivalent_diameter_area)
    return (
        eqd_px >= cfg.split_trigger_diameter_px
        or axis_ratio >= cfg.split_trigger_axis_ratio
        or float(region.solidity) <= cfg.split_trigger_solidity_below
    )


def _split_region_if_needed(region, region_mask: np.ndarray, cfg: ViewerConfig) -> list[np.ndarray]:
    if not _should_try_split(region, cfg):
        return [region_mask]

    min_row, min_col, max_row, max_col = region.bbox
    pad = max(4, cfg.split_min_distance_px)
    r0 = max(0, min_row - pad)
    c0 = max(0, min_col - pad)
    r1 = min(region_mask.shape[0], max_row + pad)
    c1 = min(region_mask.shape[1], max_col + pad)

    local_mask = region_mask[r0:r1, c0:c1]
    if not local_mask.any():
        return [region_mask]

    distance = ndi.distance_transform_edt(local_mask)
    peaks = peak_local_max(
        distance,
        labels=local_mask.astype(np.uint8),
        min_distance=cfg.split_min_distance_px,
        threshold_abs=cfg.split_threshold_px,
        exclude_border=False,
    )

    if peaks.shape[0] < cfg.split_min_peak_count or peaks.shape[0] > cfg.split_max_peak_count:
        return [region_mask]

    markers = np.zeros(local_mask.shape, dtype=np.int32)
    for idx, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = idx

    markers = ndi.label(markers > 0)[0]
    split_labels = watershed(-distance, markers, mask=local_mask)

    child_masks: list[np.ndarray] = []
    for child_label in range(1, int(split_labels.max()) + 1):
        child = split_labels == child_label
        area = int(child.sum())
        if area < cfg.split_min_child_area_px:
            return [region_mask]
        eqd = float(np.sqrt(4.0 * area / np.pi))
        if eqd < cfg.split_min_child_diameter_px or eqd > cfg.split_max_child_diameter_px:
            return [region_mask]
        child_masks.append(child)

    if len(child_masks) < cfg.split_min_peak_count:
        return [region_mask]

    out: list[np.ndarray] = []
    for child in child_masks:
        full = np.zeros(region_mask.shape, dtype=bool)
        full[r0:r1, c0:c1] = child
        out.append(full)
    return out


def _region_xy_diameters(region) -> tuple[float, float]:
    rows, cols = region.coords[:, 0], region.coords[:, 1]
    y_px = float(rows.max() - rows.min() + 1)
    x_px = float(cols.max() - cols.min() + 1)
    return x_px, y_px


def _classify_measurements(
    labels: np.ndarray, metadata: SEMMetadata, cfg: ViewerConfig
) -> tuple[list[BeadMeasurement], np.ndarray, np.ndarray]:
    pixel_size_m = metadata.mean_pixel_size_m
    provisional = []
    h, w = labels.shape

    for region in regionprops(labels):
        if region.area < cfg.min_object_area_px:
            continue

        x_px, y_px = _region_xy_diameters(region)
        eqd_px = float(region.equivalent_diameter_area)
        min_row, min_col, max_row, max_col = region.bbox
        edge_touched = (
            min_row <= cfg.edge_touch_margin_px
            or min_col <= cfg.edge_touch_margin_px
            or max_row >= h - cfg.edge_touch_margin_px
            or max_col >= w - cfg.edge_touch_margin_px
        )
        axis_ratio = max(x_px, y_px) / max(min(x_px, y_px), 1e-6)
        anisotropic = axis_ratio > cfg.outlier_axis_ratio
        provisional.append(
            {
                "label_id": int(region.label),
                "centroid_rc": (float(region.centroid[0]), float(region.centroid[1])),
                "equivalent_diameter_px": eqd_px,
                "x_diameter_px": x_px,
                "y_diameter_px": y_px,
                "area_px": int(region.area),
                "bbox": tuple(int(v) for v in region.bbox),
                "solidity": float(region.solidity),
                "eccentricity": float(region.eccentricity),
                "edge_touched": bool(edge_touched),
                "anisotropic": bool(anisotropic),
                "global_outlier": False,
                "reasons": [],
            }
        )

    if not provisional:
        shape = labels.shape
        return [], np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)

    # Global size outlier rejection is reporting-metric based.  Build neutral
    # provisional measurements first so this path uses the same selected-size
    # vector helper as statistics, summaries, and histograms.  Their final
    # validity is determined below after every rejection rule is evaluated.
    provisional_measurements = [
        BeadMeasurement(
            label_id=item["label_id"],
            centroid_rc=item["centroid_rc"],
            equivalent_diameter_px=item["equivalent_diameter_px"],
            equivalent_diameter_m=(item["equivalent_diameter_px"] * pixel_size_m) if pixel_size_m else None,
            x_diameter_px=item["x_diameter_px"],
            y_diameter_px=item["y_diameter_px"],
            x_diameter_m=(item["x_diameter_px"] * pixel_size_m) if pixel_size_m else None,
            y_diameter_m=(item["y_diameter_px"] * pixel_size_m) if pixel_size_m else None,
            area_px=item["area_px"],
            bbox=item["bbox"],
            solidity=item["solidity"],
            eccentricity=item["eccentricity"],
            edge_touched=item["edge_touched"],
            anisotropic=item["anisotropic"],
            global_outlier=False,
            rejected=False,
            reasons=(),
        )
        for item in provisional
    ]
    global_size_outliers = global_size_outlier_flags(provisional_measurements, cfg)

    measurements: list[BeadMeasurement] = []
    valid_mask = np.zeros(labels.shape, dtype=bool)
    outlier_mask = np.zeros(labels.shape, dtype=bool)

    for idx, item in enumerate(provisional):
        reasons = list(item["reasons"])
        if cfg.diameter_size_limits and item["equivalent_diameter_px"] < cfg.min_diameter_px:
            reasons.append("too_small")
        if cfg.diameter_size_limits and item["equivalent_diameter_px"] > cfg.max_diameter_px:
            reasons.append("too_large")
        if item["anisotropic"]:
            reasons.append("anisotropic_xy")
        if item["eccentricity"] > cfg.max_eccentricity:
            reasons.append("high_eccentricity")
        if item["solidity"] < cfg.min_solidity:
            reasons.append("low_solidity")
        if item["edge_touched"] and not cfg.include_edge_candidates:
            reasons.append("edge_touch")
        if cfg.global_size_outliers and global_size_outliers[idx]:
            reasons.append("global_size_outlier")
            item["global_outlier"] = True

        rejected = len(reasons) > 0
        label_mask = labels == item["label_id"]
        if rejected:
            outlier_mask |= label_mask
        else:
            valid_mask |= label_mask

        x_m = item["x_diameter_px"] * pixel_size_m if pixel_size_m else None
        y_m = item["y_diameter_px"] * pixel_size_m if pixel_size_m else None
        measurements.append(
            BeadMeasurement(
                label_id=item["label_id"],
                centroid_rc=item["centroid_rc"],
                equivalent_diameter_px=item["equivalent_diameter_px"],
                equivalent_diameter_m=(item["equivalent_diameter_px"] * pixel_size_m) if pixel_size_m else None,
                x_diameter_px=item["x_diameter_px"],
                y_diameter_px=item["y_diameter_px"],
                x_diameter_m=x_m,
                y_diameter_m=y_m,
                area_px=item["area_px"],
                bbox=item["bbox"],
                solidity=item["solidity"],
                eccentricity=item["eccentricity"],
                edge_touched=item["edge_touched"],
                anisotropic=item["anisotropic"],
                global_outlier=bool(item["global_outlier"]),
                rejected=rejected,
                reasons=tuple(reasons),
            )
        )

    return measurements, valid_mask, outlier_mask


def analyze_bead_image(image_path: str | Path, config: ViewerConfig = ViewerConfig()) -> BeadAnalysisResult:
    # Validate reporting configuration centrally without affecting segmentation.
    normalize_size_distribution_metric(config.size_distribution_metric)
    image_path = Path(image_path)
    hdr_path = _paired_hdr_path(image_path)
    metadata = _read_hdr_metadata(hdr_path) if hdr_path else SEMMetadata(None, None, None, None, "", "", "", "")

    raw = imread(str(image_path))
    cropped, crop_row = _crop_infobar(raw, config, strip_rows=metadata.image_strip_size_px)
    display = _scale_for_display(cropped, config.display_percentiles)

    feature, candidate_mask, labels = _segment_candidates(display, config)
    measurements, valid_mask, outlier_mask = _classify_measurements(labels, metadata, config)

    return BeadAnalysisResult(
        image_path=image_path,
        hdr_path=hdr_path,
        raw=raw,
        cropped=cropped,
        display=display,
        feature=feature,
        candidate_mask=candidate_mask,
        valid_mask=valid_mask,
        outlier_mask=outlier_mask,
        labels=labels,
        measurements=measurements,
        metadata=metadata,
        crop_row=int(crop_row),
    )


def _format_length_m(value_m: Optional[float]) -> str:
    if value_m is None or not math.isfinite(value_m):
        return "n/a"

    abs_val = abs(value_m)
    if abs_val < 1e-6:
        return f"{value_m * 1e9:.0f} nm"
    if abs_val < 1e-3:
        return f"{value_m * 1e6:.2f} um"
    return f"{value_m * 1e3:.3f} mm"


def _format_length_value(value_m: Optional[float]) -> str:
    if value_m is None or not math.isfinite(value_m):
        return "n/a"

    abs_val = abs(value_m)
    if abs_val < 1e-6:
        return f"{value_m * 1e9:.0f}"
    if abs_val < 1e-3:
        return f"{value_m * 1e6:.2f}"
    return f"{value_m * 1e3:.3f}"


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _nice_scale_length_m(target_m: float) -> float:
    if target_m <= 0:
        return 0.0

    exponent = math.floor(math.log10(target_m))
    fraction = target_m / (10 ** exponent)
    for base in (1.0, 2.0, 5.0, 10.0):
        if fraction <= base:
            return base * (10 ** exponent)
    return 10 ** (exponent + 1)


class BeadDatasetViewer:
    def __init__(self, folder: str | Path, config: ViewerConfig = ViewerConfig()):
        self.folder = Path(folder)
        self.config = config
        self.image_paths = sort_paths(list(self.folder.glob("*.tif")))
        if not self.image_paths:
            raise FileNotFoundError(f"No TIFF files found in '{self.folder}'.")

        self._cache: dict[Path, BeadAnalysisResult] = {}
        self.index = 0
        self.show_scale = config.default_show_scale
        self.show_boundaries = config.default_show_boundaries
        self.show_measures = config.default_show_measures

        self.fig = None
        self.ax_image = None
        self.ax_hist = None
        self.ax_info = None
        self.image_artist = None
        self.scale_artists: list[object] = []
        self.measure_artists: list[object] = []
        self.check_buttons: Optional[CheckButtons] = None

    def _get_result(self, index: int) -> BeadAnalysisResult:
        path = self.image_paths[index]
        if path not in self._cache:
            self._cache[path] = analyze_bead_image(path, self.config)
        return self._cache[path]

    def _display_image(self, res: BeadAnalysisResult) -> np.ndarray:
        base = np.dstack([res.display, res.display, res.display]).astype(np.float32)
        if not self.show_boundaries:
            return base

        valid_edges = find_boundaries(res.valid_mask, mode="outer")
        outlier_edges = find_boundaries(res.outlier_mask, mode="outer")
        base[valid_edges] = (0.0, 1.0, 0.0)
        base[outlier_edges] = (1.0, 0.1, 0.1)
        return base

    def _make_scale_overlay(self, res: BeadAnalysisResult) -> list[object]:
        pixel_size_m = res.metadata.mean_pixel_size_m
        if pixel_size_m is None:
            return []

        h, w = res.display.shape
        target_m = w * pixel_size_m * 0.22
        scale_length_m = _nice_scale_length_m(target_m)
        if scale_length_m <= 0:
            return []

        scale_length_px = scale_length_m / pixel_size_m
        x0 = w * 0.06
        y0 = h * 0.92

        bg = Rectangle(
            (x0 - 8, y0 - 28),
            scale_length_px + 16,
            36,
            facecolor=(0.0, 0.0, 0.0, 0.35),
            edgecolor="none",
        )
        bar = Line2D([x0, x0 + scale_length_px], [y0, y0], color="white", linewidth=3)
        tick_l = Line2D([x0, x0], [y0 - 7, y0 + 7], color="white", linewidth=1.5)
        tick_r = Line2D([x0 + scale_length_px, x0 + scale_length_px], [y0 - 7, y0 + 7], color="white", linewidth=1.5)
        label_artist = self.ax_image.text(
            x0,
            y0 - 11,
            _format_length_m(scale_length_m),
            color="white",
            fontsize=10,
            va="bottom",
            ha="left",
            bbox={"facecolor": (0.0, 0.0, 0.0, 0.0), "edgecolor": "none", "pad": 0.0},
        )
        return [bg, bar, tick_l, tick_r, label_artist]

    def _make_measure_overlay(self, res: BeadAnalysisResult) -> list[object]:
        artists: list[object] = []
        for meas in res.measurements:
            row, col = meas.centroid_rc
            color = "cyan" if meas.valid else "red"
            x_half = meas.x_diameter_px / 2.0
            y_half = meas.y_diameter_px / 2.0

            hline = Line2D([col - x_half, col + x_half], [row, row], color=color, linewidth=1.2, alpha=0.95)
            vline = Line2D([col, col], [row - y_half, row + y_half], color=color, linewidth=1.2, alpha=0.95)

            if meas.mean_diameter_m is not None:
                label_text = f"x={_format_length_value(meas.x_diameter_m)}  y={_format_length_value(meas.y_diameter_m)} um"
            else:
                label_text = f"x={meas.x_diameter_px:.1f}  y={meas.y_diameter_px:.1f} px"

            if not meas.valid and meas.reasons:
                label_text += "  !"

            text = self.ax_image.text(
                col,
                row - y_half - 7,
                label_text,
                color=color,
                fontsize=7,
                ha="center",
                va="bottom",
                bbox={"facecolor": (0.0, 0.0, 0.0, 0.45), "edgecolor": "none", "pad": 1.5},
            )
            artists.extend([hline, vline, text])
        return artists

    def _clear_artists(self, artists: list[object]) -> None:
        for artist in artists:
            try:
                artist.remove()
            except ValueError:
                pass
        artists.clear()

    def _valid_diameters(self, res: BeadAnalysisResult) -> np.ndarray:
        """Backward-compatible calibrated selected-metric vector."""

        values_m, _values_px = valid_bead_size_vectors(
            res.measurements, self.config.size_distribution_metric
        )
        return values_m

    def _valid_size_vectors(self, res: BeadAnalysisResult) -> tuple[np.ndarray, np.ndarray]:
        return valid_bead_size_vectors(res.measurements, self.config.size_distribution_metric)

    def _update_hist(self, res: BeadAnalysisResult) -> None:
        self.ax_hist.clear()
        vals_m, vals_px = self._valid_size_vectors(res)
        if vals_m.size or vals_px.size:
            values, unit = (vals_m * 1e6, "µm") if vals_m.size else (vals_px, "px")
            bins = min(max(5, values.size // 2), 12)
            self.ax_hist.hist(values, bins=bins, color="#4cc9f0", edgecolor="#0b1f2a")
            self.ax_hist.set_xlabel(bead_size_metric_label(self.config.size_distribution_metric, unit))
            self.ax_hist.set_ylabel("Count")
        else:
            self.ax_hist.text(0.5, 0.5, "No valid bead measurements", ha="center", va="center", transform=self.ax_hist.transAxes)
            self.ax_hist.set_xticks([])
            self.ax_hist.set_yticks([])
        self.ax_hist.set_title("Size Distribution")

    def _update_info(self, res: BeadAnalysisResult) -> None:
        self.ax_info.clear()
        self.ax_info.axis("off")

        valid_m, valid_px = self._valid_size_vectors(res)
        valid_arr, unit = (valid_m, "m") if valid_m.size else (valid_px, "px")
        outlier_count = sum(1 for m in res.measurements if not m.valid)

        summary = [
            f"File: {res.image_path.name}",
            f"Sample: {res.image_path.parent.name}",
            f"Candidates: {len(res.measurements)}",
            f"Used for stats: {valid_arr.size}",
            f"Flagged red: {outlier_count}",
            f"Size metric: {'equivalent diameter' if self.config.size_distribution_metric == 'equivalent_diameter' else 'mean X/Y diameter'}",
        ]

        if valid_arr.size:
            mean_value = float(valid_arr.mean())
            std_value = float(valid_arr.std(ddof=1)) if valid_arr.size > 1 else 0.0
            sem_value = float(std_value / math.sqrt(valid_arr.size)) if valid_arr.size > 1 else 0.0
            format_value = _format_length_m if unit == "m" else lambda value: f"{value:.2f} px"
            summary.extend(
                [
                    f"Mean diameter: {format_value(mean_value)}",
                    f"SD: {format_value(std_value)}",
                    f"SEM: {format_value(sem_value)}",
                    f"Median: {format_value(float(np.median(valid_arr)))}",
                    f"Range: {format_value(float(valid_arr.min()))} .. {format_value(float(valid_arr.max()))}",
                ]
            )
        else:
            summary.append("No valid measurements after filtering")

        if res.metadata.mean_pixel_size_m:
            summary.append(f"Pixel size: {_format_length_m(res.metadata.mean_pixel_size_m)} / px")
        if res.metadata.device:
            summary.append(f"Instrument: {res.metadata.device}")
        if res.metadata.magnification:
            summary.append(f"Magnification: {res.metadata.magnification:.0f}x")
        if res.metadata.date or res.metadata.time:
            summary.append(f"Acquired: {res.metadata.date} {res.metadata.time}".strip())

        flagged = [m for m in res.measurements if not m.valid][:4]
        if flagged:
            summary.append("")
            summary.append("Flagged examples:")
            for m in flagged:
                summary.append(f"- #{m.label_id}: {', '.join(m.reasons[:2])}")

        self.ax_info.text(
            0.02,
            0.98,
            "\n".join(summary),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            transform=self.ax_info.transAxes,
        )

    def _reset_image_limits(self, res: BeadAnalysisResult) -> None:
        h, w = res.display.shape
        self.ax_image.set_autoscale_on(False)
        self.ax_image.set_xlim(-0.5, w - 0.5)
        self.ax_image.set_ylim(h - 0.5, -0.5)
        self.ax_image.set_aspect("equal")

    def _render_current(self) -> None:
        res = self._get_result(self.index)
        self.image_artist.set_data(self._display_image(res))
        self.ax_image.set_title(f"{self.index + 1}/{len(self.image_paths)}  {res.image_path.name}", fontsize=11)
        self._reset_image_limits(res)

        self._clear_artists(self.scale_artists)
        self._clear_artists(self.measure_artists)

        if self.show_scale:
            self.scale_artists = self._make_scale_overlay(res)
            for artist in self.scale_artists:
                if getattr(artist, "axes", None) is None:
                    self.ax_image.add_artist(artist)

        if self.show_measures:
            self.measure_artists = self._make_measure_overlay(res)
            for artist in self.measure_artists:
                if getattr(artist, "axes", None) is None:
                    self.ax_image.add_artist(artist)

        self._update_hist(res)
        self._update_info(res)
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
        self.show_scale = bool(status[0])
        self.show_boundaries = bool(status[1])
        self.show_measures = bool(status[2])
        self._render_current()

    def show(self) -> None:
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_image = self.fig.add_axes([0.04, 0.10, 0.60, 0.80])
        self.ax_hist = self.fig.add_axes([0.70, 0.54, 0.28, 0.36])
        self.ax_info = self.fig.add_axes([0.70, 0.10, 0.18, 0.30])
        ax_checks = self.fig.add_axes([0.90, 0.10, 0.08, 0.30])

        first = self._get_result(self.index)
        self.image_artist = self.ax_image.imshow(self._display_image(first), vmin=0.0, vmax=1.0)
        self.ax_image.axis("off")

        ax_checks.set_title("Overlays", fontsize=10)
        self.check_buttons = CheckButtons(
            ax_checks,
            ["Scale", "Boundaries", "Measures"],
            [self.show_scale, self.show_boundaries, self.show_measures],
        )
        for text in self.check_buttons.labels:
            text.set_fontsize(10)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.check_buttons.on_clicked(self._on_checks)

        self._render_current()
        self.fig.suptitle("SEM Bead Viewer  |  left/right = next image", fontsize=12)
        plt.show()


def build_bead_summary(folder: str | Path, config: ViewerConfig) -> dict:
    folder = Path(folder)
    image_paths = sort_paths(list(folder.glob("*.tif")))
    images = []
    metric = normalize_size_distribution_metric(config.size_distribution_metric)
    global_values_m: list[float] = []
    global_values_px: list[float] = []

    def _stats(values: Sequence[float]) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if not values:
            return None, None, None, None
        array = np.asarray(values, dtype=np.float64)
        sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        sem = float(sd / math.sqrt(array.size)) if array.size > 1 else 0.0
        return float(np.mean(array)), float(np.median(array)), sd, sem

    for image_path in image_paths:
        res = analyze_bead_image(image_path, config)
        valid = [m for m in res.measurements if m.valid]
        equivalent_px = [float(m.equivalent_diameter_px) for m in valid]
        equivalent_m = [float(m.equivalent_diameter_m) for m in valid if m.equivalent_diameter_m is not None]
        mean_xy_px = [m.mean_xy_diameter_px for m in valid]
        mean_xy_m = [float(m.mean_xy_diameter_m) for m in valid if m.mean_xy_diameter_m is not None]
        selected_m, selected_px = valid_bead_size_vectors(valid, metric)
        selected_values_m = [float(value) for value in selected_m]
        selected_values_px = [float(value) for value in selected_px]

        mean_m, median_m, sd_m, sem_m = _stats(selected_values_m)
        mean_px, median_px, sd_px, sem_px = _stats(selected_values_px)
        image_entry = {
            "file": image_path.name,
            "sample": image_path.parent.name,
            "crop_row": int(res.crop_row),
            "pixel_size_m": _safe_float(res.metadata.mean_pixel_size_m),
            "candidate_count": len(res.measurements),
            "used_count": len(valid),
            "flagged_count": len(res.measurements) - len(valid),
            "date": res.metadata.date,
            "time": res.metadata.time,
            "device": res.metadata.device,
            "magnification": _safe_float(res.metadata.magnification),
            # ``diameters_m`` is retained as a backward-compatible selected
            # calibrated vector; explicit vectors make new summaries unambiguous.
            "diameters_m": [_safe_float(v) for v in selected_values_m],
            "equivalent_diameters_px": [_safe_float(v) for v in equivalent_px],
            "equivalent_diameters_m": [_safe_float(v) for v in equivalent_m],
            "mean_xy_diameters_px": [_safe_float(v) for v in mean_xy_px],
            "mean_xy_diameters_m": [_safe_float(v) for v in mean_xy_m],
            "size_distribution_metric": metric,
            "histogram_metric": metric,
            "selected_diameters_px": [_safe_float(v) for v in selected_values_px],
            "selected_diameters_m": [_safe_float(v) for v in selected_values_m],
            "mean_diameter_m": _safe_float(mean_m),
            "median_diameter_m": _safe_float(median_m),
            "sd_diameter_m": _safe_float(sd_m),
            "sem_diameter_m": _safe_float(sem_m),
            "selected_mean_diameter_m": _safe_float(mean_m),
            "selected_median_diameter_m": _safe_float(median_m),
            "selected_sd_diameter_m": _safe_float(sd_m),
            "selected_sem_diameter_m": _safe_float(sem_m),
            "mean_diameter_px": _safe_float(mean_px),
            "median_diameter_px": _safe_float(median_px),
            "sd_diameter_px": _safe_float(sd_px),
            "sem_diameter_px": _safe_float(sem_px),
            "flagged": [
                {
                    "label_id": int(m.label_id),
                    "reasons": list(m.reasons),
                    "equivalent_diameter_m": _safe_float(m.equivalent_diameter_m),
                    "mean_xy_diameter_m": _safe_float(m.mean_xy_diameter_m),
                }
                for m in res.measurements
                if not m.valid
            ],
        }
        images.append(image_entry)
        global_values_m.extend(value for value in selected_values_m if math.isfinite(value))
        global_values_px.extend(value for value in selected_values_px if math.isfinite(value))

    global_mean_m, global_median_m, global_sd_m, global_sem_m = _stats(global_values_m)
    global_mean_px, global_median_px, global_sd_px, global_sem_px = _stats(global_values_px)

    global_summary = {
        "image_count": len(images),
        "images_with_valid_measurements": sum(1 for image in images if image["used_count"] > 0),
        "total_used_beads": len(global_values_px),
        "size_distribution_metric": metric,
        "selected_diameters_px": [_safe_float(value) for value in global_values_px],
        "selected_diameters_m": [_safe_float(value) for value in global_values_m],
        "mean_diameter_m": _safe_float(global_mean_m),
        "median_diameter_m": _safe_float(global_median_m),
        "sd_diameter_m": _safe_float(global_sd_m),
        "sem_diameter_m": _safe_float(global_sem_m),
        "mean_diameter_px": _safe_float(global_mean_px),
        "median_diameter_px": _safe_float(global_median_px),
        "sd_diameter_px": _safe_float(global_sd_px),
        "sem_diameter_px": _safe_float(global_sem_px),
    }

    return {
        "folder": str(folder),
        "viewer_config": asdict(config),
        "global_summary": global_summary,
        "images": images,
    }


def write_bead_summary_json(folder: str | Path, config: ViewerConfig, output_path: str | Path) -> None:
    summary = build_bead_summary(folder, config)
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


DEFAULT_CONFIG_PATH = Path("sem_bead_viewer_config.json")


def run_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    folder_override: str | Path | None = None,
) -> None:
    """Run the bead viewer from one config file and temporary CLI overrides."""

    config_path = expand_user_path(config_path)
    if not config_path.exists():
        save_default_config(
            config_path,
            Path(__file__).resolve().parent / "testData" / "100226" / "10 kDa, bare",
        )
    app_cfg = load_app_config(config_path)
    effective_folder = (
        expand_user_path(folder_override)
        if folder_override is not None
        else resolve_existing_input_path(
            app_cfg.folder, config_path=config_path, description="image folder"
        )
    )
    if app_cfg.summary_json_path:
        write_bead_summary_json(
            effective_folder, app_cfg.viewer, expand_user_path(app_cfg.summary_json_path)
        )
    BeadDatasetViewer(effective_folder, app_cfg.viewer).show()


def main(config_path: str | Path = "sem_bead_viewer_config.json") -> None:
    run_from_config(config_path)


if __name__ == "__main__":
    main()

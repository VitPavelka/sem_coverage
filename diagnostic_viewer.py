"""Unified interactive diagnostics for image-analysis modes.

The GUI in this module is intentionally analysis-agnostic.  Mode adapters own
configuration metadata, validation, result rendering, overlays, and summaries.
Only the SEM bead adapter is implemented at present.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import CancelledError, Future
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
from pathlib import Path
from queue import Empty, Queue
from threading import RLock, Thread
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import CloseEvent, KeyEvent, MouseEvent
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import (
    Button,
    CheckButtons,
    RadioButtons,
    RangeSlider,
    Slider,
    TextBox,
)
from matplotlib.patches import Circle, Wedge
import numpy as np
from skimage.segmentation import find_boundaries
from tifffile import imread

from sem_bead_viewer import (
    AppConfig,
    BeadAnalysisResult,
    ViewerConfig,
    _format_length_m,
    _nice_scale_length_m,
    analyze_bead_image,
    bead_size_metric_label,
    normalize_size_distribution_metric,
    valid_bead_size_vectors,
    load_app_config,
)
from sem_coverage_viewer import (
    BeadCoverageResult,
    BeadMetrics,
    CoverageAppConfig,
    CoverageImageResult,
    CoverageSegmentationFailure,
    CoverageSegmentationDiagnostics,
    CoverageViewerConfig,
    FailedImagePreview,
    _format_length_m as _format_coverage_length_m,
    _include_roi_in_global_summary,
    _load_preprocessed_image,
    _nice_scale_length_m as _nice_coverage_scale_length_m,
    _paired_hdr_path,
    _read_hdr_metadata,
    analyze_coverage_image,
    load_app_config as load_coverage_app_config,
    load_failed_image_preview,
)
from coverage_cap import (
    annular_cap_profile,
    compute_coverage_cap_metrics,
    cumulative_cap_sweep,
    normalize_cap_coverage_metric,
    normalize_sphere_diameter_metric,
    sphere_geometry_from_mask,
    maximum_inscribed_circle,
)
from coverage_homogeneity import compute_homogeneity
from coverage_local_heterogeneity import (
    aggregate_local_rotation_results,
    compute_local_grid_at_rotation,
    compute_local_heterogeneity,
    compute_local_rotation_series,
    local_grid_index_maps,
    local_grid_indices,
)
from sem_coverage import AnalyzerConfig, SEMCoverageAnalyzer
from path_utils import discover_images_recursive, path_to_config_text, resolve_optional_file_in_folder

LOGGER = logging.getLogger(__name__)


def _coverage_branch_label(config: CoverageViewerConfig) -> str:
    """Describe the existing coverage-mask branch without changing its logic."""

    return "secondary coverage branch" if config.ag_enable_secondary_coverage else "primary-only coverage"

ParameterKind = Literal["float", "int", "bool", "range", "text", "choice"]
ActivityPredicate = Callable[[Any], tuple[bool, str] | bool]
ConditionalNote = Callable[[Any], str | None]


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata for one diagnostic control."""

    name: str
    config_names: tuple[str, ...]
    config_paths: tuple[tuple[str, ...], ...] | None
    label: str
    group: str
    kind: ParameterKind
    minimum: float | None
    maximum: float | None
    step: float | None
    requires_analysis: bool
    help_text: str
    text_parser: Callable[[str], Any] | None = None
    choices: tuple[tuple[str, str], ...] | None = None
    format_value: Callable[[Any], str] | None = None
    placeholder: str | None = None
    active_when: ActivityPredicate | None = None
    note_when: ConditionalNote | None = None


@dataclass(frozen=True)
class BoundaryLayer:
    """One named boundary overlay layer."""

    control_label: str
    mask: np.ndarray
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class AxisChord:
    """One mask-constrained measurement chord."""

    start_row: float
    start_col: float
    end_row: float
    end_col: float
    length_px: float


@dataclass(frozen=True)
class MeasurementOverlay:
    """One measurement overlay with independent line and label controls."""

    row: float
    col: float
    x_diameter_px: float
    y_diameter_px: float
    label_text: str
    x_color: str
    y_color: str
    label_color: str
    line_control_label: str
    label_control_label: str
    x_start_row: float | None = None
    x_start_col: float | None = None
    x_end_row: float | None = None
    x_end_col: float | None = None
    y_start_row: float | None = None
    y_start_col: float | None = None
    y_end_row: float | None = None
    y_end_col: float | None = None
    x_label_text: str | None = None
    x_label_row: float | None = None
    x_label_col: float | None = None
    x_label_color: str | None = None
    x_label_ha: str = "center"
    x_label_va: str = "bottom"
    y_label_text: str | None = None
    y_label_row: float | None = None
    y_label_col: float | None = None
    y_label_color: str | None = None
    y_label_ha: str = "left"
    y_label_va: str = "center"


@dataclass(frozen=True)
class PointOverlay:
    """One point-marker overlay layer."""

    control_label: str
    rows: np.ndarray
    cols: np.ndarray
    color: str
    marker: str = "."
    markersize: float = 4.0


@dataclass(frozen=True)
class TextOverlay:
    """One text annotation overlay."""

    control_label: str
    row: float
    col: float
    text: str
    color: str
    fontsize: float = 8.0
    ha: str = "center"
    va: str = "bottom"
    boxed: bool = False


@dataclass(frozen=True)
class OverlayData:
    """Mode-independent data used by the generic overlay renderer."""

    boundary_layers: tuple[BoundaryLayer, ...]
    measurements: tuple[MeasurementOverlay, ...]
    point_layers: tuple[PointOverlay, ...]
    text_overlays: tuple[TextOverlay, ...]
    pixel_size_m: float | None
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class AnalysisCompletion:
    """One completed, cached, or failed analysis request."""

    generation: int
    image_path: Path
    config: Any
    result: Any | None
    duration_s: float
    error: BaseException | None = None
    cached: bool = False
    wait_s: float = 0.0


class DiagnosticAdapter(Protocol):
    """Interface implemented by each supported diagnostic analysis mode."""

    mode_name: str
    stages: tuple[str, ...]
    overlay_stage: str | None
    inactive_fields: tuple[str, ...]
    measurement_line_label: str | None
    measurement_text_label: str | None
    scale_bar_label: str | None
    supports_roi_selection: bool

    def load_config(self, config_path: Path) -> tuple[Any, Any, dict[str, Any]]:
        """Load app/config objects and the untouched JSON mapping."""

    def resolve_images(
        self,
        folder: Path,
        app_config: Any,
        selected_file: str | Path | None = None,
    ) -> list[Path]:
        """Return supported images from a folder."""

    def analyze(self, image_path: Path, config: Any) -> Any:
        """Run the production analysis implementation."""

    def parameter_groups(self) -> Mapping[str, tuple[ParameterSpec, ...]]:
        """Return ordered parameter groups."""

    def update_config(self, base_config: Any, values: Mapping[str, Any]) -> Any:
        """Build an immutable effective config from control values."""

    def validate_config(self, config: Any, image_path: Path | None = None) -> str | None:
        """Return a user-facing validation message, or ``None``."""

    def render_stage(
        self,
        result: Any,
        stage: str,
        roi_selection: int | None = None,
        current_config: Any | None = None,
    ) -> np.ndarray:
        """Build display pixels for one stage without rerunning analysis."""

    def make_overlay_data(
        self,
        result: Any,
        roi_selection: int | None = None,
        current_config: Any | None = None,
    ) -> OverlayData:
        """Extract generic overlay data from a result."""

    def summarize_result(
        self,
        result: Any,
        duration_s: float,
        roi_selection: int | None = None,
        current_config: Any | None = None,
    ) -> str:
        """Return the compact numerical status text."""

    def overlay_labels(self, config: Any) -> tuple[str, ...]:
        """Return overlay checkbox labels for the current mode."""

    def default_overlay_state(self, config: Any) -> dict[str, bool]:
        """Return default overlay visibility keyed by checkbox label."""

    def initial_preview(self, image_path: Path, config: Any) -> np.ndarray | None:
        """Return a cheap full-size preview image before analysis completes."""

    def load_failed_preview(self, image_path: Path, config: Any) -> Any | None:
        """Return a mode-specific failure preview, if supported."""

    def render_failed_preview(
        self, preview: Any, stage: str, current_config: Any | None = None
    ) -> np.ndarray:
        """Render one stage for a failure preview without running analysis."""
        

    def summarize_failed_preview(
        self,
        preview: Any | None,
        error: BaseException,
        duration_s: float,
        stage: str,
    ) -> str:
        """Return status text when only a failure preview is available."""

    def roi_options(self, result: Any | None) -> tuple[int, ...]:
        """Return selectable ROI identifiers for the current result."""

    def format_roi_selection(self, roi_selection: int | None) -> str | None:
        """Return a concise title/status label for the current ROI selection."""

    def stage_message(
        self,
        result: Any | None,
        stage: str,
        pixels: np.ndarray,
        roi_selection: int | None = None,
        preview: Any | None = None,
    ) -> str:
        """Return an optional stage-specific message."""


def _spec(
    name: str,
    label: str,
    group: str,
    kind: ParameterKind,
    minimum: float | None,
    maximum: float | None,
    step: float | None,
    help_text: str,
    *,
    config_names: tuple[str, ...] | None = None,
    config_paths: tuple[tuple[str, ...], ...] | None = None,
    requires_analysis: bool = True,
    text_parser: Callable[[str], Any] | None = None,
    choices: tuple[tuple[str, str], ...] | None = None,
    format_value: Callable[[Any], str] | None = None,
    placeholder: str | None = None,
    active_when: ActivityPredicate | None = None,
    note_when: ConditionalNote | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        config_names=config_names or (name,),
        config_paths=config_paths,
        label=label,
        group=group,
        kind=kind,
        minimum=minimum,
        maximum=maximum,
        step=step,
        requires_analysis=requires_analysis,
        help_text=help_text,
        text_parser=text_parser,
        choices=choices,
        format_value=format_value,
        placeholder=placeholder,
        active_when=active_when,
        note_when=note_when,
    )


def _get_by_path(config: Any, path: Sequence[str]) -> Any:
    current = config
    for part in path:
        current = getattr(current, part)
    return current


def _replace_by_path(config: Any, path: Sequence[str], value: Any) -> Any:
    if len(path) == 1:
        return replace(config, **{path[0]: value})
    child = getattr(config, path[0])
    replaced = _replace_by_path(child, path[1:], value)
    return replace(config, **{path[0]: replaced})


def _scale_display_image(
    image: np.ndarray, percentiles: tuple[float, float]
) -> np.ndarray:
    """Scale one grayscale image with the production display-percentile formula."""

    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError(f"Display scaling requires a non-empty 2D image, got {gray.shape}.")
    low_p, high_p = map(float, percentiles)
    low, high = np.percentile(gray, (low_p, high_p))
    high = max(float(high), float(low) + 1e-6)
    scaled = (gray - float(low)) / (high - float(low))
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _contiguous_true_run(line: np.ndarray, anchor_index: int) -> tuple[int, int] | None:
    """Return the inclusive True-run containing ``anchor_index``."""

    if line.ndim != 1 or line.size == 0:
        return None
    if anchor_index < 0 or anchor_index >= line.size or not bool(line[anchor_index]):
        return None
    start = int(anchor_index)
    end = int(anchor_index)
    while start > 0 and bool(line[start - 1]):
        start -= 1
    while end + 1 < line.size and bool(line[end + 1]):
        end += 1
    return start, end


def _mask_xy_chords(
    mask: np.ndarray,
    centroid_rc: tuple[float, float],
) -> tuple[AxisChord | None, AxisChord | None]:
    """Return horizontal and vertical mask chords anchored near the centroid."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.shape[0] < 1 or binary.shape[1] < 1:
        raise ValueError(f"Chord mask must be a non-empty 2D array, got {binary.shape}.")
    mask_points = np.argwhere(binary)
    if mask_points.size == 0:
        raise ValueError("Chord mask does not contain any foreground pixels.")

    centroid_row = float(centroid_rc[0])
    centroid_col = float(centroid_rc[1])
    anchor_row = int(np.clip(np.rint(centroid_row), 0, binary.shape[0] - 1))
    anchor_col = int(np.clip(np.rint(centroid_col), 0, binary.shape[1] - 1))
    if not binary[anchor_row, anchor_col]:
        deltas = mask_points.astype(np.float64)
        deltas[:, 0] -= centroid_row
        deltas[:, 1] -= centroid_col
        nearest_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        anchor_row, anchor_col = map(int, mask_points[nearest_index])

    horizontal: AxisChord | None = None
    vertical: AxisChord | None = None
    horizontal_run = _contiguous_true_run(binary[anchor_row, :], anchor_col)
    if horizontal_run is not None:
        start_col, end_col = horizontal_run
        horizontal = AxisChord(
            start_row=float(anchor_row),
            start_col=float(start_col),
            end_row=float(anchor_row),
            end_col=float(end_col),
            length_px=float(end_col - start_col + 1),
        )
    vertical_run = _contiguous_true_run(binary[:, anchor_col], anchor_row)
    if vertical_run is not None:
        start_row, end_row = vertical_run
        vertical = AxisChord(
            start_row=float(start_row),
            start_col=float(anchor_col),
            end_row=float(end_row),
            end_col=float(anchor_col),
            length_px=float(end_row - start_row + 1),
        )
    return horizontal, vertical


class BeadsDiagnosticAdapter:
    """Diagnostic adapter backed exclusively by the production bead analysis."""

    mode_name = "beads"
    overlay_stage = "overlay"
    measurement_line_label = "Dimension lines"
    measurement_text_label = "Measurement labels"
    scale_bar_label = "Scale bar"
    supports_roi_selection = False
    stages = (
        "display",
        "feature",
        "candidate_mask",
        "labels",
        "valid_mask",
        "outlier_mask",
        "overlay",
    )
    inactive_fields = ("peak_min_distance_px", "peak_threshold_px", "boundary_linewidth")

    _GROUPS: dict[str, tuple[ParameterSpec, ...]] = {
        "Preprocessing": (
            _spec(
                "infobar_tail_rows", "Infobar tail rows", "Preprocessing", "int", 10, 1000, 1,
                "Rows searched for the SEM information bar. Increase to search farther upward; "
                "decrease to limit cropping to the image tail.",
            ),
            _spec(
                "infobar_k_mad", "Infobar MAD factor", "Preprocessing", "float", 1.0, 30.0, 0.25,
                "Brightness threshold for automatic infobar detection. Increase for less sensitive "
                "cropping; decrease when the infobar is not detected.",
            ),
            _spec(
                "infobar_min_run", "Infobar minimum run", "Preprocessing", "int", 1, 100, 1,
                "Consecutive bright rows required for an infobar. Increase to reject short artifacts; "
                "decrease when a narrow infobar is missed.",
            ),
            _spec(
                "display_percentiles", "Display percentiles", "Preprocessing", "range", 0.0, 100.0, 0.1,
                "Percentile contrast range used by display and segmentation. Narrow it for stronger "
                "contrast; widen it to preserve intensity extremes.",
            ),
        ),
        "Detection": (
            _spec(
                "dog_sigma_small", "DoG small sigma", "Detection", "float", 0.1, 12.0, 0.1,
                "Fine Difference-of-Gaussians scale. Increase to suppress fine noise; decrease to "
                "retain smaller bead features.",
            ),
            _spec(
                "dog_sigma_large", "DoG large sigma", "Detection", "float", 0.5, 40.0, 0.25,
                "Broad Difference-of-Gaussians scale. Increase for broader background removal; "
                "decrease to emphasize smaller structures.",
            ),
            _spec(
                "dog_foreground_percentile", "DoG foreground percentile", "Detection", "float",
                0.0, 100.0, 0.5,
                "Selects strong DoG responses used for threshold estimation. Increase for stricter "
                "features; decrease to include weaker features and more noise.",
            ),
            _spec(
                "intensity_percentile", "Intensity percentile", "Detection", "float", 0.0, 100.0, 0.1,
                "Requires candidates to belong to the brightest image pixels. Increase for stricter "
                "bright-object selection; decrease when dim beads are missed.",
            ),
        ),
        "Morphology and size": (
            _spec(
                "closing_radius", "Closing radius", "Morphology and size", "int", 0, 20, 1,
                "Closes small gaps and can connect nearby foreground pixels. Increase when bead masks "
                "are fragmented; decrease when neighboring beads merge.",
            ),
            _spec(
                "opening_radius", "Opening radius", "Morphology and size", "int", 0, 20, 1,
                "Removes small noise and thin bridges. Increase to clean masks; decrease when small "
                "beads are eroded or lost.",
            ),
            _spec(
                "min_object_area_px", "Minimum object area [px]", "Morphology and size", "int",
                1, 5000, 1,
                "Objects smaller than this area are removed. Increase to suppress small noise; "
                "decrease when small beads are missed.",
            ),
            _spec(
                "diameter_size_limits", "Use diameter limits", "Morphology and size", "bool",
                None, None, None,
                "Enables minimum and maximum equivalent-diameter rejection. Enable for strict size "
                "filtering; disable to retain all sizes.",
            ),
            _spec(
                "diameter_px_range", "Diameter limits [px]", "Morphology and size", "range",
                1.0, 250.0, 0.5,
                "Accepted equivalent-diameter interval. Raise the low limit to reject small objects; "
                "raise the high limit to retain larger objects.",
                config_names=("min_diameter_px", "max_diameter_px"),
            ),
            _spec(
                "size_distribution_metric", "Size distribution metric", "Morphology and size", "choice",
                None, None, None,
                "Selects the scalar used for reported bead distributions, statistics, and global MAD "
                "size-outlier rejection. It does not affect segmentation, equivalent-diameter filters, "
                "or X/Y overlays.",
                requires_analysis=True,
                choices=(
                    ("equivalent_diameter", "Equivalent diameter"),
                    ("mean_xy_diameter", "Mean X/Y diameter"),
                ),
            ),
        ),
        "Watershed splitting": (
            _spec(
                "use_watershed_split", "Use watershed split", "Watershed splitting", "bool",
                None, None, None,
                "Enables splitting of connected candidates. Enable for touching beads; disable if "
                "single beads are being divided.",
            ),
            _spec(
                "split_only_suspicious", "Split only suspicious", "Watershed splitting", "bool",
                None, None, None,
                "Limits splitting to large, elongated, or low-solidity candidates. Enable to reduce "
                "over-splitting; disable to test every candidate.",
            ),
            _spec(
                "split_min_distance_px", "Marker spacing [px]", "Watershed splitting", "int",
                1, 80, 1,
                "Minimum spacing between watershed markers. Decrease to permit more aggressive "
                "splitting; increase to prevent over-splitting.",
            ),
            _spec(
                "split_threshold_px", "Peak threshold [px]", "Watershed splitting", "float",
                0.0, 50.0, 0.25,
                "Minimum distance-transform peak height. Increase to reject shallow split markers; "
                "decrease to split less distinct contacts.",
            ),
            _spec(
                "split_peak_count_range", "Allowed peak count", "Watershed splitting", "range",
                1, 12, 1,
                "Allowed watershed-marker count. Raise the minimum to require larger clusters; raise "
                "the maximum to permit more children.",
                config_names=("split_min_peak_count", "split_max_peak_count"),
            ),
            _spec(
                "split_min_child_area_px", "Minimum child area [px]", "Watershed splitting", "int",
                1, 5000, 1,
                "Minimum area accepted for every split child. Increase to reject tiny fragments; "
                "decrease for genuinely small touching beads.",
            ),
            _spec(
                "split_child_diameter_range", "Child diameter [px]", "Watershed splitting", "range",
                1.0, 250.0, 0.5,
                "Accepted equivalent-diameter interval for split children. Narrow it to reject "
                "implausible splits; widen it to accept more child sizes.",
                config_names=("split_min_child_diameter_px", "split_max_child_diameter_px"),
            ),
            _spec(
                "split_trigger_diameter_px", "Trigger diameter [px]", "Watershed splitting", "float",
                1.0, 250.0, 0.5,
                "Diameter that marks a candidate as suspicious. Decrease to attempt splitting more "
                "often; increase to leave large single beads intact.",
            ),
            _spec(
                "split_trigger_axis_ratio", "Trigger axis ratio", "Watershed splitting", "float",
                1.0, 5.0, 0.01,
                "Axis ratio that triggers splitting. Decrease to inspect mildly elongated objects; "
                "increase to target only strongly elongated clusters.",
            ),
            _spec(
                "split_trigger_solidity_below", "Trigger solidity below", "Watershed splitting",
                "float", 0.0, 1.0, 0.01,
                "Solidity below which splitting is attempted. Increase to split more irregular "
                "objects; decrease to target only deeply concave clusters.",
            ),
        ),
        "Filtering": (
            _spec(
                "outlier_axis_ratio", "Maximum axis ratio", "Filtering", "float", 1.0, 5.0, 0.01,
                "Maximum accepted x/y dimension ratio. Increase to retain elongated objects; decrease "
                "for stricter roundness.",
            ),
            _spec(
                "global_size_outliers", "Global size outliers", "Filtering", "bool",
                None, None, None,
                "Enables robust diameter-outlier rejection across the image. Enable to reject unusual "
                "sizes; disable when mixed bead sizes are expected.",
            ),
            _spec(
                "outlier_mad_zscore", "MAD z-score limit", "Filtering", "float", 0.1, 15.0, 0.1,
                "Robust size-outlier threshold. Increase to retain more size variation; decrease for "
                "stricter rejection.",
            ),
            _spec(
                "min_solidity", "Minimum solidity", "Filtering", "float", 0.0, 1.0, 0.01,
                "Minimum object-to-convex-hull area ratio. Increase to reject lobed clusters; decrease "
                "to retain irregular beads.",
            ),
            _spec(
                "max_eccentricity", "Maximum eccentricity", "Filtering", "float", 0.0, 1.0, 0.01,
                "Maximum accepted ellipse eccentricity. Decrease for stricter circularity; increase "
                "to retain elongated objects.",
            ),
            _spec(
                "edge_touch_margin_px", "Edge margin [px]", "Filtering", "int", 0, 100, 1,
                "Margin used to identify edge-touching candidates. Increase to flag objects near an "
                "edge; decrease to flag only direct contacts.",
            ),
            _spec(
                "include_edge_candidates", "Include edge candidates", "Filtering", "bool",
                None, None, None,
                "Allows candidates touching the edge to remain valid. Disable to reject cropped "
                "objects; enable when edge objects should be measured.",
            ),
        ),
    }

    def load_config(self, config_path: Path) -> tuple[AppConfig, ViewerConfig, dict[str, Any]]:
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: '{config_path}'.") from None
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in '{config_path}' at line {exc.lineno}, column {exc.colno}."
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration root in '{config_path}' must be a JSON object.")
        try:
            app_config = load_app_config(config_path)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bead configuration in '{config_path}': {exc}") from exc
        for field_name in self.inactive_fields:
            LOGGER.info("Inactive bead configuration field omitted from diagnostics: %s", field_name)
        return app_config, app_config.viewer, raw

    def resolve_images(
        self,
        folder: Path,
        app_config: AppConfig,
        selected_file: str | Path | None = None,
    ) -> list[Path]:
        if not folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: '{folder}'.")
        if not folder.is_dir():
            raise NotADirectoryError(f"Image folder is not a directory: '{folder}'.")
        file_choice = selected_file if selected_file is not None else app_config.file
        if file_choice:
            selected = resolve_optional_file_in_folder(folder, file_choice, description="selected image")
            if selected is None or not selected.is_file():
                raise FileNotFoundError(f"Selected image is not a file: '{file_choice}'.")
            return [selected.resolve()]
        return discover_images_recursive(folder)

    def analyze(self, image_path: Path, config: ViewerConfig) -> BeadAnalysisResult:
        return analyze_bead_image(image_path, config=config)

    def parameter_groups(self) -> Mapping[str, tuple[ParameterSpec, ...]]:
        return self._GROUPS

    def update_config(
        self, base_config: ViewerConfig, values: Mapping[str, Any]
    ) -> ViewerConfig:
        config = base_config
        for specs in self._GROUPS.values():
            for spec in specs:
                if spec.name not in values:
                    continue
                paths = spec.config_paths or tuple((name,) for name in spec.config_names)
                value = values[spec.name]
                if spec.kind == "range":
                    low, high = value
                    convert = (
                        (lambda item: int(round(item)))
                        if spec.step == 1
                        else (lambda item: float(item))
                    )
                    converted = (convert(low), convert(high))
                    if len(paths) == 1:
                        config = _replace_by_path(config, paths[0], converted)
                    else:
                        config = _replace_by_path(config, paths[0], converted[0])
                        config = _replace_by_path(config, paths[1], converted[1])
                elif spec.kind == "int":
                    config = _replace_by_path(config, paths[0], int(round(value)))
                elif spec.kind == "bool":
                    config = _replace_by_path(config, paths[0], bool(value))
                elif spec.kind == "text":
                    parsed = spec.text_parser(value) if spec.text_parser else value
                    config = _replace_by_path(config, paths[0], parsed)
                elif spec.kind == "choice":
                    config = _replace_by_path(config, paths[0], str(value))
                else:
                    config = _replace_by_path(config, paths[0], float(value))
        return config

    def validate_config(
        self, config: ViewerConfig, image_path: Path | None = None
    ) -> str | None:
        try:
            normalize_size_distribution_metric(config.size_distribution_metric)
        except ValueError as exc:
            return str(exc)
        if config.infobar_tail_rows < 1 or config.infobar_min_run < 1:
            return "Infobar row counts must be positive."
        if config.infobar_k_mad <= 0:
            return "The infobar MAD factor must be positive."
        low_percentile, high_percentile = config.display_percentiles
        if not (0.0 <= low_percentile < high_percentile <= 100.0):
            return "Display percentiles must satisfy 0 <= low < high <= 100."
        if config.dog_sigma_small <= 0 or config.dog_sigma_large <= 0:
            return "DoG sigma values must be positive."
        if config.dog_sigma_small >= config.dog_sigma_large:
            return "DoG small sigma must be lower than DoG large sigma."
        if not (
            0.0 <= config.dog_foreground_percentile <= 100.0
            and 0.0 <= config.intensity_percentile <= 100.0
        ):
            return "Detection percentiles must be between 0 and 100."
        if config.closing_radius < 0 or config.opening_radius < 0:
            return "Morphology radii must be non-negative."
        if config.min_object_area_px < 1:
            return "Minimum object area must be positive."
        if config.min_diameter_px < 0 or config.max_diameter_px < 0:
            return "Diameter limits must be non-negative."
        if config.min_diameter_px > config.max_diameter_px:
            return "Minimum bead diameter must not exceed maximum bead diameter."
        if config.split_min_distance_px < 1 or config.split_threshold_px < 0:
            return "Watershed spacing must be positive and its threshold non-negative."
        if config.split_min_peak_count < 1 or config.split_max_peak_count < 1:
            return "Watershed peak counts must be positive."
        if config.split_min_child_area_px < 1:
            return "Minimum watershed child area must be positive."
        if (
            config.split_min_child_diameter_px < 0
            or config.split_max_child_diameter_px < 0
        ):
            return "Watershed child diameters must be non-negative."
        if config.split_min_child_diameter_px > config.split_max_child_diameter_px:
            return "Minimum child diameter must not exceed maximum child diameter."
        if config.split_min_peak_count > config.split_max_peak_count:
            return "Minimum peak count must not exceed maximum peak count."
        if config.split_trigger_diameter_px < 0:
            return "Watershed trigger diameter must be non-negative."
        if config.split_trigger_axis_ratio < 1:
            return "Watershed trigger axis ratio must be at least 1."
        if not 0.0 <= config.split_trigger_solidity_below <= 1.0:
            return "Watershed trigger solidity must be between 0 and 1."
        if config.outlier_axis_ratio < 1:
            return "Maximum outlier axis ratio must be at least 1."
        if config.outlier_mad_zscore <= 0:
            return "The MAD z-score threshold must be positive."
        if not 0.0 <= config.min_solidity <= 1.0:
            return "Minimum solidity must be between 0 and 1."
        if not 0.0 <= config.max_eccentricity <= 1.0:
            return "Maximum eccentricity must be between 0 and 1."
        if config.edge_touch_margin_px < 0:
            return "Edge-touch margin must be non-negative."
        return None

    def render_stage(
        self,
        result: BeadAnalysisResult,
        stage: str,
        roi_selection: int | None = None,
        current_config: ViewerConfig | None = None,
    ) -> np.ndarray:
        if stage in ("display", "overlay"):
            return np.dstack((result.display, result.display, result.display)).astype(np.float32)
        if stage == "feature":
            feature = result.feature.astype(np.float32)
            lo, hi = np.percentile(feature, (1.0, 99.0))
            if hi <= lo:
                return np.zeros((*feature.shape, 3), dtype=np.float32)
            normalized = np.clip((feature - lo) / (hi - lo), 0.0, 1.0)
            return plt.get_cmap("magma")(normalized)[..., :3].astype(np.float32)
        if stage == "labels":
            labels = result.labels
            rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
            positive = labels > 0
            if positive.any():
                rgb[positive] = plt.get_cmap("tab20")((labels[positive] - 1) % 20)[..., :3]
            return rgb
        mask_by_stage = {
            "candidate_mask": result.candidate_mask,
            "valid_mask": result.valid_mask,
            "outlier_mask": result.outlier_mask,
        }
        if stage not in mask_by_stage:
            raise ValueError(f"Unknown beads diagnostic stage: '{stage}'.")
        gray = mask_by_stage[stage].astype(np.float32)
        return np.dstack((gray, gray, gray))

    def make_overlay_data(
        self,
        result: BeadAnalysisResult,
        roi_selection: int | None = None,
        current_config: ViewerConfig | None = None,
    ) -> OverlayData:
        measurements = []
        for measurement in result.measurements:
            row, col = measurement.centroid_rc
            if measurement.mean_diameter_m is not None:
                label = (
                    f"x={_format_length_m(measurement.x_diameter_m)}  "
                    f"y={_format_length_m(measurement.y_diameter_m)}"
                )
            else:
                label = (
                    f"x={measurement.x_diameter_px:.1f}  "
                    f"y={measurement.y_diameter_px:.1f} px"
                )
            if not measurement.valid and measurement.reasons:
                label += "  !"
            color = "cyan" if measurement.valid else "red"
            measurements.append(
                MeasurementOverlay(
                    row=float(row),
                    col=float(col),
                    x_diameter_px=float(measurement.x_diameter_px),
                    y_diameter_px=float(measurement.y_diameter_px),
                    label_text=label,
                    x_color=color,
                    y_color=color,
                    label_color=color,
                    line_control_label=self.measurement_line_label,
                    label_control_label=self.measurement_text_label,
                )
            )
        return OverlayData(
            boundary_layers=(
                BoundaryLayer(
                    control_label="Valid boundaries",
                    mask=find_boundaries(result.valid_mask, mode="outer"),
                    color=(0.0, 1.0, 0.0, 1.0),
                ),
                BoundaryLayer(
                    control_label="Rejected boundaries",
                    mask=find_boundaries(result.outlier_mask, mode="outer"),
                    color=(1.0, 0.1, 0.1, 1.0),
                ),
                BoundaryLayer(
                    control_label="Candidate boundaries",
                    mask=find_boundaries(result.candidate_mask, mode="outer"),
                    color=(1.0, 0.8, 0.0, 1.0),
                ),
            ),
            measurements=tuple(measurements),
            point_layers=(),
            text_overlays=(),
            pixel_size_m=result.metadata.mean_pixel_size_m,
            image_shape=result.display.shape,
        )

    def summarize_result(
        self,
        result: BeadAnalysisResult,
        duration_s: float,
        roi_selection: int | None = None,
        current_config: ViewerConfig | None = None,
    ) -> str:
        config = current_config or ViewerConfig()
        metric = normalize_size_distribution_metric(config.size_distribution_metric)
        valid = [measurement for measurement in result.measurements if measurement.valid]
        rejected = len(result.measurements) - len(valid)
        physical, pixels = valid_bead_size_vectors(valid, metric)
        metric_text = "equivalent diameter" if metric == "equivalent_diameter" else "mean X/Y diameter"
        if physical.size:
            mean_text = _format_length_m(float(np.mean(physical)))
            median_text = _format_length_m(float(np.median(physical)))
        else:
            mean_text = f"{float(np.mean(pixels)):.2f} px" if pixels.size else "n/a"
            median_text = f"{float(np.median(pixels)):.2f} px" if pixels.size else "n/a"
        pixel_text = (
            f" | pixel {_format_length_m(result.metadata.mean_pixel_size_m)}/px"
            if result.metadata.mean_pixel_size_m
            else ""
        )
        return (
            f"Size metric {metric_text} | Candidates {len(result.measurements)} | valid {len(valid)} | rejected {rejected} | "
            f"mean {mean_text} | median {median_text} | analysis {duration_s:.3f} s{pixel_text}"
        )

    def overlay_labels(self, config: ViewerConfig) -> tuple[str, ...]:
        return (
            "Valid boundaries",
            "Rejected boundaries",
            "Candidate boundaries",
            self.measurement_line_label,
            self.measurement_text_label,
            self.scale_bar_label,
        )

    def default_overlay_state(self, config: ViewerConfig) -> dict[str, bool]:
        return {
            "Valid boundaries": bool(getattr(config, "default_show_boundaries", True)),
            "Rejected boundaries": bool(getattr(config, "default_show_boundaries", True)),
            "Candidate boundaries": False,
            self.measurement_line_label: bool(getattr(config, "default_show_measures", True)),
            self.measurement_text_label: bool(getattr(config, "default_show_measures", True)),
            self.scale_bar_label: bool(getattr(config, "default_show_scale", True)),
        }

    def initial_preview(self, image_path: Path, config: ViewerConfig) -> np.ndarray | None:
        try:
            raw = np.asarray(imread(str(image_path)))
        except (OSError, ValueError) as exc:
            LOGGER.debug("Could not load bead diagnostic preview for %s", image_path, exc_info=exc)
            return None
        raw = np.squeeze(raw)
        if raw.size == 0:
            return None
        if raw.ndim == 3:
            raw = raw[..., 0]
        if raw.ndim != 2:
            return None
        image = raw.astype(np.float32, copy=False)
        low, high = map(float, config.display_percentiles)
        lo, hi = np.percentile(image, (low, high))
        if hi <= lo:
            normalized = np.zeros(image.shape, dtype=np.float32)
        else:
            normalized = np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        return np.dstack((normalized, normalized, normalized))

    def load_failed_preview(self, image_path: Path, config: ViewerConfig) -> Any | None:
        return None

    def render_failed_preview(
        self, preview: Any, stage: str, current_config: ViewerConfig | None = None
    ) -> np.ndarray:
        raise ValueError("Beads mode does not provide failed previews.")

    def summarize_failed_preview(
        self,
        preview: Any | None,
        error: BaseException,
        duration_s: float,
        stage: str,
    ) -> str:
        return f"Analysis failed after {duration_s:.3f} s | {error}"

    def roi_options(self, result: Any | None) -> tuple[int, ...]:
        return ()

    def format_roi_selection(self, roi_selection: int | None) -> str | None:
        return None

    def stage_message(
        self,
        result: BeadAnalysisResult | None,
        stage: str,
        pixels: np.ndarray,
        roi_selection: int | None = None,
        preview: Any | None = None,
    ) -> str:
        if stage.endswith("_mask") and not np.any(pixels):
            return f"{stage.replace('_', ' ').capitalize()}: empty"
        return ""


def _parse_positive_int_list(text: str) -> list[int] | None:
    value = text.strip()
    if value == "":
        return None
    items: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped or not stripped.isdigit():
            raise ValueError(
                "Coverage top-hat radii must be a comma-separated list of positive integers."
            )
        parsed = int(stripped)
        if parsed <= 0:
            raise ValueError("Coverage top-hat radii must be positive integers.")
        items.append(parsed)
    return sorted(set(items))


def _format_positive_int_list(value: Any) -> str:
    if not value:
        return ""
    return ", ".join(str(int(item)) for item in value)


def _coverage_rgb(image: np.ndarray) -> np.ndarray:
    gray = image.astype(np.float32)
    return np.dstack((gray, gray, gray)).astype(np.float32)


def _secondary_coverage_enabled(config: CoverageViewerConfig) -> tuple[bool, str]:
    if config.ag_enable_secondary_coverage:
        return True, ""
    return (
        False,
        "Inactive because the secondary coverage branch is disabled. "
        "The count mask is currently used as the coverage mask.",
    )


def _single_coverage_radius_active(config: CoverageViewerConfig) -> tuple[bool, str]:
    secondary_active, secondary_reason = _secondary_coverage_enabled(config)
    if not secondary_active:
        return secondary_active, secondary_reason
    if not config.ag_coverage_tophat_radii:
        return True, ""
    return (
        False,
        "Inactive because the multi-radius list overrides the single radius. "
        "Clear the Coverage top-hat radii field to use this slider.",
    )


def _adaptive_coverage_active(config: CoverageViewerConfig) -> tuple[bool, str]:
    secondary_active, secondary_reason = _secondary_coverage_enabled(config)
    if not secondary_active:
        return secondary_active, secondary_reason
    if config.ag_coverage_adaptive_threshold:
        return True, ""
    return False, "Inactive because adaptive coverage thresholding is disabled."


def _split_controls_active(config: CoverageViewerConfig) -> tuple[bool, str]:
    if config.split_touching_beads:
        return True, ""
    return False, "Inactive because bead splitting is disabled."


def _global_anisotropy_active(config: CoverageViewerConfig) -> tuple[bool, str]:
    if config.sphere_anisotropy_check:
        return True, ""
    return False, "Inactive because the global anisotropy filter is disabled."


def _global_solidity_active(config: CoverageViewerConfig) -> tuple[bool, str]:
    if config.sphere_solidity_check:
        return True, ""
    return False, "Inactive because the global solidity filter is disabled."


def _morphology_fallback_note(_config: CoverageViewerConfig) -> str:
    return (
        "Conditional effect: used only when the primary bead detector fails or "
        "produces no valid ROI."
    )


def _coverage_union_note(config: CoverageViewerConfig) -> str | None:
    if config.ag_coverage_use_union_with_count:
        return (
            "The count mask is unioned into the coverage mask. This may make "
            "coverage-detector changes appear smaller."
        )
    return None


class CoverageDiagnosticAdapter:
    """Diagnostic adapter backed exclusively by the production coverage analysis."""

    mode_name = "coverage"
    overlay_stage = "overlay"
    measurement_line_label = "Diameter lines"
    # Coverage intentionally uses one control for lines and their values.
    measurement_text_label = "Diameter lines"
    scale_bar_label = "Scale bar"
    supports_roi_selection = True
    inactive_fields: tuple[str, ...] = ()
    stages = (
        "overlay",
        "display",
        "norm",
        "bead_raw",
        "bead_refined",
        "ag_count_feature",
        "ag_coverage_feature",
        "ag_count_mask",
        "ag_coverage_mask",
        "ag_peak_map",
        "roi_index_map",
        "raw_bead_candidates",
        "rejected_bead_candidates",
    )

    INCLUDED_COLOR = (0.0, 1.0, 0.0, 1.0)
    EXCLUDED_COLOR = (1.0, 0.1, 0.1, 1.0)
    AG_COVERAGE_COLOR = (1.0, 0.0, 1.0, 1.0)
    AG_COUNT_COLOR = (1.0, 1.0, 0.0, 1.0)
    AG_PEAK_COLOR = "cyan"
    CAP_COLOR = (1.0, 0.05, 0.05, 1.0)
    DIAMETER_X_COLOR = "cyan"
    DIAMETER_Y_COLOR = "orange"

    def __init__(self) -> None:
        self._cap_graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._homogeneity_cache: dict[tuple[Any, ...], Any] = {}
        self._local_heterogeneity_cache: dict[tuple[Any, ...], Any] = {}
        self._local_heterogeneity_robust_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._local_heterogeneity_cache_result: Any | None = None
        self._inscribed_circle_cache: dict[int, Any] = {}

    def _scope_local_heterogeneity_caches(self, result: Any) -> None:
        """Keep local-grid caches bounded to the most recently used image."""

        if result is self._local_heterogeneity_cache_result:
            return
        self._local_heterogeneity_cache.clear()
        self._local_heterogeneity_robust_cache.clear()
        self._local_heterogeneity_cache_result = result

    @staticmethod
    def _put_bounded_local_cache(
        cache: dict[tuple[Any, ...], Any],
        key: tuple[Any, ...],
        value: Any,
        *,
        limit: int = 64,
    ) -> None:
        """Insert one active-image cache entry without unbounded growth."""

        cache.pop(key, None)
        cache[key] = value
        while len(cache) > limit:
            cache.pop(next(iter(cache)))

    @staticmethod
    def _effective_config(
        result_or_preview: CoverageImageResult | FailedImagePreview,
        current_config: CoverageViewerConfig | None,
    ) -> CoverageViewerConfig:
        if current_config is not None:
            return current_config
        if isinstance(result_or_preview, CoverageImageResult):
            return result_or_preview.config
        raise ValueError("Coverage failure preview rendering requires an explicit config.")

    def _display_image(
        self,
        image: np.ndarray,
        config: CoverageViewerConfig,
    ) -> np.ndarray:
        return _scale_display_image(image, config.analyzer.display_percentiles)

    def _format_chord_label(
        self, chord: AxisChord, pixel_size_m: float | None, axis_name: str
    ) -> str:
        if pixel_size_m is not None:
            return f"{axis_name} = {_format_coverage_length_m(chord.length_px * pixel_size_m)}"
        return f"{axis_name} = {chord.length_px:.1f} px"

    _GROUPS: dict[str, tuple[ParameterSpec, ...]] = {
        "Preprocessing and display": (
            _spec(
                "infobar_tail_rows",
                "Infobar tail rows",
                "Preprocessing and display",
                "int",
                10,
                1000,
                1,
                "Rows searched for the SEM information bar. Increase to search farther upward; decrease to limit cropping to the image tail.",
                config_paths=(("analyzer", "infobar_tail_rows"),),
            ),
            _spec(
                "infobar_k_mad",
                "Infobar MAD factor",
                "Preprocessing and display",
                "float",
                1.0,
                30.0,
                0.25,
                "Brightness threshold for infobar detection. Increase for less sensitive cropping; decrease when the infobar is missed.",
                config_paths=(("analyzer", "infobar_k_mad"),),
            ),
            _spec(
                "infobar_min_run",
                "Infobar minimum run",
                "Preprocessing and display",
                "int",
                1,
                100,
                1,
                "Consecutive bright rows required for an infobar. Increase to reject short artifacts; decrease when a narrow infobar is missed.",
                config_paths=(("analyzer", "infobar_min_run"),),
            ),
            _spec(
                "norm_percentiles",
                "Normalization percentiles",
                "Preprocessing and display",
                "range",
                0.0,
                100.0,
                0.1,
                "Percentile range used for normalization before bead segmentation. Narrow it for stronger segmentation contrast; widen it to preserve more raw intensity variation.",
                config_paths=(("analyzer", "norm_percentiles"),),
            ),
            _spec(
                "display_percentiles",
                "Display percentiles",
                "Preprocessing and display",
                "range",
                0.0,
                100.0,
                0.1,
                "Percentile range used only for visualization. Narrow it for stronger display contrast; widen it to preserve bright and dark extremes.",
                config_paths=(("analyzer", "display_percentiles"),),
                requires_analysis=False,
            ),
            _spec(
                "detector_choice_index",
                "Detector choice index",
                "Preprocessing and display",
                "int",
                0,
                15,
                1,
                "Selects the detector tile when metadata describes multiple detector views. Increase to inspect a later detector tile; decrease to return to earlier detector tiles.",
            ),
        ),
        "Primary bead segmentation": (
            _spec(
                "bead_blur_sigma",
                "Bead blur sigma",
                "Primary bead segmentation",
                "float",
                0.1,
                12.0,
                0.1,
                "Gaussian smoothing used before primary bead thresholding. Increase to suppress texture; decrease to preserve small or sharp bead edges.",
                config_paths=(("analyzer", "bead_blur_sigma"),),
            ),
            _spec(
                "bead_closing_radius",
                "Bead closing radius",
                "Primary bead segmentation",
                "int",
                0,
                20,
                1,
                "Closes small dark gaps in the bead mask. Increase when bead regions are fragmented; decrease when nearby structures merge.",
                config_paths=(("analyzer", "bead_closing_radius"),),
            ),
            _spec(
                "bead_opening_radius",
                "Bead opening radius",
                "Primary bead segmentation",
                "int",
                0,
                20,
                1,
                "Removes small bright bridges and specks from the bead mask. Increase to clean noise; decrease when valid bead edges are lost.",
                config_paths=(("analyzer", "bead_opening_radius"),),
            ),
            _spec(
                "bead_hole_area",
                "Bead hole area",
                "Primary bead segmentation",
                "int",
                0,
                20000,
                1,
                "Fills holes smaller than this inside bead candidates. Increase to repair incomplete beads; decrease when internal voids should remain visible.",
                config_paths=(("analyzer", "bead_hole_area"),),
            ),
            _spec(
                "min_bead_area_px",
                "Minimum bead area [px]",
                "Primary bead segmentation",
                "int",
                1,
                200000,
                1,
                "Rejects candidate bead ROIs smaller than this area. Increase to suppress debris; decrease when valid small beads are missed.",
            ),
            _spec(
                "min_roi_eq_diameter_px",
                "Minimum ROI eq. diameter [px]",
                "Primary bead segmentation",
                "float",
                1.0,
                1000.0,
                1.0,
                "Rejects candidate bead ROIs smaller than this equivalent diameter. Decrease when valid small beads are missed; increase to reject small debris.",
            ),
            _spec(
                "min_roi_solidity",
                "Minimum ROI solidity",
                "Primary bead segmentation",
                "float",
                0.0,
                1.0,
                0.01,
                "Rejects irregular or fragmented bead candidates. Increase for stricter compact-object selection; decrease when real beads have incomplete boundaries.",
            ),
            _spec(
                "max_roi_anisotropy_ratio",
                "Maximum ROI anisotropy",
                "Primary bead segmentation",
                "float",
                1.0,
                5.0,
                0.01,
                "Maximum accepted major/minor axis ratio for a bead ROI. Decrease for stricter spherical selection; increase when perspective or partial imaging makes beads elongated.",
            ),
            _spec(
                "salvage_open_radius_px",
                "Salvage open radius [px]",
                "Primary bead segmentation",
                "int",
                0,
                100,
                1,
                "Attempts to recover a valid bead core by morphological opening. Increase to remove wider connections; decrease to preserve narrow bead regions.",
            ),
        ),
        "Morphology fallback": (
            _spec(
                "bead_morph_fallback",
                "Use morphology fallback",
                "Morphology fallback",
                "bool",
                None,
                None,
                None,
                "Enables the morphology-based fallback when the primary bead segmentation fails. Disable to inspect only the primary detector branch.",
                note_when=_morphology_fallback_note,
            ),
            _spec(
                "bead_morph_downscale",
                "Morphology downscale",
                "Morphology fallback",
                "float",
                0.05,
                1.0,
                0.01,
                "Downscale factor used by the morphology fallback. Increase for more detail; decrease for stronger smoothing and faster fallback segmentation.",
                note_when=_morphology_fallback_note,
            ),
            _spec(
                "bead_morph_blur_sigma",
                "Morphology blur sigma",
                "Morphology fallback",
                "float",
                0.1,
                20.0,
                0.1,
                "Gaussian smoothing before gradient extraction in the fallback branch. Increase to suppress texture; decrease to keep sharper edges.",
                note_when=_morphology_fallback_note,
            ),
            _spec(
                "bead_morph_gradient_percentile",
                "Gradient percentile",
                "Morphology fallback",
                "float",
                0.0,
                100.0,
                0.5,
                "Gradient percentile used to threshold fallback edges. Increase for stricter edge selection; decrease to include weaker boundaries.",
                note_when=_morphology_fallback_note,
            ),
            _spec(
                "bead_morph_close_radius",
                "Fallback close radius",
                "Morphology fallback",
                "int",
                0,
                20,
                1,
                "Closing radius for the fallback edge mask. Increase to bridge gaps; decrease to reduce unintended merges.",
                note_when=_morphology_fallback_note,
            ),
            _spec(
                "bead_morph_dilate_radius",
                "Fallback dilate radius",
                "Morphology fallback",
                "int",
                0,
                20,
                1,
                "Dilation radius for the fallback edge mask. Increase to close wider boundaries; decrease to keep structures separated.",
                note_when=_morphology_fallback_note,
            ),
            _spec(
                "bead_morph_erode_radius_px",
                "Fallback erode radius [px]",
                "Morphology fallback",
                "int",
                0,
                100,
                1,
                "Final erosion applied after upscaling the fallback bead mask. Increase to remove border artifacts; decrease to retain more bead area.",
                note_when=_morphology_fallback_note,
            ),
            _spec(
                "bead_morph_min_object_area_ratio",
                "Fallback min area ratio",
                "Morphology fallback",
                "float",
                0.0,
                1.0,
                0.01,
                "Minimum fallback object area as a fraction of the full image area. Increase to reject small enclosures; decrease to keep smaller candidates.",
                note_when=_morphology_fallback_note,
            ),
        ),
        "Bead splitting": (
            _spec(
                "split_touching_beads",
                "Split touching beads",
                "Bead splitting",
                "bool",
                None,
                None,
                None,
                "Enables watershed-based splitting for suspicious bead clusters. Disable to inspect unsplit candidates.",
            ),
            _spec(
                "split_trigger_eq_diameter_px",
                "Split trigger eq. diameter [px]",
                "Bead splitting",
                "float",
                1.0,
                1500.0,
                1.0,
                "Equivalent diameter threshold that marks a bead candidate as suspicious. Decrease to split more often; increase to leave larger beads intact.",
                active_when=_split_controls_active,
            ),
            _spec(
                "split_trigger_anisotropy_ratio",
                "Split trigger anisotropy",
                "Bead splitting",
                "float",
                1.0,
                5.0,
                0.01,
                "Anisotropy threshold that triggers splitting. Decrease for more aggressive splitting; increase to target only strongly elongated clusters.",
                active_when=_split_controls_active,
            ),
            _spec(
                "split_trigger_solidity_below",
                "Split trigger solidity below",
                "Bead splitting",
                "float",
                0.0,
                1.0,
                0.01,
                "Solidity threshold below which splitting is attempted. Increase to split more irregular objects; decrease to split only deeply concave clusters.",
                active_when=_split_controls_active,
            ),
            _spec(
                "split_min_distance_px",
                "Split minimum distance [px]",
                "Bead splitting",
                "int",
                1,
                200,
                1,
                "Minimum separation between watershed marker peaks. Decrease for more aggressive splitting; increase to avoid over-splitting one bead.",
                active_when=_split_controls_active,
            ),
            _spec(
                "split_peak_threshold_rel",
                "Split peak threshold",
                "Bead splitting",
                "float",
                0.0,
                1.0,
                0.01,
                "Relative watershed peak threshold. Increase to require stronger split markers; decrease to accept weaker bead separations.",
                active_when=_split_controls_active,
            ),
            _spec(
                "split_max_peaks",
                "Split maximum peaks",
                "Bead splitting",
                "int",
                2,
                12,
                1,
                "Maximum number of watershed peaks accepted for one bead cluster. Increase to allow more child ROIs; decrease for stricter splitting.",
                active_when=_split_controls_active,
            ),
            _spec(
                "split_min_child_area_ratio",
                "Split minimum child area ratio",
                "Bead splitting",
                "float",
                0.0,
                1.0,
                0.01,
                "Minimum child area as a fraction of the parent bead area. Increase to reject tiny fragments; decrease to allow smaller valid child beads.",
                active_when=_split_controls_active,
            ),
        ),
        "Ag count detector": (
            _spec(
                "ag_tophat_radius",
                "Ag top-hat radius",
                "Ag count detector",
                "int",
                1,
                50,
                1,
                "Feature scale used for Ag counting. Increase for larger bright Ag structures; decrease for smaller nanoparticles and fine texture.",
                config_paths=(("analyzer", "ag_tophat_radius"),),
            ),
            _spec(
                "ag_min_object_size",
                "Ag min object size",
                "Ag count detector",
                "int",
                1,
                500,
                1,
                "Removes tiny objects from the Ag count mask. Increase to suppress noise; decrease when small Ag nanoparticles are missed.",
                config_paths=(("analyzer", "ag_min_object_size"),),
            ),
            _spec(
                "ag_erode_bead_radius",
                "Ag ROI erode radius",
                "Ag count detector",
                "int",
                0,
                20,
                1,
                "Erodes the bead ROI before Ag counting. Increase to avoid edge artifacts; decrease to keep more of the bead border region.",
                config_paths=(("analyzer", "ag_erode_bead_radius"),),
            ),
            _spec(
                "ag_use_log",
                "Use log intensity",
                "Ag count detector",
                "bool",
                None,
                None,
                None,
                "Applies log compression before Ag feature extraction. Enable when bright outliers dominate; disable to keep raw intensity contrast.",
                config_paths=(("analyzer", "ag_use_log"),),
            ),
            _spec(
                "count_min_distance",
                "Count minimum distance",
                "Ag count detector",
                "int",
                1,
                100,
                1,
                "Minimum spacing between Ag count peaks. Decrease to resolve closer particles; increase to prevent multiple peaks inside one particle.",
                config_paths=(("analyzer", "count_min_distance"),),
            ),
            _spec(
                "count_thr_rel",
                "Count threshold multiplier",
                "Ag count detector",
                "float",
                0.05,
                5.0,
                0.05,
                "Multiplier applied to the Ag peak threshold. Increase for stricter counting; decrease when weak particles are missed. This branch controls projected Ag count, count mask, count feature, and Ag peak markers.",
                config_paths=(("analyzer", "count_thr_rel"),),
            ),
        ),
        "Ag coverage detector": (
            _spec(
                "ag_enable_secondary_coverage",
                "Use secondary coverage branch",
                "Ag coverage detector",
                "bool",
                None,
                None,
                None,
                "Enables a separate segmentation branch for area coverage. When disabled, the Ag count mask is also used as the coverage mask.",
            ),
            _spec(
                "ag_coverage_tophat_radius",
                "Coverage top-hat radius",
                "Ag coverage detector",
                "int",
                1,
                50,
                1,
                "Primary feature scale for Ag area coverage. Increase for broader bright structures; decrease for finer local detail.",
                active_when=_single_coverage_radius_active,
            ),
            _spec(
                "ag_coverage_tophat_radii",
                "Coverage top-hat radii",
                "Ag coverage detector",
                "text",
                None,
                None,
                None,
                "Optional comma-separated list of positive top-hat radii for the coverage detector. Leave empty to use only the single coverage top-hat radius.",
                text_parser=_parse_positive_int_list,
                format_value=_format_positive_int_list,
                placeholder="7, 15, 25",
                active_when=_secondary_coverage_enabled,
            ),
            _spec(
                "ag_coverage_threshold_rel",
                "Coverage threshold multiplier",
                "Ag coverage detector",
                "float",
                0.05,
                5.0,
                0.05,
                "Multiplier applied to the global Otsu threshold of the coverage feature. Increase to reduce detected Ag area; decrease to include weaker Ag regions.",
                active_when=_secondary_coverage_enabled,
            ),
            _spec(
                "ag_coverage_adaptive_threshold",
                "Use adaptive coverage threshold",
                "Ag coverage detector",
                "bool",
                None,
                None,
                None,
                "Enables a local adaptive threshold in the coverage branch. Disable to inspect only the global coverage threshold.",
                active_when=_secondary_coverage_enabled,
            ),
            _spec(
                "ag_coverage_adaptive_block_size",
                "Adaptive block size",
                "Ag coverage detector",
                "int",
                15,
                501,
                2,
                "Odd local window size used by the adaptive coverage threshold. Increase for broader local context; decrease for more local sensitivity.",
                active_when=_adaptive_coverage_active,
            ),
            _spec(
                "ag_coverage_adaptive_k_std",
                "Adaptive k*std",
                "Ag coverage detector",
                "float",
                0.0,
                10.0,
                0.1,
                "Local mean-plus-standard-deviation threshold strength. Increase for stricter local detection; decrease to include weaker local structures.",
                active_when=_adaptive_coverage_active,
            ),
            _spec(
                "ag_coverage_min_object_size",
                "Coverage min object size",
                "Ag coverage detector",
                "int",
                1,
                500,
                1,
                "Removes tiny objects from the Ag coverage mask. Increase to suppress noise; decrease when small covered regions are missed.",
                active_when=_secondary_coverage_enabled,
            ),
            _spec(
                "ag_coverage_closing_radius",
                "Coverage closing radius",
                "Ag coverage detector",
                "int",
                0,
                20,
                1,
                "Closing radius for the Ag coverage mask. Increase to bridge small gaps; decrease to keep nearby regions separate.",
                active_when=_secondary_coverage_enabled,
            ),
            _spec(
                "ag_coverage_use_union_with_count",
                "Union with count mask",
                "Ag coverage detector",
                "bool",
                None,
                None,
                None,
                "Adds the Ag count mask to the coverage mask. Enable to guarantee counted particles contribute to coverage; disable to inspect the independent coverage detector.",
                active_when=_secondary_coverage_enabled,
                note_when=_coverage_union_note,
            ),
        ),
        "Global sphere filters": (
            _spec(
                "sphere_anisotropy_check",
                "Enable anisotropy filter",
                "Global sphere filters",
                "bool",
                None,
                None,
                None,
                "Controls inclusion in global statistics based on bead anisotropy. Disabled ROIs remain visible even when excluded from the summary.",
                requires_analysis=False,
            ),
            _spec(
                "max_global_sphere_anisotropy_ratio",
                "Maximum global anisotropy",
                "Global sphere filters",
                "float",
                1.0,
                5.0,
                0.01,
                "Excludes highly anisotropic bead ROIs from global summary statistics when the associated check is enabled.",
                requires_analysis=False,
                active_when=_global_anisotropy_active,
            ),
            _spec(
                "sphere_solidity_check",
                "Enable solidity filter",
                "Global sphere filters",
                "bool",
                None,
                None,
                None,
                "Controls inclusion in global statistics based on bead solidity. Disabled ROIs remain visible even when excluded from the summary.",
                requires_analysis=False,
            ),
            _spec(
                "min_global_sphere_solidity",
                "Minimum global solidity",
                "Global sphere filters",
                "float",
                0.0,
                1.0,
                0.01,
                "Excludes low-solidity bead ROIs from global summary statistics when the associated check is enabled.",
                requires_analysis=False,
                active_when=_global_solidity_active,
            ),
        ),
        "Sphere geometry": (
            _spec(
                "sphere_diameter_metric", "Sphere geometry", "Sphere geometry", "choice",
                None, None, None,
                "Mean X/Y and equivalent diameter use the bead-mask centroid. Maximum inscribed circle uses its own center and radius. The selected geometry is reused by cap, legacy homogeneity, and local heterogeneity.",
                requires_analysis=False,
                choices=(("mean_xy_diameter", "Mean X/Y diameter"), ("equivalent_diameter", "Equivalent diameter"), ("max_inscribed_circle", "Maximum inscribed circle")),
            ),
        ),
        "Coverage cap": (
            _spec(
                "coverage_cap_enabled", "Enable central coverage cap", "Coverage cap", "bool",
                None, None, None,
                "Uses a central circular cap for the selected coverage metric. Disable to retain legacy full projected coverage.",
                requires_analysis=False,
            ),
            _spec(
                "coverage_cap_radius_fraction", "Cap projected radius fraction", "Coverage cap", "float",
                0.01, 1.0, 0.01,
                "Projected cap radius a/R. Increase to inspect farther toward the sphere edge; decrease to focus on the central surface.",
                requires_analysis=False,
            ),
            _spec(
                "coverage_cap_min_completeness", "Minimum cap completeness", "Coverage cap", "float",
                0.0, 1.0, 0.01,
                "Minimum retained fraction of the theoretical cap circle. Increase to reject caps clipped by an incomplete bead mask.",
                requires_analysis=False,
            ),
            _spec(
                "selected_cap_coverage_metric", "Selected cap coverage metric", "Coverage cap", "choice",
                None, None, None,
                "Selects the cap value reported as the main diagnostic metric. The graph always shows all three cap curves.",
                requires_analysis=False,
                choices=(
                    ("projected_fraction", "Projected fraction"),
                    ("surface_weighted_fraction", "Surface-weighted fraction"),
                    ("projected_over_cap_surface", "Projected Ag / cap surface"),
                ),
            ),
            _spec(
                "coverage_cap_graph_mode", "Cap graph mode", "Coverage cap", "choice",
                None, None, None,
                "Cumulative shows central caps; annular shows non-overlapping local radial rings.",
                requires_analysis=False,
                choices=(
                    ("cumulative", "Cumulative"),
                    ("annular", "Annular"),
                ),
            ),
            _spec("coverage_cap_annulus_width_fraction", "Annulus width fraction", "Coverage cap", "float", 0.01, 0.5, 0.01, "Sets constant non-overlapping annulus width as a fraction of estimated sphere radius.", requires_analysis=False, active_when=lambda c: (c.coverage_cap_graph_mode == "annular", "Inactive unless graph mode is Annular.")),
            _spec("coverage_cap_sensitivity_enabled", "Enable radius sensitivity", "Coverage cap", "bool", None, None, None, "Calculates cap-radius selection sensitivity from existing masks.", requires_analysis=False),
            _spec("coverage_cap_sensitivity_half_width", "Sensitivity half width", "Coverage cap", "float", 0.0, 0.5, 0.01, "Half-width of the cap-radius sensitivity interval around the configured cap fraction.", requires_analysis=False),
            _spec("coverage_cap_sensitivity_step_fraction", "Sensitivity step", "Coverage cap", "float", 0.005, 0.2, 0.005, "Fraction step used for cap-radius sensitivity points.", requires_analysis=False),
        ),
        "Coverage homogeneity (legacy)": (
            _spec("coverage_homogeneity_enabled", "Enable legacy homogeneity", "Coverage homogeneity (legacy)", "bool", None, None, None, "Enables the established separate ring and sector homogeneity workflow.", requires_analysis=False),
            _spec("homogeneity_inner_radius_fraction", "Inner radius fraction", "Coverage homogeneity (legacy)", "float", 0.0, 0.95, 0.01, "Inner analyzed radius in units of R.", requires_analysis=False),
            _spec("homogeneity_outer_radius_fraction", "Outer radius fraction", "Coverage homogeneity (legacy)", "float", 0.05, 1.0, 0.01, "Outer analyzed radius in units of R.", requires_analysis=False),
            _spec("radial_ring_width_fraction", "Radial ring width", "Coverage homogeneity (legacy)", "float", 0.01, 0.5, 0.01, "Constant non-overlapping ring width in units of R.", requires_analysis=False),
            _spec("polar_sector_count", "Polar sector count", "Coverage homogeneity (legacy)", "int", 2, 36, 1, "Equal angular sectors over the configured annulus.", requires_analysis=False),
            _spec("homogeneity_min_segment_completeness", "Minimum segment completeness", "Coverage homogeneity (legacy)", "float", 0.0, 1.0, 0.01, "Minimum retained reference fraction for a valid ring or sector.", requires_analysis=False),
            _spec("homogeneity_view_mode", "Homogeneity view", "Coverage homogeneity (legacy)", "choice", None, None, None, "Select radial rings or polar annular sectors.", requires_analysis=False, choices=(("radial", "Radial"), ("polar", "Polar"))),
            _spec("polar_rotation_samples", "Polar rotation samples", "Coverage homogeneity (legacy)", "int", 1, 36, 1, "Number of grid rotations used for robust production polar heterogeneity.", requires_analysis=False),
            _spec("polar_display_rotation_deg", "Polar display rotation [deg]", "Coverage homogeneity (legacy)", "float", -180.0, 180.0, 1.0, "Display-only sector-grid orientation; it does not change rotation-robust production statistics.", requires_analysis=False),
        ),
        "Coverage local heterogeneity": (
            _spec("local_heterogeneity_enabled", "Enable local grid", "Coverage local heterogeneity", "bool", None, None, None, "Enables a shared radial-band by polar-sector local coverage grid.", requires_analysis=False),
            _spec("local_heterogeneity_inner_radius_fraction", "Inner radius fraction", "Coverage local heterogeneity", "float", 0.0, 0.95, 0.01, "Inner normalized radius of the local polar grid.", requires_analysis=False),
            _spec("local_heterogeneity_outer_radius_fraction", "Outer radius fraction", "Coverage local heterogeneity", "float", 0.05, 1.0, 0.01, "Outer normalized radius of the local polar grid.", requires_analysis=False),
            _spec("local_heterogeneity_radial_band_count", "Radial band count", "Coverage local heterogeneity", "int", 1, 24, 1, "Number of non-overlapping radial bands in the local grid.", requires_analysis=False),
            _spec("local_heterogeneity_polar_sector_count", "Polar sector count", "Coverage local heterogeneity", "int", 2, 48, 1, "Number of non-overlapping angular sectors in the local grid.", requires_analysis=False),
            _spec("local_heterogeneity_polar_rotation_samples", "Polar rotation samples", "Coverage local heterogeneity", "int", 1, 36, 1, "Robust local-grid rotations across one sector width.", requires_analysis=False),
            _spec("local_heterogeneity_display_rotation_deg", "Display rotation [deg]", "Coverage local heterogeneity", "float", 0.0, 360.0, 1.0, "Display-only local sector orientation; it does not change robust rotation summaries.", requires_analysis=False),
            _spec("local_heterogeneity_min_segment_completeness", "Minimum cell completeness", "Coverage local heterogeneity", "float", 0.0, 1.0, 0.01, "Minimum retained reference fraction for a local grid cell.", requires_analysis=False),
            _spec("local_heterogeneity_show_heatmap", "Show 2D heatmap", "Coverage local heterogeneity", "bool", None, None, None, "Display the shared local-grid heatmap in the graph window.", requires_analysis=False),
            _spec("local_heterogeneity_show_1d_profiles", "Show 1D profiles", "Coverage local heterogeneity", "bool", None, None, None, "Display radial and polar profiles derived from the same local grid.", requires_analysis=False),
        ),
    }

    def load_config(
        self, config_path: Path
    ) -> tuple[CoverageAppConfig, CoverageViewerConfig, dict[str, Any]]:
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: '{config_path}'.") from None
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in '{config_path}' at line {exc.lineno}, column {exc.colno}."
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Configuration root in '{config_path}' must be a JSON object.")
        try:
            app_config = load_coverage_app_config(config_path)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid coverage configuration in '{config_path}': {exc}") from exc
        return app_config, app_config.viewer, raw

    def resolve_images(
        self,
        folder: Path,
        app_config: CoverageAppConfig,
        selected_file: str | Path | None = None,
    ) -> list[Path]:
        if not folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: '{folder}'.")
        if not folder.is_dir():
            raise NotADirectoryError(f"Image folder is not a directory: '{folder}'.")
        file_choice: str | Path | None = None
        if selected_file is not None:
            file_choice = str(selected_file)
        elif app_config.file:
            file_choice = app_config.file
        if file_choice:
            selected = resolve_optional_file_in_folder(folder, file_choice, description="selected image")
            if selected is None or not selected.is_file():
                raise FileNotFoundError(f"Selected image is not a file: '{file_choice}'.")
            return [selected.resolve()]
        return discover_images_recursive(folder)

    def analyze(
        self, image_path: Path, config: CoverageViewerConfig
    ) -> CoverageImageResult:
        try:
            return analyze_coverage_image(image_path, config=config, collect_diagnostics=True)
        except CoverageSegmentationFailure as exc:
            # The generic controller caches this exact exception/config pair;
            # retaining it here lets failure stages render without rerunning.
            self._failure_by_key = getattr(self, "_failure_by_key", {})
            self._failure_by_key[(str(image_path.resolve()), repr(config))] = exc
            raise

    def parameter_groups(self) -> Mapping[str, tuple[ParameterSpec, ...]]:
        return self._GROUPS

    def update_config(
        self, base_config: CoverageViewerConfig, values: Mapping[str, Any]
    ) -> CoverageViewerConfig:
        config = base_config
        for specs in self._GROUPS.values():
            for spec in specs:
                if spec.name not in values:
                    continue
                paths = spec.config_paths or tuple((name,) for name in spec.config_names)
                value = values[spec.name]
                if spec.kind == "range":
                    low, high = value
                    convert = (
                        (lambda item: int(round(item)))
                        if spec.step == 1
                        else (lambda item: float(item))
                    )
                    converted = (convert(low), convert(high))
                    if len(paths) == 1:
                        config = _replace_by_path(config, paths[0], converted)
                    else:
                        config = _replace_by_path(config, paths[0], converted[0])
                        config = _replace_by_path(config, paths[1], converted[1])
                elif spec.kind == "int":
                    config = _replace_by_path(config, paths[0], int(round(value)))
                elif spec.kind == "bool":
                    config = _replace_by_path(config, paths[0], bool(value))
                elif spec.kind == "text":
                    parsed = spec.text_parser(value) if spec.text_parser else value
                    config = _replace_by_path(config, paths[0], parsed)
                elif spec.kind == "choice":
                    config = _replace_by_path(config, paths[0], str(value))
                else:
                    config = _replace_by_path(config, paths[0], float(value))
        return config

    def validate_config(
        self, config: CoverageViewerConfig, image_path: Path | None = None
    ) -> str | None:
        analyzer = config.analyzer
        if analyzer.infobar_tail_rows < 1 or analyzer.infobar_min_run < 1:
            return "Infobar row counts must be positive."
        if analyzer.infobar_k_mad <= 0:
            return "The infobar MAD factor must be positive."
        norm_low, norm_high = analyzer.norm_percentiles
        if not (0.0 <= norm_low < norm_high <= 100.0):
            return "Normalization percentiles must satisfy 0 <= low < high <= 100."
        display_low, display_high = analyzer.display_percentiles
        if not (0.0 <= display_low < display_high <= 100.0):
            return "Display percentiles must satisfy 0 <= low < high <= 100."
        if analyzer.bead_blur_sigma <= 0:
            return "Bead blur sigma must be positive."
        if analyzer.bead_closing_radius < 0 or analyzer.bead_opening_radius < 0:
            return "Primary bead morphology radii must be non-negative."
        if analyzer.bead_hole_area < 0:
            return "Bead hole area must be non-negative."
        if config.detector_choice_index < 0:
            return "Detector choice index must be non-negative."
        if config.min_bead_area_px < 1 or config.min_roi_eq_diameter_px < 0:
            return "Bead area and equivalent-diameter limits must be positive."
        if not 0.0 <= config.min_roi_solidity <= 1.0:
            return "Minimum ROI solidity must be between 0 and 1."
        if config.max_roi_anisotropy_ratio < 1.0:
            return "Maximum ROI anisotropy ratio must be at least 1."
        if config.salvage_open_radius_px < 0:
            return "Salvage opening radius must be non-negative."
        if not 0.05 <= config.bead_morph_downscale <= 1.0:
            return "Morphology downscale must be between 0.05 and 1.0."
        if config.bead_morph_blur_sigma <= 0:
            return "Morphology blur sigma must be positive."
        if not 0.0 <= config.bead_morph_gradient_percentile <= 100.0:
            return "Morphology gradient percentile must be between 0 and 100."
        if (
            config.bead_morph_close_radius < 0
            or config.bead_morph_dilate_radius < 0
            or config.bead_morph_erode_radius_px < 0
        ):
            return "Morphology fallback radii must be non-negative."
        if not 0.0 <= config.bead_morph_min_object_area_ratio <= 1.0:
            return "Morphology minimum object area ratio must be between 0 and 1."
        if config.split_trigger_eq_diameter_px < 0:
            return "Split trigger equivalent diameter must be non-negative."
        if config.split_trigger_anisotropy_ratio < 1.0:
            return "Split trigger anisotropy ratio must be at least 1."
        if not 0.0 <= config.split_trigger_solidity_below <= 1.0:
            return "Split trigger solidity must be between 0 and 1."
        if config.split_min_distance_px < 1:
            return "Split minimum distance must be positive."
        if not 0.0 <= config.split_peak_threshold_rel <= 1.0:
            return "Split peak threshold must be between 0 and 1."
        if config.split_max_peaks < 2:
            return "Split maximum peaks must be at least 2."
        if not 0.0 <= config.split_min_child_area_ratio <= 1.0:
            return "Split minimum child area ratio must be between 0 and 1."
        if analyzer.ag_tophat_radius < 1 or analyzer.ag_min_object_size < 1:
            return "Ag count radii and object-size limits must be positive."
        if analyzer.ag_erode_bead_radius < 0 or analyzer.count_min_distance < 1:
            return "Ag count erosion radius must be non-negative and the peak spacing must be positive."
        if analyzer.count_thr_rel <= 0:
            return "Ag count threshold multiplier must be positive."
        if config.ag_coverage_tophat_radius < 1:
            return "Coverage top-hat radius must be positive."
        if config.ag_coverage_tophat_radii is not None:
            if any(int(item) <= 0 for item in config.ag_coverage_tophat_radii):
                return "Coverage top-hat radii must contain only positive integers."
        if config.ag_coverage_threshold_rel <= 0:
            return "Coverage threshold multiplier must be positive."
        if config.ag_coverage_adaptive_block_size < 15 or config.ag_coverage_adaptive_block_size % 2 == 0:
            return "Adaptive block size must be an odd integer of at least 15."
        if config.ag_coverage_adaptive_k_std < 0:
            return "Adaptive k*std must be non-negative."
        if config.ag_coverage_min_object_size < 1 or config.ag_coverage_closing_radius < 0:
            return "Coverage object-size and closing-radius limits must be non-negative."
        if config.max_global_sphere_anisotropy_ratio < 1.0:
            return "Maximum global anisotropy ratio must be at least 1."
        if not 0.0 <= config.min_global_sphere_solidity <= 1.0:
            return "Minimum global solidity must be between 0 and 1."
        if not 0.0 < config.coverage_cap_radius_fraction <= 1.0:
            return "Coverage cap radius fraction must satisfy 0 < f <= 1."
        if not 0.0 <= config.coverage_cap_min_completeness <= 1.0:
            return "Coverage cap minimum completeness must be between 0 and 1."
        try:
            normalize_sphere_diameter_metric(config.sphere_diameter_metric)
            normalize_cap_coverage_metric(config.selected_cap_coverage_metric)
        except ValueError as exc:
            return str(exc)
        if config.coverage_cap_graph_mode not in {"cumulative", "annular"}:
            return "Coverage cap graph mode must be cumulative or annular."
        if not 0.0 < config.coverage_cap_annulus_width_fraction <= 1.0:
            return "Coverage cap annulus width must satisfy 0 < width <= 1."
        if not 0.0 <= config.coverage_cap_sensitivity_half_width < 1.0:
            return "Cap sensitivity half width must satisfy 0 <= value < 1."
        if not 0.0 < config.coverage_cap_sensitivity_step_fraction <= 1.0:
            return "Cap sensitivity step must satisfy 0 < value <= 1."
        if not (0.0 <= config.homogeneity_inner_radius_fraction < config.homogeneity_outer_radius_fraction <= 1.0):
            return "Homogeneity radii must satisfy 0 <= inner < outer <= 1."
        if not (0.0 < config.radial_ring_width_fraction <= config.homogeneity_outer_radius_fraction - config.homogeneity_inner_radius_fraction):
            return "Radial ring width must be positive and no larger than the analyzed range."
        if config.polar_sector_count < 2:
            return "Polar sector count must be at least 2."
        if not 0.0 <= config.homogeneity_min_segment_completeness <= 1.0:
            return "Homogeneity segment completeness must be between 0 and 1."
        if config.homogeneity_view_mode not in {"radial", "polar"}:
            return "Homogeneity view mode must be radial or polar."
        if not (0.0 <= config.local_heterogeneity_inner_radius_fraction < config.local_heterogeneity_outer_radius_fraction <= 1.0):
            return "Local heterogeneity radii must satisfy 0 <= inner < outer <= 1."
        if config.local_heterogeneity_radial_band_count < 1 or config.local_heterogeneity_polar_sector_count < 2:
            return "Local heterogeneity requires at least one radial band and two polar sectors."
        if not 0.0 <= config.local_heterogeneity_min_segment_completeness <= 1.0:
            return "Local heterogeneity completeness must be between 0 and 1."
        if config.local_heterogeneity_polar_rotation_samples < 1 or not np.isfinite(config.local_heterogeneity_display_rotation_deg):
            return "Local heterogeneity rotation settings are invalid."
        if config.polar_rotation_samples < 1 or not np.isfinite(config.polar_display_rotation_deg):
            return "Polar rotation settings are invalid."
        if image_path is not None:
            hdr_path = _paired_hdr_path(Path(image_path))
            if hdr_path is not None:
                metadata = _read_hdr_metadata(hdr_path)
                count_x = metadata.view_fields_count_x or 1
                count_y = metadata.view_fields_count_y or 1
                detector_count = int(count_x * count_y)
                if config.detector_choice_index >= detector_count:
                    return (
                        f"Detector choice index {config.detector_choice_index} exceeds "
                        f"available detector views 0..{detector_count - 1}."
                    )
        return None

    def _scale_feature(self, image: np.ndarray) -> np.ndarray:
        values = image[np.isfinite(image)]
        if values.size == 0:
            return np.zeros(image.shape, dtype=np.float32)
        lo = float(np.percentile(values, 1.0))
        hi = float(np.percentile(values, 99.5))
        if hi <= lo:
            return np.zeros(image.shape, dtype=np.float32)
        scaled = (image.astype(np.float32) - lo) / (hi - lo)
        return np.clip(scaled, 0.0, 1.0)

    def _selected_rois(
        self, result: CoverageImageResult, roi_selection: int | None
    ) -> list[BeadCoverageResult]:
        if roi_selection is None:
            return list(result.roi_results)
        return [roi for roi in result.roi_results if roi.roi_index == roi_selection]

    def _union_mask(
        self,
        result: CoverageImageResult,
        roi_selection: int | None,
        attribute: str,
    ) -> np.ndarray:
        rois = self._selected_rois(result, roi_selection)
        union = np.zeros(result.display.shape, dtype=bool)
        for roi in rois:
            union |= np.asarray(getattr(roi, attribute), dtype=bool)
        return union

    def _roi_index_map(
        self, result: CoverageImageResult, roi_selection: int | None
    ) -> np.ndarray:
        labels = np.zeros(result.display.shape, dtype=np.int32)
        for order, roi in enumerate(self._selected_rois(result, roi_selection), start=1):
            labels[roi.bead_mask] = order
        rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
        positive = labels > 0
        if positive.any():
            rgb[positive] = plt.get_cmap("tab20")((labels[positive] - 1) % 20)[..., :3]
        return rgb

    def _coverage_status(self, roi: BeadCoverageResult, config: CoverageViewerConfig) -> str:
        reasons: list[str] = []
        if config.sphere_anisotropy_check and float(roi.bead_metrics.anisotropy_ratio) > float(config.max_global_sphere_anisotropy_ratio):
            reasons.append("anisotropy")
        if config.sphere_solidity_check and float(roi.bead_metrics.solidity) < float(config.min_global_sphere_solidity):
            reasons.append("solidity")
        if not reasons:
            return "included"
        return "excluded: " + ", ".join(reasons)

    @staticmethod
    def _sphere_geometry(roi: BeadCoverageResult, config: CoverageViewerConfig):
        """Resolve diagnostic cap geometry from an already accepted ROI mask."""

        metrics = roi.bead_metrics
        return sphere_geometry_from_mask(
            roi.bead_mask,
            centroid_rc=metrics.centroid_rc,
            equivalent_diameter_px=metrics.equivalent_diameter_px,
            x_diameter_px=metrics.x_diameter_px,
            y_diameter_px=metrics.y_diameter_px,
            metric=config.sphere_diameter_metric,
        )

    def inscribed_circle_geometry(self, roi: BeadCoverageResult):
        """Cache the mask-only maximum circle for comparison overlays."""

        key = id(roi.bead_mask)
        geometry = self._inscribed_circle_cache.get(key)
        if geometry is None:
            geometry = maximum_inscribed_circle(roi.bead_mask)
            self._inscribed_circle_cache[key] = geometry
        return geometry

    @staticmethod
    def _cap_for_roi(roi: BeadCoverageResult, config: CoverageViewerConfig, pixel_size_m: float | None):
        """Recompute only cap post-processing from the already accepted masks."""

        geometry = CoverageDiagnosticAdapter._sphere_geometry(roi, config)
        return compute_coverage_cap_metrics(
            roi.bead_mask, roi.ag_mask, geometry.center_rc,
            geometry.radius_px, config.coverage_cap_radius_fraction,
            pixel_size_m,
            compute_surface_weighted=True,
            min_completeness=config.coverage_cap_min_completeness,
        )

    def cap_graph_data(
        self,
        result: CoverageImageResult,
        roi_selection: int | None,
        config: CoverageViewerConfig,
    ) -> dict[str, Any] | None:
        """Return cached cumulative or annular cap curves from existing ROI masks."""

        selected = self._selected_rois(result, roi_selection)
        if roi_selection is None:
            selected = [roi for roi in selected if _include_roi_in_global_summary(roi, config)]
        if not selected:
            return None
        key = (
            id(result), roi_selection, config.sphere_diameter_metric,
            config.coverage_cap_min_completeness,
        )
        cached = self._cap_graph_cache.get(key)
        if cached is not None:
            return cached
        fractions = np.linspace(0.05, 0.95, 19)
        cumulative: list[list[Any]] = []
        annular: list[list[dict[str, Any]]] = []
        legacy: list[float] = []
        for roi in selected:
            geometry = self._sphere_geometry(roi, config)
            radius = geometry.radius_px
            cumulative.append(cumulative_cap_sweep(
                roi.bead_mask, roi.ag_mask, geometry.center_rc, radius,
                fractions, result.metadata.mean_pixel_size_m,
                min_completeness=config.coverage_cap_min_completeness,
            ))
            annular.append(annular_cap_profile(
                roi.bead_mask, roi.ag_mask, geometry.center_rc, radius,
                width_fraction=config.coverage_cap_annulus_width_fraction
            ))
            legacy.append(float(roi.legacy_full_projected_coverage))
        metric_names = (
            "projected_fraction", "surface_weighted_fraction", "projected_over_cap_surface"
        )
        cumulative_values = {
            name: np.asarray([
                [np.nan if cap.selected_value(name) is None else cap.selected_value(name) for cap in row]
                for row in cumulative
            ], dtype=float)
            for name in metric_names
        }
        annular_values = {
            name: np.asarray([
                [np.nan if item[name] is None else item[name] for item in row]
                for row in annular
            ], dtype=float)
            for name in metric_names
        }
        cached = {
            "fractions": fractions,
            "annular_x": np.asarray([item["r_over_R_center"] for item in annular[0]], dtype=float),
            "cumulative": cumulative_values,
            "annular": annular_values,
            "invalid": np.asarray([[not cap.valid for cap in row] for row in cumulative], dtype=bool),
            "legacy": float(np.mean(legacy)),
        }
        self._cap_graph_cache[key] = cached
        return cached

    def homogeneity_data(self, result: CoverageImageResult, roi_selection: int | None, config: CoverageViewerConfig):
        """Return shared production-compatible ring/sector post-processing."""
        rois = self._selected_rois(result, roi_selection)
        if roi_selection is None:
            rois = [roi for roi in rois if _include_roi_in_global_summary(roi, config)]
        output=[]
        for roi in rois:
            if config == result.config and roi.homogeneity is not None:
                output.append(roi.homogeneity)
                continue
            geometry=self._sphere_geometry(roi, config)
            radius=geometry.radius_px
            key=(id(roi), config.sphere_diameter_metric, config.homogeneity_inner_radius_fraction, config.homogeneity_outer_radius_fraction, config.radial_ring_width_fraction, config.polar_sector_count, config.homogeneity_min_segment_completeness, config.selected_cap_coverage_metric, config.polar_rotation_samples, config.polar_display_rotation_deg)
            item=self._homogeneity_cache.get(key)
            if item is None:
                item=compute_homogeneity(roi.bead_mask,roi.ag_mask,geometry.center_rc,radius,inner=config.homogeneity_inner_radius_fraction,outer=config.homogeneity_outer_radius_fraction,width=config.radial_ring_width_fraction,sectors=config.polar_sector_count,min_completeness=config.homogeneity_min_segment_completeness,metric=config.selected_cap_coverage_metric,polar_rotation_samples=config.polar_rotation_samples,polar_display_rotation_deg=config.polar_display_rotation_deg)
                self._homogeneity_cache[key]=item
            output.append(item)
        return output

    def local_heterogeneity_data(
        self, result: CoverageImageResult, roi_selection: int | None, config: CoverageViewerConfig
    ) -> list[Any]:
        """Return cached joint local grids from existing ROI masks only."""

        self._scope_local_heterogeneity_caches(result)
        rois = self._selected_rois(result, roi_selection)
        if roi_selection is None:
            rois = [roi for roi in rois if _include_roi_in_global_summary(roi, config)]
        output: list[Any] = []
        for roi in rois:
            geometry = self._sphere_geometry(roi, config)
            base_key = (
                id(roi), geometry.center_rc, geometry.radius_px,
                config.local_heterogeneity_inner_radius_fraction,
                config.local_heterogeneity_outer_radius_fraction,
                config.local_heterogeneity_radial_band_count,
                config.local_heterogeneity_polar_sector_count,
                config.local_heterogeneity_min_segment_completeness,
            )
            key = base_key + (float(config.local_heterogeneity_display_rotation_deg) % 360.0,)
            item = self._local_heterogeneity_cache.get(key)
            if item is None:
                # The display grid and orientation-robust series are separate:
                # rotating the visible grid does not repeat robust rotations.
                item = compute_local_grid_at_rotation(
                    roi.bead_mask, roi.ag_mask, geometry.center_rc, geometry.radius_px,
                    inner_fraction=config.local_heterogeneity_inner_radius_fraction,
                    outer_fraction=config.local_heterogeneity_outer_radius_fraction,
                    radial_band_count=config.local_heterogeneity_radial_band_count,
                    polar_sector_count=config.local_heterogeneity_polar_sector_count,
                    min_segment_completeness=config.local_heterogeneity_min_segment_completeness,
                    metric=config.selected_cap_coverage_metric,
                    rotation_offset_deg=config.local_heterogeneity_display_rotation_deg,
                )
                robust_key = base_key + (config.local_heterogeneity_polar_rotation_samples,)
                robust = self._local_heterogeneity_robust_cache.get(robust_key)
                if robust is None:
                    rotations = compute_local_rotation_series(
                        roi.bead_mask, roi.ag_mask, geometry.center_rc, geometry.radius_px,
                        inner_fraction=config.local_heterogeneity_inner_radius_fraction,
                        outer_fraction=config.local_heterogeneity_outer_radius_fraction,
                        radial_band_count=config.local_heterogeneity_radial_band_count,
                        polar_sector_count=config.local_heterogeneity_polar_sector_count,
                        min_segment_completeness=config.local_heterogeneity_min_segment_completeness,
                        metric=config.selected_cap_coverage_metric,
                        polar_rotation_samples=config.local_heterogeneity_polar_rotation_samples,
                    )
                    robust = (rotations, aggregate_local_rotation_results(rotations))
                    self._put_bounded_local_cache(
                        self._local_heterogeneity_robust_cache,
                        robust_key,
                        robust,
                    )
                item = replace(item, polar_rotation_samples=config.local_heterogeneity_polar_rotation_samples, rotation_results=robust[0], rotation_aggregates=robust[1])
                self._put_bounded_local_cache(
                    self._local_heterogeneity_cache,
                    key,
                    item,
                )
            output.append(replace(item, metric=normalize_cap_coverage_metric(config.selected_cap_coverage_metric)))
        return output

    def render_stage(
        self,
        result: CoverageImageResult,
        stage: str,
        roi_selection: int | None = None,
        current_config: CoverageViewerConfig | None = None,
    ) -> np.ndarray:
        config = self._effective_config(result, current_config)
        display = self._display_image(result.cropped, config)
        if stage in {"overlay", "display"}:
            return _coverage_rgb(display)
        if stage == "norm":
            return _coverage_rgb(np.clip(result.norm.astype(np.float32), 0.0, 1.0))
        if stage == "bead_raw":
            mask = result.bead_raw_union if roi_selection is None else self._union_mask(result, roi_selection, "bead_mask")
            return _coverage_rgb(mask.astype(np.float32))
        if stage == "bead_refined":
            mask = self._union_mask(result, roi_selection, "bead_mask") if roi_selection is not None else result.bead_refined_union
            return _coverage_rgb(mask.astype(np.float32))
        if stage == "ag_count_feature":
            if roi_selection is None:
                return _coverage_rgb(self._scale_feature(result.ag_count_feature_union))
            rois = self._selected_rois(result, roi_selection)
            feature = np.zeros(result.display.shape, dtype=np.float32)
            if rois:
                feature = rois[0].count_feature.astype(np.float32)
            return _coverage_rgb(self._scale_feature(feature))
        if stage == "ag_coverage_feature":
            if roi_selection is None:
                return _coverage_rgb(self._scale_feature(result.ag_coverage_feature_union))
            rois = self._selected_rois(result, roi_selection)
            feature = np.zeros(result.display.shape, dtype=np.float32)
            if rois:
                feature = rois[0].coverage_feature.astype(np.float32)
            return _coverage_rgb(self._scale_feature(feature))
        if stage == "ag_count_mask":
            return _coverage_rgb(self._union_mask(result, roi_selection, "ag_count_mask").astype(np.float32))
        if stage == "ag_coverage_mask":
            return _coverage_rgb(self._union_mask(result, roi_selection, "ag_mask").astype(np.float32))
        if stage == "ag_peak_map":
            base = _coverage_rgb(display)
            for roi in self._selected_rois(result, roi_selection):
                if roi.ag_peak_coords.size == 0:
                    continue
                rows = np.clip(roi.ag_peak_coords[:, 0].astype(int), 0, base.shape[0] - 1)
                cols = np.clip(roi.ag_peak_coords[:, 1].astype(int), 0, base.shape[1] - 1)
                base[rows, cols] = (0.0, 1.0, 1.0)
            return base
        if stage == "roi_index_map":
            return self._roi_index_map(result, roi_selection)
        if stage == "raw_bead_candidates":
            diagnostics = result.diagnostics
            mask = diagnostics.raw_candidate_union if diagnostics is not None else result.bead_raw_union
            return _coverage_rgb(mask.astype(np.float32))
        if stage == "rejected_bead_candidates":
            diagnostics = result.diagnostics
            mask = diagnostics.rejected_candidate_union if diagnostics is not None else np.zeros(result.display.shape, dtype=bool)
            return _coverage_rgb(mask.astype(np.float32))
        raise ValueError(f"Unknown coverage diagnostic stage: '{stage}'.")

    def make_overlay_data(
        self,
        result: CoverageImageResult,
        roi_selection: int | None = None,
        current_config: CoverageViewerConfig | None = None,
    ) -> OverlayData:
        config = self._effective_config(result, current_config)
        rois = self._selected_rois(result, roi_selection)
        boundary_layers: list[BoundaryLayer] = []
        measurements: list[MeasurementOverlay] = []
        point_layers: list[PointOverlay] = []
        text_overlays: list[TextOverlay] = []
        peak_rows: list[np.ndarray] = []
        peak_cols: list[np.ndarray] = []
        for roi in rois:
            included = _include_roi_in_global_summary(roi, config)
            boundary_layers.append(
                BoundaryLayer(
                    control_label="Bead boundaries",
                    mask=find_boundaries(roi.bead_mask, mode="outer"),
                    color=self.INCLUDED_COLOR if included else self.EXCLUDED_COLOR,
                )
            )
            cap = self._cap_for_roi(roi, config, result.metadata.mean_pixel_size_m)
            boundary_layers.append(
                BoundaryLayer(
                    control_label="Cap boundary",
                    mask=find_boundaries(cap.geometry.theoretical_circle_mask, mode="outer"),
                    color=self.CAP_COLOR,
                )
            )
            point_layers.append(
                PointOverlay(
                    control_label="Cap center",
                    rows=np.array([cap.geometry.center_rc[0]]),
                    cols=np.array([cap.geometry.center_rc[1]]),
                    color="white", marker="+", markersize=7.0,
                )
            )
            text_overlays.append(
                TextOverlay(
                    control_label="Cap completeness status",
                    row=cap.geometry.center_rc[0],
                    col=cap.geometry.center_rc[1],
                    text=(
                        f"a/R = {cap.geometry.radius_fraction:.2f}"
                        if cap.valid else f"cap invalid: {cap.geometry.completeness:.3f}"
                    ),
                    color="white" if cap.valid else "red", fontsize=7.0, boxed=True,
                )
            )
            boundary_layers.append(
                BoundaryLayer(
                    control_label="Ag coverage boundaries",
                    mask=find_boundaries(roi.ag_mask, mode="outer"),
                    color=self.AG_COVERAGE_COLOR,
                )
            )
            boundary_layers.append(
                BoundaryLayer(
                    control_label="Ag count boundaries",
                    mask=find_boundaries(roi.ag_count_mask, mode="outer"),
                    color=self.AG_COUNT_COLOR,
                )
            )
            metrics = roi.bead_metrics
            x_chord: AxisChord | None = None
            y_chord: AxisChord | None = None
            try:
                x_chord, y_chord = _mask_xy_chords(roi.bead_mask, metrics.centroid_rc)
            except ValueError as exc:
                LOGGER.debug("Could not compute ROI chords for ROI %s", roi.roi_index, exc_info=exc)
            measurements.append(
                MeasurementOverlay(
                    row=float(metrics.centroid_rc[0]),
                    col=float(metrics.centroid_rc[1]),
                    x_diameter_px=float(metrics.x_diameter_px),
                    y_diameter_px=float(metrics.y_diameter_px),
                    label_text="",
                    x_color=self.DIAMETER_X_COLOR,
                    y_color=self.DIAMETER_Y_COLOR,
                    label_color="white" if included else "red",
                    line_control_label=self.measurement_line_label,
                    label_control_label=self.measurement_text_label,
                    x_start_row=None if x_chord is None else x_chord.start_row,
                    x_start_col=None if x_chord is None else x_chord.start_col,
                    x_end_row=None if x_chord is None else x_chord.end_row,
                    x_end_col=None if x_chord is None else x_chord.end_col,
                    y_start_row=None if y_chord is None else y_chord.start_row,
                    y_start_col=None if y_chord is None else y_chord.start_col,
                    y_end_row=None if y_chord is None else y_chord.end_row,
                    y_end_col=None if y_chord is None else y_chord.end_col,
                    x_label_text=(
                        None
                        if x_chord is None
                        else self._format_chord_label(
                            x_chord, result.metadata.mean_pixel_size_m, "x"
                        )
                    ),
                    x_label_row=(
                        None
                        if x_chord is None
                        else (x_chord.start_row + x_chord.end_row) / 2.0
                    ),
                    x_label_col=(
                        None
                        if x_chord is None
                        else x_chord.start_col - 10.0
                    ),
                    x_label_color=self.DIAMETER_X_COLOR,
                    y_label_text=(
                        None
                        if y_chord is None
                        else self._format_chord_label(
                            y_chord, result.metadata.mean_pixel_size_m, "y"
                        )
                    ),
                    y_label_row=(
                        None
                        if y_chord is None
                        else y_chord.start_row - 10.0
                    ),
                    y_label_col=(
                        None
                        if y_chord is None
                        else (y_chord.start_col + y_chord.end_col) / 2.0
                    ),
                    y_label_color=self.DIAMETER_Y_COLOR,
                )
            )
            if x_chord is None:
                LOGGER.debug("Coverage ROI %s has no horizontal chord.", roi.roi_index)
            if y_chord is None:
                LOGGER.debug("Coverage ROI %s has no vertical chord.", roi.roi_index)
            if roi.ag_peak_coords.size:
                peak_rows.append(roi.ag_peak_coords[:, 0].astype(float))
                peak_cols.append(roi.ag_peak_coords[:, 1].astype(float))
            text_overlays.append(
                TextOverlay(
                    control_label="ROI inclusion status",
                    row=float(metrics.centroid_rc[0]) + metrics.y_diameter_px / 2.0 + 10.0,
                    col=float(metrics.centroid_rc[1]),
                    text=self._coverage_status(roi, config),
                    color="white" if included else "red",
                    fontsize=7.5,
                    boxed=True,
                    va="top",
                )
            )
            text_overlays.append(
                TextOverlay(
                    control_label="ROI index labels",
                    row=float(metrics.centroid_rc[0]) - metrics.y_diameter_px / 2.0 - 18.0,
                    col=float(metrics.centroid_rc[1]),
                    text=f"ROI {roi.roi_index}",
                    color="white",
                    fontsize=8.0,
                    boxed=True,
                )
            )
        if peak_rows:
            point_layers.append(
                PointOverlay(
                    control_label="Ag peak markers",
                    rows=np.concatenate(peak_rows),
                    cols=np.concatenate(peak_cols),
                    color=self.AG_PEAK_COLOR,
                    marker=".",
                    markersize=4.0,
                )
            )
        return OverlayData(
            boundary_layers=tuple(boundary_layers),
            measurements=tuple(measurements),
            point_layers=tuple(point_layers),
            text_overlays=tuple(text_overlays),
            pixel_size_m=result.metadata.mean_pixel_size_m,
            image_shape=result.display.shape,
        )

    def summarize_result(
        self,
        result: CoverageImageResult,
        duration_s: float,
        roi_selection: int | None = None,
        current_config: CoverageViewerConfig | None = None,
    ) -> str:
        config = self._effective_config(result, current_config)
        coverage_branch = _coverage_branch_label(config)
        rois = self._selected_rois(result, roi_selection)
        pixel_text = (
            f" | pixel {_format_coverage_length_m(result.metadata.mean_pixel_size_m)}/px"
            if result.metadata.mean_pixel_size_m is not None
            else ""
        )
        if roi_selection is not None and rois:
            roi = rois[0]
            metrics = roi.bead_metrics
            eq_text = (
                _format_coverage_length_m(metrics.equivalent_diameter_m)
                if metrics.equivalent_diameter_m is not None
                else f"{metrics.equivalent_diameter_px:.1f} px"
            )
            density = (
                f"{roi.sphere_np_density_per_um2:.2f}/um^2"
                if roi.sphere_np_density_per_um2 is not None
                else "n/a"
            )
            cap = self._cap_for_roi(roi, config, result.metadata.mean_pixel_size_m)
            selected_metric = normalize_cap_coverage_metric(config.selected_cap_coverage_metric)
            selected_value = cap.selected_value(selected_metric)
            cap_text = (
                f"{selected_metric} {selected_value * 100.0:.2f}%"
                if selected_value is not None else f"cap invalid ({cap.geometry.completeness:.3f})"
            )
            cap_values = (
                f"projected {cap.projected_fraction * 100.0:.2f}% | "
                f"weighted {cap.surface_weighted_fraction * 100.0:.2f}% | "
                f"Ag/cap surface {cap.projected_over_cap_surface * 100.0:.2f}%"
                if cap.projected_fraction is not None
                and cap.surface_weighted_fraction is not None
                and cap.projected_over_cap_surface is not None else "cap invalid"
            )
            sphere_diameter = 2.0 * self._sphere_geometry(roi, config).radius_px
            local = self.local_heterogeneity_data(result, roi.roi_index, config)
            domain = local[0].domain if local else None
            domain_value = domain.components(selected_metric).value if domain is not None else None
            domain_text = (
                f"domain C {domain_value * 100.0:.2f}% over "
                f"{config.local_heterogeneity_inner_radius_fraction:.2f}–{config.local_heterogeneity_outer_radius_fraction:.2f} R "
                f"(completeness {domain.completeness:.3f}, grid Δ {local[0].metrics[selected_metric].reconstruction_delta_pp:.3g} pp)"
                if domain is not None and domain_value is not None else "domain C n/a"
            )
            fixed_cap_text = (
                f"cap C {selected_value * 100.0:.2f}% at {config.coverage_cap_radius_fraction:.2f} R"
                if selected_value is not None else "cap C invalid"
            )
            return (
                f"ROI {roi.roi_index} | {self._coverage_status(roi, config)} | "
                f"selected {cap_text if config.coverage_cap_enabled else f'legacy {roi.legacy_full_projected_coverage_percent:.2f}%'} | "
                f"legacy {roi.legacy_full_projected_coverage_percent:.2f}% | {cap_values} | "
                f"sphere ({config.sphere_diameter_metric}) {sphere_diameter:.1f} px | "
                f"{fixed_cap_text} | {domain_text} | "
                f"cap a/R {cap.geometry.radius_fraction:.2f}, a {cap.geometry.cap_radius_px:.1f} px, "
                f"angle {cap.geometry.half_angle_deg:.1f} deg, completeness {cap.geometry.completeness:.3f}, "
                f"bead/Ag cap px {cap.bead_pixel_count}/{cap.ag_pixel_count} | projected {roi.projected_ag_count} | "
                f"sphere {roi.sphere_ag_count_est:.1f} | density {density} | "
                f"eq {eq_text} | anisotropy {metrics.anisotropy_ratio:.3f} | "
                f"solidity {metrics.solidity:.3f} | count thr {roi.ag_count_threshold:.4f} | "
                f"cov thr {roi.ag_coverage_threshold:.4f} | bead px {roi.bead_area_px} | "
                f"Ag px {roi.ag_area_px} | analysis {duration_s:.3f} s{pixel_text}"
                f" | {coverage_branch}"
            )
        included = [
            roi for roi in result.roi_results if _include_roi_in_global_summary(roi, config)
        ]
        coverage_values = []
        for roi in included:
            cap = self._cap_for_roi(roi, config, result.metadata.mean_pixel_size_m)
            selected = cap.selected_value(config.selected_cap_coverage_metric)
            if config.coverage_cap_enabled and selected is not None:
                coverage_values.append(selected * 100.0)
            else:
                coverage_values.append(roi.legacy_full_projected_coverage_percent)
        densities = [
            roi.sphere_np_density_per_um2
            for roi in included
            if roi.sphere_np_density_per_um2 is not None
        ]
        mean_cov = f"{float(np.mean(coverage_values)):.2f}%" if coverage_values else "n/a"
        median_cov = f"{float(np.median(coverage_values)):.2f}%" if coverage_values else "n/a"
        total_projected = sum(int(roi.projected_ag_count) for roi in included)
        total_sphere = sum(float(roi.sphere_ag_count_est) for roi in included)
        mean_density = f"{float(np.mean(densities)):.2f}/um^2" if densities else "n/a"
        return (
            f"ROIs {len(result.roi_results)} | included {len(included)} | "
            f"mean cov {mean_cov} | median cov {median_cov} | "
            f"projected {total_projected} | sphere {total_sphere:.1f} | "
            f"mean density {mean_density} | analysis {duration_s:.3f} s{pixel_text}"
            f" | {coverage_branch}"
        )

    def overlay_labels(self, config: CoverageViewerConfig) -> tuple[str, ...]:
        return (
            "Bead boundaries",
            "ROI inclusion status",
            self.measurement_line_label,
            "Ag coverage boundaries",
            "Ag count boundaries",
            "Ag peak markers",
            "ROI index labels",
            "Cap boundary",
            "Cap center",
            "Cap completeness status",
            "Inscribed circle",
            "Local heterogeneity radial bands",
            "Local heterogeneity polar sectors",
            "Local heterogeneity coverage map",
            "Rejected candidate boundaries",
            "Raw candidate boundaries",
            "Rejected candidate labels",
            self.scale_bar_label,
        )

    def default_overlay_state(self, config: CoverageViewerConfig) -> dict[str, bool]:
        return {
            "Bead boundaries": bool(config.default_show_bead_boundary),
            "ROI inclusion status": True,
            self.measurement_line_label: bool(config.default_show_diameter_lines),
            "Ag coverage boundaries": bool(config.default_show_ag_boundary),
            "Ag count boundaries": bool(config.default_show_ag_count_boundary),
            "Ag peak markers": bool(config.default_show_ag_peaks),
            "ROI index labels": True,
            "Cap boundary": True,
            "Cap center": True,
            "Cap completeness status": True,
            "Inscribed circle": False,
            "Local heterogeneity radial bands": True,
            "Local heterogeneity polar sectors": True,
            "Local heterogeneity coverage map": False,
            "Rejected candidate boundaries": True,
            "Raw candidate boundaries": False,
            "Rejected candidate labels": True,
            self.scale_bar_label: bool(config.default_show_scale),
        }

    def initial_preview(
        self, image_path: Path, config: CoverageViewerConfig
    ) -> np.ndarray | None:
        try:
            preview = load_failed_image_preview(image_path, config)
        except (OSError, ValueError) as exc:
            LOGGER.debug(
                "Could not load coverage diagnostic preview for %s", image_path, exc_info=exc
            )
            return None
        return _coverage_rgb(preview.display)

    def load_failed_preview(
        self, image_path: Path, config: CoverageViewerConfig
    ) -> FailedImagePreview | None:
        failure = getattr(self, "_failure_by_key", {}).get((str(image_path.resolve()), repr(config)))
        if failure is not None:
            return failure.preview
        return load_failed_image_preview(image_path, config)

    def render_failed_preview(
        self,
        preview: FailedImagePreview,
        stage: str,
        current_config: CoverageViewerConfig | None = None,
    ) -> np.ndarray:
        config = self._effective_config(preview, current_config)
        failure = getattr(self, "_failure_by_key", {}).get((str(preview.image_path.resolve()), repr(config)))
        diagnostics = failure.diagnostics if failure is not None else None
        if stage == "norm":
            return _coverage_rgb(np.clip(preview.norm.astype(np.float32), 0.0, 1.0))
        if stage in {"overlay", "display", "ag_peak_map"}:
            return _coverage_rgb(self._display_image(preview.cropped, config))
        if stage == "raw_bead_candidates" and diagnostics is not None:
            return _coverage_rgb(diagnostics.raw_candidate_union.astype(np.float32))
        if stage == "rejected_bead_candidates" and diagnostics is not None:
            return _coverage_rgb(diagnostics.rejected_candidate_union.astype(np.float32))
        return _coverage_rgb(np.zeros(preview.display.shape, dtype=np.float32))

    def summarize_failed_preview(
        self,
        preview: FailedImagePreview | None,
        error: BaseException,
        duration_s: float,
        stage: str,
    ) -> str:
        pixel_text = ""
        if preview is not None and preview.metadata.mean_pixel_size_m is not None:
            pixel_text = f" | pixel {_format_coverage_length_m(preview.metadata.mean_pixel_size_m)}/px"
        return (
            f"Coverage analysis found no accepted ROI after {duration_s:.3f} s | {error} | "
            f"Adjust ROI or detector parameters to recover valid bead ROIs{pixel_text}"
        )

    def make_failed_overlay_data(
        self, preview: FailedImagePreview, config: CoverageViewerConfig
    ) -> OverlayData:
        """Render structured rejected candidates without rerunning segmentation."""

        failure = getattr(self, "_failure_by_key", {}).get((str(preview.image_path.resolve()), repr(config)))
        diagnostics = failure.diagnostics if failure is not None else None
        if diagnostics is None:
            return OverlayData((), (), (), (), preview.metadata.mean_pixel_size_m, preview.display.shape)
        layers = (
            BoundaryLayer("Rejected candidate boundaries", find_boundaries(diagnostics.rejected_candidate_union, mode="outer"), self.EXCLUDED_COLOR),
            BoundaryLayer("Raw candidate boundaries", find_boundaries(diagnostics.raw_candidate_union, mode="outer"), (1.0, 0.8, 0.0, 1.0)),
        )
        labels = tuple(
            TextOverlay("Rejected candidate labels", candidate.centroid_rc[0], candidate.centroid_rc[1],
                        f"{candidate.candidate_index}: " + "; ".join(candidate.rejection_reasons),
                        "red", fontsize=7.0, boxed=True)
            for candidate in diagnostics.rejected_candidates
        )
        return OverlayData(layers, (), (), labels, preview.metadata.mean_pixel_size_m, preview.display.shape)

    def roi_options(self, result: CoverageImageResult | None) -> tuple[int, ...]:
        if result is None:
            return ()
        return tuple(int(roi.roi_index) for roi in result.roi_results)

    def format_roi_selection(self, roi_selection: int | None) -> str | None:
        return "All ROIs" if roi_selection is None else f"ROI {roi_selection}"

    def stage_message(
        self,
        result: CoverageImageResult | None,
        stage: str,
        pixels: np.ndarray,
        roi_selection: int | None = None,
        preview: FailedImagePreview | None = None,
    ) -> str:
        if result is not None and len(result.roi_results) == 0:
            return "No valid coverage ROIs"
        labels = {
            "bead_raw": "Bead raw",
            "bead_refined": "Bead refined",
            "ag_count_mask": "Ag count mask",
            "ag_coverage_mask": "Ag coverage mask",
            "roi_index_map": "ROI index map",
            "raw_bead_candidates": "Raw bead candidates",
            "rejected_bead_candidates": "Rejected bead candidates",
        }
        if stage in labels and not np.any(pixels):
            return f"{labels[stage]}: empty"
        if stage == "ag_peak_map":
            if result is not None:
                rois = self._selected_rois(result, roi_selection)
                peak_count = sum(int(roi.ag_peak_coords.shape[0]) for roi in rois)
                if peak_count == 0:
                    return "Ag peak map: no peaks"
            elif preview is not None:
                return "Ag peak map unavailable until ROI analysis succeeds"
        if preview is not None and stage not in {"overlay", "display", "norm"}:
            return "This stage becomes available after a successful ROI analysis"
        return ""


DIAGNOSTIC_ADAPTERS: dict[str, type[DiagnosticAdapter]] = {
    "beads": BeadsDiagnosticAdapter,
    "coverage": CoverageDiagnosticAdapter,
}


def effective_config_values(
    adapter: DiagnosticAdapter, base_config: Any, changes: Mapping[str, Any]
) -> Any:
    """Return a new immutable config; useful to callers and non-GUI tests."""

    return adapter.update_config(base_config, changes)


_UNSET_DATA_SOURCE = object()


def save_tuned_config(
    raw_source_config: Mapping[str, Any], viewer_config: Any, output_path: str | Path,
    *,
    effective_folder: Path | None | object = _UNSET_DATA_SOURCE,
    effective_file: str | None | object = _UNSET_DATA_SOURCE,
) -> Path:
    """Write a tuned copy while preserving all non-viewer top-level fields."""

    def _merge_dicts(base: Any, updates: Any) -> Any:
        if isinstance(base, dict) and isinstance(updates, dict):
            merged = deepcopy(base)
            for key, value in updates.items():
                merged[key] = _merge_dicts(merged.get(key), value)
            return merged
        return deepcopy(updates)

    output_path = Path(output_path)
    payload = deepcopy(dict(raw_source_config))
    source_viewer = payload.get("viewer", {})
    viewer_payload = deepcopy(source_viewer) if isinstance(source_viewer, dict) else {}
    viewer_payload = _merge_dicts(viewer_payload, asdict(viewer_config))
    payload["viewer"] = viewer_payload
    # Direct callers retain the source fields already in the template.  The
    # diagnostic window always passes both values explicitly, including the
    # explicit no-data pair (None, None).
    if effective_folder is not _UNSET_DATA_SOURCE:
        payload["folder"] = path_to_config_text(effective_folder) if effective_folder is not None else ""
    if effective_file is not _UNSET_DATA_SOURCE:
        payload["file"] = effective_file or None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def choose_json_save_path(initial_path: Path, figure: Figure) -> Path | None:
    """Choose a JSON destination with the standard-library native Tk dialog."""
    root: Any | None = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        selected=filedialog.asksaveasfilename(parent=root, initialdir=str(initial_path.parent), initialfile=initial_path.name, defaultextension=".json", filetypes=(("JSON files","*.json"),("All files","*.*")))
    except Exception:
        LOGGER.debug("Native save dialog unavailable", exc_info=True)
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
    if not selected: return None
    path=Path(selected)
    return path if path.suffix else path.with_suffix(".json")


@dataclass(frozen=True)
class SelectedDataSource:
    folder: Path
    file: str | None


@dataclass(frozen=True)
class DataSourceDialogResult:
    """Outcome of the native image/folder chooser.

    A backend failure is deliberately distinct from an ordinary cancellation.
    """

    source: SelectedDataSource | None
    cancelled: bool
    error_message: str | None = None


def choose_data_source(initial_directory: Path | None, figure: Figure) -> DataSourceDialogResult:
    """Use a viewable, short-lived Tk modal to choose one image or a folder."""
    root: Any | None = None
    dialog: Any | None = None
    choice: dict[str, SelectedDataSource | None] = {"value": None}
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        dialog = tk.Toplevel(root)
        dialog.title("Load diagnostic data")
        dialog.transient(root)

        def finish() -> None:
            # The outer finally block owns destruction, avoiding double-destroy
            # races from Tk button callbacks.
            dialog.quit()

        def select_file() -> None:
            path = filedialog.askopenfilename(parent=dialog, initialdir=str(initial_directory or Path.cwd()), filetypes=(("Supported images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")))
            if path:
                choice["value"] = SelectedDataSource(Path(path).parent, Path(path).name)
            finish()

        def select_folder() -> None:
            path = filedialog.askdirectory(parent=dialog, initialdir=str(initial_directory or Path.cwd()))
            if path:
                choice["value"] = SelectedDataSource(Path(path), None)
            finish()

        tk.Button(dialog, text="Select image", command=select_file).pack(fill="x", padx=12, pady=(12, 4))
        tk.Button(dialog, text="Select folder", command=select_folder).pack(fill="x", padx=12, pady=4)
        tk.Button(dialog, text="Cancel", command=finish).pack(fill="x", padx=12, pady=(4, 12))
        dialog.protocol("WM_DELETE_WINDOW", finish)
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.wait_visibility()
        dialog.grab_set()
        root.mainloop()
        return DataSourceDialogResult(choice["value"], choice["value"] is None)
    except Exception as exc:
        LOGGER.exception("Could not open native data-source dialog")
        return DataSourceDialogResult(None, False, str(exc))
    finally:
        if dialog is not None:
            try:
                dialog.grab_release()
            except Exception:
                pass
            try:
                dialog.destroy()
            except Exception:
                pass
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


class _DaemonTaskRunner:
    """Minimal Future runner for disposable diagnostic calculations."""

    def __init__(self) -> None:
        self._shutdown = False

    def submit(self, function: Callable[..., Any], *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        if self._shutdown:
            future.cancel()
            return future

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(function(*args))
            except BaseException as exc:
                future.set_exception(exc)

        Thread(target=run, name="sem-coverage-diagnostic", daemon=True).start()
        return future

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """A daemon task is abandoned after the controller has been closed."""

        self._shutdown = True


class AnalysisController:
    """Bounded single-worker analysis scheduler with latest-result semantics."""

    def __init__(
        self,
        analyze: Callable[[Path, Any], Any],
        *,
        asynchronous: bool = True,
        cache_size: int = 12,
    ) -> None:
        self._analyze = analyze
        self._asynchronous = asynchronous
        self._cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[str, Any], tuple[Any, float]] = OrderedDict()
        self._failure_cache: OrderedDict[tuple[str, Any], tuple[BaseException, float]] = OrderedDict()
        self._completed: Queue[AnalysisCompletion] = Queue()
        self._executor = _DaemonTaskRunner() if asynchronous else None
        self._lock = RLock()
        self._running: Future[Any] | None = None
        self._running_meta: tuple[int, float] | None = None
        self._pending: tuple[int, Path, Any, float] | None = None
        self._closed = False

    @staticmethod
    def _config_key(config: Any) -> Any:
        values = asdict(config)

        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted((key, freeze(item)) for key, item in value.items()))
            if isinstance(value, list):
                return tuple(freeze(item) for item in value)
            return value

        return freeze(values)

    def _cache_key(self, image_path: Path, config: Any) -> tuple[str, Any]:
        return (str(image_path.resolve()), self._config_key(config))

    def submit(self, generation: int, image_path: Path, config: Any) -> None:
        """Submit immediately, retaining at most one newest pending state."""

        key = self._cache_key(image_path, config)
        submitted_at = perf_counter()
        with self._lock:
            if self._closed:
                return
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                cached_result, cached_duration = cached
                self._completed.put(
                    AnalysisCompletion(
                        generation,
                        image_path,
                        config,
                        cached_result,
                        cached_duration,
                        cached=True,
                        wait_s=0.0,
                    )
                )
                return
            cached_failure = self._failure_cache.get(key)
            if cached_failure is not None:
                self._failure_cache.move_to_end(key)
                error, duration = cached_failure
                self._completed.put(
                    AnalysisCompletion(generation, image_path, config, None, duration, error=error, cached=True, wait_s=0.0)
                )
                return
            if not self._asynchronous:
                pass
            elif self._running is not None:
                self._pending = (generation, image_path, config, submitted_at)
                return
            else:
                self._start_locked(generation, image_path, config, submitted_at)
                return

        self._run_synchronously(generation, image_path, config)

    def _run_synchronously(self, generation: int, image_path: Path, config: Any) -> None:
        started = perf_counter()
        try:
            result = self._analyze(image_path, config)
            duration = perf_counter() - started
            self._store_cache(image_path, config, result, duration)
            completion = AnalysisCompletion(
                generation, image_path, config, result, duration, wait_s=0.0
            )
        except BaseException as exc:
            duration = perf_counter() - started
            self._store_failure(image_path, config, exc, duration)
            completion = AnalysisCompletion(
                generation,
                image_path,
                config,
                None,
                duration,
                error=exc,
                wait_s=0.0,
            )
        self._completed.put(completion)

    def _start_locked(
        self, generation: int, image_path: Path, config: Any, submitted_at: float
    ) -> None:
        if self._executor is None:
            raise RuntimeError("Asynchronous executor is unavailable.")
        self._running_meta = (generation, submitted_at)
        future = self._executor.submit(
            self._execute, generation, image_path, config, submitted_at
        )
        self._running = future
        future.add_done_callback(self._on_done)

    def _execute(
        self, generation: int, image_path: Path, config: Any, submitted_at: float
    ) -> AnalysisCompletion:
        started = perf_counter()
        wait_s = max(0.0, started - submitted_at)
        try:
            result = self._analyze(image_path, config)
            return AnalysisCompletion(
                generation,
                image_path,
                config,
                result,
                perf_counter() - started,
                wait_s=wait_s,
            )
        except BaseException as exc:
            return AnalysisCompletion(
                generation,
                image_path,
                config,
                None,
                perf_counter() - started,
                error=exc,
                wait_s=wait_s,
            )

    def _on_done(self, future: Future[Any]) -> None:
        try:
            completion = future.result()
        except CancelledError:
            return
        except BaseException as exc:
            LOGGER.exception("Unexpected diagnostic worker failure")
            completion = AnalysisCompletion(-1, Path(), None, None, 0.0, error=exc)
        if completion.result is not None:
            self._store_cache(
                completion.image_path,
                completion.config,
                completion.result,
                completion.duration_s,
            )
        elif completion.error is not None and completion.config is not None:
            self._store_failure(
                completion.image_path, completion.config, completion.error, completion.duration_s
            )
        with self._lock:
            self._running = None
            self._running_meta = None
            if self._closed:
                return
            self._completed.put(completion)
            pending = self._pending
            self._pending = None
            if pending is not None:
                self._start_locked(*pending)

    def _store_cache(
        self, image_path: Path, config: Any, result: Any, duration_s: float
    ) -> None:
        key = self._cache_key(image_path, config)
        with self._lock:
            if self._closed:
                return
            self._cache[key] = (result, duration_s)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _store_failure(
        self, image_path: Path, config: Any, error: BaseException, duration_s: float
    ) -> None:
        key = self._cache_key(image_path, config)
        with self._lock:
            if self._closed:
                return
            self._failure_cache[key] = (error, duration_s)
            self._failure_cache.move_to_end(key)
            while len(self._failure_cache) > self._cache_size:
                self._failure_cache.popitem(last=False)

    def poll(self) -> list[AnalysisCompletion]:
        """Return all currently completed requests without blocking."""

        items: list[AnalysisCompletion] = []
        while True:
            try:
                items.append(self._completed.get_nowait())
            except Empty:
                return items

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._failure_cache.clear()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running is not None or self._pending is not None

    @property
    def running_generation(self) -> int | None:
        with self._lock:
            return None if self._running_meta is None else int(self._running_meta[0])

    @property
    def pending_generation(self) -> int | None:
        with self._lock:
            return None if self._pending is None else int(self._pending[0])

    @property
    def pending_wait_s(self) -> float | None:
        with self._lock:
            if self._pending is None:
                return None
            return max(0.0, perf_counter() - float(self._pending[3]))

    def close(self) -> None:
        """Cancel pending work and shut down the worker without accepting more jobs."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            running = self._running
        if running is not None:
            running.cancel()
        if self._executor is not None:
            # A running Python thread cannot be cancelled safely.  Diagnostic
            # results are ignored after closure, so never block a GUI close on
            # an in-flight calculation.
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.clear_cache()
        while True:
            try:
                self._completed.get_nowait()
            except Empty:
                break


@dataclass
class _Control:
    spec: ParameterSpec
    axis: Axes
    widget: Slider | RangeSlider | CheckButtons | TextBox | RadioButtons
    callback_id: int | None = None
    enabled: bool = True
    inactive_reason: str = ""


class DiagnosticViewer:
    """Generic Matplotlib diagnostic shell driven by a mode adapter."""

    def __init__(
        self,
        adapter: DiagnosticAdapter,
        config_path: str | Path,
        *,
        selected_file: str | Path | None = None,
        folder_override: str | Path | None = None,
        output_config: str | Path | None = None,
        asynchronous: bool = True,
        debounce_ms: int = 200,
    ) -> None:
        self.adapter = adapter
        self.config_path = Path(config_path).expanduser().resolve()
        self._cli_selected_file = Path(selected_file).expanduser() if selected_file else None
        self.selected_file = self._cli_selected_file
        self.folder_override = (
            Path(folder_override).expanduser().resolve() if folder_override else None
        )
        self.output_config = (
            Path(output_config).expanduser().resolve()
            if output_config
            else self.config_path.with_name(f"{self.config_path.stem}_tuned.json")
        )
        self.debounce_ms = int(debounce_ms)

        self.app_config, self.original_viewer_config, self.raw_source_config = (
            self.adapter.load_config(self.config_path)
        )
        self.current_viewer_config = self.original_viewer_config
        self.folder = self.folder_override or (Path(self.app_config.folder).expanduser() if self.app_config.folder else None)
        self.image_paths: list[Path] = []
        self._data_message = ""
        if self.folder is not None:
            try:
                self.image_paths = self.adapter.resolve_images(self.folder, self.app_config, self.selected_file)
            except (OSError, ValueError, FileNotFoundError) as exc:
                self._data_message = f"No data loaded: {exc}"
        if self.selected_file is None and self.app_config.file and self.image_paths:
            # A config-level file selection has the same effective source
            # semantics as a file selected through the native dialog.
            self.selected_file = Path(self.image_paths[0].name)
        if not self.image_paths and not self._data_message:
            self._data_message = "No data loaded. Use Load data to select an image or folder."
        self.index = self._resolve_start_index(self.selected_file) if self.image_paths else 0
        validation = self.adapter.validate_config(
            self.current_viewer_config, self.image_paths[self.index] if self.image_paths else None
        )
        if validation:
            raise ValueError(f"Invalid source viewer configuration: {validation}")
        initial_stage = self.adapter.overlay_stage or self.adapter.stages[0]
        self.stage_index = self.adapter.stages.index(initial_stage)
        self.current_result: Any | None = None
        self.current_preview: Any | None = None
        self.current_error: BaseException | None = None
        self.current_duration_s = 0.0
        self.current_wait_s = 0.0
        self.current_cached = False
        self._generation = 0
        self._expected_generation = 0
        self._last_displayed_generation = 0
        self._updating = False
        self._closed = False
        self._ignore_control_events = False
        self._overwrite_confirmation_pending = False
        self._message = ""
        self._help_message = ""
        self._stage_message = ""
        self._validation_message = ""
        self._current_image_shape: tuple[int, int] | None = None
        self._reset_view_on_next_result = True
        self._roi_selection: int | None = None
        self._slider_change_pending = False
        self._last_hover_target: tuple[Any, ...] | None = None
        self._supports_roi_selection = bool(getattr(self.adapter, "supports_roi_selection", False))
        self.HELP_TEXT = self._build_help_text()

        self.groups = self.adapter.parameter_groups()
        self.group_names = tuple(self.groups)
        self._spec_by_name = {
            spec.name: spec for specs in self.groups.values() for spec in specs
        }
        self.current_group = self.group_names[0]
        self._controls: dict[str, _Control] = {}
        self._source_values = self._values_from_config(self.original_viewer_config)
        self._all_values = dict(self._source_values)
        self._committed_values = dict(self._source_values)
        self._committed_viewer_config = self.original_viewer_config

        self.controller = AnalysisController(
            self.adapter.analyze, asynchronous=asynchronous
        )

        self.fig: Figure
        self.ax_image: Axes
        self.image_artist: AxesImage
        self.boundary_artist: AxesImage
        self.loading_text: Text
        self.status_text: Text
        self.group_radio: RadioButtons
        self.overlay_checks: CheckButtons
        self._buttons: list[Button] = []
        self._button_callback_ids: list[int] = []
        self._group_radio_callback_id: int | None = None
        self._overlay_checks_callback_id: int | None = None
        self._main_canvas_callback_ids: list[int] = []
        self._graph_canvas_callback_ids: list[int] = []
        self._control_slot_axes: list[Axes] = []
        self.cap_graph_axis: Axes | None = None
        self.graph_main_axis: Axes | None = None
        self.graph_heatmap_axis: Axes | None = None
        self.graph_polar_axis: Axes | None = None
        self.graph_radial_axis: Axes | None = None
        self.graph_summary_axis: Axes | None = None
        self.graph_colorbar_axis: Axes | None = None
        self.graph_fig: Figure | None = None
        self.graph_status_text: Text | None = None
        self.local_graph_hover_artist: Rectangle | None = None
        self.local_radial_hover_artist: Line2D | None = None
        self.local_polar_hover_artist: Line2D | None = None
        self.homogeneity_artists: list[Any] = []
        self.homogeneity_hover_artists: list[Any] = []
        self.local_geometry_artists: list[Any] = []
        self.inscribed_geometry_artists: list[Any] = []
        self.homogeneity_info_text: Text | None = None
        self.homogeneity_graph_hover_artist: Any = None
        self.dimension_lines: list[tuple[Line2D, Line2D]] = []
        self.measurement_labels: list[Text] = []
        self._measurement_label_groups: list[tuple[Text, Text, Text]] = []
        self._active_measurement_count = 0
        self.point_artists: list[Line2D] = []
        self.annotation_artists: list[Text] = []
        self.scale_artists: list[Any] = []
        self._debounce_timer: Any = None
        self._poll_timer: Any = None
        self.overlay_labels = self.adapter.overlay_labels(self.current_viewer_config)
        self.overlay_state = self.adapter.default_overlay_state(self.current_viewer_config)

        self._build_figure()
        if self.image_paths:
            self._schedule_analysis(immediate=True)
        else:
            self._show_no_data_state()

    def _resolve_start_index(self, selected_file: str | Path | None) -> int:
        if selected_file is None:
            return 0
        candidate = Path(selected_file).expanduser()
        if not candidate.is_absolute():
            candidate = self.folder / candidate
        candidate = candidate.resolve()
        resolved = [path.resolve() for path in self.image_paths]
        try:
            return resolved.index(candidate)
        except ValueError:
            raise FileNotFoundError(
                f"Selected image '{selected_file}' was not found in '{self.folder}'."
            ) from None

    def _values_from_config(self, config: Any) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for specs in self.groups.values():
            for spec in specs:
                paths = spec.config_paths or tuple((name,) for name in spec.config_names)
                if spec.kind == "range":
                    if len(paths) == 1:
                        values[spec.name] = tuple(_get_by_path(config, paths[0]))
                    else:
                        values[spec.name] = tuple(_get_by_path(config, path) for path in paths)
                elif spec.kind == "text":
                    raw_value = _get_by_path(config, paths[0])
                    values[spec.name] = (
                        spec.format_value(raw_value) if spec.format_value else str(raw_value)
                    )
                else:
                    values[spec.name] = _get_by_path(config, paths[0])
        return values

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15.5, 9.0))
        self.ax_image = self.fig.add_axes([0.025, 0.13, 0.65, 0.82])
        initial_image = self._load_initial_preview()
        if initial_image is None:
            initial_image = self._make_placeholder_image()
        transparent = np.zeros((*initial_image.shape[:2], 4), dtype=np.float32)
        self.image_artist = self.ax_image.imshow(initial_image, interpolation="nearest")
        self.boundary_artist = self.ax_image.imshow(transparent, interpolation="nearest", zorder=3)
        self.local_coverage_artist = self.ax_image.imshow(transparent, interpolation="nearest", zorder=1.5)
        self.local_coverage_artist.set_visible(False)
        self.radial_hover_patch = Wedge((0, 0), 1, 0, 360, width=0.5, facecolor="yellow", edgecolor="yellow", alpha=0.22, linewidth=1.8, visible=False, zorder=21)
        self.ax_image.add_patch(self.radial_hover_patch)
        self.local_hover_patch = Wedge((0, 0), 1, 0, 1, width=.5, facecolor="#ffe600", edgecolor="#ffe600", alpha=.22, linewidth=1.8, visible=False, zorder=22)
        self.ax_image.add_patch(self.local_hover_patch)
        self.ax_image.axis("off")
        self.loading_text = self.ax_image.text(
            0.5,
            0.5,
            "Loading image...",
            transform=self.ax_image.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="white",
            bbox={
                "facecolor": (0.0, 0.0, 0.0, 0.45),
                "edgecolor": "none",
                "pad": 6.0,
            },
        )
        self._set_main_image(initial_image, reset_view=True)

        self.ax_parameter_groups = self.fig.add_axes([0.70, 0.765, 0.135, 0.20])
        ax_groups = self.ax_parameter_groups
        ax_groups.set_title("Parameter group", fontsize=9)
        self.group_radio = RadioButtons(
            ax_groups, self.group_names, active=0, activecolor="#2878b5"
        )
        for text in self.group_radio.labels:
            text.set_fontsize(8)
        self._group_radio_callback_id = self.group_radio.on_clicked(self._on_group_selected)

        overlay_height = 0.20 if len(self.overlay_labels) <= 6 else 0.26
        self.ax_overlay_controls = self.fig.add_axes([0.845, 0.965 - overlay_height, 0.145, overlay_height])
        ax_overlays = self.ax_overlay_controls
        ax_overlays.set_title("Overlay", fontsize=9)
        self.overlay_checks = CheckButtons(
            ax_overlays,
            self.overlay_labels,
            [self.overlay_state[label] for label in self.overlay_labels],
        )
        for text in self.overlay_checks.labels:
            text.set_fontsize(7 if len(self.overlay_labels) > 6 else 8)
        self._overlay_checks_callback_id = self.overlay_checks.on_clicked(
            self._on_overlay_clicked
        )

        self._build_control_slots()
        # Large analytic plots live in a synchronized second figure so widget
        # axes in the control window never compete with graph typography.
        self.graph_fig = plt.figure(figsize=(9.5, 7.0))
        self.graph_fig.canvas.manager.set_window_title("SEM coverage diagnostics – graphs")
        grid = self.graph_fig.add_gridspec(2, 3, height_ratios=(1.0, 3.0), width_ratios=(4.5, 1.25, .18), left=.07, right=.96, bottom=.13, top=.93, hspace=.38, wspace=.32)
        self.graph_main_axis = self.graph_fig.add_axes([.08, .15, .84, .74])
        self.cap_graph_axis = self.graph_main_axis  # compatibility for cap/legacy helpers
        self.graph_heatmap_axis = self.graph_fig.add_subplot(grid[1, 0])
        self.graph_polar_axis = self.graph_fig.add_subplot(grid[0, 0])
        self.graph_radial_axis = self.graph_fig.add_subplot(grid[1, 1])
        self.graph_summary_axis = self.graph_fig.add_subplot(grid[0, 1])
        self.graph_colorbar_axis = self.graph_fig.add_subplot(grid[1, 2])
        for axis in (self.graph_main_axis, self.graph_heatmap_axis, self.graph_polar_axis, self.graph_radial_axis, self.graph_summary_axis, self.graph_colorbar_axis):
            axis.set_visible(False)
        graph_status_axis = self.graph_fig.add_axes([0.07, 0.035, 0.89, 0.055])
        graph_status_axis.axis("off")
        self.graph_status_text = graph_status_axis.text(0.0, 0.8, "", va="top", fontsize=9)
        self._rebuild_parameter_controls()
        self._layout_right_panel_for_group()
        self._build_buttons()

        ax_status = self.fig.add_axes([0.025, 0.012, 0.965, 0.062])
        ax_status.axis("off")
        self.status_text = ax_status.text(
            0.0, 1.0, "", va="top", ha="left", fontsize=9, wrap=True
        )

        self._main_canvas_callback_ids.extend(
            [
                self.fig.canvas.mpl_connect("key_press_event", self._on_key),
                self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
                self.fig.canvas.mpl_connect("button_release_event", self._on_button_release),
                self.fig.canvas.mpl_connect("resize_event", self._on_resize),
                self.fig.canvas.mpl_connect("close_event", self._on_close),
            ]
        )
        self._graph_canvas_callback_ids.extend([
            self.graph_fig.canvas.mpl_connect("close_event", self._on_graph_close),
            self.graph_fig.canvas.mpl_connect("motion_notify_event", self._on_graph_motion),
            self.graph_fig.canvas.mpl_connect("resize_event", self._on_resize),
        ])

        self._debounce_timer = self.fig.canvas.new_timer(interval=self.debounce_ms)
        self._debounce_timer.single_shot = True
        self._debounce_timer.add_callback(self._submit_analysis)
        self._poll_timer = self.fig.canvas.new_timer(interval=75)
        self._poll_timer.add_callback(self._poll_results)
        self._poll_timer.start()
        self._update_status()

    def _build_control_slots(self) -> None:
        max_controls = max(len(specs) for specs in self.groups.values())
        top = 0.69
        bottom = 0.145
        slot_height = 0.035
        total_height = top - bottom
        gap = max((total_height - max_controls * slot_height) / max(max_controls - 1, 1), 0.01)
        for slot in range(max_controls):
            y = top - slot * (slot_height + gap) - slot_height
            axis = self.fig.add_axes([0.705, y, 0.275, slot_height])
            axis.set_visible(False)
            self._control_slot_axes.append(axis)

    def _build_help_text(self) -> str:
        text = (
            "Left/Right image | Up/Down stage | Home/End first/last | R reset group | "
            "Shift+R reset all | S save | F5 reload | H help | Apply changes button"
        )
        if self._supports_roi_selection:
            text += " | [ prev ROI | ] next ROI | \\ all ROIs"
        return text

    def _rebuild_parameter_controls(self) -> None:
        self._release_canvas_mouse()
        self._clear_parameter_controls()
        for axis in self._control_slot_axes:
            axis.set_visible(False)
            axis.clear()

        for slot, spec in enumerate(self.groups[self.current_group]):
            axis = self._control_slot_axes[slot]
            axis.clear()
            axis.set_visible(True)
            widget = self._create_control_widget(axis, spec, self._all_values[spec.name])
            callback = (
                lambda new_value, parameter=spec.name: self._on_parameter_changed(
                    parameter, new_value
                )
            )
            if spec.kind in {"bool", "choice"}:
                callback_id = widget.on_clicked(callback)
            elif spec.kind == "text":
                callback_id = widget.on_submit(callback)
            else:
                callback_id = widget.on_changed(callback)
            self._controls[spec.name] = _Control(spec, axis, widget, callback_id)
        self._update_control_activity_states()
        self.fig.canvas.draw_idle()

    def _create_control_widget(
        self, axis: Axes, spec: ParameterSpec, value: Any
    ) -> Slider | RangeSlider | CheckButtons | TextBox | RadioButtons:
        if spec.kind == "range":
            minimum = float(spec.minimum)
            maximum = float(spec.maximum)
            value = tuple(np.clip(value, minimum, maximum))
        elif spec.kind in ("float", "int"):
            minimum = float(spec.minimum)
            maximum = float(spec.maximum)
            value = float(np.clip(float(value), minimum, maximum))
        else:
            minimum = maximum = 0.0

        if spec.kind == "bool":
            widget: Slider | RangeSlider | CheckButtons = CheckButtons(
                axis, [spec.label], [bool(value)]
            )
            widget.labels[0].set_fontsize(8)
            return widget
        if spec.kind == "text":
            axis.set_title(spec.label, fontsize=8, pad=1.0, loc="left")
            widget = TextBox(axis, "", initial=str(value))
            widget.label.set_visible(False)
            if spec.placeholder:
                axis.text(
                    1.0,
                    1.35,
                    spec.placeholder,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                    color="0.45",
                )
            return widget
        if spec.kind == "choice":
            if not spec.choices:
                raise ValueError(f"Choice control '{spec.name}' has no choices.")
            labels = [label for _stored, label in spec.choices]
            stored = [stored_value for stored_value, _label in spec.choices]
            active = stored.index(str(value)) if str(value) in stored else 0
            axis.set_title(spec.label, fontsize=8, pad=1.0, loc="left")
            widget = RadioButtons(axis, labels, active=active, activecolor="#2878b5")
            for text in widget.labels:
                text.set_fontsize(7)
            return widget
        if spec.kind == "range":
            widget = RangeSlider(
                axis,
                spec.label,
                minimum,
                maximum,
                valinit=value,
                valstep=spec.step,
            )
        else:
            widget = Slider(
                axis,
                spec.label,
                minimum,
                maximum,
                valinit=value,
                valstep=spec.step,
            )
        widget.label.set_fontsize(8)
        widget.valtext.set_fontsize(8)
        vline = getattr(widget, "vline", None)
        if vline is not None:
            vline.set_visible(False)
        return widget

    def _build_buttons(self) -> None:
        definitions: tuple[tuple[str, Callable[[], None]], ...] = (
            ("Load data", self.load_data),
            ("Previous", self.previous_image),
            ("Next", self.next_image),
        )
        if self._supports_roi_selection:
            definitions += (
                ("Previous ROI", self.previous_roi),
                ("Next ROI", self.next_roi),
                ("All ROIs", self.all_rois),
            )
        definitions += (
            ("Apply changes", self.apply_changes),
            ("Reset group", self.reset_group),
            ("Reset all", self.reset_all),
            ("Save tuned config", self.save),
            ("Reload", self.reload),
        )
        left = 0.025
        width_map = {
            "Load data": 0.080,
            "Previous": 0.070,
            "Next": 0.070,
            "Previous ROI": 0.078,
            "Next ROI": 0.068,
            "All ROIs": 0.062,
            "Apply changes": 0.100,
            "Reset group": 0.090,
            "Reset all": 0.080,
            "Save tuned config": 0.125,
            "Reload": 0.070,
        }
        gap = 0.008
        for label, callback in definitions:
            button_width = width_map[label]
            axis = self.fig.add_axes([left, 0.09, button_width, 0.035])
            button = Button(axis, label)
            button.label.set_fontsize(8)
            callback_id = button.on_clicked(lambda _event, action=callback: action())
            self._buttons.append(button)
            self._button_callback_ids.append(callback_id)
            left += button_width + gap

    def load_data(self) -> None:
        """Replace only the effective image source, retaining tuned parameters."""
        outcome = choose_data_source(self.folder, self.fig)
        if outcome.error_message:
            self._message = f"Could not open the data-source dialog: {outcome.error_message}"
            self._update_status()
            return
        if outcome.cancelled:
            self._message = "Load data cancelled."
            self._update_status()
            return
        if outcome.source is not None:
            self._apply_data_source(outcome.source.folder, outcome.source.file)

    def _apply_data_source(self, folder: Path, file: str | None) -> None:
        try:
            app = replace(self.app_config, folder=str(folder), file=file)
            paths = self.adapter.resolve_images(folder, app, Path(file) if file else None)
            if not paths:
                raise FileNotFoundError("No supported images found.")
        except (OSError, ValueError, FileNotFoundError) as exc:
            self.folder = folder
            self.selected_file = Path(file) if file else None
            self.image_paths = []
            self.index = 0
            self._data_message = f"No data loaded: {exc}"
            self._message = self._data_message
            self._show_no_data_state()
            return
        self.app_config = app
        self.folder = folder.resolve()
        self.selected_file = Path(file) if file else None
        self.image_paths = paths
        self.index = 0
        self._roi_selection = None
        self.current_result = self.current_preview = None
        self.current_error = None
        self.controller.clear_cache()
        self._data_message = ""
        self._message = f"Loaded {len(paths)} image(s)."
        self._reset_view_on_next_result = True
        self._show_loading_preview()
        self._schedule_analysis(immediate=True)

    def _on_group_selected(self, label: str) -> None:
        self.current_group = label
        self._help_message = ""
        self._rebuild_parameter_controls()
        self._layout_right_panel_for_group()
        # A group switch changes control and graph layout, not the analysed
        # image.  Keep the existing base image/boundary artists and update only
        # local-grid artists whose visibility is group-dependent.
        self._update_local_geometry_artists()
        self._update_local_coverage_map()
        self._update_cap_graph()
        self._update_status()

    def _layout_right_panel_for_group(self) -> None:
        """Lay out control widgets below the header panels in the main window."""
        header_bottom = min(self.ax_parameter_groups.get_position().y0, self.ax_overlay_controls.get_position().y0)
        top, bottom = header_bottom - 0.015, 0.145
        count = len(self.groups[self.current_group]); height = 0.033
        gap = max((top-bottom-count*height)/max(count-1,1), 0.008)
        for index, axis in enumerate(self._control_slot_axes):
            if index < count:
                axis.set_position([0.705, top-(index+1)*height-index*gap, 0.275, height])

    def _active_widget_overlaps(self) -> list[tuple[str, str]]:
        """Return visible axes intersections for layout tests and DEBUG checks."""
        axes=[(name, control.axis) for name,control in self._controls.items() if control.axis.get_visible()]
        axes += [("parameter groups",self.ax_parameter_groups),("overlay",self.ax_overlay_controls)]
        axes += [("button",button.ax) for button in self._buttons]
        out=[]
        for i,(a,aa) in enumerate(axes):
            for b,bb in axes[i+1:]:
                if aa.get_position().overlaps(bb.get_position()): out.append((a,b))
        return out

    @property
    def has_pending_changes(self) -> bool:
        return any(
            self._all_values[name] != self._committed_values.get(name)
            for name in self._all_values
        )

    def _normalize_control_value(self, spec: ParameterSpec, value: Any) -> Any:
        if spec.kind == "bool":
            return bool(self._controls[spec.name].widget.get_status()[0])
        if spec.kind == "range":
            return tuple(value)
        if spec.kind == "text":
            return str(value)
        if spec.kind == "choice":
            for stored, label in spec.choices or ():
                if value == label:
                    return stored
            return str(value)
        return value

    def _current_valid_config(self) -> Any | None:
        try:
            candidate = self.adapter.update_config(
                self.original_viewer_config, self._all_values
            )
        except ValueError:
            return None
        validation = self.adapter.validate_config(candidate, self.image_paths[self.index] if self.image_paths else None)
        if validation:
            return None
        return candidate

    def _set_validation_error(self, message: str) -> None:
        self._validation_message = message
        self._message = f"Validation: {message}"
        self._stop_debounce()
        self._updating = self.controller.running

    def _clear_validation_error(self) -> None:
        self._validation_message = ""
        if self._message.startswith("Validation:"):
            self._message = ""

    def _restore_control_value(self, name: str, value: Any) -> None:
        self._ignore_control_events = True
        try:
            self._set_control_value(name, value)
        finally:
            self._ignore_control_events = False

    def _set_widget_enabled(self, control: _Control, enabled: bool) -> None:
        control.enabled = bool(enabled)
        widget = control.widget
        if hasattr(widget, "active"):
            try:
                widget.active = bool(enabled)
            except Exception:
                LOGGER.debug("Could not set widget active state", exc_info=True)
        if hasattr(widget, "eventson"):
            widget.eventson = bool(enabled)
        if hasattr(widget, "set_active") and not isinstance(widget, (CheckButtons, RadioButtons)):
            try:
                widget.set_active(bool(enabled))
            except Exception:
                LOGGER.debug("Could not toggle widget active state", exc_info=True)
        control.axis.set_facecolor("#ffffff" if enabled else "#efefef")
        alpha = 1.0 if enabled else 0.42
        for text in control.axis.texts:
            text.set_alpha(alpha)
        for attr in ("label", "valtext"):
            artist = getattr(widget, attr, None)
            if artist is not None and hasattr(artist, "set_alpha"):
                artist.set_alpha(alpha)

    def _activity_for_control(self, control: _Control, config: Any) -> tuple[bool, str]:
        spec = control.spec
        if spec.active_when is None:
            return True, ""
        outcome = spec.active_when(config)
        if isinstance(outcome, tuple):
            return bool(outcome[0]), str(outcome[1])
        return bool(outcome), ""

    def _update_control_activity_states(self) -> None:
        candidate = self._current_valid_config() or self.current_viewer_config
        for control in self._controls.values():
            active, reason = self._activity_for_control(control, candidate)
            control.inactive_reason = reason
            self._set_widget_enabled(control, active)

    def _apply_lightweight_updates(self) -> None:
        self._clear_validation_error()
        self._message = ""
        if self.current_result is not None or self.current_preview is not None:
            self._render_current_image()
        else:
            self._update_status()

    def _commit_current_controls(self) -> None:
        if self._validation_message:
            self._update_status()
            return
        candidate = self._current_valid_config()
        if candidate is None:
            self._update_status()
            return
        changed_names = [
            name
            for name in self._all_values
            if self._all_values[name] != self._committed_values.get(name)
        ]
        if not changed_names:
            self._message = "No pending changes."
            self._update_status()
            return
        self.current_viewer_config = candidate
        requires_analysis = any(
            self._spec_by_name[name].requires_analysis for name in changed_names
        )
        self._committed_values = dict(self._all_values)
        self._committed_viewer_config = candidate
        self._slider_change_pending = False
        if requires_analysis:
            self._message = ""
            self._schedule_analysis(immediate=True)
        else:
            self._apply_lightweight_updates()

    def apply_changes(self) -> None:
        self._commit_current_controls()

    def _on_parameter_changed(self, name: str, value: Any) -> None:
        if self._ignore_control_events:
            return
        control = self._controls[name]
        spec = control.spec
        previous = self._all_values[name]
        normalized = self._normalize_control_value(spec, value)
        if not control.enabled:
            self._restore_control_value(name, previous)
            self._help_message = self._control_help_text(spec)
            self._message = control.inactive_reason or "This control is currently inactive."
            self._update_status()
            return
        self._all_values[name] = normalized
        try:
            candidate = self.adapter.update_config(
                self.original_viewer_config, self._all_values
            )
        except ValueError as exc:
            self._help_message = self._control_help_text(spec)
            self._set_validation_error(str(exc))
            self._update_status()
            return
        validation = self.adapter.validate_config(candidate, self.image_paths[self.index] if self.image_paths else None)
        self._help_message = self._control_help_text(spec)
        if validation:
            self._set_validation_error(validation)
            self._update_status()
            return
        self.current_viewer_config = candidate
        self._clear_validation_error()
        self._update_control_activity_states()
        if spec.kind in {"float", "int", "range"}:
            self._slider_change_pending = True
            self._message = "Pending changes. Release the mouse or use Apply changes."
            # Slider motion changes only the tentative value.  Cap curves and
            # local robust rotations are deliberately deferred to mouse release.
            self._update_status()
            return
        elif spec.kind == "bool":
            self._commit_current_controls()
            return
        else:
            self._commit_current_controls()
            return
        self._update_status()

    def _on_motion(self, event: MouseEvent) -> None:
        if not hasattr(event, "inaxes"):
            return
        if event.inaxes is None:
            self._clear_homogeneity_hover()
            return
        for control in self._controls.values():
            if event.inaxes is control.axis and control.axis.get_visible():
                help_text = self._control_help_text(control.spec)
                if self._help_message != help_text:
                    self._help_message = help_text
                    self._update_status()
                return
        if self.current_group == "Coverage local heterogeneity":
            self._update_local_hover_from_image(event)
        else:
            self._update_homogeneity_hover(event)

    def _clear_homogeneity_hover(self) -> None:
        visible = (
            self.radial_hover_patch.get_visible() or self.local_hover_patch.get_visible()
            or (self.local_graph_hover_artist is not None and self.local_graph_hover_artist.get_visible())
            or (self.local_radial_hover_artist is not None and self.local_radial_hover_artist.get_visible())
            or (self.local_polar_hover_artist is not None and self.local_polar_hover_artist.get_visible())
            or any(artist.get_visible() for artist in self.homogeneity_hover_artists)
        )
        self._last_hover_target = None
        if not visible:
            return
        self.radial_hover_patch.set_visible(False)
        self.local_hover_patch.set_visible(False)
        if self.local_graph_hover_artist is not None:
            self.local_graph_hover_artist.set_visible(False)
        if self.local_radial_hover_artist is not None:
            self.local_radial_hover_artist.set_data([], [])
            self.local_radial_hover_artist.set_visible(False)
        if self.local_polar_hover_artist is not None:
            self.local_polar_hover_artist.set_data([], [])
            self.local_polar_hover_artist.set_visible(False)
        for artist in self.homogeneity_hover_artists:
            artist.set_visible(False)
        self._hide_homogeneity_graph_hover()
        if not self._closed:
            self.fig.canvas.draw_idle()
            if self.graph_fig is not None:
                self.graph_fig.canvas.draw_idle()

    def _update_local_hover_from_image(self, event: MouseEvent) -> None:
        """Highlight the local-grid cell under the image cursor without recomputation."""

        if not (event.inaxes is self.ax_image and self.current_result is not None and self.current_viewer_config.local_heterogeneity_enabled):
            self._clear_homogeneity_hover(); return
        provider = getattr(self.adapter, "local_heterogeneity_data", None)
        select = getattr(self.adapter, "_selected_rois", None)
        if not callable(provider) or not callable(select) or event.xdata is None or event.ydata is None:
            return
        selection = self._roi_selection
        if selection is None:
            row, col = int(round(event.ydata)), int(round(event.xdata))
            hit = next((roi for roi in select(self.current_result, None) if 0 <= row < roi.bead_mask.shape[0] and 0 <= col < roi.bead_mask.shape[1] and roi.bead_mask[row, col]), None)
            if hit is None: self._clear_homogeneity_hover(); return
            selection = hit.roi_index
        rois, data = select(self.current_result, selection), provider(self.current_result, selection, self.current_viewer_config)
        if not rois or not data: self._clear_homogeneity_hover(); return
        roi, grid = rois[0], data[0]
        row, col = int(round(event.ydata)), int(round(event.xdata))
        if not (0 <= row < grid.r_over_R.shape[0] and 0 <= col < grid.r_over_R.shape[1]): self._clear_homogeneity_hover(); return
        indices = local_grid_indices(grid, row, col)
        if indices is None:
            self._clear_homogeneity_hover()
            return
        radial, sector = indices
        target = (roi.roi_index, radial, sector)
        if target == self._last_hover_target:
            return
        self._last_hover_target = target
        geometry = self.adapter._sphere_geometry(roi, self.current_viewer_config)
        center = (geometry.center_rc[1], geometry.center_rc[0])
        self.local_hover_patch.set_center(center)
        self.local_hover_patch.set_radius(grid.radial_edges[radial + 1] * geometry.radius_px)
        self.local_hover_patch.set_width((grid.radial_edges[radial + 1] - grid.radial_edges[radial]) * geometry.radius_px)
        self.local_hover_patch.set_theta1(float(grid.sector_edges_deg[sector]))
        self.local_hover_patch.set_theta2(float(grid.sector_edges_deg[sector + 1]))
        self.local_hover_patch.set_visible(True)
        if self.local_graph_hover_artist is not None:
            self.local_graph_hover_artist.set_bounds(float(grid.sector_edges_deg[sector]), float(grid.radial_edges[radial]), float(grid.sector_edges_deg[sector + 1] - grid.sector_edges_deg[sector]), float(grid.radial_edges[radial + 1] - grid.radial_edges[radial]))
            self.local_graph_hover_artist.set_visible(True)
        if self.local_radial_hover_artist is not None:
            self.local_radial_hover_artist.set_data([grid.radial_profile[radial] * 100.0], [(grid.radial_edges[radial] + grid.radial_edges[radial + 1]) / 2.0]); self.local_radial_hover_artist.set_visible(True)
        if self.local_polar_hover_artist is not None:
            self.local_polar_hover_artist.set_data([(grid.sector_edges_deg[sector] + grid.sector_edges_deg[sector + 1]) / 2.0], [grid.polar_profile[sector] * 100.0]); self.local_polar_hover_artist.set_visible(True)
        value = grid.values[radial, sector]
        cell = next(
            (item for item in grid.cells if item.radial_index == radial and item.sector_index == sector),
            None,
        )
        branch = _coverage_branch_label(self.current_viewer_config)
        self._message = (
            f"ROI {roi.roi_index}, local cell r{radial + 1}, φ{sector + 1}: "
            f"{value * 100.0:.2f}% | completeness "
            f"{cell.completeness:.3f} | {branch}"
            if np.isfinite(value) and cell is not None
            else f"ROI {roi.roi_index}, local cell r{radial + 1}, φ{sector + 1}: invalid | {branch}"
        )
        self._update_status(); self.fig.canvas.draw_idle()
        if self.graph_fig is not None: self.graph_fig.canvas.draw_idle()

    def _on_graph_motion(self, event: MouseEvent) -> None:
        """Map a joint-grid heatmap cell back to the corresponding image wedge."""

        if not hasattr(event, "inaxes"):
            return
        if self._closed or self.current_group != "Coverage local heterogeneity":
            return
        if event.inaxes not in {
            self.graph_heatmap_axis,
            self.graph_radial_axis,
            self.graph_polar_axis,
        } or event.xdata is None or event.ydata is None:
            self._clear_homogeneity_hover()
            return
        provider = getattr(self.adapter, "local_heterogeneity_data", None)
        select = getattr(self.adapter, "_selected_rois", None)
        if not callable(provider) or not callable(select) or self.current_result is None:
            return
        rois, data = select(self.current_result, self._roi_selection), provider(self.current_result, self._roi_selection, self.current_viewer_config)
        if not rois or not data:
            self._clear_homogeneity_hover()
            return
        grid, roi = data[0], rois[0]
        radial = sector = None
        if event.inaxes is self.graph_heatmap_axis:
            radial = int(np.searchsorted(grid.radial_edges, event.ydata, side="right") - 1)
            angle = min(max(float(event.xdata), 0.0), float(np.nextafter(360.0, 0.0)))
            sector = int(np.searchsorted(grid.sector_edges_deg, angle, side="right") - 1)
        elif event.inaxes is self.graph_radial_axis:
            radial = int(np.argmin(np.abs((grid.radial_edges[:-1] + grid.radial_edges[1:]) / 2.0 - event.ydata)))
        elif event.inaxes is self.graph_polar_axis:
            sector = int(np.argmin(np.abs((grid.sector_edges_deg[:-1] + grid.sector_edges_deg[1:]) / 2.0 - event.xdata)))
        if radial is not None and not (0 <= radial < len(grid.radial_edges) - 1):
            self._clear_homogeneity_hover()
            return
        if sector is not None and not (0 <= sector < self.current_viewer_config.local_heterogeneity_polar_sector_count):
            self._clear_homogeneity_hover()
            return
        geometry = self.adapter._sphere_geometry(roi, self.current_viewer_config)
        self.local_hover_patch.set_center((geometry.center_rc[1], geometry.center_rc[0]))
        lo_r = grid.radial_edges[radial] if radial is not None else grid.radial_edges[0]
        hi_r = grid.radial_edges[radial + 1] if radial is not None else grid.radial_edges[-1]
        lo_phi = grid.sector_edges_deg[sector] if sector is not None else grid.sector_edges_deg[0]
        hi_phi = grid.sector_edges_deg[sector + 1] if sector is not None else grid.sector_edges_deg[-1]
        self.local_hover_patch.set_radius(hi_r * geometry.radius_px)
        self.local_hover_patch.set_width((hi_r - lo_r) * geometry.radius_px)
        self.local_hover_patch.set_theta1(float(lo_phi))
        self.local_hover_patch.set_theta2(float(hi_phi))
        self.local_hover_patch.set_visible(True)
        if self.local_graph_hover_artist is not None:
            graph_x0 = float(grid.sector_edges_deg[sector]) if sector is not None else float(grid.sector_edges_deg[0])
            graph_x1 = float(grid.sector_edges_deg[sector + 1]) if sector is not None else float(grid.sector_edges_deg[-1])
            graph_y0 = float(grid.radial_edges[radial]) if radial is not None else float(grid.radial_edges[0])
            graph_y1 = float(grid.radial_edges[radial + 1]) if radial is not None else float(grid.radial_edges[-1])
            self.local_graph_hover_artist.set_bounds(graph_x0, graph_y0, graph_x1 - graph_x0, graph_y1 - graph_y0)
            self.local_graph_hover_artist.set_visible(True)
        if radial is not None and self.local_radial_hover_artist is not None:
            self.local_radial_hover_artist.set_data([grid.radial_profile[radial] * 100.0], [(grid.radial_edges[radial] + grid.radial_edges[radial + 1]) / 2.0]); self.local_radial_hover_artist.set_visible(True)
        if sector is not None and self.local_polar_hover_artist is not None:
            self.local_polar_hover_artist.set_data([(grid.sector_edges_deg[sector] + grid.sector_edges_deg[sector + 1]) / 2.0], [grid.polar_profile[sector] * 100.0]); self.local_polar_hover_artist.set_visible(True)
        if radial is not None and sector is not None:
            value = grid.values[radial, sector]
            self._message = (
                f"ROI {roi.roi_index}, local cell r{radial + 1}, φ{sector + 1}: "
                f"{value * 100.0:.2f}% | {_coverage_branch_label(self.current_viewer_config)}"
                if np.isfinite(value)
                else f"ROI {roi.roi_index}, local cell r{radial + 1}, φ{sector + 1}: invalid"
            )
        elif radial is not None:
            self._message = f"ROI {roi.roi_index}, local radial band r{radial + 1}"
        else:
            self._message = f"ROI {roi.roi_index}, local polar sector φ{(sector or 0) + 1}"
        self._update_status()
        self.fig.canvas.draw_idle()
        if self.graph_fig is not None: self.graph_fig.canvas.draw_idle()

    def _ensure_homogeneity_graph_hover_artist(self) -> Line2D | None:
        axis = self.cap_graph_axis
        if axis is None:
            return None
        artist = self.homogeneity_graph_hover_artist
        if artist is None or getattr(artist, "axes", None) is not axis:
            artist = axis.plot([], [], marker="o", linestyle="none", color="red", markersize=8, zorder=20)[0]
            artist.set_visible(False)
            self.homogeneity_graph_hover_artist = artist
        return artist

    def _hide_homogeneity_graph_hover(self) -> None:
        artist = self.homogeneity_graph_hover_artist
        if artist is not None and getattr(artist, "axes", None) is not None:
            artist.set_data([], [])
            artist.set_visible(False)

    def _set_homogeneity_graph_hover(self, x: float, y: float) -> None:
        artist = self._ensure_homogeneity_graph_hover_artist()
        if artist is not None:
            artist.set_data([x], [y])
            artist.set_visible(True)

    def _update_homogeneity_hover(self, event: MouseEvent) -> None:
        if not (event.inaxes is self.ax_image and self.current_group == "Coverage homogeneity (legacy)" and self.current_result is not None):
            self._clear_homogeneity_hover()
            return
        provider=getattr(self.adapter,"homogeneity_data",None); select=getattr(self.adapter,"_selected_rois",None)
        if not callable(provider) or not callable(select) or event.xdata is None or event.ydata is None: return
        selection=self._roi_selection
        if selection is None:
            all_rois=select(self.current_result,None)
            row_i,col_i=int(round(event.ydata)),int(round(event.xdata))
            hit=next((roi for roi in all_rois if 0<=row_i<roi.bead_mask.shape[0] and 0<=col_i<roi.bead_mask.shape[1] and roi.bead_mask[row_i,col_i]),None)
            if hit is None: self._clear_homogeneity_hover(); return
            selection=hit.roi_index
        rois=select(self.current_result,selection); items=provider(self.current_result,selection,self.current_viewer_config)
        if not rois or not items: return
        roi,item=rois[0],items[0]; row,col=event.ydata,event.xdata
        rr=float(item.r_over_R[int(round(row)),int(round(col))]) if 0<=round(row)<item.r_over_R.shape[0] and 0<=round(col)<item.r_over_R.shape[1] else -1
        if not (self.current_viewer_config.homogeneity_inner_radius_fraction <= rr <= self.current_viewer_config.homogeneity_outer_radius_fraction): return
        mode=self.current_viewer_config.homogeneity_view_mode; segments=item.rings if mode=="radial" else item.sectors
        phi=float(item.phi_rad[int(round(row)),int(round(col))]) if mode=="polar" else 0.0
        if mode == "radial":
            index=next((s.index for s in segments if s.inner <= rr <= s.outer),None)
        else:
            delta=2*np.pi/max(1, self.current_viewer_config.polar_sector_count)
            offset=np.radians(self.current_viewer_config.polar_display_rotation_deg)
            index=int(np.floor(np.mod(phi-offset,2*np.pi)/delta))
        if index is None: return
        self._clear_homogeneity_hover()
        s=next((segment for segment in segments if segment.index == index), None)
        if s is None:
            return
        x=s.center if mode=="radial" else (((s.start_angle_deg or 0.0)+(((s.end_angle_deg or 0.0)-(s.start_angle_deg or 0.0))%360.0)/2.0)%360.0)
        y=getattr(s,self.current_viewer_config.selected_cap_coverage_metric)
        if self.cap_graph_axis is not None and y is not None:
            self._set_homogeneity_graph_hover(x,y*100)
            geometry=self.adapter._sphere_geometry(roi, self.current_viewer_config); radius=geometry.radius_px; center=(geometry.center_rc[1],geometry.center_rc[0])
            if mode=="polar":
                for fraction in (self.current_viewer_config.homogeneity_inner_radius_fraction,self.current_viewer_config.homogeneity_outer_radius_fraction):
                    arc=Circle(center,fraction*radius,fill=False,edgecolor="red",linewidth=2.2,zorder=21); self.ax_image.add_patch(arc); self.homogeneity_hover_artists.append(arc)
                for phi in (np.radians(s.start_angle_deg or 0.0),np.radians(s.end_angle_deg or 0.0)):
                    line=self.ax_image.plot([center[0]+self.current_viewer_config.homogeneity_inner_radius_fraction*radius*np.cos(phi),center[0]+self.current_viewer_config.homogeneity_outer_radius_fraction*radius*np.cos(phi)],[center[1]+self.current_viewer_config.homogeneity_inner_radius_fraction*radius*np.sin(phi),center[1]+self.current_viewer_config.homogeneity_outer_radius_fraction*radius*np.sin(phi)],color="red",linewidth=2.2,zorder=21)[0]; self.homogeneity_hover_artists.append(line)
            else:
                self.radial_hover_patch.set_center(center)
                self.radial_hover_patch.set_radius(s.outer * radius)
                self.radial_hover_patch.set_width((s.outer-s.inner) * radius)
                self.radial_hover_patch.set_visible(True)
            self.fig.canvas.draw_idle()

    def _on_button_release(self, _event: MouseEvent) -> None:
        if not hasattr(_event, "inaxes"):
            return
        if self._slider_change_pending:
            self._commit_current_controls()

    def _on_resize(self, _event: Any) -> None:
        """Reposition axes only; never forward a ResizeEvent to widgets."""

        if self._closed:
            return
        self._layout_right_panel_for_group()
        # The graph axes use figure coordinates and remain independent of the
        # control figure.  Redraw both canvases without synthetic mouse events.
        self.fig.canvas.draw_idle()
        if self.graph_fig is not None:
            self.graph_fig.canvas.draw_idle()

    def _stop_debounce(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()

    def _schedule_analysis(self, *, immediate: bool = False) -> None:
        if not self.image_paths:
            self._show_no_data_state()
            return
        validation = self.adapter.validate_config(
            self._committed_viewer_config, self.image_paths[self.index]
        )
        if validation:
            self._message = f"Validation: {validation}"
            self._update_status()
            return
        self._generation += 1
        self._expected_generation = self._generation
        self._updating = True
        self.current_error = None
        self._overwrite_confirmation_pending = False
        self._stop_debounce()
        if immediate:
            self._submit_analysis(self._committed_viewer_config)
        else:
            self._debounce_timer.start()
        self._update_status()

    def _submit_analysis(self, config: Any | None = None) -> None:
        if self._closed or not self.image_paths:
            return
        generation = self._expected_generation
        path = self.image_paths[self.index]
        self.controller.submit(generation, path, config or self._committed_viewer_config)
        self._update_status()

    def _poll_results(self) -> None:
        if self._closed:
            return
        for completion in self.controller.poll():
            if completion.generation != self._expected_generation:
                LOGGER.debug(
                    "Discarded stale diagnostic result generation %s; expected %s",
                    completion.generation,
                    self._expected_generation,
                )
                continue
            self._updating = False
            self._last_displayed_generation = completion.generation
            self.current_wait_s = completion.wait_s
            self.current_cached = completion.cached
            if completion.error is not None:
                if isinstance(completion.error, CoverageSegmentationFailure):
                    LOGGER.warning("No accepted coverage ROI for %s: %s", completion.image_path.name, completion.error)
                    LOGGER.debug("Structured coverage segmentation failure", exc_info=completion.error)
                else:
                    LOGGER.error("Analysis failed for %s", completion.image_path, exc_info=completion.error)
                self.current_error = completion.error
                try:
                    self.current_preview = self.adapter.load_failed_preview(
                        completion.image_path, completion.config
                    )
                except BaseException as preview_exc:
                    LOGGER.debug(
                        "Failed to load diagnostic failure preview for %s",
                        completion.image_path,
                        exc_info=preview_exc,
                    )
                    self.current_preview = None
                self.current_result = None
                self.current_duration_s = completion.duration_s
                self._roi_selection = None
                self._message = (
                    f"Analysis failed for {completion.image_path.name}: {completion.error}"
                )
                self._render_current_image()
                continue
            self.current_result = completion.result
            self.current_preview = None
            self.current_error = None
            self.current_duration_s = completion.duration_s
            self._roi_selection = None
            self._message = "Loaded from cache." if completion.cached else ""
            self._render_current_image()
        if not self.controller.running:
            if self._updating and (self.current_result is not None or self.current_preview is not None):
                self._updating = False
                self._update_status()

    def _render_current_image(self) -> None:
        if self._closed:
            return
        stage = self.adapter.stages[self.stage_index]
        if self.current_result is not None:
            pixels = self.adapter.render_stage(
                self.current_result,
                stage,
                self._roi_selection,
                self.current_viewer_config,
            )
        elif self.current_preview is not None:
            pixels = self.adapter.render_failed_preview(
                self.current_preview, stage, self.current_viewer_config
            )
        else:
            return
        image_shape = pixels.shape[:2]
        reset_view = (
            self._reset_view_on_next_result
            or self._current_image_shape is None
            or self._current_image_shape != image_shape
        )
        self._set_main_image(pixels, reset_view=reset_view)
        self.loading_text.set_visible(False)
        self._current_image_shape = image_shape
        self._reset_view_on_next_result = False
        self._stage_message = self.adapter.stage_message(
            self.current_result,
            stage,
            pixels,
            self._roi_selection,
            self.current_preview,
        )
        self._render_overlays()
        self._update_homogeneity_artists()
        self._update_local_geometry_artists()
        self._update_local_coverage_map()
        self._update_cap_graph()
        self._update_status()
        self.fig.canvas.draw_idle()

    def _clear_homogeneity_artists(self) -> None:
        for artist in self.homogeneity_artists + self.homogeneity_hover_artists:
            try: artist.remove()
            except (ValueError, NotImplementedError): pass
        self.homogeneity_artists.clear(); self.homogeneity_hover_artists.clear()
        if self.homogeneity_info_text is not None:
            self.homogeneity_info_text.set_visible(False)

    def _clear_geometry_artists(self, artists: list[Any]) -> None:
        for artist in artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        artists.clear()

    def _update_local_geometry_artists(self) -> None:
        """Draw optional inscribed-circle and joint-grid geometry on the image."""

        self._clear_geometry_artists(self.inscribed_geometry_artists)
        self._clear_geometry_artists(self.local_geometry_artists)
        if self.current_result is None or not isinstance(self.adapter, CoverageDiagnosticAdapter):
            return
        rois = self.adapter._selected_rois(self.current_result, self._roi_selection)
        if self._roi_selection is None:
            rois = [roi for roi in rois if _include_roi_in_global_summary(roi, self.current_viewer_config)]
        for roi in rois:
            geometry = self.adapter._sphere_geometry(roi, self.current_viewer_config)
            center = (geometry.center_rc[1], geometry.center_rc[0])
            inscribed = self.adapter.inscribed_circle_geometry(roi)
            inscribed_center = (inscribed.center_rc[1], inscribed.center_rc[0])
            if self.overlay_state.get("Inscribed circle", False):
                circle = Circle(inscribed_center, inscribed.radius_px, fill=False, edgecolor="#00d9ff", linewidth=2.6 if geometry.mode == "max_inscribed_circle" else 2.0, linestyle="-.", zorder=16)
                self.ax_image.add_patch(circle); self.inscribed_geometry_artists.append(circle)
                marker = self.ax_image.plot([inscribed_center[0]], [inscribed_center[1]], marker="+", color="#00d9ff", markersize=10, markeredgewidth=2, zorder=17)[0]
                self.inscribed_geometry_artists.append(marker)
                diameter = Line2D(
                    [inscribed_center[0] - inscribed.radius_px, inscribed_center[0] + inscribed.radius_px],
                    [inscribed_center[1], inscribed_center[1]], color="#00d9ff", linewidth=1.6, zorder=17,
                )
                self.ax_image.add_line(diameter); self.inscribed_geometry_artists.append(diameter)
                pixel_size = self.current_result.metadata.mean_pixel_size_m
                diameter_m = 2.0 * inscribed.radius_px * pixel_size if pixel_size else None
                label = self.ax_image.text(
                    inscribed_center[0], inscribed_center[1] - inscribed.radius_px - 7,
                    f"d_ins = {_format_coverage_length_m(diameter_m)}" if diameter_m else f"d_ins = {2.0 * inscribed.radius_px:.1f} px",
                    color="#00d9ff", fontsize=7, ha="center", va="bottom",
                    bbox={"facecolor": (0.0, 0.0, 0.0, .45), "edgecolor": "none", "pad": 1.2}, zorder=17,
                )
                self.inscribed_geometry_artists.append(label)
            if self.current_group != "Coverage local heterogeneity" or not self.current_viewer_config.local_heterogeneity_enabled:
                continue
            inner = self.current_viewer_config.local_heterogeneity_inner_radius_fraction * geometry.radius_px
            outer = self.current_viewer_config.local_heterogeneity_outer_radius_fraction * geometry.radius_px
            if self.overlay_state.get("Local heterogeneity radial bands", True):
                bands = self.current_viewer_config.local_heterogeneity_radial_band_count
                for radius in np.linspace(inner, outer, bands + 1):
                    circle = Circle(center, radius, fill=False, edgecolor="#ff4040", linewidth=1.25, zorder=14)
                    self.ax_image.add_patch(circle); self.local_geometry_artists.append(circle)
            if self.overlay_state.get("Local heterogeneity polar sectors", True):
                sectors = self.current_viewer_config.local_heterogeneity_polar_sector_count
                offset = np.radians(self.current_viewer_config.local_heterogeneity_display_rotation_deg % 360.0)
                for angle in offset + np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False):
                    line = self.ax_image.plot(
                        [center[0] + inner * np.cos(angle), center[0] + outer * np.cos(angle)],
                        [center[1] + inner * np.sin(angle), center[1] + outer * np.sin(angle)],
                        color="#42ff79", linewidth=1.1, zorder=14,
                    )[0]
                    self.local_geometry_artists.append(line)

    def _update_local_coverage_map(self) -> None:
        """Render one transparent per-pixel local-grid map from cached results."""

        visible = (
            self.current_group == "Coverage local heterogeneity"
            and self.current_result is not None
            and self.current_viewer_config.local_heterogeneity_enabled
            and self.overlay_state.get("Local heterogeneity coverage map", False)
            and isinstance(self.adapter, CoverageDiagnosticAdapter)
        )
        if not visible:
            self.local_coverage_artist.set_visible(False)
            return
        rois = self.adapter._selected_rois(self.current_result, self._roi_selection)
        if self._roi_selection is None:
            rois = [roi for roi in rois if _include_roi_in_global_summary(roi, self.current_viewer_config)]
        grids = self.adapter.local_heterogeneity_data(self.current_result, self._roi_selection, self.current_viewer_config)
        if not grids:
            self.local_coverage_artist.set_visible(False); return
        all_values = np.concatenate([grid.values[np.isfinite(grid.values)] for grid in grids if np.isfinite(grid.values).any()]) if any(np.isfinite(grid.values).any() for grid in grids) else np.array([])
        if not all_values.size:
            self.local_coverage_artist.set_visible(False); return
        lo, hi = float(all_values.min()), float(all_values.max())
        if hi <= lo: hi = lo + 1e-12
        rgba = np.zeros((*self.current_result.display.shape, 4), dtype=np.float32)
        cmap = plt.get_cmap("viridis")
        for roi, grid in zip(rois, grids):
            radial, sectors, in_domain = local_grid_index_maps(grid)
            values = grid.values[radial, sectors]
            domain = roi.bead_mask & in_domain & np.isfinite(values)
            colors = cmap(np.clip((values - lo) / (hi - lo), 0, 1))
            rgba[domain] = colors[domain]
            rgba[..., 3][domain] = .38
        self.local_coverage_artist.set_data(rgba)
        self.local_coverage_artist.set_visible(True)

    def _update_homogeneity_artists(self) -> None:
        """Draw dedicated ring/sector geometry only in the homogeneity group."""
        self._clear_homogeneity_artists()
        if not (self.current_group == "Coverage homogeneity (legacy)" and self.current_result is not None and getattr(self.current_viewer_config, "coverage_homogeneity_enabled", False)):
            return
        provider=getattr(self.adapter,"homogeneity_data",None); select=getattr(self.adapter,"_selected_rois",None)
        if not callable(provider) or not callable(select): return
        items=provider(self.current_result,self._roi_selection,self.current_viewer_config)
        rois=select(self.current_result,self._roi_selection)
        if self._roi_selection is None:
            include=getattr(self.adapter,"_coverage_status",None)
            rois=[roi for roi in rois if include is None or not str(include(roi,self.current_viewer_config)).startswith("excluded")]
        mode=self.current_viewer_config.homogeneity_view_mode
        color="red" if mode=="radial" else "lime"
        for roi,item in zip(rois,items):
            geometry=self.adapter._sphere_geometry(roi, self.current_viewer_config)
            radius=geometry.radius_px
            center=(geometry.center_rc[1],geometry.center_rc[0])
            if mode=="radial":
                fractions=[item.rings[0].inner]+[ring.outer for ring in item.rings] if item.rings else []
                for fraction in fractions:
                    circle=Circle(center,fraction*radius,fill=False,edgecolor=color,linewidth=1.8,zorder=12); self.ax_image.add_patch(circle); self.homogeneity_artists.append(circle)
            else:
                inner=self.current_viewer_config.homogeneity_inner_radius_fraction*radius; outer=self.current_viewer_config.homogeneity_outer_radius_fraction*radius
                for rr in (inner,outer):
                    circle=Circle(center,rr,fill=False,edgecolor=color,linewidth=1.8,zorder=12); self.ax_image.add_patch(circle); self.homogeneity_artists.append(circle)
                for segment in item.sectors:
                    for phi in (np.radians(segment.start_angle_deg or 0.0),):
                        x=[center[0]+inner*np.cos(phi),center[0]+outer*np.cos(phi)]; y=[center[1]+inner*np.sin(phi),center[1]+outer*np.sin(phi)]
                        line=self.ax_image.plot(x,y,color=color,linewidth=1.5,zorder=12)[0]; self.homogeneity_artists.append(line)
        text=(f"ring width = {self.current_viewer_config.radial_ring_width_fraction:.2f} R" if mode=="radial" else f"{self.current_viewer_config.polar_sector_count} sectors\nradial range = {self.current_viewer_config.homogeneity_inner_radius_fraction:.2f}–{self.current_viewer_config.homogeneity_outer_radius_fraction:.2f} R")
        self.homogeneity_info_text=self.ax_image.text(.02,.04,text,transform=self.ax_image.transAxes,color="white",fontsize=8,bbox={"facecolor":"black","alpha":.55,"pad":2},zorder=13)

    def _update_cap_graph(self) -> None:
        """Render coverage-cap curves without submitting a production analysis."""

        axis = self.cap_graph_axis
        if axis is None:
            return
        for local_axis in (self.graph_heatmap_axis, self.graph_polar_axis, self.graph_radial_axis, self.graph_summary_axis, self.graph_colorbar_axis):
            if local_axis is not None:
                local_axis.clear()
                local_axis.set_visible(False)
        if self.current_group == "Coverage local heterogeneity" and self.current_result is not None:
            axis.set_visible(False)
            provider = getattr(self.adapter, "local_heterogeneity_data", None)
            items = provider(self.current_result, self._roi_selection, self.current_viewer_config) if callable(provider) else []
            if not items or not self.current_viewer_config.local_heterogeneity_enabled:
                self.graph_main_axis.set_visible(True)
                self.graph_main_axis.clear()
                self.graph_main_axis.text(.5, .5, "Local heterogeneity is disabled or no included ROI is available", transform=self.graph_main_axis.transAxes, ha="center", va="center")
                return
            values = np.asarray([item.values for item in items], dtype=float)
            matrix = np.nanmean(values, axis=0)
            finite_values = values[np.isfinite(values)]
            color_lo = float(finite_values.min()) if finite_values.size else 0.0
            color_hi = float(finite_values.max()) if finite_values.size else 1.0
            if color_hi <= color_lo:
                color_hi = color_lo + 1e-12
            first = items[0]
            heatmap_axis = self.graph_heatmap_axis
            polar_axis = self.graph_polar_axis
            radial_axis = self.graph_radial_axis
            summary_axis = self.graph_summary_axis
            colorbar_axis = self.graph_colorbar_axis
            show_heatmap = self.current_viewer_config.local_heterogeneity_show_heatmap
            show_profiles = self.current_viewer_config.local_heterogeneity_show_1d_profiles
            if not show_heatmap and not show_profiles:
                self.graph_main_axis.set_visible(True); self.graph_main_axis.clear()
                self.graph_main_axis.text(.5,.5,"Enable a heatmap and/or 1D profiles to display local heterogeneity.", transform=self.graph_main_axis.transAxes, ha="center", va="center")
                return
            # These are figure-level GridSpec axes, not inset axes.  Repositioning
            # them for the three display combinations keeps the graph window
            # useful without creating axes outside their parent.
            if show_heatmap and show_profiles:
                heatmap_axis.set_position([.07, .15, .62, .52])
                polar_axis.set_position([.07, .75, .62, .18])
                radial_axis.set_position([.74, .15, .16, .52])
                summary_axis.set_position([.74, .75, .16, .18])
                colorbar_axis.set_position([.92, .15, .025, .52])
            elif show_heatmap:
                heatmap_axis.set_position([.08, .15, .78, .70])
                summary_axis.set_position([.08, .87, .78, .07])
                colorbar_axis.set_position([.89, .15, .03, .70])
            else:
                polar_axis.set_position([.09, .56, .80, .31])
                radial_axis.set_position([.09, .16, .80, .31])
                summary_axis.set_position([.09, .89, .80, .06])
            heatmap_axis.set_visible(show_heatmap)
            colorbar_axis.set_visible(show_heatmap)
            polar_axis.set_visible(show_profiles)
            radial_axis.set_visible(show_profiles)
            summary_axis.set_visible(True)
            if show_heatmap:
                image = heatmap_axis.imshow(
                    matrix * 100.0, origin="lower", aspect="auto", interpolation="nearest",
                    extent=(first.sector_edges_deg[0], first.sector_edges_deg[-1], first.radial_edges[0], first.radial_edges[-1]),
                    cmap="viridis", vmin=color_lo * 100.0, vmax=color_hi * 100.0,
                )
                self.local_graph_hover_artist = Rectangle((0, 0), 0, 0, fill=False, edgecolor="#ff3030", linewidth=2.0, visible=False, zorder=20)
                heatmap_axis.add_patch(self.local_graph_hover_artist)
                heatmap_axis.set(title="Joint local coverage heterogeneity", xlabel="Azimuth angle, φ [°]", ylabel="Normalized radius, r/R")
                heatmap_axis.set_xlim(0, 360); heatmap_axis.set_xticks(np.arange(0, 361, 60))
            else:
                self.local_graph_hover_artist = None
            if show_heatmap:
                self.graph_fig.colorbar(image, cax=colorbar_axis).set_label(f"{self.current_viewer_config.selected_cap_coverage_metric} [%]")
            radial = np.nanmean(np.asarray([item.radial_profile for item in items]) * 100.0, axis=0)
            polar = np.nanmean(np.asarray([item.polar_profile for item in items]) * 100.0, axis=0)
            if show_profiles:
                radial_axis.plot(radial, (first.radial_edges[:-1] + first.radial_edges[1:]) / 2.0, "o-", color="tab:red")
                radial_axis.set(title="Radial", xlabel="%", ylabel="r/R"); radial_axis.tick_params(labelsize=7)
                polar_axis.plot((first.sector_edges_deg[:-1] + first.sector_edges_deg[1:]) / 2.0, polar, "o-", color="tab:green")
                polar_axis.set(title="Polar", xlabel="φ [°]", ylabel="%"); polar_axis.set_xlim(0, 360); polar_axis.tick_params(labelsize=7)
                self.local_radial_hover_artist = radial_axis.plot([], [], "o", color="red", markersize=8, visible=False, zorder=20)[0]
                self.local_polar_hover_artist = polar_axis.plot([], [], "o", color="red", markersize=8, visible=False, zorder=20)[0]
            summary_axis.axis("off")
            summary_axis.text(.02, .95, f"Metric: {self.current_viewer_config.selected_cap_coverage_metric}\nCoverage branch: {_coverage_branch_label(self.current_viewer_config)}\nValid grid cells: {int(np.isfinite(matrix).sum())}/{matrix.size}", va="top", fontsize=9)
            if self.graph_status_text is not None:
                total = [item.total_weighted_sd_pp for item in items if item.total_weighted_sd_pp is not None]
                residual = [item.residual_weighted_sd_pp for item in items if item.residual_weighted_sd_pp is not None]
                self.graph_status_text.set_text(
                    f"Joint grid | total local SD: {np.mean(total):.3f} pp | residual SD: {np.mean(residual):.3f} pp"
                    if total and residual else "Joint grid: insufficient valid cells"
                )
            if self.graph_fig is not None:
                self.graph_fig.canvas.draw_idle()
            return
        if self.current_group == "Coverage homogeneity (legacy)" and self.current_result is not None:
            axis.set_visible(True); axis.clear()
            self.homogeneity_graph_hover_artist = None
            provider = getattr(self.adapter, "homogeneity_data", None)
            items = provider(self.current_result, self._roi_selection, self.current_viewer_config) if callable(provider) else []
            if not items:
                axis.text(.5,.5,"No included ROI",transform=axis.transAxes,ha="center",va="center"); return
            mode=getattr(self.current_viewer_config,"homogeneity_view_mode","radial")
            profiles=[item.rings if mode=="radial" else item.sectors for item in items]
            x=np.asarray([segment.center for segment in profiles[0]],float)
            if mode=="polar":
                x=np.asarray([((segment.start_angle_deg or 0.0) + (((segment.end_angle_deg or 0.0) - (segment.start_angle_deg or 0.0)) % 360.0) / 2.0) % 360.0 for segment in profiles[0]],float)
            metric=self.current_viewer_config.selected_cap_coverage_metric
            values=np.asarray([[np.nan if getattr(segment,metric) is None else getattr(segment,metric)*100 for segment in profile] for profile in profiles])
            mean=np.nanmean(values,axis=0); axis.plot(x,mean,"o-" if mode=="radial" else "o-",color="tab:blue")
            if values.shape[0]>1: axis.fill_between(x,mean-np.nanstd(values,axis=0),mean+np.nanstd(values,axis=0),alpha=.18)
            invalid=np.all(np.isnan(values),axis=0)
            if invalid.any(): axis.scatter(x[invalid],np.zeros(invalid.sum()),marker="x",color="red")
            axis.set(title="Radial coverage homogeneity" if mode=="radial" else "Polar coverage homogeneity",xlabel="Normalized radius, r/R" if mode=="radial" else "Azimuth angle, φ [°]",ylabel="Coverage [%]"); axis.grid(alpha=.25); axis.tick_params(labelsize=9)
            axis.title.set_fontsize(12); axis.xaxis.label.set_size(10); axis.yaxis.label.set_size(10)
            if mode=="polar": axis.set_xlim(0,360); axis.set_xticks(np.arange(0,361,60))
            self._ensure_homogeneity_graph_hover_artist()
            if self.graph_status_text is not None:
                self.graph_status_text.set_text("Legacy separate radial/polar homogeneity workflow")
            if self.graph_fig is not None:
                self.graph_fig.canvas.draw_idle()
            return
        provider = getattr(self.adapter, "cap_graph_data", None)
        visible = (
            self.current_group == "Coverage cap"
            and self.current_result is not None
            and callable(provider)
        )
        axis.set_visible(visible)
        if not visible:
            return
        axis.clear()
        data = provider(self.current_result, self._roi_selection, self.current_viewer_config)
        if data is None:
            axis.text(0.5, 0.5, "No included ROI", transform=axis.transAxes, ha="center", va="center", fontsize=8)
            return
        mode = getattr(self.current_viewer_config, "coverage_cap_graph_mode", "cumulative")
        x = data["fractions"] if mode == "cumulative" else data["annular_x"]
        series = data["cumulative"] if mode == "cumulative" else data["annular"]
        labels = {
            "projected_fraction": "Projected fraction",
            "surface_weighted_fraction": "Surface weighted",
            "projected_over_cap_surface": "Ag / cap surface",
        }
        colors = {"projected_fraction": "tab:blue", "surface_weighted_fraction": "tab:green", "projected_over_cap_surface": "tab:purple"}
        for name, values in series.items():
            mean = np.nanmean(values, axis=0)
            axis.plot(x, mean * 100.0, label=labels[name], color=colors[name], linewidth=1.3)
            if values.shape[0] > 1:
                std = np.nanstd(values, axis=0)
                axis.fill_between(x, (mean - std) * 100.0, (mean + std) * 100.0, color=colors[name], alpha=0.12)
        if mode == "cumulative":
            axis.axhline(data["legacy"] * 100.0, color="0.3", linestyle=":", label="Legacy full projected")
            invalid = np.any(data["invalid"], axis=0)
            if invalid.any():
                axis.scatter(x[invalid], np.zeros(int(invalid.sum())), marker="x", color="red", label="Incomplete cap")
        axis.axvline(float(self.current_viewer_config.coverage_cap_radius_fraction), color="red", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Cap radius fraction, a/R" if mode == "cumulative" else "Annulus-center radius, r/R", fontsize=10)
        axis.set_ylabel("Coverage metric [%]", fontsize=10)
        axis.set_title("Cap coverage" if mode == "cumulative" else "Annular radial coverage", fontsize=12)
        axis.tick_params(labelsize=9)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
        if self.graph_status_text is not None:
            self.graph_status_text.set_text(
                f"Cap diagnostics | {_coverage_branch_label(self.current_viewer_config)}"
            )
        if self.graph_fig is not None:
            self.graph_fig.canvas.draw_idle()

    def _render_overlays(self) -> None:
        show = (
            (self.current_result is not None or hasattr(self.adapter, "make_failed_overlay_data"))
            and self.adapter.overlay_stage is not None
            and self.adapter.stages[self.stage_index] == self.adapter.overlay_stage
        )
        if not show:
            self.boundary_artist.set_visible(False)
            self._set_measurement_visibility()
            self._update_point_artists(())
            self._update_annotation_artists(())
            self._set_scale_visibility(False)
            self.fig.canvas.draw_idle()
            return

        if self.current_result is not None:
            overlay = self.adapter.make_overlay_data(
                self.current_result, self._roi_selection, self.current_viewer_config
            )
        elif self.current_preview is not None and hasattr(self.adapter, "make_failed_overlay_data"):
            overlay = self.adapter.make_failed_overlay_data(self.current_preview, self.current_viewer_config)
        else:
            return
        rgba = np.zeros((*overlay.image_shape, 4), dtype=np.float32)
        for layer in overlay.boundary_layers:
            if self.overlay_state.get(layer.control_label, False):
                rgba[layer.mask] = layer.color
        self.boundary_artist.set_data(rgba)
        self.boundary_artist.set_visible(bool(np.any(rgba[..., 3])))

        self._update_measurement_artists(overlay.measurements)
        self._set_measurement_visibility()
        self._update_point_artists(overlay.point_layers)
        self._update_annotation_artists(overlay.text_overlays)
        self._update_scale_artists(overlay)
        scale_label = self.adapter.scale_bar_label
        self._set_scale_visibility(bool(scale_label and self.overlay_state.get(scale_label, False)))
        self.fig.canvas.draw_idle()

    def _update_measurement_artists(
        self, measurements: Sequence[MeasurementOverlay]
    ) -> None:
        self._active_measurement_count = len(measurements)
        while len(self.dimension_lines) < len(measurements):
            hline = Line2D([], [], linewidth=1.2, alpha=0.95)
            vline = Line2D([], [], linewidth=1.2, alpha=0.95)
            self.ax_image.add_line(hline)
            self.ax_image.add_line(vline)
            label = self.ax_image.text(
                0,
                0,
                "",
                fontsize=7,
                ha="center",
                va="bottom",
                bbox={
                    "facecolor": (0.0, 0.0, 0.0, 0.45),
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )
            x_label = self.ax_image.text(
                0,
                0,
                "",
                fontsize=7,
                ha="center",
                va="bottom",
                bbox={
                    "facecolor": (0.0, 0.0, 0.0, 0.45),
                    "edgecolor": "none",
                    "pad": 1.2,
                },
            )
            y_label = self.ax_image.text(
                0,
                0,
                "",
                fontsize=7,
                ha="left",
                va="center",
                bbox={
                    "facecolor": (0.0, 0.0, 0.0, 0.45),
                    "edgecolor": "none",
                    "pad": 1.2,
                },
            )
            self.dimension_lines.append((hline, vline))
            self._measurement_label_groups.append((label, x_label, y_label))
        self.measurement_labels = [
            text for group in self._measurement_label_groups for text in group
        ]

        for index, measurement in enumerate(measurements):
            hline, vline = self.dimension_lines[index]
            if measurement.x_start_row is not None and measurement.x_start_col is not None and measurement.x_end_row is not None and measurement.x_end_col is not None:
                hline.set_data(
                    [measurement.x_start_col, measurement.x_end_col],
                    [measurement.x_start_row, measurement.x_end_row],
                )
            else:
                hline.set_data(
                    [
                        measurement.col - measurement.x_diameter_px / 2.0,
                        measurement.col + measurement.x_diameter_px / 2.0,
                    ],
                    [measurement.row, measurement.row],
                )
            if measurement.y_start_row is not None and measurement.y_start_col is not None and measurement.y_end_row is not None and measurement.y_end_col is not None:
                vline.set_data(
                    [measurement.y_start_col, measurement.y_end_col],
                    [measurement.y_start_row, measurement.y_end_row],
                )
            else:
                vline.set_data(
                    [measurement.col, measurement.col],
                    [
                        measurement.row - measurement.y_diameter_px / 2.0,
                        measurement.row + measurement.y_diameter_px / 2.0,
                    ],
                )
            hline.set_color(measurement.x_color)
            vline.set_color(measurement.y_color)
            label, x_label, y_label = self._measurement_label_groups[index]
            label.set_position(
                (
                    measurement.col,
                    measurement.row - measurement.y_diameter_px / 2.0 - 7.0,
                )
            )
            label.set_text(measurement.label_text)
            label.set_color(measurement.label_color)
            x_label.set_position(
                (
                    measurement.x_label_col
                    if measurement.x_label_col is not None
                    else measurement.col,
                    measurement.x_label_row
                    if measurement.x_label_row is not None
                    else measurement.row - measurement.y_diameter_px / 2.0 - 7.0,
                )
            )
            x_label.set_text(measurement.x_label_text or "")
            x_label.set_color(measurement.x_label_color or measurement.x_color)
            x_label.set_ha(measurement.x_label_ha)
            x_label.set_va(measurement.x_label_va)
            y_label.set_position(
                (
                    measurement.y_label_col
                    if measurement.y_label_col is not None
                    else measurement.col,
                    measurement.y_label_row
                    if measurement.y_label_row is not None
                    else measurement.row,
                )
            )
            y_label.set_text(measurement.y_label_text or "")
            y_label.set_color(measurement.y_label_color or measurement.y_color)
            y_label.set_ha(measurement.y_label_ha)
            y_label.set_va(measurement.y_label_va)
        for index in range(len(measurements), len(self.dimension_lines)):
            hline, vline = self.dimension_lines[index]
            hline.set_visible(False)
            vline.set_visible(False)
            for label in self._measurement_label_groups[index]:
                label.set_visible(False)

    def _set_measurement_visibility(self) -> None:
        line_label = self.adapter.measurement_line_label
        text_label = self.adapter.measurement_text_label
        dimensions_visible = bool(line_label and self.overlay_state.get(line_label, False))
        labels_visible = bool(text_label and self.overlay_state.get(text_label, False))
        for index, (hline, vline) in enumerate(self.dimension_lines):
            active = index < self._active_measurement_count
            hline.set_visible(active and dimensions_visible)
            vline.set_visible(active and dimensions_visible)
            if index < len(self._measurement_label_groups):
                merged, x_label, y_label = self._measurement_label_groups[index]
                merged.set_visible(active and labels_visible and bool(merged.get_text()))
                x_label.set_visible(active and labels_visible and bool(x_label.get_text()))
                y_label.set_visible(active and labels_visible and bool(y_label.get_text()))

    def _update_point_artists(self, point_layers: Sequence[PointOverlay]) -> None:
        while len(self.point_artists) < len(point_layers):
            artist = Line2D([], [], linestyle="none")
            self.ax_image.add_line(artist)
            self.point_artists.append(artist)
        for index, layer in enumerate(point_layers):
            artist = self.point_artists[index]
            visible = self.overlay_state.get(layer.control_label, False)
            artist.set_data(layer.cols, layer.rows)
            artist.set_color(layer.color)
            artist.set_marker(layer.marker)
            artist.set_markersize(layer.markersize)
            artist.set_visible(visible and layer.cols.size > 0)
        for index in range(len(point_layers), len(self.point_artists)):
            self.point_artists[index].set_visible(False)

    def _update_annotation_artists(
        self, annotations: Sequence[TextOverlay]
    ) -> None:
        while len(self.annotation_artists) < len(annotations):
            artist = self.ax_image.text(
                0,
                0,
                "",
                fontsize=8,
                ha="center",
                va="bottom",
                color="white",
            )
            self.annotation_artists.append(artist)
        for index, annotation in enumerate(annotations):
            artist = self.annotation_artists[index]
            visible = self.overlay_state.get(annotation.control_label, False)
            artist.set_position((annotation.col, annotation.row))
            artist.set_text(annotation.text)
            artist.set_color(annotation.color)
            artist.set_fontsize(annotation.fontsize)
            artist.set_ha(annotation.ha)
            artist.set_va(annotation.va)
            if annotation.boxed:
                artist.set_bbox(
                    {
                        "facecolor": (0.0, 0.0, 0.0, 0.45),
                        "edgecolor": "none",
                        "pad": 1.2,
                    }
                )
            else:
                artist.set_bbox(None)
            artist.set_visible(visible)
        for index in range(len(annotations), len(self.annotation_artists)):
            self.annotation_artists[index].set_visible(False)

    def _update_scale_artists(self, overlay: OverlayData) -> None:
        if overlay.pixel_size_m is None:
            self._set_scale_visibility(False)
            return
        h, w = overlay.image_shape
        scale_length_m = _nice_scale_length_m(w * overlay.pixel_size_m * 0.22)
        if scale_length_m <= 0:
            self._set_scale_visibility(False)
            return
        scale_length_px = scale_length_m / overlay.pixel_size_m
        x0, y0 = w * 0.06, h * 0.92
        if not self.scale_artists:
            background = Rectangle(
                (0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.35), edgecolor="none"
            )
            self.ax_image.add_patch(background)
            bar = Line2D([], [], color="white", linewidth=3)
            left_tick = Line2D([], [], color="white", linewidth=1.5)
            right_tick = Line2D([], [], color="white", linewidth=1.5)
            self.ax_image.add_line(bar)
            self.ax_image.add_line(left_tick)
            self.ax_image.add_line(right_tick)
            label = self.ax_image.text(
                0, 0, "", color="white", fontsize=10, va="bottom", ha="left"
            )
            self.scale_artists = [background, bar, left_tick, right_tick, label]
        background, bar, left_tick, right_tick, label = self.scale_artists
        background.set_xy((x0 - 8, y0 - 28))
        background.set_width(scale_length_px + 16)
        background.set_height(36)
        bar.set_data([x0, x0 + scale_length_px], [y0, y0])
        left_tick.set_data([x0, x0], [y0 - 7, y0 + 7])
        right_tick.set_data(
            [x0 + scale_length_px, x0 + scale_length_px], [y0 - 7, y0 + 7]
        )
        label.set_position((x0, y0 - 11))
        label.set_text(_format_length_m(scale_length_m))

    def _set_scale_visibility(self, visible: bool) -> None:
        for artist in self.scale_artists:
            artist.set_visible(visible)

    def _on_overlay_clicked(self, _label: str) -> None:
        statuses = self.overlay_checks.get_status()
        self.overlay_state.update(zip(self.overlay_labels, map(bool, statuses)))
        self._render_overlays()
        # These layers are deliberately independent of the overlay-stage
        # BoundaryLayer machinery and must redraw immediately when toggled.
        self._update_local_geometry_artists()
        self._update_local_coverage_map()
        self._update_status()
        self.fig.canvas.draw_idle()

    def set_overlay(self, label: str, visible: bool) -> None:
        """Set one overlay state, primarily for automation and tests."""

        if label not in self.overlay_state:
            raise KeyError(f"Unknown overlay: {label}")
        current = self.overlay_state[label]
        if current != visible:
            index = self.overlay_labels.index(label)
            self.overlay_checks.set_active(index)

    def previous_image(self) -> None:
        self._navigate_image(-1)

    def next_image(self) -> None:
        self._navigate_image(1)

    def _navigate_image(self, step: int) -> None:
        if not self.image_paths:
            self._show_no_data_state()
            return
        self.index = (self.index + step) % len(self.image_paths)
        self._roi_selection = None
        self._message = ""
        self._stage_message = ""
        self._reset_view_on_next_result = True
        self._show_loading_preview()
        self._schedule_analysis(immediate=True)

    def first_image(self) -> None:
        if not self.image_paths:
            self._show_no_data_state()
            return
        self.index = 0
        self._roi_selection = None
        self._reset_view_on_next_result = True
        self._show_loading_preview()
        self._schedule_analysis(immediate=True)

    def last_image(self) -> None:
        if not self.image_paths:
            self._show_no_data_state()
            return
        self.index = len(self.image_paths) - 1
        self._roi_selection = None
        self._reset_view_on_next_result = True
        self._show_loading_preview()
        self._schedule_analysis(immediate=True)

    def cycle_stage(self, step: int) -> None:
        self.stage_index = (self.stage_index + step) % len(self.adapter.stages)
        if self.current_result is not None or self.current_preview is not None:
            self._render_current_image()
        else:
            self._update_status()

    def previous_roi(self) -> None:
        self._cycle_roi(-1)

    def next_roi(self) -> None:
        self._cycle_roi(1)

    def all_rois(self) -> None:
        if not self._supports_roi_selection:
            return
        self._roi_selection = None
        if self.current_result is not None or self.current_preview is not None:
            self._render_current_image()
        else:
            self._update_status()

    def _cycle_roi(self, step: int) -> None:
        if not self._supports_roi_selection or self.current_result is None:
            return
        options = self.adapter.roi_options(self.current_result)
        if not options:
            self._roi_selection = None
            if self.current_result is not None or self.current_preview is not None:
                self._render_current_image()
            return
        if self._roi_selection is None:
            self._roi_selection = options[0] if step > 0 else options[-1]
        else:
            current_index = options.index(self._roi_selection) if self._roi_selection in options else 0
            self._roi_selection = options[(current_index + step) % len(options)]
        self._render_current_image()

    def _on_key(self, event: KeyEvent) -> None:
        key = (event.key or "").lower()
        if key == "left":
            self.previous_image()
        elif key == "right":
            self.next_image()
        elif key == "up":
            self.cycle_stage(-1)
        elif key == "down":
            self.cycle_stage(1)
        elif key == "home":
            self.first_image()
        elif key == "end":
            self.last_image()
        elif key == "r":
            self.reset_group()
        elif key in ("shift+r", "r+shift"):
            self.reset_all()
        elif key == "s":
            self.save()
        elif key == "f5":
            self.reload()
        elif key == "h":
            self._help_message = self.HELP_TEXT
            self._update_status()
        elif key == "[":
            self.previous_roi()
        elif key == "]":
            self.next_roi()
        elif key == "\\":
            self.all_rois()

    def _set_control_value(self, name: str, value: Any) -> None:
        control = self._controls.get(name)
        if control is None:
            return
        if control.spec.kind == "bool":
            current = bool(control.widget.get_status()[0])
            if current != bool(value):
                control.widget.set_active(0)
        elif control.spec.kind == "choice":
            choices = control.spec.choices or ()
            stored = [stored_value for stored_value, _label in choices]
            if str(value) in stored:
                desired = stored.index(str(value))
                if control.widget.value_selected != choices[desired][1]:
                    control.widget.set_active(desired)
        elif control.spec.kind == "text":
            control.widget.set_val(str(value))
        else:
            control.widget.set_val(value)

    def reset_group(self) -> None:
        self._ignore_control_events = True
        try:
            for spec in self.groups[self.current_group]:
                self._all_values[spec.name] = self._source_values[spec.name]
                self._set_control_value(spec.name, self._source_values[spec.name])
        finally:
            self._ignore_control_events = False
        self.current_viewer_config = self.adapter.update_config(
            self.original_viewer_config, self._all_values
        )
        self._clear_validation_error()
        self._update_control_activity_states()
        self._message = f"Reset group: {self.current_group}."
        self._commit_current_controls()

    def reset_all(self) -> None:
        self._all_values = dict(self._source_values)
        self._ignore_control_events = True
        try:
            for name, value in self._all_values.items():
                self._set_control_value(name, value)
        finally:
            self._ignore_control_events = False
        self.current_viewer_config = self.original_viewer_config
        self._clear_validation_error()
        self._update_control_activity_states()
        self._message = "Reset all parameters to the source configuration."
        self._commit_current_controls()

    def save(self) -> None:
        selected = choose_json_save_path(self.output_config, self.fig)
        if selected is None:
            self._message = "Save cancelled."
            self._update_status()
            return
        self.output_config = selected.expanduser().resolve()
        same_path = self.output_config == self.config_path
        if same_path and not self._overwrite_confirmation_pending:
            self._overwrite_confirmation_pending = True
            self._message = (
                "Overwrite confirmation: press Save tuned config (or S) again to replace "
                "the source configuration."
            )
            self._update_status()
            return
        try:
            saved_path = save_tuned_config(
                self.raw_source_config, self.current_viewer_config, self.output_config,
                effective_folder=self.folder,
                effective_file=self.selected_file.name if self.selected_file is not None else None,
            )
            _saved_app_config, saved_viewer_config, _saved_raw = self.adapter.load_config(saved_path)
            if saved_viewer_config != self.current_viewer_config:
                raise ValueError(
                    "Saved configuration does not round-trip to the current diagnostic settings."
                )
            if Path(_saved_app_config.folder).expanduser() != (self.folder or Path("")) or _saved_app_config.file != (self.selected_file.name if self.selected_file else None):
                raise ValueError("Saved configuration does not preserve the effective data source.")
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.exception("Failed to save tuned configuration")
            self._message = f"Save failed: {exc}"
        else:
            LOGGER.info("Saved tuned configuration to %s", self.output_config)
            self._message = f"Saved tuned configuration: {self.output_config}"
            self._overwrite_confirmation_pending = False
        self._update_status()

    def reload(self) -> None:
        """Reload source config, image list, and the selected image from disk."""

        selected = self.image_paths[self.index].resolve() if self.image_paths else None
        try:
            app_config, viewer_config, raw = self.adapter.load_config(self.config_path)
            folder = self.folder_override or (Path(app_config.folder).expanduser() if app_config.folder else None)
            configured_file = self._cli_selected_file or (Path(app_config.file) if app_config.file else None)
            image_paths = self.adapter.resolve_images(folder, app_config, configured_file) if folder is not None else []
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.exception("Diagnostic reload failed")
            self._message = f"Reload failed: {exc}"
            self._update_status()
            return

        self.app_config = app_config
        self.original_viewer_config = viewer_config
        self.current_viewer_config = viewer_config
        self.raw_source_config = raw
        self.folder = folder
        self.selected_file = Path(image_paths[0].name) if configured_file is not None and image_paths else None
        self.image_paths = image_paths
        resolved = [path.resolve() for path in image_paths]
        self.index = resolved.index(selected) if selected in resolved else 0
        validation = self.adapter.validate_config(viewer_config, self.image_paths[self.index] if self.image_paths else None)
        if validation:
            self._message = f"Reload failed: {validation}"
            self._update_status()
            return
        self._source_values = self._values_from_config(viewer_config)
        self._all_values = dict(self._source_values)
        self._committed_values = dict(self._source_values)
        self._committed_viewer_config = viewer_config
        self.overlay_labels = self.adapter.overlay_labels(viewer_config)
        self.overlay_state = self.adapter.default_overlay_state(viewer_config)
        self._sync_overlay_checkboxes()
        self._ignore_control_events = True
        try:
            for name, value in self._all_values.items():
                self._set_control_value(name, value)
        finally:
            self._ignore_control_events = False
        self._update_control_activity_states()
        self.current_result = None
        self.current_preview = None
        self.current_error = None
        self.current_wait_s = 0.0
        self.current_cached = False
        self._roi_selection = None
        self.controller.clear_cache()
        self._clear_validation_error()
        self._message = "Reloaded source configuration and image list."
        self._stage_message = ""
        self._reset_view_on_next_result = True
        self._show_loading_preview()
        if self.image_paths:
            self._schedule_analysis(immediate=True)
        else:
            self._data_message = "No data loaded. Use Load data to select an image or folder."
            self._show_no_data_state()

    @property
    def has_unsaved_changes(self) -> bool:
        current = json.dumps(asdict(self.current_viewer_config), sort_keys=True)
        original = json.dumps(asdict(self.original_viewer_config), sort_keys=True)
        return current != original

    def _ui_state_label(self) -> str:
        if self.controller.running:
            return "RUNNING"
        if self.has_pending_changes:
            return "PENDING"
        return "UP TO DATE"

    def _update_status(self) -> None:
        stage = self.adapter.stages[self.stage_index]
        state_label = self._ui_state_label()
        running = f" | {state_label}"
        if self.controller.running and self.controller.pending_generation is not None:
            running += " | newer settings pending"
        dirty = " | UNSAVED" if self.has_unsaved_changes else ""
        roi_label = self.adapter.format_roi_selection(self._roi_selection)
        roi_suffix = f" | {roi_label}" if roi_label else ""
        try:
            if self.current_result is not None:
                summary = self.adapter.summarize_result(
                    self.current_result, self.current_duration_s,
                    self._roi_selection, self.current_viewer_config,
                )
            elif self.current_error is not None:
                summary = self.adapter.summarize_failed_preview(
                    self.current_preview, self.current_error,
                    self.current_duration_s, stage,
                )
            else:
                summary = "No completed analysis yet."
        except Exception as exc:
            LOGGER.exception("Status summary formatting failed")
            summary = f"Status summary error: {type(exc).__name__}"
        timing_parts: list[str] = [f"State: {state_label}", f"Group: {self.current_group}"]
        if self.current_duration_s > 0:
            timing_parts.append(f"last analysis {self.current_duration_s:.2f} s")
        if self.current_wait_s > 0:
            timing_parts.append(f"worker wait {self.current_wait_s:.2f} s")
        if self.current_cached:
            timing_parts.append("cache hit")
        if self.controller.running:
            running_generation = self.controller.running_generation
            if running_generation is not None:
                running_text = f"running generation {running_generation}"
                pending_generation = self.controller.pending_generation
                if pending_generation is not None:
                    wait_s = self.controller.pending_wait_s
                    if wait_s is not None:
                        running_text += (
                            f"; newer settings pending (generation {pending_generation}, "
                            f"waiting {wait_s:.2f} s)"
                        )
                    else:
                        running_text += f"; newer settings pending (generation {pending_generation})"
                timing_parts.append(running_text)
        if self._validation_message:
            timing_parts.append(f"Validation: {self._validation_message}")
        details = " | ".join(timing_parts)
        if self._stage_message:
            details += f" | {self._stage_message}"
        if self._message:
            details += f" | {self._message}"
        help_text = self._help_message or "Hover over a control for parameter help; H shows keys."
        current_relative = self._relative_image_text(self.image_paths[self.index]) if self.image_paths else "no data"
        source_title = f"{self.index + 1}/{len(self.image_paths)} | {current_relative}" if self.image_paths else "no data"
        self.fig.suptitle(
            f"{self.adapter.mode_name} | {source_title} | {stage}{roi_suffix}{running}{dirty}",
            fontsize=11,
        )
        if self.graph_fig is not None:
            self.graph_fig.suptitle(f"Source: {current_relative}", fontsize=9, x=0.07, ha="left")
        if self.selected_file is not None:
            data_text = f"Data: {current_relative}"
        elif self.image_paths:
            folder_count = len({path.parent for path in self.image_paths})
            data_text = f"Data: {len(self.image_paths)} images in {folder_count} TIFF-containing folders under {self.folder}"
        else:
            data_text = "Data: none — use Load data"
        self.status_text.set_text(f"{summary}\n{data_text} | {details}\n{help_text}")
        self.fig.canvas.draw_idle()

    def _relative_image_text(self, path: Path) -> str:
        """Return a concise source identity that remains unique under a root."""

        if self.selected_file is not None:
            return path.name
        if self.folder is not None:
            try:
                return path.resolve().relative_to(self.folder.resolve()).as_posix()
            except ValueError:
                pass
        return path.name

    def _on_close(self, _event: CloseEvent | None = None) -> None:
        self.close()

    def _on_graph_close(self, _event: CloseEvent | None = None) -> None:
        """Close the shared controller when either synchronized window closes."""

        self.close()

    def close(self) -> None:
        """Stop timers and release the analysis executor."""

        if self._closed:
            return
        self._closed = True
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._clear_parameter_controls()
        self._disconnect_widget(self.group_radio, self._group_radio_callback_id)
        self._disconnect_widget(self.overlay_checks, self._overlay_checks_callback_id)
        for button, callback_id in zip(self._buttons, self._button_callback_ids):
            self._disconnect_widget(button, callback_id)
        for callback_id in self._main_canvas_callback_ids:
            self.fig.canvas.mpl_disconnect(callback_id)
        if self.graph_fig is not None:
            for callback_id in self._graph_canvas_callback_ids:
                self.graph_fig.canvas.mpl_disconnect(callback_id)
        self._release_canvas_mouse()
        self.controller.close()
        self.current_result = None
        self.current_preview = None
        self.current_error = None
        for cache_name in (
            "_cap_graph_cache",
            "_homogeneity_cache",
            "_local_heterogeneity_cache",
            "_local_heterogeneity_robust_cache",
            "_inscribed_circle_cache",
        ):
            cache = getattr(self.adapter, cache_name, None)
            if isinstance(cache, dict):
                cache.clear()
        if hasattr(self.adapter, "_local_heterogeneity_cache_result"):
            self.adapter._local_heterogeneity_cache_result = None
        # Closed figures can remain reachable briefly through test harnesses or
        # GUI backend objects.  Drop their large image arrays immediately so a
        # sequence of viewers cannot accumulate one full SEM-sized RGBA buffer
        # per closed window while cyclic collection catches up.
        for figure in (self.graph_fig, self.fig):
            if figure is None:
                continue
            for axis in tuple(figure.axes):
                for artist in tuple(axis.images):
                    artist.set_data(np.zeros((1, 1), dtype=np.float32))
            figure.clear()
        if self.graph_fig is not None and plt.fignum_exists(self.graph_fig.number):
            try:
                plt.close(self.graph_fig)
            except Exception:
                LOGGER.debug("Could not close graph figure", exc_info=True)
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            try:
                plt.close(self.fig)
            except Exception:
                LOGGER.debug("Could not close main figure", exc_info=True)

    def show(self) -> None:
        """Display the Matplotlib window."""

        plt.show()

    def _load_initial_preview(self) -> np.ndarray | None:
        if not self.image_paths:
            return None
        image_path = self.image_paths[self.index]
        return self.adapter.initial_preview(image_path, self.current_viewer_config)

    def _show_no_data_state(self) -> None:
        """Keep both windows useful when analysis parameters have no input data."""
        self.current_result = self.current_preview = None
        self.current_error = None
        image = self._make_placeholder_image()
        self._set_main_image(image, reset_view=True)
        self.loading_text.set_text("No data loaded.\nUse Load data to select an image or folder.")
        self.loading_text.set_visible(True)
        if self.graph_main_axis is not None:
            self.graph_main_axis.clear(); self.graph_main_axis.set_visible(True)
            self.graph_main_axis.text(.5, .5, "No data loaded.\nUse Load data in the main window.", transform=self.graph_main_axis.transAxes, ha="center", va="center")
        if self.graph_fig is not None:
            self.graph_fig.canvas.draw_idle()
        self._update_status()

    def _make_placeholder_image(self, shape: tuple[int, int] = (256, 256)) -> np.ndarray:
        placeholder = np.full((*shape, 3), 0.12, dtype=np.float32)
        placeholder[::16, :, :] = 0.15
        placeholder[:, ::16, :] = 0.15
        return placeholder

    def _set_main_image(self, image: np.ndarray, *, reset_view: bool) -> None:
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ValueError(f"Diagnostic image must be 2D or RGB(A), got shape {array.shape}.")
        if array.ndim == 3 and array.shape[2] not in (3, 4):
            raise ValueError(f"Diagnostic RGB image must have 3 or 4 channels, got {array.shape}.")
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError("Diagnostic image must be non-empty.")

        h, w = array.shape[:2]
        extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        self.image_artist.set_data(array)
        self.image_artist.set_extent(extent)
        self.boundary_artist.set_extent(extent)
        self.local_coverage_artist.set_extent(extent)
        self.ax_image.set_aspect("equal")
        self.ax_image.set_autoscale_on(False)
        if reset_view:
            self.ax_image.set_xlim(-0.5, w - 0.5)
            self.ax_image.set_ylim(h - 0.5, -0.5)

    def _show_loading_preview(self) -> None:
        preview = self._load_initial_preview()
        if preview is None:
            preview = self._make_placeholder_image()
        self._set_main_image(preview, reset_view=True)
        self.boundary_artist.set_data(np.zeros((*preview.shape[:2], 4), dtype=np.float32))
        self.boundary_artist.set_visible(False)
        self._active_measurement_count = 0
        self._set_measurement_visibility()
        self._update_point_artists(())
        self._update_annotation_artists(())
        self._set_scale_visibility(False)
        self.loading_text.set_visible(True)
        self.fig.canvas.draw_idle()

    def _sync_overlay_checkboxes(self) -> None:
        statuses = self.overlay_checks.get_status()
        for index, label in enumerate(self.overlay_labels):
            expected = bool(self.overlay_state[label])
            if bool(statuses[index]) != expected:
                self.overlay_checks.set_active(index)

    def _clear_parameter_controls(self) -> None:
        self._release_canvas_mouse()
        for control in self._controls.values():
            self._disconnect_widget(control.widget, control.callback_id)
            control.axis.clear()
            control.axis.set_visible(False)
        self._controls.clear()

    def _disconnect_widget(self, widget: Any, callback_id: int | None) -> None:
        if widget is None:
            return
        if callback_id is not None and hasattr(widget, "disconnect"):
            try:
                widget.disconnect(callback_id)
            except Exception:
                LOGGER.debug("Could not disconnect widget callback", exc_info=True)
        if hasattr(widget, "disconnect_events"):
            try:
                widget.disconnect_events()
            except Exception:
                LOGGER.debug("Could not disconnect widget events", exc_info=True)

    def _release_canvas_mouse(self) -> None:
        grabber = getattr(self.fig.canvas, "mouse_grabber", None)
        if grabber is not None:
            try:
                self.fig.canvas.release_mouse(grabber)
            except Exception:
                LOGGER.debug("Could not release Matplotlib mouse grab", exc_info=True)

    def _control_help_text(self, spec: ParameterSpec) -> str:
        current = self._format_control_value(self._all_values[spec.name])
        loaded = self._format_control_value(self._source_values[spec.name])
        parts = [spec.help_text]
        control = self._controls.get(spec.name)
        if control is not None and not control.enabled and control.inactive_reason:
            parts.append(control.inactive_reason)
        config = self._current_valid_config() or self.current_viewer_config
        if spec.note_when is not None:
            note = spec.note_when(config)
            if note:
                parts.append(note)
        parts.append(f"Current: {current} | Loaded: {loaded}")
        return " ".join(parts)

    @staticmethod
    def _format_control_value(value: Any) -> str:
        if isinstance(value, tuple):
            return ", ".join(DiagnosticViewer._format_control_value(item) for item in value)
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)

    def _summarize_stage_pixels(self, stage: str, pixels: np.ndarray) -> str:
        if stage.endswith("_mask") and not np.any(pixels):
            return f"{stage.replace('_', ' ').capitalize()}: empty"
        return ""

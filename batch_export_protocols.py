from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tracemalloc
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from skimage.segmentation import find_boundaries

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from export_output_summaries import export_outputs
from path_utils import (
    coverage_sample_id,
    expand_user_path,
    resolve_optional_file_in_folder,
)
from coverage_cap import sphere_geometry_from_mask
from sem_bead_viewer import (
    _format_length_m as bead_format_length_m,
    _format_length_value as bead_format_length_value,
    _nice_scale_length_m as bead_nice_scale_length_m,
    analyze_bead_image,
    build_bead_summary,
    load_app_config as load_bead_app_config,
)
from sem_coverage_viewer import (
    _format_length_m as coverage_format_length_m,
    _format_px_or_length,
    _include_roi_in_global_summary,
    _nice_scale_length_m as coverage_nice_scale_length_m,
    analyze_coverage_image,
    build_coverage_image_record,
    build_coverage_summary_from_records,
    load_app_config as load_coverage_app_config,
    load_failed_image_preview,
)
from tabular_export import sort_paths


LOGGER = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_", ".") else "_" for ch in name)
    return "_".join(safe.split())


def _find_sample_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: '{root}'.")
    sample_dirs = sort_paths(list({path.parent for path in root.rglob("*.tif")}), root=root)
    if not sample_dirs:
        raise FileNotFoundError(f"No TIFF files found under '{root}'.")
    return sample_dirs


@dataclass(frozen=True)
class ResolvedCoverageSource:
    """Coverage input selected explicitly or inherited from the config."""

    root: Path
    selected_file: Path | None
    source_origin: str


def resolve_coverage_source(
    *,
    cli_root: Path | None,
    app_config: Any,
    config_path: Path,
) -> ResolvedCoverageSource | None:
    """Resolve one batch source without mutating the loaded config object."""

    if cli_root is not None:
        root = expand_user_path(cli_root).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Coverage source supplied by --coverage-root does not exist: {root}")
        return ResolvedCoverageSource(root, None, "cli")
    folder_text = str(getattr(app_config, "folder", "") or "").strip()
    if not folder_text:
        return None
    configured_root = expand_user_path(folder_text)
    root = (
        configured_root
        if configured_root.is_absolute()
        else config_path.expanduser().resolve().parent / configured_root
    ).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Coverage source stored in config does not exist: {root}")
    file_text = getattr(app_config, "file", None)
    if file_text:
        selected = resolve_optional_file_in_folder(root, file_text, description="coverage source stored in config")
        if selected is None or not selected.is_file() or selected.suffix.lower() != ".tif":
            resolved = selected if selected is not None else root / str(file_text)
            raise FileNotFoundError(f"Coverage source stored in config does not exist: {resolved}")
        return ResolvedCoverageSource(root, selected.resolve(), "config")
    return ResolvedCoverageSource(root, None, "config")


def _count_images(sample_dirs: Iterable[Path]) -> int:
    return sum(len(list(sample_dir.glob("*.tif"))) for sample_dir in sample_dirs)


def _relative_label(root: Path, sample_dir: Path) -> str:
    rel = sample_dir.relative_to(root)
    parts = [part for part in rel.parts if part not in ("", ".")]
    return "root" if not parts else " - ".join(parts)


def _json_path(outputs_dir: Path, kind: str, root: Path, sample_dir: Path) -> Path:
    label = _relative_label(root, sample_dir)
    return outputs_dir / f"{_safe_name(label)}__{kind}.json"


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_overlay_figure(image: np.ndarray, output_path: Path, overlay_builder) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    fig_w = 10.0
    fig_h = max(2.0, fig_w * h / max(w, 1))
    fig = None
    try:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.imshow(image, vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        artists = overlay_builder(ax)
        for artist in artists:
            if getattr(artist, "axes", None) is None:
                ax.add_artist(artist)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    finally:
        if fig is not None:
            plt.close(fig)


def _make_scale_overlay(ax, shape: tuple[int, int], pixel_size_m: float | None, fmt_length, nice_length) -> list[object]:
    if pixel_size_m is None:
        return []
    h, w = shape
    scale_length_m = nice_length(w * pixel_size_m * 0.22)
    if scale_length_m <= 0:
        return []
    scale_length_px = scale_length_m / pixel_size_m
    x0 = w * 0.06
    y0 = h * 0.92
    label_artist = ax.text(x0, y0 - 11, fmt_length(scale_length_m), color="white", fontsize=10, va="bottom", ha="left")
    return [
        Rectangle((x0 - 8, y0 - 28), scale_length_px + 16, 36, facecolor=(0.0, 0.0, 0.0, 0.35), edgecolor="none"),
        Line2D([x0, x0 + scale_length_px], [y0, y0], color="white", linewidth=3),
        Line2D([x0, x0], [y0 - 7, y0 + 7], color="white", linewidth=1.5),
        Line2D([x0 + scale_length_px, x0 + scale_length_px], [y0 - 7, y0 + 7], color="white", linewidth=1.5),
        label_artist,
    ]


def _export_bead_png(image_path: Path, config, output_path: Path) -> Path:
    res = analyze_bead_image(image_path, config)
    base = np.dstack([res.display, res.display, res.display]).astype(np.float32)
    valid_edges = find_boundaries(res.valid_mask, mode="outer")
    outlier_edges = find_boundaries(res.outlier_mask, mode="outer")
    base[valid_edges] = (0.0, 1.0, 0.0)
    base[outlier_edges] = (1.0, 0.1, 0.1)

    def overlay_builder(ax):
        artists: list[object] = []
        if config.default_show_scale:
            artists.extend(
                _make_scale_overlay(
                    ax,
                    res.display.shape,
                    res.metadata.mean_pixel_size_m,
                    bead_format_length_m,
                    bead_nice_scale_length_m,
                )
            )
        if config.default_show_measures:
            for meas in res.measurements:
                row, col = meas.centroid_rc
                color = "cyan" if meas.valid else "red"
                x_half = meas.x_diameter_px / 2.0
                y_half = meas.y_diameter_px / 2.0
                if meas.mean_diameter_m is not None:
                    label_text = f"x={bead_format_length_value(meas.x_diameter_m)}  y={bead_format_length_value(meas.y_diameter_m)} um"
                else:
                    label_text = f"x={meas.x_diameter_px:.1f}  y={meas.y_diameter_px:.1f} px"
                if not meas.valid and meas.reasons:
                    label_text += "  !"
                artists.extend(
                    [
                        Line2D([col - x_half, col + x_half], [row, row], color=color, linewidth=1.2, alpha=0.95),
                        Line2D([col, col], [row - y_half, row + y_half], color=color, linewidth=1.2, alpha=0.95),
                        ax.text(
                            col,
                            row - y_half - 7,
                            label_text,
                            color=color,
                            fontsize=7,
                            ha="center",
                            va="bottom",
                            bbox={"facecolor": (0.0, 0.0, 0.0, 0.45), "edgecolor": "none", "pad": 1.5},
                        ),
                    ]
                )
        return artists

    _save_overlay_figure(base, output_path, overlay_builder)
    return output_path


def _export_coverage_png(image_path: Path, config, output_path: Path, *, result=None, branch_label: str | None = None) -> Path:
    try:
        res = result if result is not None else analyze_coverage_image(image_path, config)
        base_gray = res.display.astype(np.float32)
        base = np.dstack([base_gray, base_gray, base_gray]).astype(np.float32)
        if config.default_show_ag_boundary:
            ag_union = np.zeros(res.display.shape, dtype=bool)
            for roi in res.roi_results:
                ag_union |= roi.ag_mask
            base[find_boundaries(ag_union, mode="outer")] = (1.0, 0.0, 0.0)
        if config.default_show_ag_count_boundary:
            ag_count_union = np.zeros(res.display.shape, dtype=bool)
            for roi in res.roi_results:
                ag_count_union |= roi.ag_count_mask
            base[find_boundaries(ag_count_union, mode="outer")] = (1.0, 1.0, 0.0)

        def overlay_builder(ax):
            artists: list[object] = []
            if branch_label:
                artists.append(ax.text(0.01, 0.02, branch_label, transform=ax.transAxes, color="white", fontsize=7, ha="left", va="bottom", bbox={"facecolor": (0, 0, 0, .4), "edgecolor": "none", "pad": 1.5}))
            if config.default_show_scale:
                artists.extend(
                    _make_scale_overlay(
                        ax,
                        res.display.shape,
                        res.metadata.mean_pixel_size_m,
                        coverage_format_length_m,
                        coverage_nice_scale_length_m,
                    )
                )
            if config.default_show_diameter_lines:
                for roi in res.roi_results:
                    m = roi.bead_metrics
                    include = _include_roi_in_global_summary(roi, res.config)
                    color = "cyan" if include else "red"
                    geometry = sphere_geometry_from_mask(roi.bead_mask, centroid_rc=m.centroid_rc, equivalent_diameter_px=m.equivalent_diameter_px, x_diameter_px=m.x_diameter_px, y_diameter_px=m.y_diameter_px, metric=config.sphere_diameter_metric)
                    row, col = geometry.center_rc
                    radius = geometry.radius_px
                    if geometry.mode == "mean_xy_diameter":
                        artists.extend([Line2D([col-m.x_diameter_px/2+0.5, col+m.x_diameter_px/2-0.5], [m.centroid_rc[0]]*2, color=color, linewidth=1.2, zorder=10), Line2D([m.centroid_rc[1]]*2, [m.centroid_rc[0]-m.y_diameter_px/2+0.5, m.centroid_rc[0]+m.y_diameter_px/2-0.5], color=color, linewidth=1.2, zorder=10), ax.text(col, m.centroid_rc[0]-m.y_diameter_px/2-8, f"x={_format_px_or_length(m.x_diameter_m,m.x_diameter_px)} y={_format_px_or_length(m.y_diameter_m,m.y_diameter_px)}", color=color, fontsize=8, ha="center", zorder=40)])
                    else:
                        label_prefix = "d_eq" if geometry.mode == "equivalent_diameter" else "d_ins"
                        diameter_m = 2 * radius * res.metadata.mean_pixel_size_m if res.metadata.mean_pixel_size_m is not None else None
                        label = f"{label_prefix} = {_format_px_or_length(diameter_m, 2*radius)}"
                        artists.extend([Circle((col,row), radius, fill=False, edgecolor=color, linewidth=1.3, zorder=10), Line2D([col-radius+.5,col+radius-.5],[row,row],color=color,linewidth=1.2,zorder=10), ax.plot([col],[row],marker="+",color=color,markersize=7,zorder=10)[0], ax.text(col,row-radius-8,label,color=color,fontsize=8,ha="center",zorder=40)])
            if config.default_show_bead_boundary:
                for roi in res.roi_results:
                    include = _include_roi_in_global_summary(roi, res.config)
                    edge = find_boundaries(roi.bead_mask, mode="outer")
                    artists.append(ax.contour(edge.astype(float), levels=[0.5], colors=["lime" if include else "red"], linewidths=1.2, zorder=30))
            if config.default_show_ag_peaks:
                for roi in res.roi_results:
                    if roi.ag_peak_coords.size:
                        artists.append(ax.plot(roi.ag_peak_coords[:, 1], roi.ag_peak_coords[:, 0], "c.", markersize=4)[0])
            return artists

    except Exception:
        preview = load_failed_image_preview(image_path, config)
        base_gray = preview.display.astype(np.float32)
        base = np.dstack([base_gray, base_gray, base_gray]).astype(np.float32)

        def overlay_builder(ax):
            return _make_scale_overlay(
                ax,
                preview.display.shape,
                preview.metadata.mean_pixel_size_m,
                coverage_format_length_m,
                coverage_nice_scale_length_m,
            )

    _save_overlay_figure(base, output_path, overlay_builder)
    return output_path


def _run_bead_batch(
    root: Path, config_path: Path, outputs_dir: Path, *, sort_by: str = "name"
) -> list[Path]:
    cfg = load_bead_app_config(config_path)
    written: list[Path] = []
    png_dir = outputs_dir / "size_png"
    sample_dirs = _find_sample_dirs(root)
    progress = _progress(desc="Bead images", total=_count_images(sample_dirs))
    for sample_dir in sample_dirs:
        out_path = _json_path(outputs_dir, "bead", root, sample_dir)
        summary = build_bead_summary(sample_dir, cfg.viewer)
        _write_json(summary, out_path)
        written.append(out_path)
        label = _safe_name(_relative_label(root, sample_dir))
        for image_path in sort_paths(list(sample_dir.glob("*.tif")), sort_by=sort_by, root=root):
            written.append(_export_bead_png(image_path, cfg.viewer, png_dir / f"{label}__{_safe_name(image_path.stem)}.png"))
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return written


def _coverage_branch_configs(config, selection: str) -> list[tuple[str, str, object]]:
    """Return immutable branch viewer configs without changing the loaded config."""
    configured = bool(config.ag_enable_secondary_coverage)
    choices = {
        "configured": [("two_layers" if configured else "one_layer", configured)],
        "one-layer": [("one_layer", False)],
        "two-layers": [("two_layers", True)],
        "both": [("one_layer", False), ("two_layers", True)],
    }[selection]
    return [
        (
            branch_id,
            "secondary coverage branch enabled" if enabled else "primary-only coverage branch",
            replace(config, ag_enable_secondary_coverage=enabled),
        )
        for branch_id, enabled in choices
    ]


@dataclass
class CoverageSampleSummaryBuilder:
    """Retain JSON-compatible records, never full coverage image results."""

    sample_dir: Path
    config: Any
    image_paths: tuple[Path, ...]
    selected_file: str | None = None
    image_records: list[dict[str, object]] = field(default_factory=list)
    failures: list[dict[str, object]] = field(default_factory=list)

    def add_success(
        self,
        image_path: Path,
        serializable_record: Mapping[str, object],
    ) -> dict[str, object]:
        record = dict(serializable_record)
        record.setdefault("file", image_path.name)
        self.image_records.append(record)
        return record

    def discard_success(self, record: dict[str, object]) -> None:
        """Roll back the current record when its PNG export fails."""

        if self.image_records and self.image_records[-1] is record:
            self.image_records.pop()
            return
        for index in range(len(self.image_records) - 1, -1, -1):
            if self.image_records[index] is record:
                del self.image_records[index]
                return

    def add_failure(
        self,
        image_path: Path,
        error: Exception,
        *,
        branch_id: str,
        branch_label: str,
    ) -> None:
        self.failures.append(
            {
                "file": image_path.name,
                "sample": image_path.parent.name,
                "error_type": type(error).__name__,
                "error": str(error),
                "coverage_branch_id": branch_id,
                "coverage_branch_label": branch_label,
            }
        )

    def finalize(self) -> dict:
        return build_coverage_summary_from_records(
            self.sample_dir,
            self.config,
            file=self.selected_file,
            image_paths=self.image_paths,
            image_records=self.image_records,
            failures=self.failures,
        )


def _enable_performance_logger() -> None:
    """Enable this module's opt-in DEBUG stream without global log noise."""

    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False


def _report_image_performance(
    *,
    branch_id: str,
    sample: str,
    image_index: int,
    image_count: int,
    analysis_seconds: float,
    serialization_seconds: float,
    png_seconds: float,
    total_seconds: float,
) -> None:
    current_bytes: int | None = None
    peak_bytes: int | None = None
    if tracemalloc.is_tracing():
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    LOGGER.debug(
        "coverage-performance branch=%s sample=%s image=%d/%d "
        "analysis_s=%.6f serialization_s=%.6f png_s=%.6f total_s=%.6f "
        "figures=%d traced_current_mib=%s traced_peak_mib=%s",
        branch_id,
        sample,
        image_index,
        image_count,
        analysis_seconds,
        serialization_seconds,
        png_seconds,
        total_seconds,
        len(plt.get_fignums()),
        "n/a" if current_bytes is None else f"{current_bytes / 2**20:.3f}",
        "n/a" if peak_bytes is None else f"{peak_bytes / 2**20:.3f}",
    )


def _run_coverage_batch(
    source: ResolvedCoverageSource,
    config,
    outputs_dir: Path,
    *,
    sort_by: str = "name",
    branches: str = "both",
    debug_performance: bool = False,
) -> list[Path]:
    root = source.root
    written: list[Path] = []
    sample_dirs = [source.selected_file.parent] if source.selected_file is not None else _find_sample_dirs(root)
    image_count = 1 if source.selected_file is not None else _count_images(sample_dirs)
    branch_configs = _coverage_branch_configs(config.viewer, branches)
    progress = _progress(desc="Coverage branch images", total=image_count * len(branch_configs))
    tracing_started_here = False
    if debug_performance:
        _enable_performance_logger()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            tracing_started_here = True
    try:
        for branch_id, branch_label, viewer_config in branch_configs:
            branch_dir = outputs_dir / f"coverage_{branch_id}"
            png_dir = branch_dir / "coverage_png"
            for sample_dir in sample_dirs:
                image_paths = tuple(
                    [source.selected_file]
                    if source.selected_file is not None
                    else sort_paths(list(sample_dir.glob("*.tif")), sort_by=sort_by, root=root)
                )
                builder = CoverageSampleSummaryBuilder(
                    sample_dir=sample_dir,
                    config=viewer_config,
                    image_paths=image_paths,
                    selected_file=source.selected_file.name if source.selected_file is not None else None,
                )
                label = _safe_name(_relative_label(root, sample_dir))
                sample_id = coverage_sample_id(coverage_root=root, sample_dir=sample_dir)
                for image_index, image_path in enumerate(image_paths, start=1):
                    total_start = perf_counter()
                    analysis_seconds = serialization_seconds = png_seconds = 0.0
                    result = None
                    stored_record = None
                    try:
                        phase_start = perf_counter()
                        result = analyze_coverage_image(image_path, viewer_config)
                        analysis_seconds = perf_counter() - phase_start

                        phase_start = perf_counter()
                        image_record = build_coverage_image_record(image_path, viewer_config, result)
                        stored_record = builder.add_success(image_path, image_record)
                        serialization_seconds = perf_counter() - phase_start

                        phase_start = perf_counter()
                        try:
                            png_path = _export_coverage_png(
                                image_path,
                                viewer_config,
                                png_dir / f"{label}__{_safe_name(image_path.stem)}.png",
                                result=result,
                                branch_label=branch_label,
                            )
                        finally:
                            png_seconds = perf_counter() - phase_start
                        written.append(png_path)
                    except Exception as exc:
                        if stored_record is not None:
                            builder.discard_success(stored_record)
                        builder.add_failure(
                            image_path,
                            exc,
                            branch_id=branch_id,
                            branch_label=branch_label,
                        )
                    finally:
                        # CPython releases the array-heavy object immediately;
                        # other implementations may collect it normally.  No
                        # explicit per-image gc.collect() is needed.
                        result = None
                        stored_record = None
                        if progress is not None:
                            progress.update(1)
                        if debug_performance:
                            _report_image_performance(
                                branch_id=branch_id,
                                sample=sample_id,
                                image_index=image_index,
                                image_count=len(image_paths),
                                analysis_seconds=analysis_seconds,
                                serialization_seconds=serialization_seconds,
                                png_seconds=png_seconds,
                                total_seconds=perf_counter() - total_start,
                            )

                summary = builder.finalize()
                source_metadata = {"coverage_source_origin": source.source_origin, "coverage_source_root": str(root), "coverage_source_file": None if source.selected_file is None else str(source.selected_file)}
                summary.update({"coverage_branch_id": branch_id, "coverage_branch_label": branch_label, "ag_enable_secondary_coverage": bool(viewer_config.ag_enable_secondary_coverage), **source_metadata})
                summary["sample"] = sample_id
                summary.setdefault("global_summary", {}).update({"coverage_branch_id": branch_id, "coverage_branch_label": branch_label, "ag_enable_secondary_coverage": bool(viewer_config.ag_enable_secondary_coverage), **source_metadata})
                for failure in summary.get("failed_images", []):
                    failure_path = (sample_dir / str(failure.get("file") or "")).resolve()
                    failure.update({
                        "sample": sample_id,
                        "source_path": str(failure_path),
                        "coverage_branch_id": branch_id,
                        "coverage_branch_label": branch_label,
                        "ag_enable_secondary_coverage": bool(viewer_config.ag_enable_secondary_coverage),
                        **source_metadata,
                    })
                for image in summary.get("images", []):
                    image_path = (sample_dir / str(image.get("file") or "")).resolve()
                    image.update({
                        "sample": sample_id,
                        "source_path": str(image_path),
                        "coverage_branch_id": branch_id,
                        "coverage_branch_label": branch_label,
                        "ag_enable_secondary_coverage": bool(viewer_config.ag_enable_secondary_coverage),
                        **source_metadata,
                    })
                    for roi in image.get("rois", []):
                        roi.update({
                            "sample": sample_id,
                            "source_path": str(image_path),
                            "coverage_branch_id": branch_id,
                            "coverage_branch_label": branch_label,
                            "ag_enable_secondary_coverage": bool(viewer_config.ag_enable_secondary_coverage),
                            **source_metadata,
                        })
                out_path = _json_path(branch_dir, "coverage", root, sample_dir)
                _write_json(summary, out_path)
                written.append(out_path)
                # Do not carry the just-written sample's rich serializable
                # records into the next sample.  The JSON on disk is now the
                # source consumed by table export; only lightweight output
                # paths remain branch-wide.
                builder.image_records.clear()
                builder.failures.clear()
                image_record = None
                image = None
                roi = None
                del summary
                del builder
                if debug_performance:
                    # Matplotlib figures are cyclic object graphs.  A
                    # diagnostic-only collection distinguishes unreachable
                    # figure cycles from objects genuinely retained by the
                    # batch pipeline; normal processing still relies on the
                    # interpreter's collector rather than forcing a
                    # collection after every image.
                    collected = gc.collect()
                    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                    LOGGER.debug(
                        "coverage-sample-release branch=%s sample=%s "
                        "collected=%d figures=%d traced_current_mib=%.3f "
                        "traced_peak_mib=%.3f",
                        branch_id,
                        sample_id,
                        collected,
                        len(plt.get_fignums()),
                        current_bytes / 2**20,
                        peak_bytes / 2**20,
                    )
    finally:
        if progress is not None:
            progress.close()
        if tracing_started_here:
            tracemalloc.stop()
    return written


def _remove_outputs(outputs_dir: Path) -> None:
    if not outputs_dir.exists():
        return
    for path in outputs_dir.glob("*.json"):
        path.unlink()
    for path in outputs_dir.glob("*.csv"):
        path.unlink()
    for path in outputs_dir.glob("*.xlsx"):
        path.unlink()
    hist_dir = outputs_dir / "bead_histograms"
    if hist_dir.exists():
        for path in hist_dir.glob("*.png"):
            path.unlink()
        for path in hist_dir.glob("*.txt"):
            path.unlink()
    for png_dir_name in ("size_png", "coverage_png"):
        png_dir = outputs_dir / png_dir_name
        if png_dir.exists():
            for path in png_dir.glob("*.png"):
                path.unlink()
    for branch_name in ("coverage_one_layer", "coverage_two_layers"):
        branch = outputs_dir / branch_name
        if branch.exists():
            for path in branch.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".json", ".csv", ".xlsx", ".png"}:
                    path.unlink()


def _print_paths(paths: Iterable[Path]) -> None:
    for path in sort_paths(list(paths), sort_by="path"):
        print(path)


def _progress(*, desc: str, total: int):
    if tqdm is None:
        return None
    return tqdm(desc=desc, total=total, unit="image")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-run SEM bead and coverage summaries for all TIFF-containing subfolders, then export CSV summaries."
    )
    parser.add_argument("--bead-root", type=Path, help="Root folder for bead protocol samples.")
    parser.add_argument(
        "--bead-config",
        type=Path,
        default=Path("sem_bead_viewer_config.json"),
        help="Config file used as a parameter template for bead analysis.",
    )
    parser.add_argument("--coverage-root", type=Path, help="Optional recursive coverage root. Overrides the folder/file source stored in --coverage-config.")
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=Path("sem_coverage_viewer_config.json"),
        help="Coverage parameter template; its top-level folder/file fields may also supply the input source.",
    )
    parser.add_argument(
        "--coverage-branches",
        choices=("both", "configured", "one-layer", "two-layers"),
        default="both",
        help="Coverage-mask branches to process. Default: %(default)s",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs2"),
        help="Directory where per-sample JSONs and exported CSVs will be written.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing JSON/CSV/PNG outputs in outputs-dir before running.")
    parser.add_argument("--no-export", action="store_true", help="Only write per-sample JSON summaries, skip CSV/histogram export.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write SEM CSV summary files.")
    parser.add_argument("--no-bead-csv", action="store_true", help="Do not write bead_global_summaries.csv.")
    parser.add_argument("--no-coverage-csv", action="store_true", help="Do not write coverage_global_summaries.csv.")
    parser.add_argument("--no-histograms", action="store_true", help="Do not write bead histogram PNG files.")
    parser.add_argument(
        "--table-format",
        choices=("csv", "xlsx", "both", "none"),
        default="csv",
        help="Summary table export format. Default: %(default)s",
    )
    parser.add_argument(
        "--sort-by",
        choices=("name", "path", "none"),
        default="name",
        help="Deterministic natural sorting for samples, TIFFs, and summary tables. Default: %(default)s",
    )
    parser.add_argument(
        "--debug-performance",
        action="store_true",
        help=(
            "Log per-image coverage analysis, serialization, PNG, figure-count, "
            "and traced-memory diagnostics."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command_args = list(sys.argv[1:] if argv is None else argv)
    coverage_config_requested = any(
        item == "--coverage-config" or item.startswith("--coverage-config=")
        for item in command_args
    )
    coverage_config_path = expand_user_path(args.coverage_config)
    coverage_app_config = None
    coverage_source = None
    if args.coverage_root is not None or coverage_config_requested:
        coverage_app_config = load_coverage_app_config(coverage_config_path)
        coverage_source = resolve_coverage_source(
            cli_root=args.coverage_root,
            app_config=coverage_app_config,
            config_path=coverage_config_path,
        )
    if not args.bead_root and coverage_source is None:
        raise SystemExit("No input source was provided. Supply --bead-root, --coverage-root, or a coverage config containing a valid top-level folder/file source.")

    outputs_dir = expand_user_path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _remove_outputs(outputs_dir)

    written: list[Path] = []
    if args.bead_root:
        written.extend(
            _run_bead_batch(
                expand_user_path(args.bead_root),
                expand_user_path(args.bead_config),
                outputs_dir,
                sort_by=args.sort_by,
            )
        )
    if coverage_source is not None:
        if coverage_source.source_origin == "config":
            print("Coverage source from config: single file" if coverage_source.selected_file else "Coverage source from config: recursive folder")
        else:
            print(f"Coverage source from CLI: recursive folder ({coverage_source.root})")
        written.extend(
            _run_coverage_batch(
                coverage_source,
                coverage_app_config,
                outputs_dir,
                sort_by=args.sort_by,
                branches=args.coverage_branches,
                debug_performance=args.debug_performance,
            )
        )
    if not args.no_export:
        if args.bead_root:
            written.extend(
                export_outputs(
                    outputs_dir,
                    csv=not args.no_csv,
                    bead=True,
                    coverage=False,
                    bead_csv=not args.no_bead_csv,
                    coverage_csv=False,
                    histograms=not args.no_histograms,
                    table_format=args.table_format,
                    sort_by=args.sort_by,
                )
            )
        if coverage_source is not None:
            branch_names = (
                ("coverage_one_layer", "coverage_two_layers")
                if args.coverage_branches == "both"
                else (("coverage_one_layer",) if args.coverage_branches == "one-layer" else ("coverage_two_layers",) if args.coverage_branches == "two-layers" else ("coverage_two_layers", "coverage_one_layer"))
            )
            for name in branch_names:
                branch_dir = outputs_dir / name
                if branch_dir.exists() and any(branch_dir.glob("*.json")):
                    written.extend(export_outputs(branch_dir, csv=not args.no_csv, bead=False, coverage=True, bead_csv=False, coverage_csv=not args.no_coverage_csv, histograms=False, table_format=args.table_format, sort_by=args.sort_by))

    _print_paths(written)


if __name__ == "__main__":
    main()

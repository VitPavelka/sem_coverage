from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tracemalloc
from dataclasses import asdict, dataclass, field, replace
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
    resolve_existing_input_path,
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


@dataclass(frozen=True)
class ResolvedBeadSource:
    """Bead input selected explicitly or inherited from the config."""

    root: Path
    source_origin: str


@dataclass(frozen=True)
class CoverageBatchAssignment:
    """One validated tuned config and its canonical TIFF population."""

    config_name: str
    config_path: Path
    app_config: Any
    source: ResolvedCoverageSource
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CoverageBatchGroup:
    """One scientific sample assembled from one or more tuned configs."""

    sample_id: str
    assignments: tuple[CoverageBatchAssignment, ...]


@dataclass(frozen=True)
class CoverageBatchManifest:
    """A completely preflighted grouped coverage manifest."""

    path: Path
    groups: tuple[CoverageBatchGroup, ...]


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
    root = resolve_existing_input_path(
        folder_text,
        config_path=config_path,
        description="coverage source stored in config",
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


def resolve_bead_source(
    *,
    cli_root: Path | None,
    app_config: Any,
    config_path: Path,
) -> ResolvedBeadSource | None:
    """Resolve bead input with the same CLI-override/config fallback contract."""

    if cli_root is not None:
        root = expand_user_path(cli_root).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(
                f"Bead source supplied by --bead-root does not exist: {root}"
            )
        return ResolvedBeadSource(root, "cli")
    folder_text = str(getattr(app_config, "folder", "") or "").strip()
    if not folder_text:
        return None
    root = resolve_existing_input_path(
        folder_text,
        config_path=config_path,
        description="bead source stored in config",
    ).resolve()
    if not root.is_dir():
        raise NotADirectoryError(
            f"Bead source stored in config is not a directory: {root}"
        )
    return ResolvedBeadSource(root, "config")


def _coverage_source_image_paths(
    source: ResolvedCoverageSource,
    *,
    sort_by: str = "name",
) -> tuple[Path, ...]:
    """Return the exact canonical TIFF population for one resolved source."""

    if source.selected_file is not None:
        return (source.selected_file.resolve(),)
    paths: list[Path] = []
    for sample_dir in _find_sample_dirs(source.root):
        paths.extend(
            sort_paths(
                list(sample_dir.glob("*.tif")),
                sort_by=sort_by,
                root=source.root,
            )
        )
    return tuple(path.resolve() for path in paths)


def _resolve_manifest_config_path(configs_root: Path, config_name: str) -> Path:
    """Resolve one explicit manifest config name without fuzzy guessing."""

    name = config_name.strip()
    if not name:
        raise ValueError("Coverage config names must be non-empty strings.")
    relative = Path(name)
    if relative.is_absolute():
        raise ValueError(
            f"Coverage config name must be relative to configs_root: {config_name!r}."
        )
    if relative.suffix:
        if relative.suffix.lower() != ".json":
            raise ValueError(
                f"Coverage config name must end in .json or omit the suffix: {config_name!r}."
            )
    else:
        relative = relative.with_suffix(".json")
    resolved = (configs_root / relative).resolve()
    try:
        resolved.relative_to(configs_root)
    except ValueError:
        raise ValueError(
            f"Coverage config name escapes configs_root: {config_name!r}."
        ) from None
    return resolved


def load_coverage_batch_manifest(
    manifest_path: str | Path,
    *,
    sort_by: str = "name",
) -> CoverageBatchManifest:
    """Load and completely preflight a grouped coverage manifest.

    This performs no image analysis.  Every config, source, TIFF assignment,
    duplicate constraint, and output group identity is validated before the
    caller creates or cleans output directories.
    """

    path = expand_user_path(manifest_path).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Coverage batch manifest not found: '{path}'.") from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed JSON in coverage batch manifest '{path}' at line "
            f"{exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "Coverage batch manifest must be a non-empty JSON object keyed by scientific sample ID."
        )

    groups: list[CoverageBatchGroup] = []
    safe_group_ids: dict[str, str] = {}
    for raw_sample_id, raw_group in data.items():
        if not isinstance(raw_sample_id, str) or not raw_sample_id.strip():
            raise ValueError("Coverage batch scientific sample IDs must be non-empty strings.")
        sample_id = raw_sample_id.strip()
        safe_id = _safe_name(sample_id)
        prior_id = safe_group_ids.get(safe_id.casefold())
        if prior_id is not None:
            raise ValueError(
                f"Scientific sample IDs {prior_id!r} and {sample_id!r} map to the same output name {safe_id!r}."
            )
        safe_group_ids[safe_id.casefold()] = sample_id
        if not isinstance(raw_group, dict):
            raise ValueError(
                f"Coverage batch group {sample_id!r} must be a JSON object."
            )
        root_text = raw_group.get("configs_root")
        if not isinstance(root_text, str) or not root_text.strip():
            raise ValueError(
                f"Coverage batch group {sample_id!r} requires a non-empty configs_root."
            )
        root_value = expand_user_path(root_text)
        configs_root = (
            root_value if root_value.is_absolute() else path.parent / root_value
        ).resolve()
        if not configs_root.exists() or not configs_root.is_dir():
            raise FileNotFoundError(
                f"configs_root for coverage batch group {sample_id!r} does not exist or is not a directory: '{configs_root}'."
            )
        config_names = raw_group.get("config_names")
        if not isinstance(config_names, list) or not config_names:
            raise ValueError(
                f"Coverage batch group {sample_id!r} requires a non-empty config_names list."
            )

        assignments: list[CoverageBatchAssignment] = []
        assigned_tiffs: dict[Path, Path] = {}
        for raw_config_name in config_names:
            if not isinstance(raw_config_name, str):
                raise ValueError(
                    f"Coverage batch group {sample_id!r} config_names entries must be strings."
                )
            config_path = _resolve_manifest_config_path(configs_root, raw_config_name)
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Coverage config for scientific sample {sample_id!r} does not exist: '{config_path}'."
                )
            try:
                app_config = load_coverage_app_config(config_path)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
                raise ValueError(
                    f"Invalid coverage config '{config_path}' assigned to scientific sample {sample_id!r}: {exc}"
                ) from exc
            try:
                source = resolve_coverage_source(
                    cli_root=None,
                    app_config=app_config,
                    config_path=config_path,
                )
                if source is None:
                    raise ValueError("the config does not define a folder source")
                image_paths = _coverage_source_image_paths(source, sort_by=sort_by)
            except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
                raise ValueError(
                    f"Unusable coverage source in config '{config_path}' assigned to scientific sample {sample_id!r}: {exc}"
                ) from exc
            if not image_paths:
                raise ValueError(
                    f"Coverage config '{config_path}' assigned to scientific sample {sample_id!r} identifies no analyzable TIFF data."
                )
            for image_path in image_paths:
                canonical = image_path.resolve()
                prior_config = assigned_tiffs.get(canonical)
                if prior_config is not None:
                    raise ValueError(
                        f"Duplicate TIFF assignment for scientific sample {sample_id!r}: "
                        f"'{canonical}' is assigned by both '{prior_config}' and '{config_path}'."
                    )
                assigned_tiffs[canonical] = config_path
            assignments.append(
                CoverageBatchAssignment(
                    config_name=config_path.stem,
                    config_path=config_path,
                    app_config=app_config,
                    source=source,
                    image_paths=image_paths,
                )
            )
        groups.append(CoverageBatchGroup(sample_id, tuple(assignments)))
    return CoverageBatchManifest(path, tuple(groups))


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
    configured_label = (
        "configured secondary coverage branch enabled"
        if configured
        else "configured primary-only coverage branch"
    )
    if selection == "configured":
        # Preserve the loaded object exactly: this mode delegates the branch
        # choice to each individual tuned config, including grouped manifests.
        return [("configured", configured_label, config)]
    choices = {
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


def _coverage_branch_output_dir(outputs_dir: Path, branch_id: str) -> Path:
    """Keep configured output flat; explicit comparison branches stay isolated."""

    return outputs_dir if branch_id == "configured" else outputs_dir / f"coverage_{branch_id}"


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
    branches: str = "configured",
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
            branch_dir = _coverage_branch_output_dir(outputs_dir, branch_id)
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


def _group_branch_plans(
    group: CoverageBatchGroup,
    selection: str,
) -> list[tuple[str, str, list[tuple[CoverageBatchAssignment, Any]]]]:
    """Arrange per-config immutable branch configs into scientific branches."""

    by_branch: dict[
        str, tuple[str, list[tuple[CoverageBatchAssignment, Any]]]
    ] = {}
    for assignment in group.assignments:
        for branch_id, branch_label, viewer_config in _coverage_branch_configs(
            assignment.app_config.viewer, selection
        ):
            if branch_id not in by_branch:
                by_branch[branch_id] = (branch_label, [])
            by_branch[branch_id][1].append((assignment, viewer_config))
    plans = [
        (branch_id, label, assignments)
        for branch_id, (label, assignments) in by_branch.items()
    ]
    if selection == "configured" and plans:
        return [
            (
                "configured",
                "coverage branch selected by each analysis config",
                plans[0][2],
            )
        ]
    return plans


def _meaningful_assignment_source(assignment: CoverageBatchAssignment) -> Path:
    """Return the sample-level source path used by existing global summaries.

    Even when a config selects one file, the established global ``source_path``
    is its containing source folder.  Exact selected-file provenance remains in
    ``coverage_source_file`` and each image's ``source_path``.
    """

    return assignment.source.root.resolve()


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        canonical = path.resolve()
        if canonical not in seen:
            seen.add(canonical)
            unique.append(canonical)
    return unique


def _group_png_names(group: CoverageBatchGroup) -> dict[Path, str]:
    """Choose deterministic TIFF-derived PNG names without silent collisions."""

    items = [
        (assignment, image_path.resolve())
        for assignment in group.assignments
        for image_path in assignment.image_paths
    ]
    bases = [_safe_name(path.stem) or "image" for _assignment, path in items]
    counts: dict[str, int] = {}
    for base in bases:
        counts[base.casefold()] = counts.get(base.casefold(), 0) + 1

    used: set[str] = set()
    names: dict[Path, str] = {}
    for (assignment, image_path), base in zip(items, bases):
        if counts[base.casefold()] == 1:
            candidate = base
        else:
            config_part = _safe_name(assignment.config_name) or "config"
            candidate = f"{base}__{config_part}"
            if candidate.casefold() in used:
                parent_part = _safe_name(image_path.parent.name) or "source"
                candidate = f"{base}__{config_part}__{parent_part}"
            sequence = 2
            unsuffixed = candidate
            while candidate.casefold() in used:
                candidate = f"{unsuffixed}__{sequence}"
                sequence += 1
        used.add(candidate.casefold())
        names[image_path] = f"{candidate}.png"
    return names


def _assignment_provenance(
    assignment: CoverageBatchAssignment,
) -> dict[str, object]:
    return {
        "analysis_config_name": assignment.config_name,
        "analysis_config_path": str(assignment.config_path.resolve()),
        "coverage_source_origin": "batch_manifest",
        "coverage_source_root": str(assignment.source.root.resolve()),
        "coverage_source_file": (
            None
            if assignment.source.selected_file is None
            else str(assignment.source.selected_file.resolve())
        ),
    }


def _common_value(items: list[Any]) -> Any:
    """Return one truthful shared scalar, or None when values differ."""

    if not items:
        return None
    first = items[0]
    return first if all(item == first for item in items[1:]) else None


def _run_grouped_coverage_batch(
    manifest: CoverageBatchManifest,
    outputs_dir: Path,
    *,
    branches: str = "configured",
    debug_performance: bool = False,
) -> list[Path]:
    """Stream preflighted tuned-config assignments into pooled sample summaries."""

    written: list[Path] = []
    total = sum(
        len(assignment.image_paths)
        for group in manifest.groups
        for _branch_id, _label, branch_assignments in _group_branch_plans(
            group, branches
        )
        for assignment, _viewer_config in branch_assignments
    )
    progress = _progress(desc="Grouped coverage branch images", total=total)
    tracing_started_here = False
    if debug_performance:
        _enable_performance_logger()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            tracing_started_here = True
    try:
        for group in manifest.groups:
            png_names = _group_png_names(group)
            source_paths = _unique_paths(
                _meaningful_assignment_source(assignment)
                for assignment in group.assignments
            )
            source_path = str(source_paths[0]) if len(source_paths) == 1 else None
            source_roots = _unique_paths(
                assignment.source.root for assignment in group.assignments
            )
            source_root = str(source_roots[0]) if len(source_roots) == 1 else None
            for branch_id, branch_label, branch_assignments in _group_branch_plans(
                group, branches
            ):
                branch_dir = _coverage_branch_output_dir(outputs_dir, branch_id)
                png_dir = branch_dir / "coverage_png" / _safe_name(group.sample_id)
                image_paths = tuple(
                    path
                    for assignment, _viewer_config in branch_assignments
                    for path in assignment.image_paths
                )
                branch_viewers = [
                    viewer_config
                    for _assignment, viewer_config in branch_assignments
                ]
                builder = CoverageSampleSummaryBuilder(
                    sample_dir=Path(source_path or ""),
                    config=branch_viewers[0],
                    image_paths=image_paths,
                )
                image_number = 0
                for assignment, viewer_config in branch_assignments:
                    provenance = _assignment_provenance(assignment)
                    image_branch_label = (
                        (
                            "configured secondary coverage branch enabled"
                            if viewer_config.ag_enable_secondary_coverage
                            else "configured primary-only coverage branch"
                        )
                        if branch_id == "configured"
                        else branch_label
                    )
                    for image_path in assignment.image_paths:
                        image_number += 1
                        total_start = perf_counter()
                        analysis_seconds = serialization_seconds = png_seconds = 0.0
                        result = None
                        stored_record = None
                        image_record = None
                        try:
                            phase_start = perf_counter()
                            result = analyze_coverage_image(image_path, viewer_config)
                            analysis_seconds = perf_counter() - phase_start

                            phase_start = perf_counter()
                            image_record = dict(
                                build_coverage_image_record(
                                    image_path, viewer_config, result
                                )
                            )
                            record_metadata = {
                                "sample": group.sample_id,
                                "source_path": str(image_path.resolve()),
                                "coverage_branch_id": branch_id,
                                "coverage_branch_label": image_branch_label,
                                "ag_enable_secondary_coverage": bool(
                                    viewer_config.ag_enable_secondary_coverage
                                ),
                                **provenance,
                            }
                            image_record.update(record_metadata)
                            for roi in image_record.get("rois", []):
                                if isinstance(roi, dict):
                                    roi.update(record_metadata)
                            stored_record = builder.add_success(
                                image_path, image_record
                            )
                            serialization_seconds = perf_counter() - phase_start

                            phase_start = perf_counter()
                            try:
                                png_path = _export_coverage_png(
                                    image_path,
                                    viewer_config,
                                    png_dir / png_names[image_path.resolve()],
                                    result=result,
                                    branch_label=image_branch_label,
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
                                branch_label=image_branch_label,
                            )
                            builder.failures[-1].update(
                                {
                                    "sample": group.sample_id,
                                    "source_path": str(image_path.resolve()),
                                    "ag_enable_secondary_coverage": bool(
                                        viewer_config.ag_enable_secondary_coverage
                                    ),
                                    **provenance,
                                }
                            )
                        finally:
                            result = None
                            stored_record = None
                            image_record = None
                            if progress is not None:
                                progress.update(1)
                            if debug_performance:
                                _report_image_performance(
                                    branch_id=branch_id,
                                    sample=group.sample_id,
                                    image_index=image_number,
                                    image_count=len(image_paths),
                                    analysis_seconds=analysis_seconds,
                                    serialization_seconds=serialization_seconds,
                                    png_seconds=png_seconds,
                                    total_seconds=perf_counter() - total_start,
                                )

                summary = builder.finalize()
                config_dicts = [asdict(viewer) for viewer in branch_viewers]
                summary["viewer_config"] = _common_value(config_dicts)
                summary["folder"] = source_path or ""
                summary["file"] = None
                summary["sample"] = group.sample_id
                summary["source_path"] = source_path
                summary["source_paths"] = [str(path) for path in source_paths]
                summary["coverage_branch_id"] = branch_id
                summary["coverage_branch_label"] = branch_label
                configured_secondary = _common_value(
                    [
                        bool(viewer.ag_enable_secondary_coverage)
                        for viewer in branch_viewers
                    ]
                )
                summary["ag_enable_secondary_coverage"] = configured_secondary
                summary["coverage_source_origin"] = "batch_manifest"
                summary["coverage_source_root"] = source_root
                summary["coverage_source_file"] = None
                summary["batch_manifest_path"] = str(manifest.path)
                summary["analysis_configs"] = [
                    {
                        **_assignment_provenance(assignment),
                        "source_path": str(_meaningful_assignment_source(assignment)),
                    }
                    for assignment, _viewer_config in branch_assignments
                ]
                global_summary = summary.setdefault("global_summary", {})
                global_summary.update(
                    {
                        "coverage_branch_id": branch_id,
                        "coverage_branch_label": branch_label,
                        "ag_enable_secondary_coverage": configured_secondary,
                        "coverage_source_origin": "batch_manifest",
                        "coverage_source_root": source_root,
                        "coverage_source_file": None,
                        "source_path": source_path,
                        "source_paths": [str(path) for path in source_paths],
                    }
                )
                for field_name in (
                    "sphere_anisotropy_check",
                    "max_global_sphere_anisotropy_ratio",
                    "sphere_solidity_check",
                    "min_global_sphere_solidity",
                    "selected_cap_coverage_metric",
                    "coverage_cap_radius_fraction",
                ):
                    global_summary[field_name] = _common_value(
                        [getattr(viewer, field_name) for viewer in branch_viewers]
                    )
                out_path = branch_dir / f"{_safe_name(group.sample_id)}__coverage.json"
                _write_json(summary, out_path)
                written.append(out_path)

                builder.image_records.clear()
                builder.failures.clear()
                del summary
                del builder
                if debug_performance:
                    collected = gc.collect()
                    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                    LOGGER.debug(
                        "coverage-sample-release branch=%s sample=%s "
                        "collected=%d figures=%d traced_current_mib=%.3f "
                        "traced_peak_mib=%.3f",
                        branch_id,
                        group.sample_id,
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
            for path in png_dir.rglob("*.png"):
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
    parser.add_argument("--bead-root", type=Path, help="Optional bead root override. Without it, an explicitly selected --bead-config may supply its folder.")
    parser.add_argument(
        "--bead-config",
        type=Path,
        default=Path("sem_bead_viewer_config.json"),
        help="Config file used for bead parameters and, when --bead-root is omitted, its top-level folder source.",
    )
    parser.add_argument("--coverage-root", type=Path, help="Optional recursive coverage root. Overrides the folder/file source stored in --coverage-config; incompatible with --batch-config.")
    coverage_source_group = parser.add_mutually_exclusive_group()
    coverage_source_group.add_argument(
        "--coverage-config",
        type=Path,
        default=Path("sem_coverage_viewer_config.json"),
        help="Coverage parameter template; its top-level folder/file fields may also supply the input source.",
    )
    coverage_source_group.add_argument(
        "--batch-config",
        type=Path,
        help=(
            "Grouped coverage manifest. Top-level keys are scientific sample IDs; "
            "each entry supplies configs_root and a non-empty config_names list."
        ),
    )
    parser.add_argument(
        "--coverage-branches",
        choices=("both", "configured", "one-layer", "two-layers"),
        default="configured",
        help=(
            "Coverage-mask mode. 'configured' (default) uses each coverage "
            "config's ag_enable_secondary_coverage value and writes directly "
            "to outputs-dir; explicit comparison branches use separate directories."
        ),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    command_args = list(sys.argv[1:] if argv is None else argv)
    if args.batch_config is not None and args.coverage_root is not None:
        parser.error("--coverage-root cannot be combined with --batch-config; each manifest subconfig supplies its own source.")
    bead_config_requested = any(
        item == "--bead-config" or item.startswith("--bead-config=")
        for item in command_args
    )
    coverage_config_requested = any(
        item == "--coverage-config" or item.startswith("--coverage-config=")
        for item in command_args
    )

    bead_source = None
    bead_config_path = expand_user_path(args.bead_config)
    if args.bead_root is not None or bead_config_requested:
        bead_app_config = load_bead_app_config(bead_config_path)
        bead_source = resolve_bead_source(
            cli_root=args.bead_root,
            app_config=bead_app_config,
            config_path=bead_config_path,
        )
        if bead_source is None:
            raise SystemExit(
                "No bead input source was provided. Supply --bead-root or a bead config containing a valid top-level folder source."
            )
        # Validate the selected bead population before cleaning or creating outputs.
        _find_sample_dirs(bead_source.root)

    coverage_config_path = expand_user_path(args.coverage_config)
    coverage_app_config = None
    coverage_source = None
    coverage_manifest = None
    if args.batch_config is not None:
        coverage_manifest = load_coverage_batch_manifest(
            args.batch_config,
            sort_by=args.sort_by,
        )
    elif args.coverage_root is not None or coverage_config_requested:
        coverage_app_config = load_coverage_app_config(coverage_config_path)
        coverage_source = resolve_coverage_source(
            cli_root=args.coverage_root,
            app_config=coverage_app_config,
            config_path=coverage_config_path,
        )
    if bead_source is None and coverage_source is None and coverage_manifest is None:
        raise SystemExit(
            "No input source was provided. Supply --bead-root/--bead-config, "
            "--coverage-root/--coverage-config, or --batch-config."
        )

    outputs_dir = expand_user_path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _remove_outputs(outputs_dir)

    written: list[Path] = []
    if bead_source is not None:
        if bead_source.source_origin == "config":
            print(f"Bead source from config: recursive folder ({bead_source.root})")
        else:
            print(f"Bead source from CLI: recursive folder ({bead_source.root})")
        written.extend(
            _run_bead_batch(
                bead_source.root,
                bead_config_path,
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
    if coverage_manifest is not None:
        print(
            f"Coverage source from grouped manifest: {len(coverage_manifest.groups)} scientific sample group(s)"
        )
        written.extend(
            _run_grouped_coverage_batch(
                coverage_manifest,
                outputs_dir,
                branches=args.coverage_branches,
                debug_performance=args.debug_performance,
            )
        )
    if not args.no_export:
        if bead_source is not None:
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
        if coverage_source is not None or coverage_manifest is not None:
            coverage_export_dirs = (
                (outputs_dir,)
                if args.coverage_branches == "configured"
                else tuple(
                    outputs_dir / name
                    for name in ("coverage_one_layer", "coverage_two_layers")
                )
            )
            for branch_dir in coverage_export_dirs:
                if branch_dir.exists() and any(branch_dir.glob("*.json")):
                    written.extend(export_outputs(branch_dir, csv=not args.no_csv, bead=False, coverage=True, bead_csv=False, coverage_csv=not args.no_coverage_csv, histograms=False, table_format=args.table_format, sort_by=args.sort_by))

    _print_paths(written)


if __name__ == "__main__":
    main()

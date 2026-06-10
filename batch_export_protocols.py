from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from export_output_summaries import export_outputs
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
    build_coverage_summary,
    load_app_config as load_coverage_app_config,
    load_failed_image_preview,
)


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_", ".") else "_" for ch in name)
    return "_".join(safe.split())


def _find_sample_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: '{root}'.")
    sample_dirs = sorted({path.parent for path in root.rglob("*.tif")})
    if not sample_dirs:
        raise FileNotFoundError(f"No TIFF files found under '{root}'.")
    return sample_dirs


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


def _export_coverage_png(image_path: Path, config, output_path: Path) -> Path:
    try:
        res = analyze_coverage_image(image_path, config)
        base_gray = res.display.astype(np.float32)
        base = np.dstack([base_gray, base_gray, base_gray]).astype(np.float32)
        if config.default_show_bead_boundary:
            for roi in res.roi_results:
                include = _include_roi_in_global_summary(roi, res.config)
                base[find_boundaries(roi.bead_mask, mode="outer")] = (0.0, 1.0, 0.0) if include else (1.0, 0.0, 0.0)
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
                    row, col = m.centroid_rc
                    x_half = m.x_diameter_px / 2.0
                    y_half = m.y_diameter_px / 2.0
                    artists.extend(
                        [
                            Line2D([col - x_half, col + x_half], [row, row], color=color, linewidth=1.2),
                            Line2D([col, col], [row - y_half, row + y_half], color=color, linewidth=1.2),
                            ax.text(
                                col,
                                row - y_half - 8,
                                f"x={_format_px_or_length(m.x_diameter_m, m.x_diameter_px)}  y={_format_px_or_length(m.y_diameter_m, m.y_diameter_px)}",
                                color=color,
                                fontsize=8,
                                ha="center",
                                va="bottom",
                                bbox={"facecolor": (0.0, 0.0, 0.0, 0.45), "edgecolor": "none", "pad": 1.5},
                            ),
                        ]
                    )
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


def _run_bead_batch(root: Path, config_path: Path, outputs_dir: Path) -> list[Path]:
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
        for image_path in sorted(sample_dir.glob("*.tif")):
            written.append(_export_bead_png(image_path, cfg.viewer, png_dir / f"{label}__{_safe_name(image_path.stem)}.png"))
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return written


def _run_coverage_batch(root: Path, config_path: Path, outputs_dir: Path) -> list[Path]:
    cfg = load_coverage_app_config(config_path)
    written: list[Path] = []
    png_dir = outputs_dir / "coverage_png"
    sample_dirs = _find_sample_dirs(root)
    progress = _progress(desc="Coverage images", total=_count_images(sample_dirs))
    for sample_dir in sample_dirs:
        out_path = _json_path(outputs_dir, "coverage", root, sample_dir)
        summary = build_coverage_summary(sample_dir, cfg.viewer)
        _write_json(summary, out_path)
        written.append(out_path)
        label = _safe_name(_relative_label(root, sample_dir))
        for image_path in sorted(sample_dir.glob("*.tif")):
            written.append(_export_coverage_png(image_path, cfg.viewer, png_dir / f"{label}__{_safe_name(image_path.stem)}.png"))
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return written


def _remove_outputs(outputs_dir: Path) -> None:
    if not outputs_dir.exists():
        return
    for path in outputs_dir.glob("*.json"):
        path.unlink()
    for path in outputs_dir.glob("*.csv"):
        path.unlink()
    hist_dir = outputs_dir / "bead_histograms"
    if hist_dir.exists():
        for path in hist_dir.glob("*.png"):
            path.unlink()
    for png_dir_name in ("size_png", "coverage_png"):
        png_dir = outputs_dir / png_dir_name
        if png_dir.exists():
            for path in png_dir.glob("*.png"):
                path.unlink()


def _print_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        print(path)


def _progress(*, desc: str, total: int):
    if tqdm is None:
        return None
    return tqdm(desc=desc, total=total, unit="image")


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--coverage-root", type=Path, help="Root folder for coverage protocol samples.")
    parser.add_argument(
        "--coverage-config",
        type=Path,
        default=Path("sem_coverage_viewer_config.json"),
        help="Config file used as a parameter template for coverage analysis.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs2"),
        help="Directory where per-sample JSONs and exported CSVs will be written.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing JSON/CSV/PNG outputs in outputs-dir before running.")
    parser.add_argument("--no-export", action="store_true", help="Only write per-sample JSON summaries, skip CSV/histogram export.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.bead_root and not args.coverage_root:
        raise SystemExit("At least one of --bead-root or --coverage-root must be provided.")

    outputs_dir = args.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _remove_outputs(outputs_dir)

    written: list[Path] = []
    if args.bead_root:
        written.extend(_run_bead_batch(args.bead_root, args.bead_config, outputs_dir))
    if args.coverage_root:
        written.extend(_run_coverage_batch(args.coverage_root, args.coverage_config, outputs_dir))
    if not args.no_export:
        written.extend(export_outputs(outputs_dir))

    _print_paths(written)


if __name__ == "__main__":
    main()

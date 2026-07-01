from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from path_utils import expand_user_path, resolve_optional_file_in_folder
from tabular_export import natural_sort_key, sort_paths, sort_rows, write_csv_table, write_xlsx_workbook
from tem_particle_viewer import (
    TEMAnalysisResult,
    _make_measure_overlay,
    _make_scale_overlay,
    _measurements_to_dicts,
    _overlay_image,
    _resolve_image_paths,
    _safe_float,
    analyze_tem_image,
    build_tem_summary,
    build_tem_summary_from_paths,
    load_app_config,
    load_failed_image_preview,
)


TEM_GLOBAL_CSV_NAME = "tem_global_summaries.csv"
TEM_IMAGE_CSV_NAME = "tem_image_summaries.csv"
TEM_WORKBOOK_NAME = "tem_summaries.xlsx"
TEM_HISTOGRAM_DIR_NAME = "tem_histograms"


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_", ".") else "_" for ch in name)
    return "_".join(safe.split())


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _discover_source_images(root: Path, outputs_dir: Path) -> list[Path]:
    exclude_dirs = {
        outputs_dir.name,
        "tem_png",
        "tem_json",
        TEM_HISTOGRAM_DIR_NAME,
        "__pycache__",
    }
    paths = _resolve_image_paths(root, exclude_dirs=exclude_dirs)
    outputs_dir_resolved = outputs_dir.resolve()
    return sort_paths(
        [path for path in paths if not _is_within(path.resolve(), outputs_dir_resolved)],
        root=root,
    )


def _group_images_by_sample(
    root: Path, image_paths: list[Path], *, sort_by: str = "name"
) -> list[tuple[Path, list[Path]]]:
    grouped: dict[Path, list[Path]] = {}
    root = root.resolve()
    for path in image_paths:
        sample_dir = path.parent.resolve()
        grouped.setdefault(sample_dir, []).append(path)
    sample_dirs = sort_paths(list(grouped), sort_by=sort_by, root=root)
    return [
        (sample_dir, sort_paths(grouped[sample_dir], sort_by=sort_by, root=root))
        for sample_dir in sample_dirs
    ]


def _sample_label(root: Path, sample_dir: Path) -> str:
    try:
        rel = sample_dir.resolve().relative_to(root.resolve())
        parts = [part for part in rel.parts if part not in ("", ".")]
        return "root" if not parts else " - ".join(parts)
    except ValueError:
        return sample_dir.name


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


def _export_tem_png(res: TEMAnalysisResult, viewer_cfg, output_path: Path, show_measures: bool = True, show_scale: bool = True) -> Path:
    base = _overlay_image(res, show_boundaries=True)

    def overlay_builder(ax):
        artists: list[object] = []
        if show_scale:
            artists.extend(_make_scale_overlay(ax, res.display.shape, res.metadata.pixel_size_nm))
        if show_measures:
            artists.extend(_make_measure_overlay(ax, res.measurements, viewer_cfg))
        return artists

    _save_overlay_figure(base, output_path, overlay_builder)
    return output_path


def _preview_png(preview, output_path: Path) -> Path:
    base_gray = preview.display.astype(np.float32)
    base = np.dstack([base_gray, base_gray, base_gray]).astype(np.float32)

    def overlay_builder(ax):
        return _make_scale_overlay(ax, preview.display.shape, preview.metadata.pixel_size_nm)

    _save_overlay_figure(base, output_path, overlay_builder)
    return output_path


def _per_image_json(res: TEMAnalysisResult) -> dict:
    valid_mean_nm = [m.mean_axis_length_nm for m in res.measurements if m.valid and m.mean_axis_length_nm is not None]
    valid_mean_px = [m.mean_axis_length_px for m in res.measurements if m.valid]
    valid_eq_nm = [m.equivalent_diameter_nm for m in res.measurements if m.valid and m.equivalent_diameter_nm is not None]
    valid_eq_px = [m.equivalent_diameter_px for m in res.measurements if m.valid]
    return {
        "file": res.image_path.name,
        "sample": res.image_path.parent.name,
        "status": "ok",
        "pixel_size_nm": _safe_float(res.metadata.pixel_size_nm),
        "fov_nm": _safe_float(res.metadata.fov_nm),
        "detector": res.metadata.detector,
        "crop_row": int(res.metadata.crop_row),
        "particle_count": sum(1 for m in res.measurements if m.valid),
        "flagged_count": sum(1 for m in res.measurements if not m.valid),
        "mean_axis_length_px": _safe_float(float(np.mean(valid_mean_px))) if valid_mean_px else None,
        "median_mean_axis_length_px": _safe_float(float(np.median(valid_mean_px))) if valid_mean_px else None,
        "sd_mean_axis_length_px": _safe_float(float(np.std(valid_mean_px, ddof=1))) if len(valid_mean_px) > 1 else 0.0 if valid_mean_px else None,
        "mean_axis_length_nm": _safe_float(float(np.mean(valid_mean_nm))) if valid_mean_nm else None,
        "median_mean_axis_length_nm": _safe_float(float(np.median(valid_mean_nm))) if valid_mean_nm else None,
        "sd_mean_axis_length_nm": _safe_float(float(np.std(valid_mean_nm, ddof=1))) if len(valid_mean_nm) > 1 else 0.0 if valid_mean_nm else None,
        "mean_eq_diameter_px": _safe_float(float(np.mean(valid_eq_px))) if valid_eq_px else None,
        "median_eq_diameter_px": _safe_float(float(np.median(valid_eq_px))) if valid_eq_px else None,
        "sd_eq_diameter_px": _safe_float(float(np.std(valid_eq_px, ddof=1))) if len(valid_eq_px) > 1 else 0.0 if valid_eq_px else None,
        "mean_eq_diameter_nm": _safe_float(float(np.mean(valid_eq_nm))) if valid_eq_nm else None,
        "median_eq_diameter_nm": _safe_float(float(np.median(valid_eq_nm))) if valid_eq_nm else None,
        "sd_eq_diameter_nm": _safe_float(float(np.std(valid_eq_nm, ddof=1))) if len(valid_eq_nm) > 1 else 0.0 if valid_eq_nm else None,
        "particles": _measurements_to_dicts(res.measurements),
    }


def _global_summary_row(global_summary: dict, source_name: str, json_file: str) -> dict:
    return {
        "name": source_name,
        "json_file": json_file,
        **global_summary,
    }


def _image_summary_rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for image in summary.get("images", []):
        row = {key: value for key, value in image.items() if key != "particles"}
        rows.append(row)
    return rows


def _histogram_metric_values(summary: dict, histogram_metric: str) -> tuple[np.ndarray, str]:
    particles = []
    for image in summary.get("images", []):
        if image.get("status") != "ok":
            continue
        particles.extend([p for p in image.get("particles", []) if p.get("valid")])

    if histogram_metric == "eq_diameter":
        nm_vals = [float(p["eq_diameter_nm"]) for p in particles if p.get("eq_diameter_nm") is not None]
        if nm_vals:
            return np.array(nm_vals, dtype=np.float64), "Equivalent diameter [nm]"
        return np.array([float(p["eq_diameter_px"]) for p in particles], dtype=np.float64), "Equivalent diameter [px]"
    if histogram_metric == "major_axis":
        nm_vals = [float(p["display_major_axis_length_nm"]) for p in particles if p.get("display_major_axis_length_nm") is not None]
        if nm_vals:
            return np.array(nm_vals, dtype=np.float64), "Major axis [nm]"
        return np.array([float(p["display_major_axis_length_px"]) for p in particles], dtype=np.float64), "Major axis [px]"
    if histogram_metric == "minor_axis":
        nm_vals = [float(p["display_minor_axis_length_nm"]) for p in particles if p.get("display_minor_axis_length_nm") is not None]
        if nm_vals:
            return np.array(nm_vals, dtype=np.float64), "Minor axis [nm]"
        return np.array([float(p["display_minor_axis_length_px"]) for p in particles], dtype=np.float64), "Minor axis [px]"
    nm_vals = [float(p["mean_axis_length_nm"]) for p in particles if p.get("mean_axis_length_nm") is not None]
    if nm_vals:
        return np.array(nm_vals, dtype=np.float64), "Mean axes [nm]"
    return np.array([float(p["mean_axis_length_px"]) for p in particles], dtype=np.float64), "Mean axes [px]"


def _format_hist_stats(values: np.ndarray, unit_label: str) -> str:
    if values.size == 0:
        return "No valid particles"
    mean = float(np.mean(values))
    median = float(np.median(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else float("nan")
    return "\n".join(
        [
            f"n = {values.size}",
            f"mean = {mean:.2f} {unit_label}",
            f"median = {median:.2f} {unit_label}",
            f"SD = {sd:.2f} {unit_label}",
            f"CV = {cv:.1f} %",
        ]
    )


def _plot_histogram(values: np.ndarray, xlabel: str, title: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if values.size:
        bins = min(max(8, int(np.sqrt(values.size))), 40)
        ax.hist(values, bins=bins, color="#4cc9f0", edgecolor="#0b1f2a", alpha=0.9)
        ax.axvline(float(np.mean(values)), color="#d00000", linewidth=1.5)
        ax.axvline(float(np.median(values)), color="#2d6a4f", linewidth=1.5, linestyle="--")
    else:
        ax.text(0.5, 0.5, "No valid particles", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    unit_label = xlabel.split("[", 1)[1].rstrip("]") if "[" in xlabel else "px"
    ax.text(
        0.98,
        0.98,
        _format_hist_stats(values, unit_label),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.85, "pad": 6},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _export_tables(
    summary_rows: list[dict],
    image_rows: list[dict],
    outputs_dir: Path,
    *,
    table_format: str,
    csv_enabled: bool,
    sort_by: str,
) -> list[Path]:
    written: list[Path] = []
    if table_format in {"csv", "both"} and csv_enabled:
        global_csv = write_csv_table(
            summary_rows,
            outputs_dir / TEM_GLOBAL_CSV_NAME,
            preferred_columns=("name", "json_file"),
            sort_by=sort_by,
        )
        if global_csv is not None:
            written.append(global_csv)
        image_csv = write_csv_table(
            image_rows,
            outputs_dir / TEM_IMAGE_CSV_NAME,
            preferred_columns=("sample", "file", "status"),
            sort_by=sort_by,
        )
        if image_csv is not None:
            written.append(image_csv)
    if table_format in {"xlsx", "both"}:
        workbook = write_xlsx_workbook(
            {
                "Samples": summary_rows,
                "Images": image_rows,
            },
            outputs_dir / TEM_WORKBOOK_NAME,
            preferred_columns={
                "Samples": ("name", "json_file"),
                "Images": ("sample", "file", "status"),
            },
            sort_by=sort_by,
        )
        if workbook is not None:
            written.append(workbook)
    return written


def _export_histograms(sample_summaries: list[tuple[str, dict]], outputs_dir: Path, histogram_metric: str) -> list[Path]:
    written: list[Path] = []
    hist_dir = outputs_dir / TEM_HISTOGRAM_DIR_NAME
    suffix = _safe_name(histogram_metric)
    combined: list[np.ndarray] = []
    for sample_name, summary in sample_summaries:
        values, xlabel = _histogram_metric_values(summary, histogram_metric)
        combined.append(values)
        written.append(
            _plot_histogram(
                values,
                xlabel,
                sample_name,
                hist_dir / f"{_safe_name(sample_name)}_{suffix}_histogram.png",
            )
        )
    if combined:
        all_values = np.concatenate([vals for vals in combined if vals.size]) if any(vals.size for vals in combined) else np.array([], dtype=np.float64)
        if sample_summaries:
            _, xlabel = _histogram_metric_values(sample_summaries[0][1], histogram_metric)
        else:
            xlabel = "Mean axes [px]"
        written.append(
            _plot_histogram(
                all_values,
                xlabel,
                f"All TEM samples ({histogram_metric})",
                hist_dir / f"all_particles_{suffix}_histogram.png",
            )
        )
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
    for subdir in ("tem_png", "tem_json"):
        target = outputs_dir / subdir
        if target.exists():
            for path in target.glob("*"):
                if path.is_file():
                    path.unlink()
    hist_dir = outputs_dir / TEM_HISTOGRAM_DIR_NAME
    if hist_dir.exists():
        for path in hist_dir.glob("*.png"):
            path.unlink()


def _print_paths(paths: Iterable[Path]) -> None:
    for path in sort_paths(list(paths), sort_by="path"):
        print(path)


def _progress(desc: str, total: int):
    if tqdm is None:
        return None
    return tqdm(desc=desc, total=total, unit="image")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-run TEM particle sizing export for PNG/JPG images.")
    parser.add_argument("--root", type=Path, required=True, help="TEM image folder.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tem_particle_viewer_config.json"),
        help="TEM config used as a parameter template.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs_tem"),
        help="Directory where PNG previews and JSON summaries will be written.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing TEM outputs in outputs-dir before running.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write TEM CSV summary files.")
    parser.add_argument("--no-histograms", action="store_true", help="Do not write TEM histogram PNG files.")
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
        help="Deterministic natural sorting for samples, images, and rows. Default: %(default)s",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app_cfg = load_app_config(expand_user_path(args.config))
    outputs_dir = expand_user_path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _remove_outputs(outputs_dir)

    root = expand_user_path(args.root)
    image_paths = _discover_source_images(root, outputs_dir)
    if app_cfg.file:
        requested = resolve_optional_file_in_folder(root, app_cfg.file).resolve()
        image_paths = [path for path in image_paths if path.resolve() == requested]
        if not image_paths:
            raise FileNotFoundError(f"Requested TEM file was not found under root '{root}': '{requested}'.")
    sample_groups = _group_images_by_sample(root, image_paths, sort_by=args.sort_by)
    png_dir = outputs_dir / "tem_png"
    json_dir = outputs_dir / "tem_json"
    written: list[Path] = []
    progress = _progress("TEM images", len(image_paths))
    sample_summary_rows: list[dict] = []
    image_csv_rows: list[dict] = []
    sample_summaries: list[tuple[str, dict]] = []

    for sample_dir, sample_paths in sample_groups:
        sample_label = _sample_label(root, sample_dir)
        sample_summary = build_tem_summary_from_paths(sample_paths, app_cfg.viewer, sample_dir)
        sample_summary["file"] = app_cfg.file if len(sample_paths) == 1 and app_cfg.file else None
        sample_summary_path = outputs_dir / f"{_safe_name(sample_label)}__tem_summary.json"
        _write_json(sample_summary, sample_summary_path)
        written.append(sample_summary_path)
        sample_summaries.append((sample_label, sample_summary))
        sample_summary_rows.append(_global_summary_row(sample_summary.get("global_summary", {}), sample_label, sample_summary_path.name))
        image_csv_rows.extend(_image_summary_rows(sample_summary))

        for image_path in sample_paths:
            stem = _safe_name(image_path.stem)
            try:
                res = analyze_tem_image(image_path, app_cfg.viewer)
                written.append(_export_tem_png(res, app_cfg.viewer, png_dir / f"{_safe_name(sample_label)}__{stem}.png"))
                per_image = _per_image_json(res)
            except Exception as exc:
                preview = load_failed_image_preview(image_path, app_cfg.viewer)
                written.append(_preview_png(preview, png_dir / f"{_safe_name(sample_label)}__{stem}.png"))
                per_image = {
                    "file": image_path.name,
                    "sample": image_path.parent.name,
                    "status": "failed",
                    "error": str(exc),
                    "pixel_size_nm": _safe_float(preview.metadata.pixel_size_nm),
                    "fov_nm": _safe_float(preview.metadata.fov_nm),
                    "detector": preview.metadata.detector,
                    "crop_row": int(preview.metadata.crop_row),
                    "particle_count": 0,
                    "flagged_count": 0,
                    "particles": [],
                }
            per_image_path = json_dir / f"{_safe_name(sample_label)}__{stem}.json"
            _write_json(per_image, per_image_path)
            written.append(per_image_path)
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    global_summary = build_tem_summary_from_paths(image_paths, app_cfg.viewer, root)
    global_summary["file"] = app_cfg.file
    global_path = outputs_dir / "tem_global_summary.json"
    _write_json(global_summary, global_path)
    written.append(global_path)
    if args.table_format != "none":
        written.extend(
            _export_tables(
                sample_summary_rows,
                image_csv_rows,
                outputs_dir,
                table_format=args.table_format,
                csv_enabled=not args.no_csv,
                sort_by=args.sort_by,
            )
        )
    if not args.no_histograms:
        written.extend(_export_histograms(sample_summaries, outputs_dir, app_cfg.viewer.histogram_metric))
    _print_paths(written)


if __name__ == "__main__":
    main()

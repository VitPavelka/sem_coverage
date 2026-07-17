"""Supplementary central-cap sweep using one production segmentation per image."""

from __future__ import annotations

import csv
from dataclasses import asdict
import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coverage_cap import annular_cap_profile, compute_coverage_cap_metrics
from sem_coverage_viewer import (
    CoverageSegmentationFailure,
    CoverageViewerConfig,
    _resolve_image_paths,
    analyze_coverage_image,
)
from tabular_export import natural_sort_key

LOGGER = logging.getLogger(__name__)

SWEEP_COLUMNS = (
    "source_file", "roi_index", "cap_radius_fraction", "cap_half_angle_deg",
    "cap_radius_px", "cap_radius_um", "cap_completeness", "cap_valid",
    "legacy_full_projected_coverage_percent", "cap_projected_coverage_percent",
    "cap_surface_weighted_coverage_percent", "cap_projected_over_cap_surface_percent",
    "cap_projected_area_um2", "cap_surface_area_um2",
)
PROFILE_COLUMNS = (
    "r_over_R_inner", "r_over_R_outer", "r_over_R_center", "bead_pixel_count",
    "ag_pixel_count", "projected_fraction_percent", "surface_weighted_fraction_percent",
    "projected_over_cap_surface_percent",
)


def parse_fractions(values: str | None, start: float | None = None, stop: float | None = None, step: float | None = None) -> list[float]:
    """Parse explicit comma-separated fractions or an inclusive numeric range."""

    if values:
        try:
            fractions = [float(item.strip()) for item in values.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("Fractions must be comma-separated numbers.") from exc
    elif start is not None or stop is not None or step is not None:
        if start is None or stop is None or step is None or step <= 0:
            raise ValueError("Range mode requires positive --fraction-start, --fraction-stop, and --fraction-step.")
        count = int(math.floor((stop - start) / step + 1e-9)) + 1
        fractions = [start + index * step for index in range(max(0, count))]
    else:
        fractions = [0.10, 0.15, math.sin(math.radians(10.0)), 0.20, 0.25, 0.30, 0.40, 0.50]
    fractions = sorted({round(value, 12) for value in fractions})
    if not fractions or any(not (0.0 < value <= 1.0) for value in fractions):
        raise ValueError("Every cap fraction must satisfy 0 < f <= 1.")
    return fractions


def _write_rows(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def radial_coverage_profile(bead_mask: np.ndarray, ag_mask: np.ndarray, center_rc: tuple[float, float], radius_px: float, bins: int = 10) -> list[dict]:
    """Backward-compatible wrapper around the shared annular profile helper."""

    rows = annular_cap_profile(bead_mask, ag_mask, center_rc, radius_px, bins=bins)
    for row in rows:
        row["projected_fraction_percent"] = _percent(row.pop("projected_fraction"))
        row["surface_weighted_fraction_percent"] = _percent(row.pop("surface_weighted_fraction"))
        row["projected_over_cap_surface_percent"] = _percent(row.pop("projected_over_cap_surface"))
    return rows


def _percent(value: float | None) -> float | None:
    return None if value is None else float(value * 100.0)


def evaluate_roi_sweep(roi, source_file: Path, pixel_size_m: float | None, fractions: Sequence[float], *, include_surface_weighted: bool) -> list[dict]:
    """Evaluate fractions from cached masks without rerunning segmentation."""

    rows: list[dict] = []
    for fraction in fractions:
        cap = compute_coverage_cap_metrics(
            roi.bead_mask, roi.ag_mask, roi.bead_metrics.centroid_rc,
            roi.bead_metrics.sphere_radius_px, fraction, pixel_size_m,
            compute_surface_weighted=True, min_completeness=0.0,
        )
        rows.append({
            "source_file": source_file.name, "roi_index": roi.roi_index,
            "cap_radius_fraction": fraction, "cap_half_angle_deg": cap.geometry.half_angle_deg,
            "cap_radius_px": cap.geometry.cap_radius_px,
            "cap_radius_um": cap.cap_radius_m * 1e6 if cap.cap_radius_m is not None else None,
            "cap_completeness": cap.geometry.completeness, "cap_valid": cap.valid,
            "legacy_full_projected_coverage_percent": roi.legacy_full_projected_coverage_percent,
            "cap_projected_coverage_percent": cap.projected_coverage * 100.0 if cap.projected_coverage is not None else None,
            "cap_surface_weighted_coverage_percent": cap.surface_weighted_coverage * 100.0 if cap.surface_weighted_coverage is not None else None,
            "cap_projected_over_cap_surface_percent": cap.projected_over_cap_surface * 100.0 if cap.projected_over_cap_surface is not None else None,
            "cap_projected_area_um2": cap.projected_area_m2 * 1e12 if cap.projected_area_m2 is not None else None,
            "cap_surface_area_um2": cap.surface_area_m2 * 1e12 if cap.surface_area_m2 is not None else None,
        })
    return rows


def _plot_sweep(rows: Sequence[dict], output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fractions = [row["cap_radius_fraction"] for row in rows]
    projected = [row["cap_projected_coverage_percent"] for row in rows]
    ax.plot(fractions, projected, "o-", label="cap projected")
    weighted = [row["cap_surface_weighted_coverage_percent"] for row in rows]
    if any(value is not None for value in weighted):
        ax.plot(fractions, weighted, "s--", label="experimental surface weighted")
    over_surface = [row["cap_projected_over_cap_surface_percent"] for row in rows]
    if any(value is not None for value in over_surface):
        ax.plot(fractions, over_surface, "^:", label="Ag projected / cap surface")
    ax.axhline(rows[0]["legacy_full_projected_coverage_percent"], color="0.35", linestyle=":", label="legacy full projected")
    invalid = [row for row in rows if not row["cap_valid"]]
    if invalid:
        ax.scatter([row["cap_radius_fraction"] for row in invalid], [row["cap_projected_coverage_percent"] for row in invalid], color="red", marker="x", label="incomplete")
    ax.set(xlabel="Cap radius fraction a/R", ylabel="Coverage [%]", title=title)
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output_path, dpi=180); plt.close(fig)
    return output_path


def _plot_profile(rows: Sequence[dict], output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [row["r_over_R_center"] for row in rows]
    ax.plot(x, [row["projected_fraction_percent"] for row in rows], "o-", label="projected")
    ax.plot(x, [row["surface_weighted_fraction_percent"] for row in rows], "s--", label="surface weighted")
    ax.plot(x, [row["projected_over_cap_surface_percent"] for row in rows], "^:", label="Ag / zone surface")
    ax.set(xlabel="Normalized radius r/R", ylabel="Local projected coverage [%]", title=title)
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output_path, dpi=180); plt.close(fig)
    return output_path


def run_cap_sweep(image_paths: Iterable[Path], config: CoverageViewerConfig, output_dir: Path, fractions: Sequence[float], *, include_surface_weighted: bool = False) -> list[Path]:
    """Run one coverage analysis per image and export lightweight cap studies."""

    written: list[Path] = []
    combined: list[dict] = []
    failures: list[dict] = []
    for image_path in sorted(image_paths, key=natural_sort_key):
        try:
            result = analyze_coverage_image(image_path, config, collect_diagnostics=True)
        except CoverageSegmentationFailure as exc:
            failures.append({"source_file": image_path.name, "error": str(exc), "rejections": " | ".join(reason for candidate in exc.diagnostics.rejected_candidates for reason in candidate.rejection_reasons)})
            LOGGER.warning("No accepted ROI for %s: %s", image_path.name, exc)
            continue
        except Exception as exc:
            failures.append({"source_file": image_path.name, "error": str(exc), "rejections": ""})
            LOGGER.error("Coverage sweep failed for %s", image_path.name, exc_info=True)
            continue
        if not result.roi_results:
            failures.append({"source_file": image_path.name, "error": "No accepted coverage ROI", "rejections": ""})
            LOGGER.warning("No accepted coverage ROI for %s", image_path.name)
            continue
        for roi in result.roi_results:
            stem = f"{image_path.stem}__roi_{roi.roi_index:03d}"
            rows = evaluate_roi_sweep(roi, image_path, result.metadata.mean_pixel_size_m, fractions, include_surface_weighted=include_surface_weighted)
            combined.extend(rows)
            written.append(_write_rows(output_dir / f"{stem}__coverage_cap_sweep.csv", SWEEP_COLUMNS, rows))
            written.append(_plot_sweep(rows, output_dir / f"{stem}__coverage_vs_cap_fraction.png", stem))
            profile = radial_coverage_profile(roi.bead_mask, roi.ag_mask, roi.bead_metrics.centroid_rc, roi.bead_metrics.sphere_radius_px)
            written.append(_write_rows(output_dir / f"{stem}__radial_coverage_profile.csv", PROFILE_COLUMNS, profile))
            written.append(_plot_profile(profile, output_dir / f"{stem}__radial_coverage_profile.png", stem))
    written.append(_write_rows(output_dir / "coverage_cap_sweep_summary.csv", SWEEP_COLUMNS, combined))
    if failures:
        written.append(_write_rows(output_dir / "coverage_cap_sweep_failures.csv", ("source_file", "error", "rejections"), failures))
    return written

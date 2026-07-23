from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from path_utils import coverage_sample_id, expand_user_path
from sem_bead_viewer import BEAD_SIZE_METRICS, bead_size_metric_label, normalize_size_distribution_metric
from tabular_export import natural_sort_key, sort_paths, write_csv_table, write_xlsx_workbook


OUTPUTS_DIR = Path(r"/home/vitpavelka/Projects/sem_coverage/testData/outputs2")
BEAD_CSV_NAME = "bead_global_summaries.csv"
COVERAGE_CSV_NAME = "coverage_global_summaries.csv"
COVERAGE_ROI_CSV_NAME = "coverage_roi_details.csv"
COVERAGE_IMAGE_CSV_NAME = "coverage_image_summaries.csv"
COVERAGE_LOCAL_CELL_CSV_NAME = "coverage_local_cell_details.csv"
COVERAGE_RADIAL_PROFILE_CSV_NAME = "coverage_radial_profile_details.csv"
COVERAGE_POLAR_SECTOR_CSV_NAME = "coverage_polar_sector_details.csv"
COVERAGE_POLAR_ROTATION_CSV_NAME = "coverage_polar_rotation_summaries.csv"
COVERAGE_FAILURE_CSV_NAME = "coverage_failures.csv"
SEM_WORKBOOK_NAME = "sem_global_summaries.xlsx"
BEAD_HISTOGRAM_DIR_NAME = "bead_histograms"

_COMPACT_METRICS = (("proj", "projected_fraction"), ("caps", "projected_over_cap_surface"), ("surfw", "surface_weighted_fraction"))
COVERAGE_GLOBAL_COLUMNS = (
    "sample", "source_path", "coverage_branch",
    "input_image_count", "successful_image_count", "failed_image_count", "detected_bead_count", "included_bead_count",
    "sphere_geometry",
    "selected_sphere_diameter_mean_um", "selected_sphere_diameter_sd_um",
    *tuple(item for short, _metric in _COMPACT_METRICS for item in (f"cap_{short}_mean_pct", f"cap_{short}_median_pct", f"cap_{short}_sd_pp")),
    *tuple(item for short, _metric in _COMPACT_METRICS for item in (f"homo_{short}_mean_pct", f"homo_{short}_median_pct", f"homo_{short}_sd_pp")),
    *tuple(item for family in ("rad", "polar", "local_total", "local_residual") for short, _metric in _COMPACT_METRICS for item in (f"{family}_{short}_mean_pp", f"{family}_{short}_median_pp", f"{family}_{short}_between_bead_sd_pp")),
    *tuple(f"cap_sens_{short}_median_pp" for short, _metric in _COMPACT_METRICS),
)
COVERAGE_IMAGE_COLUMNS = (
    "sample", "source_path", "coverage_branch", "detected_bead_count", "included_bead_count", "sphere_geometry", "selected_sphere_diameter_mean_um",
    *tuple(f"cap_{short}_mean_pct" for short, _ in _COMPACT_METRICS),
    *tuple(f"homo_{short}_mean_pct" for short, _ in _COMPACT_METRICS),
    *tuple(f"{family}_{short}_mean_pp" for family in ("rad", "polar", "local_total", "local_residual") for short, _ in _COMPACT_METRICS),
    *tuple(f"cap_sens_{short}_mean_pp" for short, _ in _COMPACT_METRICS),
)
COVERAGE_ROI_COLUMNS = (
    "sample", "source_path", "coverage_branch", "roi_index", "included", "exclusion_reasons", "cap_valid", "homo_valid", "local_heterogeneity_valid", "cap_completeness", "homo_completeness", "sphere_geometry", "diam_xy_mean_um", "diam_eq_um", "diam_inscribed_um", "selected_sphere_diameter_um",
    *tuple(f"cap_{short}_pct" for short, _ in _COMPACT_METRICS),
    *tuple(f"homo_{short}_pct" for short, _ in _COMPACT_METRICS),
    *tuple(f"{family}_{short}_pp" for family in ("rad", "polar", "local_total", "local_residual") for short, _ in _COMPACT_METRICS),
    *tuple(f"rad_slope_{short}_pp_per_R" for short, _ in _COMPACT_METRICS),
    *tuple(f"cap_sens_{short}_pp" for short, _ in _COMPACT_METRICS),
)
COVERAGE_RADIAL_PROFILE_COLUMNS = (
    "sample",
    "source_path",
    "coverage_branch",
    "roi_index",
    "radial_band_index",
    "radial_inner_fraction",
    "radial_outer_fraction",
    "radial_center_fraction",
    "valid",
    "completeness",
    "proj_pct",
    "caps_pct",
    "surfw_pct",
)
COVERAGE_LOCAL_CELL_COLUMNS = (
    "sample",
    "source_path",
    "coverage_branch",
    "roi_index",
    "rotation_offset_deg",
    "radial_band_index",
    "polar_sector_index",
    "radial_inner_fraction",
    "radial_outer_fraction",
    "polar_start_deg",
    "polar_end_deg",
    "valid",
    "completeness",
    "reference_pixel_count",
    "ag_pixel_count",
    "proj_pct",
    "caps_pct",
    "surfw_pct",
)
COVERAGE_POLAR_SECTOR_COLUMNS = (
    "sample",
    "source_path",
    "coverage_branch",
    "roi_index",
    "rotation_index",
    "rotation_offset_deg",
    "polar_sector_index",
    "polar_start_deg",
    "polar_end_deg",
    "valid",
    "completeness",
    "proj_pct",
    "caps_pct",
    "surfw_pct",
)
COVERAGE_POLAR_ROTATION_COLUMNS = (
    "sample",
    "source_path",
    "coverage_branch",
    "roi_index",
    "metric",
    "rotation_index",
    "rotation_offset_deg",
    "valid_sector_count",
    "valid_cell_count",
    "polar_sd_pp",
    "polar_mad_pp",
    "local_total_sd_pp",
    "local_total_mad_pp",
    "local_residual_sd_pp",
    "local_residual_mad_pp",
    "reconstructed_coverage_pct",
    "reconstruction_delta_pp",
)


def _stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(np.median(array)), float(np.std(array, ddof=1)) if array.size >= 2 else None


def _resolved_coverage_path(folder: Path, value: object) -> Path:
    """Resolve a serialized coverage path without consulting the CWD twice."""

    candidate = Path(str(value or "")).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (folder / candidate).resolve()


def _canonical_coverage_identity(
    data: Mapping[str, Any],
    image: Mapping[str, Any] | None = None,
) -> tuple[str, str, object]:
    """Return one canonical sample, concrete source, and compact branch ID."""

    folder_text = data.get("folder") or data.get("coverage_source_root") or ""
    folder = Path(str(folder_text)).expanduser().resolve()
    root_text = data.get("coverage_source_root") or folder
    root = Path(str(root_text)).expanduser().resolve()
    selected_file = data.get("coverage_source_file") or data.get("file")

    if image is not None:
        image_value = image.get("source_path") or image.get("file")
        source_path = _resolved_coverage_path(folder, image_value)
        sample_dir = source_path.parent
    elif selected_file:
        selected_path = _resolved_coverage_path(folder, selected_file)
        sample_dir = selected_path.parent
        source_path = sample_dir
    else:
        source_path = folder
        sample_dir = folder

    stored_sample = data.get("sample")
    if stored_sample not in (None, "", ".", "root"):
        sample = str(stored_sample).replace("\\", "/")
    else:
        sample = coverage_sample_id(coverage_root=root, sample_dir=sample_dir)
    branch = data.get("coverage_branch_id")
    if branch is None and image is not None:
        branch = image.get("coverage_branch_id")
    return sample, str(source_path), branch


def build_compact_coverage_global_row(data: Mapping[str, Any], *, json_path: Path) -> dict[str, object]:
    """Build the deliberate scientific sample schema from rich coverage JSON."""
    global_summary = data.get("global_summary") or {}
    config = data.get("viewer_config") or {}
    sample, source_path, branch = _canonical_coverage_identity(data)
    rois = [roi for image in data.get("images", []) for roi in image.get("rois", [])]
    included = [roi for roi in rois if roi.get("included_in_global_summary")]
    local_valid = [
        roi
        for roi in included
        if (roi.get("local_heterogeneity") or {}).get("valid") is True
    ]
    row: dict[str, object] = {column: None for column in COVERAGE_GLOBAL_COLUMNS}
    row.update({
        "sample": sample, "source_path": source_path,
        "coverage_branch": branch or global_summary.get("coverage_branch_id"),
        "input_image_count": global_summary.get("input_image_count", len(data.get("images", [])) + len(data.get("failed_images", []))),
        "successful_image_count": global_summary.get("image_count", len(data.get("images", []))),
        "failed_image_count": global_summary.get("failed_image_count", len(data.get("failed_images", []))),
        "detected_bead_count": len(rois) if rois else global_summary.get("total_roi_count"),
        "included_bead_count": len(included) if rois else global_summary.get("included_roi_count"),
        "sphere_geometry": config.get("sphere_diameter_metric") or global_summary.get("sphere_diameter_metric"),
    })
    diameters = [float(roi["sphere_diameter_m"]) * 1e6 for roi in included if roi.get("sphere_diameter_m") is not None]
    row["selected_sphere_diameter_mean_um"], _median, row["selected_sphere_diameter_sd_um"] = _stats(diameters)
    for short, metric in _COMPACT_METRICS:
        cap = [float(roi[f"cap_{metric}_percent"]) for roi in included if roi.get("coverage_cap_valid") and roi.get(f"cap_{metric}_percent") is not None]
        domain = [float(roi[f"homogeneity_domain_{metric}_percent"]) for roi in included if roi.get("homogeneity_domain_valid") and roi.get(f"homogeneity_domain_{metric}_percent") is not None]
        for prefix, values in (("cap", cap), ("homo", domain)):
            mean, median, sd = _stats(values); row[f"{prefix}_{short}_mean_pct"] = mean; row[f"{prefix}_{short}_median_pct"] = median; row[f"{prefix}_{short}_sd_pp"] = sd
        radial = [((roi.get("local_heterogeneity") or {}).get(f"{metric}_radial_weighted_sd_pp")) for roi in local_valid]
        polar = [((roi.get("local_heterogeneity") or {}).get(f"{metric}_polar_sd_median_pp")) for roi in local_valid]
        total = [
            (roi.get("local_heterogeneity") or {}).get(
                f"{metric}_total_local_sd_median_pp"
            )
            for roi in local_valid
        ]
        residual = [
            (roi.get("local_heterogeneity") or {}).get(
                f"{metric}_residual_sd_median_pp"
            )
            for roi in local_valid
        ]
        for prefix, values in (("rad", radial), ("polar", polar), ("local_total", total), ("local_residual", residual)):
            mean, median, sd = _stats([float(value) for value in values if value is not None]); row[f"{prefix}_{short}_mean_pp"] = mean; row[f"{prefix}_{short}_median_pp"] = median; row[f"{prefix}_{short}_between_bead_sd_pp"] = sd
        sensitivity = [
            roi.get(f"{metric}_sensitivity_q10_q90_half_width_pp")
            for roi in included
            if roi.get("coverage_cap_valid")
        ]
        values = [float(value) for value in sensitivity if value is not None]
        row[f"cap_sens_{short}_median_pp"] = float(np.median(values)) if values else None
    return row


def _compact_identification(data: Mapping[str, Any], image: Mapping[str, Any]) -> tuple[str, str, object]:
    """Return the canonical serialized sample and concrete image source path."""

    return _canonical_coverage_identity(data, image)


def build_compact_coverage_image_row(data: Mapping[str, Any], image: Mapping[str, Any]) -> dict[str, object]:
    sample, source, branch = _compact_identification(data, image)
    rois = [roi for roi in image.get("rois", []) if roi.get("included_in_global_summary")]
    local_valid = [
        roi
        for roi in rois
        if (roi.get("local_heterogeneity") or {}).get("valid") is True
    ]
    row = {column: None for column in COVERAGE_IMAGE_COLUMNS}
    row.update({"sample": sample, "source_path": source, "coverage_branch": branch, "detected_bead_count": image.get("roi_count", len(image.get("rois", []))), "included_bead_count": image.get("included_roi_count", len(rois)), "sphere_geometry": (data.get("viewer_config") or {}).get("sphere_diameter_metric")})
    diameters = [float(roi.get("selected_sphere_diameter_um")) for roi in rois if roi.get("selected_sphere_diameter_um") is not None]
    row["selected_sphere_diameter_mean_um"] = float(np.mean(diameters)) if diameters else None
    for short, metric in _COMPACT_METRICS:
        cap = [float(roi[f"cap_{metric}_percent"]) for roi in rois if roi.get("coverage_cap_valid") and roi.get(f"cap_{metric}_percent") is not None]
        homo = [float(roi[f"homogeneity_domain_{metric}_percent"]) for roi in rois if roi.get("homogeneity_domain_valid") and roi.get(f"homogeneity_domain_{metric}_percent") is not None]
        row[f"cap_{short}_mean_pct"] = float(np.mean(cap)) if cap else None; row[f"homo_{short}_mean_pct"] = float(np.mean(homo)) if homo else None
        values = {"rad": [(roi.get("local_heterogeneity") or {}).get(f"{metric}_radial_weighted_sd_pp") for roi in local_valid], "polar": [(roi.get("local_heterogeneity") or {}).get(f"{metric}_polar_sd_median_pp") for roi in local_valid], "local_total": [(roi.get("local_heterogeneity") or {}).get(f"{metric}_total_local_sd_median_pp") for roi in local_valid], "local_residual": [(roi.get("local_heterogeneity") or {}).get(f"{metric}_residual_sd_median_pp") for roi in local_valid]}
        for family, items in values.items():
            good = [float(item) for item in items if item is not None]; row[f"{family}_{short}_mean_pp"] = float(np.mean(good)) if good else None
        sens = [float(roi[f"{metric}_sensitivity_q10_q90_half_width_pp"]) for roi in rois if roi.get("coverage_cap_valid") and roi.get(f"{metric}_sensitivity_q10_q90_half_width_pp") is not None]; row[f"cap_sens_{short}_mean_pp"] = float(np.mean(sens)) if sens else None
    return row


def build_compact_coverage_roi_row(data: Mapping[str, Any], image: Mapping[str, Any], roi: Mapping[str, Any]) -> dict[str, object]:
    sample, source, branch = _compact_identification(data, image)
    row = {column: None for column in COVERAGE_ROI_COLUMNS}
    local = roi.get("local_heterogeneity") or {}
    row.update({"sample": sample, "source_path": source, "coverage_branch": branch, "roi_index": roi.get("roi_index"), "included": roi.get("included_in_global_summary"), "exclusion_reasons": "; ".join(roi.get("exclusion_reasons") or []), "cap_valid": roi.get("coverage_cap_valid"), "homo_valid": roi.get("homogeneity_domain_valid"), "local_heterogeneity_valid": local.get("valid", False), "cap_completeness": roi.get("coverage_cap_completeness"), "homo_completeness": roi.get("homogeneity_domain_completeness"), "sphere_geometry": roi.get("sphere_diameter_metric"), "diam_xy_mean_um": roi.get("diam_xy_mean_um"), "diam_eq_um": roi.get("diam_eq_um"), "diam_inscribed_um": roi.get("diam_inscribed_um"), "selected_sphere_diameter_um": roi.get("selected_sphere_diameter_um")})
    for short, metric in _COMPACT_METRICS:
        row[f"cap_{short}_pct"] = roi.get(f"cap_{metric}_percent"); row[f"homo_{short}_pct"] = roi.get(f"homogeneity_domain_{metric}_percent")
        row[f"rad_{short}_pp"] = local.get(f"{metric}_radial_weighted_sd_pp")
        row[f"polar_{short}_pp"] = local.get(f"{metric}_polar_sd_median_pp")
        row[f"local_total_{short}_pp"] = local.get(f"{metric}_total_local_sd_median_pp")
        row[f"local_residual_{short}_pp"] = local.get(f"{metric}_residual_sd_median_pp")
        row[f"rad_slope_{short}_pp_per_R"] = local.get(f"{metric}_radial_slope_pp_per_R")
        row[f"cap_sens_{short}_pp"] = roi.get(f"{metric}_sensitivity_q10_q90_half_width_pp")
    return row


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_summary(data: dict) -> str | None:
    global_summary = data.get("global_summary", {})
    keys = set(global_summary.keys())
    if {
        "mean_diameter_m",
        "sd_diameter_m",
        "total_used_beads",
        "size_distribution_metric",
    } & keys:
        return "bead"
    if {"mean_coverage", "mean_projected_ag_count", "mean_sphere_ag_count_est"} & keys or data.get("coverage_branch_id") or any("rois" in image for image in data.get("images", [])):
        return "coverage"
    return None


def _rows_for_kind(paths: Iterable[Path], kind: str) -> list[dict]:
    rows: list[dict] = []
    for path in sort_paths(list(paths)):
        data = _load_json(path)
        if _classify_summary(data) != kind:
            continue
        row = build_compact_coverage_global_row(data, json_path=path) if kind == "coverage" else {"name": path.stem, "json_file": path.name, **data.get("global_summary", {})}
        rows.append(row)
    return rows


_DETAIL_METRICS = (
    ("proj", "projected_fraction"),
    ("caps", "projected_over_cap_surface"),
    ("surfw", "surface_weighted_fraction"),
)
_DETAIL_SHORT_BY_METRIC = {
    **{metric: short for short, metric in _DETAIL_METRICS},
    "projected_over_surface": "caps",
}


def _exact_detail_row(columns: tuple[str, ...], values: Mapping[str, Any]) -> dict[str, object]:
    """Keep compact detail rows immune to enriched JSON's extra metadata."""

    return {column: values.get(column) for column in columns}


def _metric_percent(record: Mapping[str, Any], metric: str) -> object:
    """Read only canonical joint-grid percentage fields, never legacy aliases."""

    return record.get(f"{metric}_percent")


def _detail_identity(data: Mapping[str, Any], image: Mapping[str, Any], roi: Mapping[str, Any]) -> dict[str, object]:
    sample, source, branch = _compact_identification(data, image)
    return {
        "sample": sample,
        "source_path": source,
        "coverage_branch": branch,
        "roi_index": roi.get("roi_index"),
    }


def build_compact_coverage_radial_profile_row(
    data: Mapping[str, Any], image: Mapping[str, Any], roi: Mapping[str, Any], profile: Mapping[str, Any],
) -> dict[str, object]:
    values = {
        **_detail_identity(data, image, roi),
        "radial_band_index": profile.get("radial_band_index"),
        "radial_inner_fraction": profile.get("radial_inner_fraction"),
        "radial_outer_fraction": profile.get("radial_outer_fraction"),
        "radial_center_fraction": profile.get("radial_center_fraction"),
        "valid": profile.get("valid"),
        "completeness": profile.get("completeness"),
    }
    for short, metric in _DETAIL_METRICS:
        values[f"{short}_pct"] = _metric_percent(profile, metric)
    return _exact_detail_row(COVERAGE_RADIAL_PROFILE_COLUMNS, values)


def build_compact_coverage_local_cell_row(
    data: Mapping[str, Any], image: Mapping[str, Any], roi: Mapping[str, Any], cell: Mapping[str, Any],
) -> dict[str, object]:
    local = roi.get("local_heterogeneity") or {}
    values = {
        **_detail_identity(data, image, roi),
        "rotation_offset_deg": cell.get("rotation_offset_deg", local.get("display_rotation_deg")),
        "radial_band_index": cell.get("radial_index", cell.get("radial_band_index")),
        "polar_sector_index": cell.get("sector_index", cell.get("polar_sector_index")),
        "radial_inner_fraction": cell.get("inner_radius_fraction", cell.get("radial_inner_fraction")),
        "radial_outer_fraction": cell.get("outer_radius_fraction", cell.get("radial_outer_fraction")),
        "polar_start_deg": cell.get("start_angle_deg", cell.get("polar_start_deg")),
        "polar_end_deg": cell.get("end_angle_deg", cell.get("polar_end_deg")),
        "valid": cell.get("valid"),
        "completeness": cell.get("completeness"),
        "reference_pixel_count": cell.get("reference_pixel_count"),
        "ag_pixel_count": cell.get("ag_pixel_count"),
    }
    for short, metric in _DETAIL_METRICS:
        values[f"{short}_pct"] = _metric_percent(cell, metric)
    return _exact_detail_row(COVERAGE_LOCAL_CELL_COLUMNS, values)


def _sector_group_rows(
    data: Mapping[str, Any], image: Mapping[str, Any], roi: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Reduce serialized robust cells to one numerator/denominator sector row.

    New summaries serialize one compact record per sector.  Grouping here also
    accepts early new-grid JSON that stored one record per radial cell, without
    consulting any legacy polar-sector records.
    """

    grouped: dict[tuple[object, object, object], list[Mapping[str, Any]]] = {}
    for profile in roi.get("local_rotation_sector_profiles", []) or []:
        if not isinstance(profile, Mapping):
            continue
        key = (
            profile.get("rotation_index"),
            profile.get("rotation_offset_deg"),
            profile.get("polar_sector_index"),
        )
        grouped.setdefault(key, []).append(profile)
    rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (
        -1 if item[0] is None else int(item[0]),
        -1 if item[2] is None else int(item[2]),
    )):
        entries = grouped[key]
        first = entries[0]
        valid_entries = [entry for entry in entries if entry.get("valid")]
        values: dict[str, object] = {
            **_detail_identity(data, image, roi),
            "rotation_index": first.get("rotation_index"),
            "rotation_offset_deg": first.get("rotation_offset_deg"),
            "polar_sector_index": first.get("polar_sector_index"),
            "polar_start_deg": first.get("polar_start_deg"),
            "polar_end_deg": first.get("polar_end_deg"),
            "valid": bool(valid_entries),
            "completeness": first.get("completeness"),
        }
        for short, metric in _DETAIL_METRICS:
            numerator = sum(float(entry.get(f"{metric}_numerator") or 0.0) for entry in valid_entries)
            denominator = sum(float(entry.get(f"{metric}_denominator") or 0.0) for entry in valid_entries)
            values[f"{short}_pct"] = numerator / denominator * 100.0 if denominator > 0 else None
        rows.append(_exact_detail_row(COVERAGE_POLAR_SECTOR_COLUMNS, values))
    return rows


def build_compact_coverage_polar_rotation_row(
    data: Mapping[str, Any], image: Mapping[str, Any], roi: Mapping[str, Any], rotation: Mapping[str, Any],
) -> dict[str, object] | None:
    short_metric = _DETAIL_SHORT_BY_METRIC.get(str(rotation.get("metric") or ""))
    if short_metric is None:
        return None
    values = {
        **_detail_identity(data, image, roi),
        "metric": short_metric,
        "rotation_index": rotation.get("rotation_index"),
        "rotation_offset_deg": rotation.get("rotation_offset_deg"),
        "valid_sector_count": rotation.get("valid_sector_count"),
        "valid_cell_count": rotation.get("valid_cell_count"),
        "polar_sd_pp": rotation.get("polar_sd_pp"),
        "polar_mad_pp": rotation.get("polar_mad_pp"),
        "local_total_sd_pp": rotation.get("local_total_sd_pp"),
        "local_total_mad_pp": rotation.get("local_total_mad_pp"),
        "local_residual_sd_pp": rotation.get("local_residual_sd_pp"),
        "local_residual_mad_pp": rotation.get("local_residual_mad_pp"),
        "reconstructed_coverage_pct": rotation.get("reconstructed_coverage_pct"),
        "reconstruction_delta_pp": rotation.get("reconstruction_delta_pp"),
    }
    return _exact_detail_row(COVERAGE_POLAR_ROTATION_COLUMNS, values)


def _coverage_detail_rows(paths: Iterable[Path]) -> dict[str, list[dict]]:
    """Build standard coverage details only from serialized joint-grid data."""

    output = {
        "images": [],
        "rois": [],
        "cells": [],
        "radial_profiles": [],
        "sectors": [],
        "rotations": [],
        "failures": [],
    }
    for path in sort_paths(list(paths)):
        data = _load_json(path)
        if _classify_summary(data) != "coverage":
            continue
        for failure in data.get("failed_images", []):
            output["failures"].append({"name": path.stem, "json_file": path.name, **failure})
        for image in data.get("images", []):
            output["images"].append(build_compact_coverage_image_row(data, image))
            for roi in image.get("rois", []):
                output["rois"].append(build_compact_coverage_roi_row(data, image, roi))
                local = roi.get("local_heterogeneity") or {}
                for profile in local.get("radial_profile_details", []) or []:
                    if isinstance(profile, Mapping):
                        output["radial_profiles"].append(
                            build_compact_coverage_radial_profile_row(data, image, roi, profile)
                        )
                for cell in roi.get("local_grid_cells", []) or []:
                    if isinstance(cell, Mapping):
                        output["cells"].append(
                            build_compact_coverage_local_cell_row(data, image, roi, cell)
                        )
                output["sectors"].extend(_sector_group_rows(data, image, roi))
                for rotation in roi.get("local_rotation_summaries", []) or []:
                    if isinstance(rotation, Mapping):
                        row = build_compact_coverage_polar_rotation_row(data, image, roi, rotation)
                        if row is not None:
                            output["rotations"].append(row)
    return output


def export_table_summaries(
    outputs_dir: Path = OUTPUTS_DIR,
    *,
    bead: bool = True,
    coverage: bool = True,
    table_format: str = "csv",
    bead_csv: bool = True,
    coverage_csv: bool = True,
    sort_by: str = "name",
) -> list[Path]:
    json_paths = sort_paths(list(outputs_dir.glob("*.json")), sort_by=sort_by)
    written: list[Path] = []
    bead_rows = _rows_for_kind(json_paths, "bead") if bead else []
    coverage_rows = _rows_for_kind(json_paths, "coverage") if coverage else []
    coverage_details = _coverage_detail_rows(json_paths) if coverage else {
        "images": [], "rois": [], "cells": [], "radial_profiles": [],
        "sectors": [], "rotations": [], "failures": [],
    }
    if table_format in {"csv", "both"}:
        if bead and bead_csv:
            bead_csv_path = write_csv_table(
                bead_rows,
                outputs_dir / BEAD_CSV_NAME,
                preferred_columns=("name", "json_file", "size_distribution_metric"),
                sort_by=sort_by,
            )
            if bead_csv_path is not None:
                written.append(bead_csv_path)
        if coverage and coverage_csv:
            coverage_csv_path = write_csv_table(
                coverage_rows,
                outputs_dir / COVERAGE_CSV_NAME,
                exact_columns=COVERAGE_GLOBAL_COLUMNS,
                sort_by=sort_by,
            )
            if coverage_csv_path is not None:
                written.append(coverage_csv_path)
        if coverage and coverage_csv:
            for filename, key, exact in (
                (COVERAGE_IMAGE_CSV_NAME, "images", COVERAGE_IMAGE_COLUMNS),
                (COVERAGE_ROI_CSV_NAME, "rois", COVERAGE_ROI_COLUMNS),
                (COVERAGE_LOCAL_CELL_CSV_NAME, "cells", COVERAGE_LOCAL_CELL_COLUMNS),
                (COVERAGE_RADIAL_PROFILE_CSV_NAME, "radial_profiles", COVERAGE_RADIAL_PROFILE_COLUMNS),
                (COVERAGE_POLAR_SECTOR_CSV_NAME, "sectors", COVERAGE_POLAR_SECTOR_COLUMNS),
                (COVERAGE_POLAR_ROTATION_CSV_NAME, "rotations", COVERAGE_POLAR_ROTATION_COLUMNS),
                (COVERAGE_FAILURE_CSV_NAME, "failures", None),
            ):
                detail_path = write_csv_table(
                    coverage_details[key],
                    outputs_dir / filename,
                    preferred_columns=("name", "sample", "file", "error"),
                    exact_columns=exact,
                    sort_by=sort_by,
                )
                if detail_path is not None:
                    written.append(detail_path)
    if table_format in {"xlsx", "both"}:
        workbook_sheets: dict[str, list[dict]] = {}
        preferred_columns: dict[str, tuple[str, ...]] = {}
        exact_sheet_columns: dict[str, tuple[str, ...]] = {}
        coverage_number_format_sheets: list[str] = []
        if bead and bead_rows:
            workbook_sheets["Beads"] = bead_rows
            preferred_columns["Beads"] = ("name", "json_file", "size_distribution_metric")
        if coverage and coverage_rows:
            coverage_sheet = "Coverage samples" if not bead else "Coverage"
            workbook_sheets[coverage_sheet] = coverage_rows
            preferred_columns[coverage_sheet] = COVERAGE_GLOBAL_COLUMNS
            exact_sheet_columns[coverage_sheet] = COVERAGE_GLOBAL_COLUMNS
            coverage_number_format_sheets.append(coverage_sheet)
        if coverage:
            for sheet, key, exact in (
                ("Coverage images", "images", COVERAGE_IMAGE_COLUMNS),
                ("Coverage ROIs", "rois", COVERAGE_ROI_COLUMNS),
                ("Local cells", "cells", COVERAGE_LOCAL_CELL_COLUMNS),
                ("Radial profile", "radial_profiles", COVERAGE_RADIAL_PROFILE_COLUMNS),
                ("Polar sectors", "sectors", COVERAGE_POLAR_SECTOR_COLUMNS),
                ("Polar rotations", "rotations", COVERAGE_POLAR_ROTATION_COLUMNS),
                ("Coverage failures", "failures", ("name", "sample", "file", "error")),
            ):
                if coverage_details[key]:
                    workbook_sheets[sheet] = coverage_details[key]
                    preferred_columns[sheet] = exact
                    coverage_number_format_sheets.append(sheet)
                    if key != "failures":
                        exact_sheet_columns[sheet] = exact
        workbook_name = "coverage_summaries.xlsx" if coverage and not bead else SEM_WORKBOOK_NAME
        workbook_path = write_xlsx_workbook(
            workbook_sheets,
            outputs_dir / workbook_name,
            preferred_columns=preferred_columns,
            exact_columns=exact_sheet_columns,
            semantic_number_format_sheets=coverage_number_format_sheets,
            sort_by=sort_by,
        )
        if workbook_path is not None:
            written.append(workbook_path)
    return written


def export_csv_summaries(
    outputs_dir: Path = OUTPUTS_DIR,
    bead: bool = True,
    coverage: bool = True,
    *,
    sort_by: str = "name",
) -> list[Path]:
    """Backward-compatible CSV export wrapper."""

    return export_table_summaries(
        outputs_dir,
        bead=bead,
        coverage=coverage,
        table_format="csv",
        bead_csv=True,
        coverage_csv=True,
        sort_by=sort_by,
    )


def _summary_metric(data: dict) -> str:
    """Resolve an explicit new metric or legacy mean-X/Y summary convention."""

    for mapping in (data.get("global_summary", {}), data.get("viewer_config", {})):
        if isinstance(mapping, dict) and mapping.get("size_distribution_metric"):
            return normalize_size_distribution_metric(mapping["size_distribution_metric"])
    images = data.get("images", [])
    legacy_metric = next(
        (image.get("histogram_metric") for image in images if isinstance(image, dict) and image.get("histogram_metric")),
        None,
    )
    if legacy_metric == "mean_xy_diameter" or any(
        isinstance(image, dict) and "mean_xy_diameters_px" in image for image in images
    ):
        return "mean_xy_diameter"
    return "equivalent_diameter"


def bead_histogram_values(data: dict) -> tuple[np.ndarray, str, str]:
    """Return values, TXT header, and normalized metric for one bead summary."""

    metric = _summary_metric(data)
    vals_m: list[float] = []
    vals_px: list[float] = []
    key_m = f"{metric}s_m" if metric == "equivalent_diameter" else "mean_xy_diameters_m"
    key_px = f"{metric}s_px" if metric == "equivalent_diameter" else "mean_xy_diameters_px"
    for image in data.get("images", []):
        if not isinstance(image, dict):
            continue
        # New summaries carry explicit selected vectors. Old mean-X/Y
        # summaries used ``diameters_m`` plus mean-X/Y pixel values.
        values_m = image.get("selected_diameters_m")
        values_px = image.get("selected_diameters_px")
        if values_m is None:
            values_m = image.get(key_m)
            if values_m is None and metric == "mean_xy_diameter":
                values_m = image.get("diameters_m", [])
        if values_px is None:
            values_px = image.get(key_px, [])
        vals_m.extend(float(value) for value in values_m or [] if value is not None and math.isfinite(float(value)))
        vals_px.extend(float(value) for value in values_px or [] if value is not None and math.isfinite(float(value)))
    if vals_m:
        return np.asarray(vals_m, dtype=np.float64) * 1e6, f"{metric}_um", metric
    return np.asarray(vals_px, dtype=np.float64), f"{metric}_px", metric


def _bead_diameters_um(data: dict) -> np.ndarray:
    """Backward-compatible calibrated histogram-vector helper."""

    values, header, _metric = bead_histogram_values(data)
    return values if header.endswith("_um") else np.array([], dtype=np.float64)


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_", ".") else "_" for ch in name)
    return "_".join(safe.split())


def _format_stats_text(values: np.ndarray, unit: str) -> str:
    if values.size == 0:
        return "No valid diameters"
    mean = float(np.mean(values))
    median = float(np.median(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else float("nan")
    return "\n".join(
        [
            f"n = {values.size}",
            f"mean = {mean:.3f} {unit}",
            f"median = {median:.3f} {unit}",
            f"SD = {sd:.3f} {unit}",
            f"CV = {cv:.1f} %",
        ]
    )


def _plot_bead_histogram(values: np.ndarray, title: str, output_path: Path, header: str = "equivalent_diameter_um") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if values.size:
        bins = min(max(8, int(np.sqrt(values.size))), 40)
        ax.hist(values, bins=bins, color="#4cc9f0", edgecolor="#0b1f2a", alpha=0.9)
        ax.axvline(float(np.mean(values)), color="#d00000", linewidth=1.5)
        ax.axvline(float(np.median(values)), color="#2d6a4f", linewidth=1.5, linestyle="--")
    else:
        ax.text(0.5, 0.5, "No valid diameters", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    unit = "µm" if header.endswith("_um") else "px"
    metric = header.removesuffix("_um").removesuffix("_px")
    ax.set_xlabel(bead_size_metric_label(metric, unit))
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98,
        0.98,
        _format_stats_text(values, unit),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.85, "pad": 6},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_histogram_values(values: np.ndarray, header: str, output_path: Path) -> Path:
    """Write the same immutable vector supplied to the matching histogram."""

    lines = [header]
    lines.extend(format(float(value), ".17g") for value in values)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def export_bead_histograms(outputs_dir: Path = OUTPUTS_DIR, hist_dir: Path | None = None) -> list[Path]:
    hist_dir = hist_dir or outputs_dir / BEAD_HISTOGRAM_DIR_NAME
    hist_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    combined: dict[str, list[np.ndarray]] = {
        f"{metric}_{unit}": [] for metric in BEAD_SIZE_METRICS for unit in ("um", "px")
    }

    for json_path in sort_paths(list(outputs_dir.glob("*.json"))):
        data = _load_json(json_path)
        if _classify_summary(data) != "bead":
            continue
        values, header, metric = bead_histogram_values(data)
        combined[header].append(values)
        unit_suffix = "um" if header.endswith("_um") else "px"
        stem = f"{_safe_name(json_path.stem)}_bead_{metric}_{unit_suffix}"
        out_path = hist_dir / f"{stem}_histogram.png"
        _plot_bead_histogram(values, json_path.stem, out_path, header)
        written.append(out_path)
        written.append(_write_histogram_values(values, header, hist_dir / f"{stem}_values.txt"))

    for header, vectors in combined.items():
        if not vectors:
            continue
        all_vals = np.concatenate([vals for vals in vectors if vals.size]) if any(vals.size for vals in vectors) else np.array([], dtype=np.float64)
        unit_suffix = "um" if header.endswith("_um") else "px"
        metric = header.removesuffix("_um").removesuffix("_px")
        stem = f"all_bead_{metric}_{unit_suffix}"
        out_path = hist_dir / f"{stem}_histogram.png"
        _plot_bead_histogram(all_vals, "All bead outputs", out_path, header)
        written.append(out_path)
        written.append(_write_histogram_values(all_vals, header, hist_dir / f"{stem}_values.txt"))
    return written


def export_outputs(
    outputs_dir: Path = OUTPUTS_DIR,
    *,
    csv: bool = True,
    bead: bool = True,
    coverage: bool = True,
    bead_csv: bool = True,
    coverage_csv: bool = True,
    histograms: bool = True,
    table_format: str = "csv",
    sort_by: str = "name",
) -> list[Path]:
    written: list[Path] = []
    if table_format in {"csv", "xlsx", "both"}:
        csv_format = table_format if csv else ("xlsx" if table_format == "both" else "none")
        if csv_format != "none":
            written.extend(
                export_table_summaries(
                    outputs_dir,
                    bead=bead,
                    coverage=coverage,
                    table_format=csv_format,
                    bead_csv=bead_csv,
                    coverage_csv=coverage_csv,
                    sort_by=sort_by,
                )
            )
    if histograms and bead:
        written.extend(export_bead_histograms(outputs_dir))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export SEM output JSON global summaries to CSV, XLSX, or both, "
            "and optionally regenerate bead size histograms."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Directory containing output JSON files.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV summary files.")
    parser.add_argument("--no-bead-csv", action="store_true", help="Do not write bead_global_summaries.csv.")
    parser.add_argument("--no-coverage-csv", action="store_true", help="Do not write coverage_global_summaries.csv.")
    parser.add_argument("--no-histograms", action="store_true", help="Do not write bead diameter histogram PNGs.")
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
        help="Deterministic natural sorting for summary rows. Default: %(default)s",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    written = export_outputs(
        expand_user_path(args.outputs_dir),
        csv=not args.no_csv,
        bead_csv=not args.no_bead_csv,
        coverage_csv=not args.no_coverage_csv,
        histograms=not args.no_histograms,
        table_format=args.table_format,
        sort_by=args.sort_by,
    )
    for path in sort_paths(list(written), sort_by="path"):
        print(path)


if __name__ == "__main__":
    main()

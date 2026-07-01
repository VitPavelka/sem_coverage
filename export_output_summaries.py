from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from path_utils import expand_user_path
from tabular_export import natural_sort_key, sort_paths, write_csv_table, write_xlsx_workbook


OUTPUTS_DIR = Path(r"/home/vitpavelka/Projects/sem_coverage/testData/outputs2")
BEAD_CSV_NAME = "bead_global_summaries.csv"
COVERAGE_CSV_NAME = "coverage_global_summaries.csv"
SEM_WORKBOOK_NAME = "sem_global_summaries.xlsx"
BEAD_HISTOGRAM_DIR_NAME = "bead_histograms"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_summary(data: dict) -> str | None:
    global_summary = data.get("global_summary", {})
    keys = set(global_summary.keys())
    if {"mean_diameter_m", "sd_diameter_m", "total_used_beads"} & keys:
        return "bead"
    if {"mean_coverage", "mean_projected_ag_count", "mean_sphere_ag_count_est"} & keys:
        return "coverage"
    return None


def _rows_for_kind(paths: Iterable[Path], kind: str) -> list[dict]:
    rows: list[dict] = []
    for path in sort_paths(list(paths)):
        data = _load_json(path)
        if _classify_summary(data) != kind:
            continue
        row = {
            "name": path.stem,
            "json_file": path.name,
            **data.get("global_summary", {}),
        }
        rows.append(row)
    return rows


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
    if table_format in {"csv", "both"}:
        if bead and bead_csv:
            bead_csv_path = write_csv_table(
                bead_rows,
                outputs_dir / BEAD_CSV_NAME,
                preferred_columns=("name", "json_file"),
                sort_by=sort_by,
            )
            if bead_csv_path is not None:
                written.append(bead_csv_path)
        if coverage and coverage_csv:
            coverage_csv_path = write_csv_table(
                coverage_rows,
                outputs_dir / COVERAGE_CSV_NAME,
                preferred_columns=("name", "json_file"),
                sort_by=sort_by,
            )
            if coverage_csv_path is not None:
                written.append(coverage_csv_path)
    if table_format in {"xlsx", "both"}:
        workbook_sheets: dict[str, list[dict]] = {}
        preferred_columns: dict[str, tuple[str, ...]] = {}
        if bead and bead_rows:
            workbook_sheets["Beads"] = bead_rows
            preferred_columns["Beads"] = ("name", "json_file")
        if coverage and coverage_rows:
            workbook_sheets["Coverage"] = coverage_rows
            preferred_columns["Coverage"] = ("name", "json_file")
        workbook_path = write_xlsx_workbook(
            workbook_sheets,
            outputs_dir / SEM_WORKBOOK_NAME,
            preferred_columns=preferred_columns,
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


def _bead_diameters_um(data: dict) -> np.ndarray:
    vals_m = []
    for image in data.get("images", []):
        vals_m.extend(v for v in image.get("diameters_m", []) if v is not None)
    vals_um = [float(v) * 1e6 for v in vals_m if math.isfinite(float(v))]
    return np.array(vals_um, dtype=np.float64)


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_", ".") else "_" for ch in name)
    return "_".join(safe.split())


def _format_stats_text(vals_um: np.ndarray) -> str:
    if vals_um.size == 0:
        return "No valid diameters"
    mean = float(np.mean(vals_um))
    median = float(np.median(vals_um))
    sd = float(np.std(vals_um, ddof=1)) if vals_um.size > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else float("nan")
    return "\n".join(
        [
            f"n = {vals_um.size}",
            f"mean = {mean:.3f} um",
            f"median = {median:.3f} um",
            f"SD = {sd:.3f} um",
            f"CV = {cv:.1f} %",
        ]
    )


def _plot_bead_histogram(vals_um: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if vals_um.size:
        bins = min(max(8, int(np.sqrt(vals_um.size))), 40)
        ax.hist(vals_um, bins=bins, color="#4cc9f0", edgecolor="#0b1f2a", alpha=0.9)
        ax.axvline(float(np.mean(vals_um)), color="#d00000", linewidth=1.5)
        ax.axvline(float(np.median(vals_um)), color="#2d6a4f", linewidth=1.5, linestyle="--")
    else:
        ax.text(0.5, 0.5, "No valid diameters", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("Diameter [um]")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98,
        0.98,
        _format_stats_text(vals_um),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.85, "pad": 6},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_bead_histograms(outputs_dir: Path = OUTPUTS_DIR, hist_dir: Path | None = None) -> list[Path]:
    hist_dir = hist_dir or outputs_dir / BEAD_HISTOGRAM_DIR_NAME
    hist_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    combined: list[np.ndarray] = []

    for json_path in sort_paths(list(outputs_dir.glob("*.json"))):
        data = _load_json(json_path)
        if _classify_summary(data) != "bead":
            continue
        vals_um = _bead_diameters_um(data)
        combined.append(vals_um)
        out_path = hist_dir / f"{_safe_name(json_path.stem)}_diameter_histogram.png"
        _plot_bead_histogram(vals_um, json_path.stem, out_path)
        written.append(out_path)

    if combined:
        all_vals = np.concatenate([vals for vals in combined if vals.size]) if any(vals.size for vals in combined) else np.array([], dtype=np.float64)
        out_path = hist_dir / "all_bead_diameter_histogram.png"
        _plot_bead_histogram(all_vals, "All bead outputs", out_path)
        written.append(out_path)
    return written


def export_outputs(
    outputs_dir: Path = OUTPUTS_DIR,
    *,
    csv: bool = True,
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
                    bead=True,
                    coverage=True,
                    table_format=csv_format,
                    bead_csv=bead_csv,
                    coverage_csv=coverage_csv,
                    sort_by=sort_by,
                )
            )
    if histograms:
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

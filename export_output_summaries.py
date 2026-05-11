from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUTS_DIR = Path(r"C:\Users\pavel\Desktop\AVCR\codes\sem_coverage\testData\outputs")
BEAD_CSV_NAME = "bead_global_summaries.csv"
COVERAGE_CSV_NAME = "coverage_global_summaries.csv"
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
    for path in sorted(paths):
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


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = ["name", "json_file"]
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extra_keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_csv_summaries(outputs_dir: Path = OUTPUTS_DIR, bead: bool = True, coverage: bool = True) -> list[Path]:
    json_paths = sorted(outputs_dir.glob("*.json"))
    written: list[Path] = []
    if bead:
        bead_rows = _rows_for_kind(json_paths, "bead")
        bead_csv = outputs_dir / BEAD_CSV_NAME
        _write_csv(bead_rows, bead_csv)
        written.append(bead_csv)
    if coverage:
        coverage_rows = _rows_for_kind(json_paths, "coverage")
        coverage_csv = outputs_dir / COVERAGE_CSV_NAME
        _write_csv(coverage_rows, coverage_csv)
        written.append(coverage_csv)
    return written


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

    for json_path in sorted(outputs_dir.glob("*.json")):
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
) -> list[Path]:
    written: list[Path] = []
    if csv:
        written.extend(export_csv_summaries(outputs_dir, bead=bead_csv, coverage=coverage_csv))
    if histograms:
        written.extend(export_bead_histograms(outputs_dir))
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SEM output JSON global summaries and bead size histograms.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR, help="Directory containing output JSON files.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV summary files.")
    parser.add_argument("--no-bead-csv", action="store_true", help="Do not write bead_global_summaries.csv.")
    parser.add_argument("--no-coverage-csv", action="store_true", help="Do not write coverage_global_summaries.csv.")
    parser.add_argument("--no-histograms", action="store_true", help="Do not write bead diameter histogram PNGs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    written = export_outputs(
        args.outputs_dir,
        csv=not args.no_csv,
        bead_csv=not args.no_bead_csv,
        coverage_csv=not args.no_coverage_csv,
        histograms=not args.no_histograms,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

"""Command-line entry point for supplementary central-cap coverage sweeps."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coverage_cap_sweep import parse_fractions, run_cap_sweep
from path_utils import expand_user_path, resolve_existing_input_path
from sem_coverage_viewer import _resolve_image_paths, load_app_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate central coverage caps after one segmentation per SEM image.")
    parser.add_argument("--config", type=Path, default=Path("sem_coverage_viewer_config.json"))
    parser.add_argument("--file", type=Path, help="One TIFF image; overrides the config file selection.")
    parser.add_argument("--folder", type=Path, help="Temporarily override the configured folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("coverage_cap_sweep_output"))
    parser.add_argument("--fractions", help="Comma-separated cap radius fractions, for example 0.10,0.25,0.50.")
    parser.add_argument("--fraction-start", type=float)
    parser.add_argument("--fraction-stop", type=float)
    parser.add_argument("--fraction-step", type=float)
    parser.add_argument("--include-surface-weighted", action="store_true", help="Also calculate the experimental curvature-weighted comparison.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    config_path = expand_user_path(args.config)
    app = load_app_config(config_path)
    folder = expand_user_path(args.folder) if args.folder else resolve_existing_input_path(app.folder, config_path=config_path, description="coverage folder")
    if args.file:
        supplied = expand_user_path(args.file)
        selected = str(supplied.resolve()) if supplied.exists() else str(args.file)
    else:
        selected = app.file
    paths = _resolve_image_paths(folder, selected)
    if not paths:
        raise FileNotFoundError(f"No lowercase .tif files found in '{folder}'.")
    fractions = parse_fractions(args.fractions, args.fraction_start, args.fraction_stop, args.fraction_step)
    written = run_cap_sweep(paths, app.viewer, expand_user_path(args.output_dir), fractions, include_surface_weighted=args.include_surface_weighted)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

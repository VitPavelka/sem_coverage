"""Command-line entry point for the TEM particle viewer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

from tem_particle_viewer import DEFAULT_CONFIG_PATH, run_from_config, setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the TEM particle viewer CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Open the TEM particle viewer. No argument is required; the default "
            f"JSON config is '{DEFAULT_CONFIG_PATH}'. The config provides the image "
            "folder, optional file selection, analysis parameters, display defaults, "
            "and optional summary JSON output. CLI overrides are temporary and do "
            "not modify the source config."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON configuration to load. Default: %(default)s",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Temporarily override the folder from the JSON config.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Temporarily select one input image without modifying the source config.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TEM particle viewer CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(logging, args.log_level))
    try:
        run_from_config(
            args.config,
            folder_override=args.folder,
            file_override=args.file,
        )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

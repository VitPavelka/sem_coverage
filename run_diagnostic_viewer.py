"""Command-line entry point for the unified diagnostic viewer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

from diagnostic_viewer import DIAGNOSTIC_ADAPTERS, DiagnosticViewer


def build_parser() -> argparse.ArgumentParser:
    """Build the diagnostic viewer argument parser."""

    parser = argparse.ArgumentParser(
        description="Interactive parameter diagnostics for image-analysis modes."
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="Diagnostic mode. Currently available: "
        + ", ".join(sorted(DIAGNOSTIC_ADAPTERS)),
    )
    parser.add_argument("--config", required=True, type=Path, help="Analysis JSON config.")
    parser.add_argument(
        "--file", type=Path, help="Start with this image from the configured folder."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Override the configured folder without modifying the source config.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        help="Destination for the explicit Save tuned config action.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Run recalculation synchronously for diagnostics.",
    )
    return parser


def resolve_adapter(mode: str):
    """Resolve a registered adapter or raise a clear CLI-level error."""

    try:
        return DIAGNOSTIC_ADAPTERS[mode]
    except KeyError:
        available = ", ".join(sorted(DIAGNOSTIC_ADAPTERS))
        raise ValueError(
            f"Unsupported diagnostic mode '{mode}'. Available modes: {available}."
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected diagnostic application."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    viewer: DiagnosticViewer | None = None
    try:
        adapter_class = resolve_adapter(args.mode)
        viewer = DiagnosticViewer(
            adapter_class(),
            args.config,
            selected_file=args.file,
            folder_override=args.folder,
            output_config=args.output_config,
            asynchronous=not args.no_async,
        )
        viewer.show()
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        return 130
    finally:
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

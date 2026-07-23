from __future__ import annotations

import os
from pathlib import Path

from tabular_export import sort_paths


def expand_user_path(value: str | Path) -> Path:
    """Expand environment variables, whitespace, and ``~`` in one path value."""

    text = os.path.expandvars(str(value).strip())
    return Path(text).expanduser()


def path_to_config_text(value: str | Path) -> str:
    """Serialize one path-like value to forward-slash JSON text when practical."""

    return str(value).strip().replace("\\", "/")


def coverage_sample_id(*, coverage_root: Path, sample_dir: Path) -> str:
    """Return the canonical human-readable ID for one coverage sample.

    Direct children and nested TIFF-containing directories are identified
    relative to the recursive input root.  When TIFFs live directly in that
    root (including single-file mode), the real directory name is used rather
    than the technical relative path ``'.'``.
    """

    root = expand_user_path(coverage_root).resolve()
    sample = expand_user_path(sample_dir).resolve()
    if sample == root:
        return sample.name
    try:
        relative = sample.relative_to(root)
    except ValueError:
        return sample.name
    text = relative.as_posix()
    return sample.name if text in ("", ".") else text


def resolve_existing_input_path(
    value: str | Path,
    *,
    config_path: str | Path,
    description: str = "input path",
) -> Path:
    """Resolve one existing input path with legacy-CWD priority and config fallback."""

    candidate = expand_user_path(value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"{description.capitalize()} does not exist: '{candidate}'."
        )

    cwd_candidate = candidate
    if cwd_candidate.exists():
        return cwd_candidate

    config_candidate = expand_user_path(config_path).resolve().parent / candidate
    if config_candidate.exists():
        return config_candidate

    raise FileNotFoundError(
        f"{description.capitalize()} does not exist. Tried '{cwd_candidate}' "
        f"and '{config_candidate}'."
    )


def resolve_optional_file_in_folder(
    folder: str | Path,
    file_value: str | Path | None,
    *,
    description: str = "input file",
) -> Path | None:
    """Resolve one optional file path, preferring a path inside ``folder`` when relative."""

    if file_value in (None, ""):
        return None
    candidate = expand_user_path(file_value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"{description.capitalize()} does not exist: '{candidate}'."
        )
    resolved = expand_user_path(folder) / candidate
    if resolved.exists():
        return resolved
    raise FileNotFoundError(
        f"{description.capitalize()} does not exist: '{resolved}'."
    )


def discover_images_recursive(
    root: Path,
    *,
    patterns: tuple[str, ...] = ("*.tif",),
) -> list[Path]:
    """Discover supported images below one selected root deterministically.

    The returned paths are resolved and naturally sorted by their paths
    relative to ``root``.  A selected single file intentionally bypasses this
    helper in callers that support a file-mode data source.
    """

    root = expand_user_path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image folder does not exist: '{root}'.")
    if not root.is_dir():
        raise NotADirectoryError(f"Image folder is not a directory: '{root}'.")
    paths: dict[Path, None] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                paths[path.resolve()] = None
    if not paths:
        suffixes = ", ".join(patterns)
        raise FileNotFoundError(f"No supported images ({suffixes}) found below '{root}'.")
    return sort_paths(list(paths), sort_by="path", root=root)

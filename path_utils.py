from __future__ import annotations

import os
from pathlib import Path

from tabular_export import sort_paths


class ConfiguredSourceError(FileNotFoundError):
    """A configured folder/file source could not be resolved during preflight."""

    def __init__(
        self,
        message: str,
        *,
        config_path: Path,
        configured_folder: str,
        configured_file: str | None,
        resolved_path: Path,
    ) -> None:
        super().__init__(message)
        self.config_path = config_path
        self.configured_folder = configured_folder
        self.configured_file = configured_file
        self.resolved_path = resolved_path


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


def resolve_configured_image_source(
    folder_value: str | Path,
    file_value: str | Path | None,
    *,
    config_path: str | Path,
    description: str = "image source",
    allowed_suffixes: tuple[str, ...] | None = None,
) -> tuple[Path, Path | None]:
    """Resolve and preflight one config-owned folder and optional file.

    Folder resolution deliberately retains :func:`resolve_existing_input_path`
    semantics. A relative file is always interpreted inside the resolved
    folder. Failures include all source fields needed to repair a stale config.
    """

    config = expand_user_path(config_path).resolve()
    configured_folder = str(folder_value).strip()
    configured_file = None if file_value in (None, "") else str(file_value).strip()

    folder_candidate = expand_user_path(configured_folder)
    if folder_candidate.is_absolute():
        attempted_folder = folder_candidate
    else:
        cwd_candidate = folder_candidate.resolve()
        config_candidate = (config.parent / folder_candidate).resolve()
        attempted_folder = cwd_candidate if folder_candidate.exists() else config_candidate

    def fail(reason: str, resolved_path: Path) -> ConfiguredSourceError:
        file_text = (
            ""
            if configured_file is None
            else f", configured file={configured_file!r}"
        )
        return ConfiguredSourceError(
            f"Invalid {description} in config '{config}': configured folder="
            f"{configured_folder!r}{file_text}; resolved path='{resolved_path}'. "
            f"{reason}",
            config_path=config,
            configured_folder=configured_folder,
            configured_file=configured_file,
            resolved_path=resolved_path,
        )

    try:
        folder = resolve_existing_input_path(
            configured_folder,
            config_path=config,
            description=f"{description} folder",
        ).resolve()
    except FileNotFoundError as exc:
        raise fail("The configured folder does not exist.", attempted_folder) from exc
    if not folder.is_dir():
        raise fail("The configured folder is not a directory.", folder)

    if configured_file is None:
        return folder, None

    file_candidate = expand_user_path(configured_file)
    resolved_file = (
        file_candidate.resolve()
        if file_candidate.is_absolute()
        else (folder / file_candidate).resolve()
    )
    if not resolved_file.exists():
        raise fail("The configured file does not exist.", resolved_file)
    if not resolved_file.is_file():
        raise fail("The configured file is not a regular file.", resolved_file)
    if allowed_suffixes is not None:
        normalized_suffixes = {suffix.lower() for suffix in allowed_suffixes}
        if resolved_file.suffix.lower() not in normalized_suffixes:
            expected = ", ".join(sorted(normalized_suffixes))
            raise fail(
                f"The configured file type is unsupported; expected one of: {expected}.",
                resolved_file,
            )
    return folder, resolved_file


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

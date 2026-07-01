from __future__ import annotations

import os
from pathlib import Path


def expand_user_path(value: str | Path) -> Path:
    """Expand environment variables, whitespace, and ``~`` in one path value."""

    text = os.path.expandvars(str(value).strip())
    return Path(text).expanduser()


def path_to_config_text(value: str | Path) -> str:
    """Serialize one path-like value to forward-slash JSON text when practical."""

    return str(value).strip().replace("\\", "/")


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

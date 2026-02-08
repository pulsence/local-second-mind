"""
Global path helpers for Local Second Mind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


GLOBAL_FOLDER_ENV_VAR = "LSM_GLOBAL_FOLDER"


def get_global_folder(override: Optional[str | Path] = None) -> Path:
    """
    Resolve the global LSM folder path.

    Priority:
    1. Explicit override argument
    2. LSM_GLOBAL_FOLDER environment variable
    3. <home>/Local Second Mind
    """
    candidate: str | Path | None = override
    if candidate is None:
        candidate = os.environ.get(GLOBAL_FOLDER_ENV_VAR)
    if candidate is None:
        candidate = Path.home() / "Local Second Mind"
    return Path(candidate).expanduser().resolve()


def get_chats_folder(global_folder: Optional[str | Path] = None) -> Path:
    """Return the global chats folder path."""
    return get_global_folder(global_folder) / "Chats"


def get_mode_chats_folder(
    mode_name: str,
    global_folder: Optional[str | Path] = None,
    base_dir: str | Path = "Chats",
) -> Path:
    """
    Return the chat transcript folder path for a specific mode.
    """
    base_path = Path(base_dir).expanduser()
    if base_path.is_absolute():
        return (base_path / mode_name).resolve()
    return (get_global_folder(global_folder) / base_path / mode_name).resolve()


def get_notes_folder(global_folder: Optional[str | Path] = None) -> Path:
    """Return the global notes folder path."""
    return get_global_folder(global_folder) / "Notes"


def resolve_relative_path(path: str | Path, global_folder: Optional[str | Path] = None) -> Path:
    """
    Resolve a possibly-relative path against the global folder.
    """
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (get_global_folder(global_folder) / value).resolve()


def ensure_global_folders(global_folder: Optional[str | Path] = None) -> None:
    """
    Ensure the default global folder structure exists.
    """
    root = get_global_folder(global_folder)
    for folder in (root, get_chats_folder(root), get_notes_folder(root)):
        folder.mkdir(parents=True, exist_ok=True)

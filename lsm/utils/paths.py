"""
Common path helpers for Local Second Mind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def resolve_path(
    path: Path | str,
    *,
    base_dir: Optional[Path] = None,
    expand_user: bool = True,
    strict: bool = False,
) -> Path:
    """
    Resolve a path, optionally relative to a base directory.
    """
    value = Path(path)
    if expand_user:
        value = value.expanduser()
    if base_dir is not None and not value.is_absolute():
        value = base_dir / value
    if strict:
        return value.resolve()
    return value.resolve(strict=False)


def resolve_paths(
    paths: Iterable[Path | str],
    *,
    base_dir: Optional[Path] = None,
    expand_user: bool = True,
    strict: bool = False,
) -> list[Path]:
    """
    Resolve multiple paths with consistent options.
    """
    return [
        resolve_path(path, base_dir=base_dir, expand_user=expand_user, strict=strict)
        for path in paths
    ]


def canonical_path(path: Path | str) -> str:
    """
    Normalize a path into a lowercase absolute string.
    """
    return str(resolve_path(path)).lower()


def safe_filename(value: str, *, default: str = "item", max_length: int = 80) -> str:
    """
    Build a filesystem-safe filename slug.
    """
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    cleaned = cleaned[:max_length].strip("_")
    return cleaned or default

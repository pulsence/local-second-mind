from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from lsm.config.models.ingest import RootConfig
from lsm.logging import get_logger

logger = get_logger(__name__)


# -----------------------------
# Crawl
# -----------------------------
def iter_files(
    roots: List[RootConfig],
    exts: set[str],
    exclude_dirs: set[str],
) -> Iterable[Tuple[Path, RootConfig]]:
    """Yield ``(file_path, root_config)`` tuples for files matching filters.

    Args:
        roots: Root configurations to scan.
        exts: Allowed file extensions (lowercased, with leading dot).
        exclude_dirs: Directory names to skip.

    Yields:
        Tuples of (resolved file path, originating RootConfig).
    """
    for root_cfg in roots:
        root = root_cfg.path.expanduser().resolve()
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Skip excluded dirs by checking any parent folder name
            if any(parent.name in exclude_dirs for parent in p.parents):
                continue
            if p.suffix.lower() in exts:
                yield p, root_cfg


# -----------------------------
# Folder Tags
# -----------------------------
def collect_folder_tags(file_path: Path, root_path: Path) -> List[str]:
    """Collect tags from ``.lsm_tags.json`` files between *root_path* and *file_path*.

    Walks from *root_path* down to *file_path*'s parent directory, reading any
    ``.lsm_tags.json`` files encountered.  Tags are accumulated root-first
    (most general first, most specific folder last) and deduplicated.

    Args:
        file_path: The file whose folder tags to collect.
        root_path: The ingest root this file belongs to.

    Returns:
        Deduplicated list of folder tags, ordered root-to-leaf.
    """
    root = root_path.expanduser().resolve()
    parent = file_path.resolve().parent

    # Build list of directories from root down to parent
    dirs_to_check: List[Path] = []
    current = parent
    while True:
        dirs_to_check.append(current)
        if current == root:
            break
        next_parent = current.parent
        if next_parent == current:
            # Reached filesystem root without finding root_path
            break
        current = next_parent

    # Reverse so root comes first
    dirs_to_check.reverse()

    seen: set[str] = set()
    tags: List[str] = []
    for d in dirs_to_check:
        tag_file = d / ".lsm_tags.json"
        if not tag_file.is_file():
            continue
        try:
            data = json.loads(tag_file.read_text(encoding="utf-8"))
            for tag in data.get("tags", []):
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Ignoring invalid .lsm_tags.json at %s: %s", d, exc)

    return tags

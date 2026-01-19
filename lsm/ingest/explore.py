"""
Helpers for exploring indexed files in an ingest collection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def normalize_query_path(value: str) -> str:
    """
    Normalize a path query for cross-platform matching.

    Args:
        value: Raw path query string

    Returns:
        Normalized path with consistent separators
    """
    normalized = value.strip().lower()
    normalized = normalized.replace("/", os.sep).replace("\\", os.sep)
    return normalized.strip(os.sep)


def parse_explore_query(
    query: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], str, bool]:
    """
    Parse explore command query into filter components.

    Args:
        query: Raw query string from user

    Returns:
        Tuple of (path_filter, ext_filter, pattern, display_root, full_path)
    """
    if not query:
        return None, None, None, "All files", False

    tokens = query.strip().split()
    full_path = False
    if "--full-path" in tokens:
        full_path = True
        tokens.remove("--full-path")
    if "full_path" in tokens:
        full_path = True
        tokens.remove("full_path")
    raw = " ".join(tokens).strip()
    if not raw:
        return None, None, None, "All files", full_path
    raw_lower = raw.lower()
    pattern = None
    ext_filter = None
    path_filter = None
    display_root = raw

    if raw_lower.startswith("ext:") or raw_lower.startswith("type:"):
        ext_filter = raw.split(":", 1)[1].strip()
    elif any(ch in raw for ch in ("*", "?", "[")):
        pattern = raw
    elif raw.startswith(".") and "/" not in raw and "\\" not in raw:
        ext_filter = raw
    elif "/" in raw or "\\" in raw:
        path_filter = raw
    else:
        path_filter = raw

    if ext_filter and not ext_filter.startswith("."):
        ext_filter = f".{ext_filter}"

    if path_filter:
        display_root = path_filter

    return (
        normalize_query_path(path_filter) if path_filter else None,
        ext_filter.lower() if ext_filter else None,
        pattern.lower() if pattern else None,
        display_root,
        full_path,
    )


def new_tree_node(name: str) -> Dict[str, Any]:
    """
    Create a new tree node for file exploration.

    Args:
        name: Name of the node

    Returns:
        Dictionary representing the tree node
    """
    return {"name": name, "children": {}, "files": {}, "file_count": 0, "chunk_count": 0}


def build_tree(
    file_stats: Dict[str, Dict[str, Any]],
    base_filter: Optional[str],
    common_parts: Tuple[str, ...],
) -> Dict[str, Any]:
    """
    Build a tree structure from file statistics.

    Args:
        file_stats: Dictionary of file paths to stats
        base_filter: Optional path filter
        common_parts: Common path prefix parts to strip

    Returns:
        Root tree node
    """
    root = new_tree_node("root")

    for source_path, info in file_stats.items():
        chunk_count = info["chunk_count"]
        source_norm = source_path.lower()

        if base_filter:
            idx = source_norm.find(base_filter)
            if idx == -1:
                continue
            rel = source_norm[idx + len(base_filter):].lstrip("\\/")
            rel_parts = Path(rel).parts if rel else (Path(source_path).name,)
        elif common_parts:
            rel_parts = Path(source_path).parts[len(common_parts):] or (Path(source_path).name,)
        else:
            rel_parts = Path(source_path).parts

        if not rel_parts:
            continue

        node = root
        node["file_count"] += 1
        node["chunk_count"] += chunk_count

        for part in rel_parts[:-1]:
            node = node["children"].setdefault(part, new_tree_node(part))
            node["file_count"] += 1
            node["chunk_count"] += chunk_count

        node["files"][rel_parts[-1]] = chunk_count

    return root


def compute_common_parts(paths: Dict[str, Dict[str, Any]]) -> Tuple[str, ...]:
    """
    Compute the common path prefix for a set of paths.

    Args:
        paths: Dictionary of file paths

    Returns:
        Tuple of common path parts
    """
    if not paths:
        return ()
    raw_paths = list(paths.keys())
    try:
        common = os.path.commonpath(raw_paths)
    except ValueError:
        return ()
    if not common:
        return ()
    return Path(common).parts

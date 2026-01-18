"""
Display utilities for the ingest REPL.

Contains banner printing, help text, tree formatting, and output helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def print_banner() -> None:
    """Print welcome banner for ingest REPL."""
    print()
    print("=" * 60)
    print(" Local Second Mind - Ingest Management")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /info              - Show collection information")
    print("  /stats             - Show detailed statistics")
    print("  /explore [query]   - Explore indexed files")
    print("  /show <path>       - Show chunks for a file")
    print("  /search <query>    - Search metadata")
    print("  /build [--force]   - Run ingest pipeline")
    print("  /tag [--max N]     - Run AI tagging on untagged chunks")
    print("  /tags              - Show all tags in collection")
    print("  /vectordb-providers - List available vector DB providers")
    print("  /vectordb-status   - Show vector DB provider status")
    print("  /wipe              - Clear collection (requires confirmation)")
    print("  /help              - Show this help")
    print("  /exit              - Exit")
    print()
    print("Tip: Type /query to switch to query mode (in the unified shell).")
    print()


def print_help() -> None:
    """Print detailed help text."""
    help_text = """
INGEST REPL COMMANDS

/info
    Display basic collection information (name, count, ID).

/stats
    Display detailed statistics including:
    - Total chunks and unique files
    - File type distribution
    - Top files by chunk count
    - Author and metadata statistics
    - Ingestion timeline

/explore [query]
    Browse indexed files. Optional query filters by path.
    Examples:
        /explore              - Show all files
        /explore python       - Show files with "python" in path
        /explore papers/ai    - Explore a specific folder
        /explore .pdf         - Show all PDF files
        /explore "*.pdf"      - Wildcard file match
        /explore --full-path  - Show full folder prefixes

/show <path>
    Display all chunks for a specific file path.
    Example:
        /show /docs/readme.md

/search <query>
    Search metadata by file path substring.
    Example:
        /search tutorial

/build [--force]
    Run the ingest pipeline to add/update documents.
    - Without --force: Only processes new/changed files (incremental)
    - With --force: Re-processes all files (full rebuild)

    Examples:
        /build           - Incremental update
        /build --force   - Full rebuild

/tag [--max N]
    Run AI tagging on chunks that haven't been AI-tagged yet.
    This is incremental - only untagged chunks are processed.
    Tags are stored in the 'ai_tags' metadata field.

    Examples:
        /tag             - Tag all untagged chunks
        /tag --max 100   - Tag at most 100 chunks

/tags
    Display all unique tags in the collection.
    Shows both AI-generated tags and user-provided tags separately.

/vectordb-providers
    List available vector DB providers.

/vectordb-status
    Show current vector DB provider status and stats.

/wipe
    Delete all chunks from the collection.
    WARNING: This is destructive and requires confirmation.

/help
    Display this help message.

/exit
    Exit the REPL.

TIPS:
- Use /info for a quick overview
- Use /stats for detailed analysis
- Use /explore to browse your knowledge base
- Run /build regularly to keep your collection up-to-date
- Run /tag after ingesting to add AI-generated tags for better organization
- Type /query to switch to query mode (unified shell)
"""
    print(help_text)


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


def print_tree(root: Dict[str, Any], label: str, max_entries: int = 200) -> None:
    """
    Print a file tree to stdout.

    Args:
        root: Root tree node
        label: Label for the tree root
        max_entries: Maximum number of entries to print before truncating
    """
    printed = 0
    truncated = False

    def add_line(line: str) -> bool:
        nonlocal printed, truncated
        if printed >= max_entries:
            truncated = True
            return False
        print(line)
        printed += 1
        return True

    add_line(f"{label}/ ({root['file_count']:,} files, {root['chunk_count']:,} chunks)")

    def walk(node: Dict[str, Any], prefix: str) -> None:
        nonlocal printed, truncated
        children = sorted(node["children"].values(), key=lambda n: n["name"].lower())
        files = sorted(node["files"].items(), key=lambda item: item[0].lower())

        entries = [(child["name"] + "/", child, True) for child in children] + [
            (name, count, False) for name, count in files
        ]

        for idx, (name, payload, is_dir) in enumerate(entries):
            is_last = idx == len(entries) - 1
            connector = "`-- " if is_last else "|-- "
            line_prefix = prefix + connector

            if is_dir:
                line = (
                    f"{line_prefix}{name} "
                    f"({payload['file_count']:,} files, {payload['chunk_count']:,} chunks)"
                )
                if not add_line(line):
                    return
                extension = "    " if is_last else "|   "
                walk(payload, prefix + extension)
            else:
                line = f"{line_prefix}{name} ({payload:,} chunks)"
                if not add_line(line):
                    return

    walk(root, "")

    if truncated:
        print("\nOutput truncated. Use /explore <path> or /explore <pattern> to narrow results.")


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

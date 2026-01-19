"""
Display utilities for ingest operations.

Contains banner, help text, tree formatting, and output helpers.
All functions return strings - the UI layer decides how to display them.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_banner() -> str:
    """
    Get the ingest management welcome banner.

    Returns:
        Banner text string
    """
    lines = [
        "",
        "=" * 60,
        " Local Second Mind - Ingest Management",
        "=" * 60,
        "",
        "Commands:",
        "  /info              - Show collection information",
        "  /stats             - Show detailed statistics",
        "  /explore [query]   - Explore indexed files",
        "  /show <path>       - Show chunks for a file",
        "  /search <query>    - Search metadata",
        "  /build [--force]   - Run ingest pipeline",
        "  /tag [--max N]     - Run AI tagging on untagged chunks",
        "  /tags              - Show all tags in collection",
        "  /vectordb-providers - List available vector DB providers",
        "  /vectordb-status   - Show vector DB provider status",
        "  /wipe              - Clear collection (requires confirmation)",
        "  /help              - Show this help",
        "  /exit              - Exit",
        "",
        "Tip: Use the Query tab (Ctrl+Q) in the TUI to run queries.",
        "",
    ]
    return "\n".join(lines)


def get_help() -> str:
    """
    Get the ingest help text.

    Returns:
        Help text string
    """
    return """
INGEST COMMANDS

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
    Exit the TUI.

TIPS:
- Use /info for a quick overview
- Use /stats for detailed analysis
- Use /explore to browse your knowledge base
- Run /build regularly to keep your collection up-to-date
- Run /tag after ingesting to add AI-generated tags for better organization
- Use the Query tab (Ctrl+Q) in the TUI to run queries
"""


def format_tree(root: Dict[str, Any], label: str, max_entries: int = 200) -> str:
    """
    Format a file tree as a string.

    Args:
        root: Root tree node
        label: Label for the tree root
        max_entries: Maximum number of entries before truncating

    Returns:
        Formatted tree string
    """
    lines: List[str] = []
    truncated = False

    def add_line(line: str) -> bool:
        nonlocal truncated
        if len(lines) >= max_entries:
            truncated = True
            return False
        lines.append(line)
        return True

    add_line(f"{label}/ ({root['file_count']:,} files, {root['chunk_count']:,} chunks)")

    def walk(node: Dict[str, Any], prefix: str) -> None:
        nonlocal truncated
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
        lines.append("\nOutput truncated. Use /explore <path> or /explore <pattern> to narrow results.")

    return "\n".join(lines)


# -----------------------------
# Backwards compatibility aliases
# -----------------------------
def print_banner() -> None:
    """Print welcome banner. DEPRECATED: Use get_banner() instead."""
    print(get_banner())


def print_help() -> None:
    """Print help text. DEPRECATED: Use get_help() instead."""
    print(get_help())


def print_tree(root: Dict[str, Any], label: str, max_entries: int = 200) -> None:
    """Print a file tree. DEPRECATED: Use format_tree() instead."""
    print(format_tree(root, label, max_entries))

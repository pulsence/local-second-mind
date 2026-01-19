"""
Command handlers for ingest operations.

Contains business logic handlers that return results as strings.
UI layers (TUI, shell) are responsible for display and user confirmation.
"""

from __future__ import annotations

import fnmatch
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from chromadb.api.models.Collection import Collection

from lsm.config.models import LSMConfig
from lsm.logging import get_logger
from lsm.ingest.stats import (
    get_collection_info,
    get_collection_stats,
    format_stats_report,
    search_metadata,
    get_file_chunks,
    iter_collection_metadatas,
)
from lsm.vectordb import create_vectordb_provider, list_available_providers
from lsm.vectordb.utils import require_chroma_collection
from lsm.ingest.pipeline import ingest
from lsm.ingest.tagging import (
    tag_chunks,
    get_all_tags,
)
from lsm.ingest.explore import (
    parse_explore_query,
    build_tree,
    compute_common_parts,
)
from .display import format_tree, get_help

logger = get_logger(__name__)


def _source_matches_pattern(source_path: str, pattern: str) -> bool:
    """Check if a source path matches a glob pattern."""
    name = Path(source_path).name
    return (
        fnmatch.fnmatch(source_path.lower(), pattern.lower())
        or fnmatch.fnmatch(name.lower(), pattern.lower())
    )


# -----------------------------
# Command Result Types
# -----------------------------
class CommandResult:
    """Result from a command handler."""

    def __init__(
        self,
        output: str = "",
        handled: bool = True,
        should_exit: bool = False,
        action: Optional[str] = None,
        action_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize command result.

        Args:
            output: Text output to display
            handled: True if command was handled
            should_exit: True if the user requested exit
            action: Optional action type for UI to handle
            action_data: Optional data for the action
        """
        self.output = output
        self.handled = handled
        self.should_exit = should_exit
        self.action = action
        self.action_data = action_data or {}


# -----------------------------
# Info Command
# -----------------------------
def format_info(collection: Collection) -> str:
    """
    Format collection info as a string.

    Args:
        collection: Vector DB collection

    Returns:
        Formatted info string
    """
    # Check if it's a non-ChromaDB provider
    if getattr(collection, "name", "") != "chromadb":
        stats = collection.get_stats()
        lines = [
            "",
            "=" * 60,
            "VECTOR DB INFO",
            "=" * 60,
            f"Provider: {stats.get('provider', 'unknown')}",
            f"Status:   {stats.get('status', 'unknown')}",
            f"Count:    {collection.count():,}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)

    info = get_collection_info(collection)

    lines = [
        "",
        "=" * 60,
        "COLLECTION INFO",
        "=" * 60,
        f"Name:     {info['name']}",
        f"ID:       {info['id']}",
        f"Chunks:   {info['count']:,}",
    ]

    if "metadata" in info and info["metadata"]:
        lines.append("\nMetadata:")
        for key, value in info["metadata"].items():
            lines.append(f"  {key}: {value}")

    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Stats Command
# -----------------------------
def format_stats(
    collection: Collection,
    config: LSMConfig,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> str:
    """
    Format collection statistics as a string.

    Args:
        collection: Vector DB collection
        config: LSM configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Formatted stats string
    """
    lines = ["\nAnalyzing collection... (this may take a moment)"]

    total = None
    try:
        total = collection.count()
    except Exception:
        total = None

    last_report = [time.time()]

    def report_progress(analyzed: int) -> None:
        if time.time() - last_report[0] >= 2.0:
            if total:
                pct = (analyzed / total) * 100 if total > 0 else 0.0
                lines.append(f"  analyzed chunks: {analyzed:,} / {total:,} ({pct:.1f}%)")
            else:
                lines.append(f"  analyzed chunks: {analyzed:,}")
            if progress_callback:
                progress_callback(analyzed, total)
            last_report[0] = time.time()

    error_report_path = config.ingest.manifest.parent / "ingest_error_report.json"
    try:
        stats = get_collection_stats(
            collection,
            limit=None,
            error_report_path=error_report_path,
            progress_callback=report_progress,
        )
    except Exception as e:
        return f"Error: {e}"

    if "error" in stats:
        return f"Error: {stats['error']}"

    if "message" in stats:
        return stats["message"]

    if progress_callback and total:
        progress_callback(total, total)

    report = format_stats_report(stats)
    lines.append(report)

    return "\n".join(lines)


# -----------------------------
# Explore Command
# -----------------------------
def format_explore(
    collection: Collection,
    query: Optional[str] = None,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> str:
    """
    Format explore results as a string.

    Args:
        collection: Vector DB collection
        query: Optional filter query
        progress_callback: Optional callback for progress updates

    Returns:
        Formatted explore string with file tree
    """
    lines = ["\nExploring collection...", "Scanning metadata... (this may take a moment)"]

    path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query(query)
    where = {"ext": {"$eq": ext_filter}} if ext_filter else None

    file_stats: Dict[str, Dict[str, Any]] = {}
    scanned = 0
    last_report = time.time()
    report_every = 2.0

    total = None
    try:
        total = collection.count()
    except Exception:
        total = None

    for meta in iter_collection_metadatas(collection, where=where, batch_size=5000):
        scanned += 1
        if time.time() - last_report >= report_every:
            if total:
                pct = (scanned / total) * 100 if total > 0 else 0.0
                lines.append(f"  scanned chunks: {scanned:,} / {total:,} ({pct:.1f}%)")
            else:
                lines.append(f"  scanned chunks: {scanned:,}")
            if progress_callback:
                progress_callback(scanned, total)
            last_report = time.time()
        source_path = meta.get("source_path", "")
        if not source_path:
            continue

        source_norm = source_path.lower()
        if path_filter and path_filter not in source_norm:
            continue
        if pattern and not _source_matches_pattern(source_path, pattern):
            continue

        entry = file_stats.setdefault(
            source_path,
            {"ext": meta.get("ext", ""), "chunk_count": 0},
        )
        entry["chunk_count"] += 1

    if progress_callback:
        progress_callback(scanned, total)

    if not file_stats:
        lines.append("No files found.")
        return "\n".join(lines)

    common_parts = ()
    if not path_filter and not full_path:
        common_parts = compute_common_parts(file_stats)

    tree = build_tree(file_stats, path_filter, common_parts)
    lines.append(format_tree(tree, display_root))
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Show Command
# -----------------------------
def format_show(collection: Collection, file_path: str) -> str:
    """
    Format chunks for a file as a string.

    Args:
        collection: Vector DB collection
        file_path: Path to the file

    Returns:
        Formatted chunks string
    """
    if not file_path:
        return "Usage: /show <file_path>"

    lines = [f"\nFetching chunks for: {file_path}"]

    chunks = get_file_chunks(collection, file_path)

    if not chunks:
        lines.append("No chunks found for this file.")
        lines.append("\nTip: Use /explore to find the exact file path.")
        return "\n".join(lines)

    lines.append(f"\nFound {len(chunks)} chunks")
    lines.append("=" * 60)

    for chunk in chunks:
        idx = chunk.get("chunk_index", 0)
        text = chunk.get("text", "")
        author = chunk.get("author", "")
        title = chunk.get("title", "")

        lines.append(f"\nChunk {idx + 1}:")
        if title:
            lines.append(f"Title: {title}")
        if author:
            lines.append(f"Author: {author}")

        lines.append("-" * 60)
        # Truncate long chunks
        if len(text) > 500:
            lines.append(text[:500] + "...")
        else:
            lines.append(text)

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


# -----------------------------
# Search Command
# -----------------------------
def format_search(collection: Collection, query: str) -> str:
    """
    Format search results as a string.

    Args:
        collection: Vector DB collection
        query: Search query

    Returns:
        Formatted search results string
    """
    if not query:
        return "Usage: /search <query>"

    lines = [f"\nSearching for: {query}"]

    results = search_metadata(collection, query=query, limit=50)

    if not results:
        lines.append("No results found.")
        return "\n".join(lines)

    lines.append(f"\nFound {len(results)} results")
    lines.append("-" * 60)

    for i, meta in enumerate(results[:20], 1):
        source_path = meta.get("source_path", "unknown")
        chunk_idx = meta.get("chunk_index", 0)
        ext = meta.get("ext", "")

        # Shorten path
        display_path = Path(source_path).name
        if len(source_path) > 50:
            display_path = "..." + source_path[-47:]

        lines.append(f"{i:3}. [{ext:5s}] {display_path} (chunk {chunk_idx})")

    if len(results) > 20:
        lines.append(f"\n... and {len(results) - 20} more results")

    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Build Command
# -----------------------------
def get_build_confirmation_prompt(force: bool) -> str:
    """
    Get the confirmation prompt for build command.

    Args:
        force: Whether --force flag was used

    Returns:
        Prompt string to display before confirmation
    """
    if force:
        return "\nWARNING: --force will re-process ALL files\n"
    return ""


def run_build(config: LSMConfig, force: bool = False) -> str:
    """
    Run the ingest pipeline.

    Args:
        config: LSM configuration
        force: Whether to force full rebuild

    Returns:
        Result string
    """
    lines = [
        "\nStarting ingest pipeline...",
        "-" * 60,
        "Press Ctrl+C to stop ingest.",
    ]

    try:
        ingest(
            roots=config.ingest.roots,
            chroma_flush_interval=config.ingest.chroma_flush_interval,
            embed_model_name=config.embed_model,
            device=config.device,
            batch_size=config.batch_size,
            manifest_path=config.ingest.manifest,
            exts=config.ingest.exts,
            exclude_dirs=config.ingest.exclude_set,
            vectordb_config=config.vectordb,
            dry_run=False,
            enable_ocr=config.ingest.enable_ocr,
            skip_errors=config.ingest.skip_errors,
            chunk_size=config.ingest.chunk_size,
            chunk_overlap=config.ingest.chunk_overlap,
        )

        lines.append("\nIngest completed successfully!")

    except Exception as e:
        lines.append(f"\nError during ingest: {e}")

    return "\n".join(lines)


# -----------------------------
# Wipe Command
# -----------------------------
def get_wipe_warning(collection: Collection) -> str:
    """
    Get the wipe warning message.

    Args:
        collection: Vector DB collection

    Returns:
        Warning message string
    """
    lines = [
        "",
        "=" * 60,
        "WARNING: DESTRUCTIVE OPERATION",
        "=" * 60,
        "",
        "This will DELETE ALL CHUNKS from the collection.",
        "This action CANNOT be undone.",
        "",
    ]

    try:
        chroma = require_chroma_collection(collection, "/wipe")
        count = chroma.count()
        lines.append(f"Current collection has {count:,} chunks")
        lines.append("")
    except Exception as e:
        lines.append(f"Error: {e}")

    return "\n".join(lines)


def run_wipe(collection: Collection) -> str:
    """
    Execute the wipe operation.

    Args:
        collection: Vector DB collection

    Returns:
        Result string
    """
    lines = ["\nDeleting all chunks..."]

    try:
        chroma = require_chroma_collection(collection, "/wipe")
        results = chroma.get(include=[])
        ids = results.get("ids", [])

        if ids:
            chroma.delete(ids=ids)
            lines.append(f"Deleted {len(ids):,} chunks")
        else:
            lines.append("Collection was already empty")

        lines.append("Collection cleared successfully.")

    except Exception as e:
        lines.append(f"Error during wipe: {e}")

    return "\n".join(lines)


# -----------------------------
# Tag Command
# -----------------------------
def parse_tag_args(args: str) -> tuple[Optional[int], Optional[str]]:
    """
    Parse /tag command arguments.

    Args:
        args: Command arguments string

    Returns:
        Tuple of (max_chunks, error_message)
    """
    max_chunks = None
    if "--max" in args.lower():
        parts = args.split()
        try:
            max_idx = parts.index("--max")
            if max_idx + 1 < len(parts):
                max_chunks = int(parts[max_idx + 1])
        except (ValueError, IndexError):
            return None, "Invalid --max argument. Usage: /tag --max N"
    return max_chunks, None


def get_tag_confirmation_prompt(config: LSMConfig, max_chunks: Optional[int]) -> str:
    """
    Get the confirmation prompt for tag command.

    Args:
        config: LSM configuration
        max_chunks: Maximum chunks to tag (or None)

    Returns:
        Prompt string to display before confirmation
    """
    tagging_config = config.llm.get_tagging_config()

    lines = [
        "",
        "=" * 60,
        "AI CHUNK TAGGING",
        "=" * 60,
        "",
        "This will tag chunks that haven't been AI-tagged yet.",
        "Tags are generated using the configured LLM model.",
        "",
        f"Using model: {tagging_config.model}",
        f"Provider: {tagging_config.provider}",
    ]

    if max_chunks:
        lines.append(f"Max chunks to tag: {max_chunks}")
    else:
        lines.append("Max chunks: unlimited")

    lines.append("")

    return "\n".join(lines)


def run_tag(collection: Collection, config: LSMConfig, max_chunks: Optional[int]) -> str:
    """
    Run AI tagging on chunks.

    Args:
        collection: Vector DB collection
        config: LSM configuration
        max_chunks: Maximum chunks to tag (or None)

    Returns:
        Result string
    """
    lines = ["\nStarting AI tagging...", "-" * 60]

    try:
        tagging_config = config.llm.get_tagging_config()
        tagged, failed = tag_chunks(
            collection=collection,
            llm_config=tagging_config,
            num_tags=3,
            batch_size=100,
            max_chunks=max_chunks,
            dry_run=False,
        )

        lines.append("")
        lines.append("=" * 60)
        lines.append("TAGGING COMPLETE")
        lines.append("=" * 60)
        lines.append(f"Successfully tagged: {tagged} chunks")
        lines.append(f"Failed: {failed} chunks")
        lines.append("")

    except Exception as e:
        lines.append(f"\nError during tagging: {e}")

    return "\n".join(lines)


# -----------------------------
# Tags Command
# -----------------------------
def format_tags(collection: Collection) -> str:
    """
    Format all tags in collection as a string.

    Args:
        collection: Vector DB collection

    Returns:
        Formatted tags string
    """
    lines = ["\nFetching all tags..."]

    try:
        all_tags = get_all_tags(collection)

        ai_tags = all_tags.get("ai_tags", [])
        user_tags = all_tags.get("user_tags", [])

        lines.append("")
        lines.append("=" * 60)
        lines.append("COLLECTION TAGS")
        lines.append("=" * 60)
        lines.append("")

        if ai_tags:
            lines.append(f"AI-Generated Tags ({len(ai_tags)} unique):")
            lines.append("-" * 60)
            # Display in columns for better readability
            for i in range(0, len(ai_tags), 3):
                row = ai_tags[i:i+3]
                lines.append("  " + " | ".join(f"{tag:20s}" for tag in row))
            lines.append("")
        else:
            lines.append("No AI-generated tags found.")
            lines.append("Run /tag to generate tags for your chunks.")
            lines.append("")

        if user_tags:
            lines.append(f"User Tags ({len(user_tags)} unique):")
            lines.append("-" * 60)
            for i in range(0, len(user_tags), 3):
                row = user_tags[i:i+3]
                lines.append("  " + " | ".join(f"{tag:20s}" for tag in row))
            lines.append("")
        else:
            lines.append("No user tags found.")
            lines.append("")

        lines.append("=" * 60)

    except Exception as e:
        lines.append(f"Error fetching tags: {e}")

    return "\n".join(lines)


# -----------------------------
# VectorDB Commands
# -----------------------------
def format_vectordb_providers(config: LSMConfig) -> str:
    """Format available vector DB providers as a string."""
    lines = [
        "",
        "=" * 60,
        "AVAILABLE VECTOR DB PROVIDERS",
        "=" * 60,
        "",
    ]

    providers = list_available_providers()
    if not providers:
        lines.append("No vector DB providers registered.")
        lines.append("")
        return "\n".join(lines)

    current_provider = config.vectordb.provider
    lines.append(f"Current Provider: {current_provider}")
    lines.append(f"Collection:       {config.vectordb.collection}")
    lines.append("")

    lines.append(f"Available Providers ({len(providers)}):")
    lines.append("")

    for provider_name in providers:
        is_current = "ACTIVE" if provider_name == current_provider else ""
        status = ""
        if provider_name == current_provider:
            try:
                provider = create_vectordb_provider(config.vectordb)
                status = "ok" if provider.is_available() else "unavailable"
            except Exception as e:
                status = f"error ({e})"
        lines.append(f"  {provider_name:20s} {status:20s} {is_current}")

    lines.append("")

    return "\n".join(lines)


def format_vectordb_status(config: LSMConfig) -> str:
    """Format vector DB provider health and stats."""
    lines = [
        "",
        "=" * 60,
        "VECTOR DB STATUS",
        "=" * 60,
        "",
    ]

    try:
        provider = create_vectordb_provider(config.vectordb)
        health = provider.health_check()
        stats = provider.get_stats()

        lines.append(f"Provider: {health.get('provider', 'unknown')}")
        lines.append(f"Status:   {health.get('status', 'unknown')}")
        if health.get("error"):
            lines.append(f"Error:    {health.get('error')}")
        lines.append(f"Count:    {stats.get('count', 'n/a')}")
    except Exception as e:
        lines.append(f"Error: {e}")

    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Main Command Handler
# -----------------------------
def handle_command(
    line: str,
    collection: Collection,
    config: LSMConfig,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> CommandResult:
    """
    Handle an ingest command.

    Args:
        line: Input line
        collection: Vector DB collection
        config: LSM configuration
        progress_callback: Optional progress callback

    Returns:
        CommandResult with output and action data
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit":
        return CommandResult(output="\nGoodbye!", should_exit=True)

    if cmd == "/help":
        return CommandResult(output=get_help())

    if cmd == "/info":
        return CommandResult(output=format_info(collection))

    if cmd == "/stats":
        return CommandResult(output=format_stats(collection, config, progress_callback))

    if cmd == "/explore":
        query = args.strip() if args else None
        return CommandResult(output=format_explore(collection, query, progress_callback))

    if cmd == "/show":
        return CommandResult(output=format_show(collection, args.strip()))

    if cmd == "/search":
        return CommandResult(output=format_search(collection, args.strip()))

    if cmd == "/build":
        force = "--force" in args.lower()
        # Return action for UI to handle confirmation
        return CommandResult(
            output=get_build_confirmation_prompt(force),
            action="build_confirm" if force else "build_run",
            action_data={"force": force, "config": config},
        )

    if cmd == "/tag":
        max_chunks, error = parse_tag_args(args.strip())
        if error:
            return CommandResult(output=error)
        # Return action for UI to handle confirmation
        return CommandResult(
            output=get_tag_confirmation_prompt(config, max_chunks),
            action="tag_confirm",
            action_data={"max_chunks": max_chunks, "collection": collection, "config": config},
        )

    if cmd == "/tags":
        return CommandResult(output=format_tags(collection))

    if cmd == "/vectordb-providers":
        return CommandResult(output=format_vectordb_providers(config))

    if cmd == "/vectordb-status":
        return CommandResult(output=format_vectordb_status(config))

    if cmd == "/wipe":
        # Return action for UI to handle confirmation
        return CommandResult(
            output=get_wipe_warning(collection),
            action="wipe_confirm",
            action_data={"collection": collection},
        )

    return CommandResult(output=f"Unknown command: {cmd}\nType /help for available commands.")


# -----------------------------
# Backwards Compatibility
# -----------------------------
def handle_info_command(collection: Collection) -> None:
    """Handle /info command. DEPRECATED: Use format_info() instead."""
    print(format_info(collection))


def handle_stats_command(
    collection: Collection,
    config: LSMConfig,
    progress_callback=None,
) -> None:
    """Handle /stats command. DEPRECATED: Use format_stats() instead."""
    print(format_stats(collection, config, progress_callback))


def handle_explore_command(
    collection: Collection,
    query: Optional[str] = None,
    progress_callback=None,
    emit_tree=None,
) -> None:
    """Handle /explore command. DEPRECATED: Use format_explore() instead."""
    if emit_tree:
        # Old callback pattern - do the work manually
        path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query(query)
        where = {"ext": {"$eq": ext_filter}} if ext_filter else None

        file_stats: Dict[str, Dict[str, Any]] = {}
        scanned = 0
        last_report = time.time()
        report_every = 2.0

        total = None
        try:
            total = collection.count()
        except Exception:
            total = None

        print("\nExploring collection...")
        print("Scanning metadata... (this may take a moment)")

        for meta in iter_collection_metadatas(collection, where=where, batch_size=5000):
            scanned += 1
            if time.time() - last_report >= report_every:
                if total:
                    pct = (scanned / total) * 100 if total > 0 else 0.0
                    print(f"  scanned chunks: {scanned:,} / {total:,} ({pct:.1f}%)")
                else:
                    print(f"  scanned chunks: {scanned:,}")
                if progress_callback:
                    progress_callback(scanned, total)
                last_report = time.time()
            source_path = meta.get("source_path", "")
            if not source_path:
                continue

            source_norm = source_path.lower()
            if path_filter and path_filter not in source_norm:
                continue
            if pattern and not _source_matches_pattern(source_path, pattern):
                continue

            entry = file_stats.setdefault(
                source_path,
                {"ext": meta.get("ext", ""), "chunk_count": 0},
            )
            entry["chunk_count"] += 1

        if progress_callback:
            progress_callback(scanned, total)

        if not file_stats:
            print("No files found.")
            return

        common_parts = ()
        if not path_filter and not full_path:
            common_parts = compute_common_parts(file_stats)

        tree = build_tree(file_stats, path_filter, common_parts)
        emit_tree(tree, display_root)
        print()
    else:
        print(format_explore(collection, query, progress_callback))


def handle_show_command(collection: Collection, file_path: str) -> None:
    """Handle /show command. DEPRECATED: Use format_show() instead."""
    print(format_show(collection, file_path))


def handle_search_command(collection: Collection, query: str) -> None:
    """Handle /search command. DEPRECATED: Use format_search() instead."""
    print(format_search(collection, query))


def handle_build_command(config: LSMConfig, force: bool = False) -> None:
    """Handle /build command. DEPRECATED: Use run_build() instead."""
    print()
    if force:
        print("WARNING: --force will re-process ALL files")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return
    print(run_build(config, force))


def handle_wipe_command(collection: Collection) -> None:
    """Handle /wipe command. DEPRECATED: Use run_wipe() with UI confirmation."""
    print(get_wipe_warning(collection))

    try:
        require_chroma_collection(collection, "/wipe")
    except Exception:
        return

    confirm1 = input("Type 'DELETE' to confirm: ").strip()
    if confirm1 != "DELETE":
        print("Cancelled.")
        return

    confirm2 = input("Are you absolutely sure? (yes/no): ").strip().lower()
    if confirm2 != "yes":
        print("Cancelled.")
        return

    print(run_wipe(collection))


def handle_tag_command(collection: Collection, config: LSMConfig, args: str) -> None:
    """Handle /tag command. DEPRECATED: Use run_tag() with UI confirmation."""
    max_chunks, error = parse_tag_args(args)
    if error:
        print(error)
        return

    print(get_tag_confirmation_prompt(config, max_chunks))

    confirm = input("Proceed with tagging? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    print(run_tag(collection, config, max_chunks))


def handle_tags_command(collection: Collection) -> None:
    """Handle /tags command. DEPRECATED: Use format_tags() instead."""
    print(format_tags(collection))


def handle_vectordb_providers_command(config: LSMConfig) -> None:
    """Handle /vectordb-providers command. DEPRECATED: Use format_vectordb_providers() instead."""
    print(format_vectordb_providers(config))


def handle_vectordb_status_command(config: LSMConfig) -> None:
    """Handle /vectordb-status command. DEPRECATED: Use format_vectordb_status() instead."""
    print(format_vectordb_status(config))

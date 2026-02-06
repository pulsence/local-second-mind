"""Ingest command parsing + formatting for TUI layer."""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from chromadb.api.models.Collection import Collection

from lsm.config.models import LSMConfig
from lsm.ingest.api import (
    get_collection_info as api_get_collection_info,
    get_collection_stats as api_get_collection_stats,
    run_ingest,
    wipe_collection,
)
from lsm.ingest.explore import build_tree, compute_common_parts, parse_explore_query
from lsm.ingest.stats import (
    get_file_chunks,
    iter_collection_metadatas,
    search_metadata,
)
from lsm.ingest.tagging import get_all_tags, tag_chunks
from lsm.ui.utils import format_ingest_tree, get_ingest_help
from lsm.vectordb import create_vectordb_provider, list_available_providers
from lsm.vectordb.utils import require_chroma_collection


@dataclass
class CommandResult:
    """Result from ingest command parsing/dispatch."""

    output: str = ""
    handled: bool = True
    should_exit: bool = False
    action: Optional[str] = None
    action_data: Dict[str, Any] = field(default_factory=dict)


def _source_matches_pattern(source_path: str, pattern: str) -> bool:
    name = Path(source_path).name
    return (
        fnmatch.fnmatch(source_path.lower(), pattern.lower())
        or fnmatch.fnmatch(name.lower(), pattern.lower())
    )


def parse_tag_args(args: str) -> tuple[Optional[int], Optional[str]]:
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


def get_build_confirmation_prompt(force: bool) -> str:
    if force:
        return "\nWARNING: --force will re-process ALL files\n"
    return ""


def get_tag_confirmation_prompt(config: LSMConfig, max_chunks: Optional[int]) -> str:
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


def get_wipe_warning(provider: Any) -> str:
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
        count = provider.count()
        lines.append(f"Current collection has {count:,} chunks")
        lines.append("")
    except Exception as e:
        lines.append(f"Error: {e}")

    return "\n".join(lines)


def run_build(
    config: LSMConfig,
    force: bool = False,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> str:
    lines = [
        "\nStarting ingest pipeline...",
        "-" * 60,
        "Press Ctrl+C to stop ingest.",
    ]

    try:
        result = run_ingest(config=config, force=force, progress_callback=progress_callback)
        lines.append("")
        lines.append("Ingest completed successfully!")
        lines.append(f"Total files: {result.total_files:,}")
        lines.append(f"Completed files: {result.completed_files:,}")
        lines.append(f"Skipped files: {result.skipped_files:,}")
        lines.append(f"Chunks added: {result.chunks_added:,}")
        lines.append(f"Elapsed seconds: {result.elapsed_seconds:.2f}")
        if result.errors:
            lines.append(f"Errors: {len(result.errors):,}")

    except Exception as e:
        lines.append(f"\nError during ingest: {e}")

    return "\n".join(lines)


def run_wipe(config: LSMConfig) -> str:
    lines = ["\nDeleting all chunks..."]
    try:
        deleted = wipe_collection(config)
        if deleted:
            lines.append(f"Deleted {deleted:,} chunks")
        else:
            lines.append("Collection was already empty")
        lines.append("Collection cleared successfully.")
    except Exception as e:
        lines.append(f"Error during wipe: {e}")
    return "\n".join(lines)


def run_tag(provider: Any, config: LSMConfig, max_chunks: Optional[int]) -> str:
    lines = ["\nStarting AI tagging...", "-" * 60]
    try:
        collection = require_chroma_collection(provider, "/tag")
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


def format_info(provider: Any, config: LSMConfig) -> str:
    if getattr(provider, "name", "") != "chromadb":
        stats = provider.get_stats()
        lines = [
            "",
            "=" * 60,
            "VECTOR DB INFO",
            "=" * 60,
            f"Provider: {stats.get('provider', 'unknown')}",
            f"Status:   {stats.get('status', 'unknown')}",
            f"Count:    {provider.count():,}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)

    info = api_get_collection_info(config)
    lines = [
        "",
        "=" * 60,
        "COLLECTION INFO",
        "=" * 60,
        f"Name:     {info.name}",
        f"Chunks:   {info.chunk_count:,}",
        f"Provider: {info.provider}",
    ]

    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)


def format_stats(
    provider: Any,
    config: LSMConfig,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> str:
    lines = ["\nAnalyzing collection... (this may take a moment)"]

    try:
        total = provider.count()
    except Exception:
        total = None

    last_report = [time.time()]

    def report_progress(analyzed: int, current_total: Optional[int]) -> None:
        if time.time() - last_report[0] >= 2.0:
            active_total = current_total if current_total is not None else total
            if active_total:
                pct = (analyzed / active_total) * 100 if active_total > 0 else 0.0
                lines.append(f"  analyzed chunks: {analyzed:,} / {active_total:,} ({pct:.1f}%)")
            else:
                lines.append(f"  analyzed chunks: {analyzed:,}")
            if progress_callback:
                progress_callback(analyzed, active_total)
            last_report[0] = time.time()

    try:
        stats = api_get_collection_stats(
            config,
            progress_callback=report_progress,
        )
    except Exception as e:
        return f"Error: {e}"

    if progress_callback and total:
        progress_callback(total, total)

    lines.extend(
        [
            "=" * 60,
            "COLLECTION STATISTICS",
            "=" * 60,
            "",
            f"Total chunks:        {stats.chunk_count:,}",
            f"Unique files:        {stats.unique_files:,}",
        ]
    )

    if stats.file_types:
        lines.append("")
        lines.append("-" * 60)
        lines.append("FILE TYPES")
        lines.append("-" * 60)
        for ext, count in stats.file_types.items():
            pct = (count / stats.chunk_count * 100) if stats.chunk_count > 0 else 0.0
            lines.append(f"  {ext:10s}  {count:7,} chunks  ({pct:5.1f}%)")

    if stats.top_files:
        lines.append("")
        lines.append("-" * 60)
        lines.append("TOP FILES (by chunk count)")
        lines.append("-" * 60)
        for entry in stats.top_files[:10]:
            file_path = entry.get("source_path", "unknown")
            count = int(entry.get("chunk_count", 0))
            display_path = Path(file_path).name
            if len(file_path) > 50:
                display_path = "..." + file_path[-47:]
            lines.append(f"  {count:5,} chunks  {display_path}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_explore(
    provider: Any,
    query: Optional[str] = None,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> str:
    lines = ["\nExploring collection...", "Scanning metadata... (this may take a moment)"]
    collection = require_chroma_collection(provider, "/explore")

    path_filter, ext_filter, pattern, display_root, full_path = parse_explore_query(query)
    where = {"ext": {"$eq": ext_filter}} if ext_filter else None

    file_stats: Dict[str, Dict[str, Any]] = {}
    scanned = 0
    last_report = time.time()
    report_every = 2.0

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
    lines.append(format_ingest_tree(tree, display_root))
    lines.append("")

    return "\n".join(lines)


def format_show(provider: Any, file_path: str) -> str:
    if not file_path:
        return "Usage: /show <file_path>"

    collection = require_chroma_collection(provider, "/show")
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
        if len(text) > 500:
            lines.append(text[:500] + "...")
        else:
            lines.append(text)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def format_search(provider: Any, query: str) -> str:
    if not query:
        return "Usage: /search <query>"

    collection = require_chroma_collection(provider, "/search")
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

        display_path = Path(source_path).name
        if len(source_path) > 50:
            display_path = "..." + source_path[-47:]

        lines.append(f"{i:3}. [{ext:5s}] {display_path} (chunk {chunk_idx})")

    if len(results) > 20:
        lines.append(f"\n... and {len(results) - 20} more results")

    lines.append("")
    return "\n".join(lines)


def format_tags(provider: Any) -> str:
    lines = ["\nFetching all tags..."]
    collection = require_chroma_collection(provider, "/tags")

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


def format_vectordb_providers(config: LSMConfig) -> str:
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


def handle_command(
    line: str,
    provider: Any,
    config: LSMConfig,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> CommandResult:
    """Handle ingest command parsing and action dispatch metadata."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit":
        return CommandResult(output="\nGoodbye!", should_exit=True)

    if cmd == "/help":
        return CommandResult(output=get_ingest_help())

    if cmd == "/info":
        return CommandResult(output=format_info(provider, config))

    if cmd == "/stats":
        return CommandResult(output=format_stats(provider, config, progress_callback))

    if cmd == "/explore":
        query = args.strip() if args else None
        return CommandResult(output=format_explore(provider, query, progress_callback))

    if cmd == "/show":
        return CommandResult(output=format_show(provider, args.strip()))

    if cmd == "/search":
        return CommandResult(output=format_search(provider, args.strip()))

    if cmd == "/build":
        force = "--force" in args.lower()
        return CommandResult(
            output=get_build_confirmation_prompt(force),
            action="build_confirm" if force else "build_run",
            action_data={"force": force, "config": config},
        )

    if cmd == "/tag":
        max_chunks, error = parse_tag_args(args.strip())
        if error:
            return CommandResult(output=error)
        return CommandResult(
            output=get_tag_confirmation_prompt(config, max_chunks),
            action="tag_confirm",
            action_data={"max_chunks": max_chunks, "provider": provider, "config": config},
        )

    if cmd == "/tags":
        return CommandResult(output=format_tags(provider))

    if cmd == "/vectordb-providers":
        return CommandResult(output=format_vectordb_providers(config))

    if cmd == "/vectordb-status":
        return CommandResult(output=format_vectordb_status(config))

    if cmd == "/wipe":
        return CommandResult(
            output=get_wipe_warning(provider),
            action="wipe_confirm",
            action_data={"config": config},
        )

    return CommandResult(output=f"Unknown command: {cmd}\nType /help for available commands.")

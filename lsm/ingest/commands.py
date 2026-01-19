"""
Command handlers for ingest operations.

Contains handlers for: /info, /stats, /explore, /show, /search, /build, /tag, /tags, /wipe,
/vectordb-providers, /vectordb-status.
"""

from __future__ import annotations

import fnmatch
import time
from pathlib import Path
from typing import Any, Dict, Optional

from chromadb.api.models.Collection import Collection

from lsm.config.models import LSMConfig
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
from .display import print_tree


def _source_matches_pattern(source_path: str, pattern: str) -> bool:
    """Check if a source path matches a glob pattern."""
    name = Path(source_path).name
    return (
        fnmatch.fnmatch(source_path.lower(), pattern.lower())
        or fnmatch.fnmatch(name.lower(), pattern.lower())
    )


def handle_info_command(collection: Collection) -> None:
    """Handle /info command."""
    if getattr(collection, "name", "") != "chromadb":
        stats = collection.get_stats()
        print()
        print("=" * 60)
        print("VECTOR DB INFO")
        print("=" * 60)
        print(f"Provider: {stats.get('provider', 'unknown')}")
        print(f"Status:   {stats.get('status', 'unknown')}")
        print(f"Count:    {collection.count():,}")
        print("=" * 60)
        print()
        return

    info = get_collection_info(collection)

    print()
    print("=" * 60)
    print("COLLECTION INFO")
    print("=" * 60)
    print(f"Name:     {info['name']}")
    print(f"ID:       {info['id']}")
    print(f"Chunks:   {info['count']:,}")

    if "metadata" in info and info["metadata"]:
        print("\nMetadata:")
        for key, value in info["metadata"].items():
            print(f"  {key}: {value}")

    print("=" * 60)
    print()


def handle_stats_command(
    collection: Collection,
    config: LSMConfig,
    progress_callback=None,
) -> None:
    """Handle /stats command."""
    print("\nAnalyzing collection... (this may take a moment)")
    last_report = time.time()
    total = None
    try:
        total = collection.count()
    except Exception:
        total = None

    def report_progress(analyzed: int) -> None:
        nonlocal last_report
        if time.time() - last_report >= 2.0:
            if total:
                pct = (analyzed / total) * 100 if total > 0 else 0.0
                print(f"  analyzed chunks: {analyzed:,} / {total:,} ({pct:.1f}%)")
            else:
                print(f"  analyzed chunks: {analyzed:,}")
            if progress_callback:
                progress_callback(analyzed, total)
            last_report = time.time()

    error_report_path = config.ingest.manifest.parent / "ingest_error_report.json"
    try:
        stats = get_collection_stats(
            collection,
            limit=None,
            error_report_path=error_report_path,
            progress_callback=report_progress,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    if "message" in stats:
        print(stats["message"])
        return

    if progress_callback and total:
        progress_callback(total, total)

    report = format_stats_report(stats)
    print(report)


def handle_explore_command(
    collection: Collection,
    query: Optional[str] = None,
    progress_callback=None,
    emit_tree=None,
) -> None:
    """Handle /explore command."""
    print("\nExploring collection...")
    print("Scanning metadata... (this may take a moment)")

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
    if emit_tree:
        emit_tree(tree, display_root)
    else:
        print_tree(tree, display_root)
    print()


def handle_show_command(collection: Collection, file_path: str) -> None:
    """Handle /show command."""
    if not file_path:
        print("Usage: /show <file_path>")
        return

    print(f"\nFetching chunks for: {file_path}")

    chunks = get_file_chunks(collection, file_path)

    if not chunks:
        print("No chunks found for this file.")
        print("\nTip: Use /explore to find the exact file path.")
        return

    print(f"\nFound {len(chunks)} chunks")
    print("=" * 60)

    for chunk in chunks:
        idx = chunk.get("chunk_index", 0)
        text = chunk.get("text", "")
        author = chunk.get("author", "")
        title = chunk.get("title", "")

        print(f"\nChunk {idx + 1}:")
        if title:
            print(f"Title: {title}")
        if author:
            print(f"Author: {author}")

        print("-" * 60)
        # Truncate long chunks
        if len(text) > 500:
            print(text[:500] + "...")
        else:
            print(text)

    print("\n" + "=" * 60)


def handle_search_command(collection: Collection, query: str) -> None:
    """Handle /search command."""
    if not query:
        print("Usage: /search <query>")
        return

    print(f"\nSearching for: {query}")

    results = search_metadata(collection, query=query, limit=50)

    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} results")
    print("-" * 60)

    for i, meta in enumerate(results[:20], 1):
        source_path = meta.get("source_path", "unknown")
        chunk_idx = meta.get("chunk_index", 0)
        ext = meta.get("ext", "")

        # Shorten path
        display_path = Path(source_path).name
        if len(source_path) > 50:
            display_path = "..." + source_path[-47:]

        print(f"{i:3}. [{ext:5s}] {display_path} (chunk {chunk_idx})")

    if len(results) > 20:
        print(f"\n... and {len(results) - 20} more results")

    print()


def handle_build_command(config: LSMConfig, force: bool = False) -> None:
    """Handle /build command."""
    print()
    if force:
        print("WARNING: --force will re-process ALL files")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return

    print("\nStarting ingest pipeline...")
    print("-" * 60)
    print("Press Ctrl+C to stop ingest.")

    try:
        # If force, we could clear manifest here
        # For now, the incremental logic handles it

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

        print("\nIngest completed successfully!")

    except Exception as e:
        print(f"\nError during ingest: {e}")


def handle_wipe_command(collection: Collection) -> None:
    """Handle /wipe command."""
    print()
    print("=" * 60)
    print("WARNING: DESTRUCTIVE OPERATION")
    print("=" * 60)
    print()
    print("This will DELETE ALL CHUNKS from the collection.")
    print("This action CANNOT be undone.")
    print()

    try:
        chroma = require_chroma_collection(collection, "/wipe")
    except Exception as e:
        print(f"Error: {e}")
        return

    count = chroma.count()
    print(f"Current collection has {count:,} chunks")
    print()

    confirm1 = input("Type 'DELETE' to confirm: ").strip()

    if confirm1 != "DELETE":
        print("Cancelled.")
        return

    confirm2 = input("Are you absolutely sure? (yes/no): ").strip().lower()

    if confirm2 != "yes":
        print("Cancelled.")
        return

    print("\nDeleting all chunks...")

    try:
        # Get all IDs and delete
        results = chroma.get(include=[])
        ids = results.get("ids", [])

        if ids:
            chroma.delete(ids=ids)
            print(f"Deleted {len(ids):,} chunks")
        else:
            print("Collection was already empty")

        print("Collection cleared successfully.")

    except Exception as e:
        print(f"Error during wipe: {e}")


def handle_tag_command(collection: Collection, config: LSMConfig, args: str) -> None:
    """Handle /tag command."""
    # Parse --max flag
    max_chunks = None
    if "--max" in args.lower():
        parts = args.split()
        try:
            max_idx = parts.index("--max")
            if max_idx + 1 < len(parts):
                max_chunks = int(parts[max_idx + 1])
        except (ValueError, IndexError):
            print("Invalid --max argument. Usage: /tag --max N")
            return

    print()
    print("=" * 60)
    print("AI CHUNK TAGGING")
    print("=" * 60)
    print()
    print("This will tag chunks that haven't been AI-tagged yet.")
    print("Tags are generated using the configured LLM model.")
    print()

    # Get tagging config
    tagging_config = config.llm.get_tagging_config()
    print(f"Using model: {tagging_config.model}")
    print(f"Provider: {tagging_config.provider}")

    if max_chunks:
        print(f"Max chunks to tag: {max_chunks}")
    else:
        print("Max chunks: unlimited")

    print()
    confirm = input("Proceed with tagging? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("Cancelled.")
        return

    print("\nStarting AI tagging...")
    print("-" * 60)

    try:
        tagged, failed = tag_chunks(
            collection=collection,
            llm_config=tagging_config,
            num_tags=3,
            batch_size=100,
            max_chunks=max_chunks,
            dry_run=False,
        )

        print()
        print("=" * 60)
        print("TAGGING COMPLETE")
        print("=" * 60)
        print(f"Successfully tagged: {tagged} chunks")
        print(f"Failed: {failed} chunks")
        print()

    except Exception as e:
        print(f"\nError during tagging: {e}")


def handle_tags_command(collection: Collection) -> None:
    """Handle /tags command."""
    print("\nFetching all tags...")

    try:
        all_tags = get_all_tags(collection)

        ai_tags = all_tags.get("ai_tags", [])
        user_tags = all_tags.get("user_tags", [])

        print()
        print("=" * 60)
        print("COLLECTION TAGS")
        print("=" * 60)
        print()

        if ai_tags:
            print(f"AI-Generated Tags ({len(ai_tags)} unique):")
            print("-" * 60)
            # Display in columns for better readability
            for i in range(0, len(ai_tags), 3):
                row = ai_tags[i:i+3]
                print("  " + " | ".join(f"{tag:20s}" for tag in row))
            print()
        else:
            print("No AI-generated tags found.")
            print("Run /tag to generate tags for your chunks.")
            print()

        if user_tags:
            print(f"User Tags ({len(user_tags)} unique):")
            print("-" * 60)
            for i in range(0, len(user_tags), 3):
                row = user_tags[i:i+3]
                print("  " + " | ".join(f"{tag:20s}" for tag in row))
            print()
        else:
            print("No user tags found.")
            print()

        print("=" * 60)

    except Exception as e:
        print(f"Error fetching tags: {e}")


def handle_vectordb_providers_command(config: LSMConfig) -> None:
    """Handle /vectordb-providers command."""
    print()
    print("=" * 60)
    print("AVAILABLE VECTOR DB PROVIDERS")
    print("=" * 60)
    print()

    providers = list_available_providers()
    if not providers:
        print("No vector DB providers registered.")
        print()
        return

    current_provider = config.vectordb.provider
    print(f"Current Provider: {current_provider}")
    print(f"Collection:       {config.vectordb.collection}")
    print()

    print(f"Available Providers ({len(providers)}):")
    print()

    for provider_name in providers:
        is_current = "ACTIVE" if provider_name == current_provider else ""
        status = ""
        if provider_name == current_provider:
            try:
                provider = create_vectordb_provider(config.vectordb)
                status = "ok" if provider.is_available() else "unavailable"
            except Exception as e:
                status = f"error ({e})"
        print(f"  {provider_name:20s} {status:20s} {is_current}")

    print()


def handle_vectordb_status_command(config: LSMConfig) -> None:
    """Handle /vectordb-status command."""
    print()
    print("=" * 60)
    print("VECTOR DB STATUS")
    print("=" * 60)
    print()

    try:
        provider = create_vectordb_provider(config.vectordb)
        health = provider.health_check()
        stats = provider.get_stats()

        print(f"Provider: {health.get('provider', 'unknown')}")
        print(f"Status:   {health.get('status', 'unknown')}")
        if health.get("error"):
            print(f"Error:    {health.get('error')}")
        print(f"Count:    {stats.get('count', 'n/a')}")
    except Exception as e:
        print(f"Error: {e}")

    print()

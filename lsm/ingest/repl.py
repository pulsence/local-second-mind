"""
Interactive REPL for ingest pipeline management.

Provides commands for exploring, managing, and introspecting the knowledge base.
"""

from __future__ import annotations

import sys
import fnmatch
import os
import time
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

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
    add_user_tags,
    remove_user_tags,
    get_all_tags,
)


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


def handle_stats_command(collection: Collection, config: LSMConfig) -> None:
    """Handle /stats command."""
    print("\nAnalyzing collection... (this may take a moment)")
    last_report = time.time()

    def report_progress(analyzed: int) -> None:
        nonlocal last_report
        if time.time() - last_report >= 2.0:
            print(f"  analyzed chunks: {analyzed:,}")
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

    report = format_stats_report(stats)
    print(report)


def _normalize_query_path(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("/", os.sep).replace("\\", os.sep)
    return normalized.strip(os.sep)


def _parse_explore_query(query: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], str, bool]:
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
        _normalize_query_path(path_filter) if path_filter else None,
        ext_filter.lower() if ext_filter else None,
        pattern.lower() if pattern else None,
        display_root,
        full_path,
    )


def _source_matches_pattern(source_path: str, pattern: str) -> bool:
    name = Path(source_path).name
    return (
        fnmatch.fnmatch(source_path.lower(), pattern.lower())
        or fnmatch.fnmatch(name.lower(), pattern.lower())
    )


def _new_tree_node(name: str) -> Dict[str, Any]:
    return {"name": name, "children": {}, "files": {}, "file_count": 0, "chunk_count": 0}


def _build_tree(
    file_stats: Dict[str, Dict[str, Any]],
    base_filter: Optional[str],
    common_parts: Tuple[str, ...],
) -> Dict[str, Any]:
    root = _new_tree_node("root")

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
            node = node["children"].setdefault(part, _new_tree_node(part))
            node["file_count"] += 1
            node["chunk_count"] += chunk_count

        node["files"][rel_parts[-1]] = chunk_count

    return root


def _print_tree(root: Dict[str, Any], label: str, max_entries: int = 200) -> None:
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


def _compute_common_parts(paths: Dict[str, Dict[str, Any]]) -> Tuple[str, ...]:
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


def handle_explore_command(collection: Collection, query: Optional[str] = None) -> None:
    """Handle /explore command."""
    print("\nExploring collection...")
    print("Scanning metadata... (this may take a moment)")

    path_filter, ext_filter, pattern, display_root, full_path = _parse_explore_query(query)
    where = {"ext": {"$eq": ext_filter}} if ext_filter else None

    file_stats: Dict[str, Dict[str, Any]] = {}
    scanned = 0
    last_report = time.time()
    report_every = 2.0

    for meta in iter_collection_metadatas(collection, where=where, batch_size=5000):
        scanned += 1
        if time.time() - last_report >= report_every:
            print(f"  scanned chunks: {scanned:,}")
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

    if not file_stats:
        print("No files found.")
        return

    common_parts = ()
    if not path_filter and not full_path:
        common_parts = _compute_common_parts(file_stats)

    tree = _build_tree(file_stats, path_filter, common_parts)
    _print_tree(tree, display_root)
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


def handle_command(line: str, collection: Collection, config: LSMConfig) -> bool:
    """
    Handle a command from the REPL.

    Args:
        line: Input line
        collection: ChromaDB collection
        config: LSM configuration

    Returns:
        True if command was handled, False if should exit
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit":
        print("\nGoodbye!")
        return False

    elif cmd == "/help":
        print_help()

    elif cmd == "/info":
        handle_info_command(collection)

    elif cmd == "/stats":
        handle_stats_command(collection, config)

    elif cmd == "/explore":
        query = args.strip() if args else None
        handle_explore_command(collection, query)

    elif cmd == "/show":
        handle_show_command(collection, args.strip())

    elif cmd == "/search":
        handle_search_command(collection, args.strip())

    elif cmd == "/build":
        force = "--force" in args.lower()
        handle_build_command(config, force)

    elif cmd == "/tag":
        handle_tag_command(collection, config, args.strip())

    elif cmd == "/tags":
        handle_tags_command(collection)

    elif cmd == "/vectordb-providers":
        handle_vectordb_providers_command(config)

    elif cmd == "/vectordb-status":
        handle_vectordb_status_command(config)

    elif cmd == "/wipe":
        handle_wipe_command(collection)

    else:
        print(f"Unknown command: {cmd}")
        print("Type /help for available commands.")

    return True


def run_ingest_repl(config: LSMConfig) -> int:
    """
    Run the interactive ingest REPL.

    Args:
        config: LSM configuration

    Returns:
        Exit code (0 for success)
    """
    # Get provider
    try:
        collection = create_vectordb_provider(config.vectordb)
    except Exception as e:
        print(f"Error: Could not connect to vector DB: {e}")
        return 1

    # Print banner
    print_banner()

    # Show initial info
    if getattr(collection, "name", "") == "chromadb":
        info = get_collection_info(collection)
        print(f"Connected to collection: {info['name']}")
        print(f"Current chunks: {info['count']:,}")
    else:
        stats = collection.get_stats()
        print(f"Connected to vector DB: {stats.get('provider', 'unknown')}")
        print(f"Current chunks: {collection.count():,}")
    print()
    print("Type /help for available commands.")
    print()

    # Main loop
    try:
        while True:
            try:
                line = input("> ").strip()

                if not line:
                    continue

                # Check if it's a command
                if line.startswith("/"):
                    should_continue = handle_command(line, collection, config)
                    if not should_continue:
                        break
                else:
                    print("Commands must start with '/'")
                    print("Type /help for available commands.")

            except EOFError:
                print("\nGoodbye!")
                break

            except KeyboardInterrupt:
                print("\nUse /exit to quit")
                continue

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0

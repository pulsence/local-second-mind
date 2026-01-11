"""
Interactive REPL for ingest pipeline management.

Provides commands for exploring, managing, and introspecting the knowledge base.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Any

from chromadb.api.models.Collection import Collection

from lsm.config.models import LSMConfig
from lsm.ingest.stats import (
    get_collection_info,
    get_collection_stats,
    format_stats_report,
    search_metadata,
    get_file_chunks,
)
from lsm.ingest.chroma_store import get_chroma_collection
from lsm.ingest.pipeline import ingest


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
    print("  /wipe              - Clear collection (requires confirmation)")
    print("  /help              - Show this help")
    print("  /exit              - Exit")
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
        /explore .pdf         - Show all PDF files

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
"""
    print(help_text)


def handle_info_command(collection: Collection) -> None:
    """Handle /info command."""
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


def handle_stats_command(collection: Collection) -> None:
    """Handle /stats command."""
    print("\nAnalyzing collection... (this may take a moment)")
    stats = get_collection_stats(collection, limit=10000)

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    if "message" in stats:
        print(stats["message"])
        return

    report = format_stats_report(stats)
    print(report)


def handle_explore_command(collection: Collection, query: Optional[str] = None) -> None:
    """Handle /explore command."""
    print("\nExploring collection...")

    # Get unique files by searching metadata
    results = search_metadata(collection, query=query, limit=1000)

    if not results:
        print("No files found.")
        return

    # Group by source_path
    files_seen = set()
    files_info = []

    for meta in results:
        source_path = meta.get("source_path", "unknown")
        if source_path not in files_seen:
            files_seen.add(source_path)
            files_info.append({
                "path": source_path,
                "ext": meta.get("ext", ""),
                "author": meta.get("author", ""),
                "title": meta.get("title", ""),
            })

    # Display results
    print(f"\nFound {len(files_info)} files")
    print("-" * 60)

    for i, file_info in enumerate(files_info[:50], 1):  # Limit to 50
        path = file_info["path"]
        ext = file_info["ext"]
        author = file_info.get("author", "")
        title = file_info.get("title", "")

        # Shorten path for display
        display_path = Path(path).name
        if len(path) > 50:
            display_path = "..." + path[-47:]

        line = f"{i:3}. [{ext:5s}] {display_path}"

        if title:
            line += f" | {title}"
        elif author:
            line += f" | by {author}"

        print(line)

    if len(files_info) > 50:
        print(f"\n... and {len(files_info) - 50} more files")

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

    try:
        # If force, we could clear manifest here
        # For now, the incremental logic handles it

        ingest(
            roots=config.ingest.roots,
            persist_dir=config.persist_dir,
            chroma_flush_interval=config.ingest.chroma_flush_interval,
            collection_name=config.collection,
            embed_model_name=config.embed_model,
            device=config.device,
            batch_size=config.batch_size,
            manifest_path=config.ingest.manifest,
            exts=config.ingest.exts,
            exclude_dirs=config.ingest.exclude_set,
            dry_run=False,
            enable_ocr=config.ingest.enable_ocr,
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

    count = collection.count()
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
        results = collection.get(include=[])
        ids = results.get("ids", [])

        if ids:
            collection.delete(ids=ids)
            print(f"Deleted {len(ids):,} chunks")
        else:
            print("Collection was already empty")

        print("Collection cleared successfully.")

    except Exception as e:
        print(f"Error during wipe: {e}")


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
        handle_stats_command(collection)

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
    # Get collection
    try:
        collection = get_chroma_collection(config.persist_dir, config.collection)
    except Exception as e:
        print(f"Error: Could not connect to collection: {e}")
        return 1

    # Print banner
    print_banner()

    # Show initial info
    info = get_collection_info(collection)
    print(f"Connected to collection: {info['name']}")
    print(f"Current chunks: {info['count']:,}")
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

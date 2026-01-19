"""
Interactive REPL for ingest pipeline management.

Deprecated: interactive ingest runs only in the TUI; command handlers remain
for reuse by the TUI screens.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from lsm.config.models import LSMConfig
from .display import print_help
from .commands import (
    handle_info_command,
    handle_stats_command,
    handle_explore_command,
    handle_show_command,
    handle_search_command,
    handle_build_command,
    handle_tag_command,
    handle_tags_command,
    handle_wipe_command,
    handle_vectordb_providers_command,
    handle_vectordb_status_command,
)


def handle_command(
    line: str,
    collection: Collection,
    config: LSMConfig,
    stats_progress_callback=None,
    explore_progress_callback=None,
    explore_emit_tree=None,
) -> bool:
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
        handle_stats_command(
            collection,
            config,
            progress_callback=stats_progress_callback,
        )

    elif cmd == "/explore":
        query = args.strip() if args else None
        handle_explore_command(
            collection,
            query,
            progress_callback=explore_progress_callback,
            emit_tree=explore_emit_tree,
        )

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
    print("Interactive ingest is now TUI-only. Run `lsm` to launch the TUI.")
    return 2

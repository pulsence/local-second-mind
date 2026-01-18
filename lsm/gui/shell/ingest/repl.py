"""
Interactive REPL for ingest pipeline management.

Provides the main loop and command dispatcher for the ingest REPL.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from lsm.config.models import LSMConfig
from lsm.ingest.stats import get_collection_info
from lsm.vectordb import create_vectordb_provider
from .display import print_banner, print_help
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

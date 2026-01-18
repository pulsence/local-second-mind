"""
Unified interactive shell for LSM.

Provides a single REPL where users can switch between ingest and query contexts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Literal

from lsm.config import load_config_from_file
from lsm.config.models import LSMConfig
from lsm.gui.shell.logging import get_logger

logger = get_logger(__name__)

ContextType = Literal["ingest", "query"]


class UnifiedShell:
    """
    Unified interactive shell that supports both ingest and query contexts.

    Users can switch between contexts using /ingest and /query commands.
    """

    def __init__(self, config: LSMConfig):
        """
        Initialize the unified shell.

        Args:
            config: LSM configuration
        """
        self.config = config
        self.current_context: Optional[ContextType] = None

        # Lazy-loaded context objects
        self._ingest_provider = None
        self._query_embedder = None
        self._query_provider = None
        self._query_state = None

    def print_banner(self) -> None:
        """Print welcome banner for unified shell."""
        print()
        print("=" * 70)
        print(" " * 20 + "Local Second Mind (LSM)")
        print("=" * 70)
        print()
        print("Welcome to the unified LSM shell!")
        print()
        print("Global Commands:")
        print("  /ingest         - Switch to ingest context")
        print("  /query          - Switch to query context")
        print("  /help           - Show context-specific help")
        print("  /exit           - Exit shell")
        print()
        print("Type /ingest or /query to get started.")
        print()

    def _init_ingest_context(self) -> None:
        """Initialize ingest context (lazy)."""
        if self._ingest_provider is not None:
            return

        logger.info("Initializing ingest context")
        from lsm.vectordb import create_vectordb_provider

        try:
            self._ingest_provider = create_vectordb_provider(self.config.vectordb)
            logger.info("Ingest context initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ingest context: {e}")
            raise

    def _init_query_context(self) -> None:
        """Initialize query context (lazy)."""
        if self._query_provider is not None:
            return

        logger.info("Initializing query context")
        from lsm.query.retrieval import init_embedder
        from lsm.query.session import SessionState
        from lsm.query.cost_tracking import CostTracker
        from lsm.vectordb import create_vectordb_provider

        try:
            # Initialize embedder
            self._query_embedder = init_embedder(
                self.config.embed_model,
                device=self.config.device
            )

            # Initialize vector DB provider
            if self.config.vectordb.provider == "chromadb":
                persist_dir = Path(self.config.persist_dir)
                if not persist_dir.exists():
                    print(f"Error: ChromaDB directory not found: {persist_dir}")
                    print("Run /ingest first to create the database.")
                    raise FileNotFoundError(f"Persist directory not found: {persist_dir}")

            self._query_provider = create_vectordb_provider(self.config.vectordb)

            # Check collection has data
            count = self._query_provider.count()
            if count == 0:
                print(f"Warning: Collection '{self.config.collection}' is empty.")
                print("Run /ingest to populate the database.")

            # Initialize session state
            query_config = self.config.llm.get_query_config()
            self._query_state = SessionState(
                model=query_config.model,
                cost_tracker=CostTracker(),
            )

            logger.info(f"Query context initialized with {count} chunks")

        except Exception as e:
            logger.error(f"Failed to initialize query context: {e}")
            raise

    def switch_to_ingest(self) -> None:
        """Switch to ingest context."""
        try:
            self._init_ingest_context()
            self.current_context = "ingest"

            # Show info
            from lsm.ingest.stats import get_collection_info
            print()
            print("=" * 70)
            print("Switched to INGEST context")
            print("=" * 70)
            if getattr(self._ingest_provider, "name", "") == "chromadb":
                info = get_collection_info(self._ingest_provider)
                print(f"Collection: {info['name']}")
                print(f"Chunks:     {info['count']:,}")
            else:
                stats = self._ingest_provider.get_stats()
                print(f"Vector DB:  {stats.get('provider', 'unknown')}")
                print(f"Chunks:     {self._ingest_provider.count():,}")
            print()
            print("Type /help for ingest commands, or /query to switch to query mode.")
            print()

        except Exception as e:
            print(f"Error switching to ingest context: {e}")
            logger.error(f"Failed to switch to ingest: {e}", exc_info=True)

    def switch_to_query(self) -> None:
        """Switch to query context."""
        try:
            self._init_query_context()
            self.current_context = "query"

            count = self._query_provider.count()

            print()
            print("=" * 70)
            print("Switched to QUERY context")
            print("=" * 70)
            print(f"Collection: {self.config.collection}")
            print(f"Chunks:     {count:,}")
            feature_map = self.config.llm.get_feature_provider_map()
            for feature in ("query", "tagging", "ranking"):
                if feature not in feature_map:
                    continue
                cfg = {
                    "query": self.config.llm.get_query_config(),
                    "tagging": self.config.llm.get_tagging_config(),
                    "ranking": self.config.llm.get_ranking_config(),
                }[feature]
                label = {"query": "query", "tagging": "tag", "ranking": "rerank"}[feature]
                provider = cfg.provider
                if provider in {"anthropic", "claude"}:
                    provider = "claude"
                print(f"{label:7s} {provider}/{cfg.model}")
            print()
            print("Type your question or /help for commands, or /ingest to switch to ingest mode.")
            print()

        except Exception as e:
            print(f"Error switching to query context: {e}")
            logger.error(f"Failed to switch to query: {e}", exc_info=True)

    def handle_ingest_command(self, line: str) -> bool:
        """
        Handle command in ingest context.

        Args:
            line: Input line

        Returns:
            True to continue, False to exit
        """
        from lsm.gui.shell.ingest import handle_command

        # Check for context switch commands
        if line.strip().lower() in ("/query", "/q"):
            self.switch_to_query()
            return True

        # Delegate to ingest command handler
        return handle_command(line, self._ingest_provider, self.config)

    def handle_query_command(self, line: str) -> bool:
        """
        Handle command or question in query context.

        Args:
            line: Input line

        Returns:
            True to continue, False to exit
        """
        # Check for context switch commands
        line_lower = line.strip().lower()
        if line_lower in ("/ingest", "/i"):
            self.switch_to_ingest()
            return True

        # Handle query commands and questions
        from lsm.gui.shell.query.commands import handle_command
        from lsm.gui.shell.query.repl import run_query_turn

        try:
            # Check if it's a command
            is_command = handle_command(
                line,
                self._query_state,
                self.config,
                self._query_embedder,
                self._query_provider,
            )

            # If not a command, treat as question
            if not is_command:
                run_query_turn(
                    line,
                    self.config,
                    self._query_state,
                    self._query_embedder,
                    self._query_provider,
                )

            return True

        except SystemExit:
            # User wants to exit
            return False

        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Query error: {e}", exc_info=True)
            return True

    def run(self) -> int:
        """
        Run the unified shell.

        Returns:
            Exit code (0 for success)
        """
        self.print_banner()

        # Main loop
        try:
            while True:
                try:
                    # Show context-aware prompt
                    if self.current_context == "ingest":
                        prompt = "[ingest] > "
                    elif self.current_context == "query":
                        prompt = "[query] > "
                    else:
                        prompt = "> "

                    line = input(prompt).strip()

                    if not line:
                        continue

                    # Handle global commands
                    line_lower = line.lower()

                    if line_lower in ("/exit", "/quit"):
                        print("\nGoodbye!")
                        return 0

                    elif line_lower in ("/ingest", "/i"):
                        self.switch_to_ingest()
                        continue

                    elif line_lower in ("/query", "/q"):
                        self.switch_to_query()
                        continue

                    elif line_lower == "/help":
                        self.show_help()
                        continue

                    # Dispatch to context-specific handler
                    if self.current_context == "ingest":
                        should_continue = self.handle_ingest_command(line)
                        if not should_continue:
                            print("\nGoodbye!")
                            return 0

                    elif self.current_context == "query":
                        should_continue = self.handle_query_command(line)
                        if not should_continue:
                            print("\nGoodbye!")
                            return 0

                    else:
                        # No context selected yet
                        if line.startswith("/"):
                            print("Please select a context first: /ingest or /query")
                        else:
                            print("Please switch to a context: /ingest or /query")

                except EOFError:
                    print("\nGoodbye!")
                    return 0

                except KeyboardInterrupt:
                    print("\nUse /exit to quit")
                    continue

        except Exception as e:
            logger.error(f"Shell error: {e}", exc_info=True)
            print(f"\nUnexpected error: {e}")
            return 1

    def show_help(self) -> None:
        """Show context-specific help."""
        if self.current_context == "ingest":
            from lsm.gui.shell.ingest.display import print_help
            print_help()
        elif self.current_context == "query":
            from lsm.gui.shell.query.display import print_help
            print_help()
        else:
            print()
            print("Global Commands:")
            print("  /ingest, /i     - Switch to ingest context")
            print("  /query, /q      - Switch to query context")
            print("  /help           - Show this help")
            print("  /exit, /quit    - Exit shell")
            print()
            print("Switch to a context to see context-specific commands.")
            print()


def run_unified_shell(config: LSMConfig) -> int:
    """
    Run the unified interactive shell.

    Args:
        config: LSM configuration

    Returns:
        Exit code (0 for success)
    """
    shell = UnifiedShell(config)
    return shell.run()

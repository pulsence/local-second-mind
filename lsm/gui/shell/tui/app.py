"""
Main TUI Application for LSM.

Provides a rich terminal interface using Textual with:
- Tabbed interface for Ingest/Query/Settings
- Keyboard shortcuts for navigation
- Reactive status display
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, TabbedContent, TabPane
from textual.reactive import reactive

from lsm.gui.shell.logging import get_logger

if TYPE_CHECKING:
    from lsm.config.models import LSMConfig

logger = get_logger(__name__)

# Type alias for context
ContextType = Literal["ingest", "query", "settings"]


class LSMApp(App):
    """
    Local Second Mind TUI Application.

    A rich terminal interface for document ingestion and semantic querying.
    """

    TITLE = "Local Second Mind"
    SUB_TITLE = "Personal Knowledge Management"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+i", "switch_ingest", "Ingest", show=True),
        Binding("ctrl+q", "switch_query", "Query", show=True),
        Binding("ctrl+s", "switch_settings", "Settings", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+d", "quit", "Quit", show=False),
    ]

    # Reactive properties for state
    current_context: reactive[ContextType] = reactive("query")
    chunk_count: reactive[int] = reactive(0)
    current_mode: reactive[str] = reactive("grounded")
    total_cost: reactive[float] = reactive(0.0)

    def __init__(
        self,
        config: "LSMConfig",
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the LSM TUI application.

        Args:
            config: LSM configuration
        """
        super().__init__(*args, **kwargs)
        self.config = config

        # Lazy-loaded providers (same pattern as UnifiedShell)
        self._ingest_provider = None
        self._query_embedder = None
        self._query_provider = None
        self._query_state = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        with TabbedContent(initial="query"):
            with TabPane("Query", id="query"):
                # Import here to avoid circular imports
                from lsm.gui.shell.tui.screens.query import QueryScreen
                yield QueryScreen(id="query-screen")

            with TabPane("Ingest", id="ingest"):
                from lsm.gui.shell.tui.screens.ingest import IngestScreen
                yield IngestScreen(id="ingest-screen")

            with TabPane("Settings", id="settings"):
                from lsm.gui.shell.tui.screens.settings import SettingsScreen
                yield SettingsScreen(id="settings-screen")

        yield Footer()

    async def on_mount(self) -> None:
        """Handle application mount - initialize providers."""
        logger.info("LSM TUI application mounted")

        # Try to initialize query context by default (most common use case)
        try:
            await self._async_init_query_context()
        except Exception as e:
            logger.warning(f"Could not initialize query context on mount: {e}")
            self.notify(
                f"Query context unavailable: {e}",
                severity="warning",
                timeout=5,
            )

    # -------------------------------------------------------------------------
    # Provider Initialization (lazy, async-friendly)
    # -------------------------------------------------------------------------

    async def _async_init_ingest_context(self) -> None:
        """Initialize ingest context asynchronously."""
        if self._ingest_provider is not None:
            return

        logger.info("Initializing ingest context")
        from lsm.vectordb import create_vectordb_provider

        try:
            # Run sync provider creation in thread to not block UI
            self._ingest_provider = await self.run_in_thread(
                lambda: create_vectordb_provider(self.config.vectordb)
            )
            logger.info("Ingest context initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ingest context: {e}")
            raise

    async def _async_init_query_context(self) -> None:
        """Initialize query context asynchronously."""
        if self._query_provider is not None:
            return

        logger.info("Initializing query context")
        from lsm.query.retrieval import init_embedder
        from lsm.query.session import SessionState
        from lsm.query.cost_tracking import CostTracker
        from lsm.vectordb import create_vectordb_provider

        try:
            # Initialize embedder (can be slow)
            self._query_embedder = await self.run_in_thread(
                lambda: init_embedder(
                    self.config.embed_model,
                    device=self.config.device,
                )
            )

            # Check persist directory for ChromaDB
            if self.config.vectordb.provider == "chromadb":
                persist_dir = Path(self.config.persist_dir)
                if not persist_dir.exists():
                    raise FileNotFoundError(
                        f"ChromaDB directory not found: {persist_dir}. "
                        "Run ingest first to create the database."
                    )

            # Initialize vector DB provider
            self._query_provider = await self.run_in_thread(
                lambda: create_vectordb_provider(self.config.vectordb)
            )

            # Get chunk count
            count = self._query_provider.count()
            self.chunk_count = count

            if count == 0:
                self.notify(
                    f"Collection '{self.config.collection}' is empty. Run ingest first.",
                    severity="warning",
                    timeout=5,
                )

            # Initialize session state
            query_config = self.config.llm.get_query_config()
            self._query_state = SessionState(
                model=query_config.model,
                cost_tracker=CostTracker(),
            )

            # Set initial mode
            self.current_mode = self.config.query.mode

            logger.info(f"Query context initialized with {count} chunks")

        except Exception as e:
            logger.error(f"Failed to initialize query context: {e}")
            raise

    # -------------------------------------------------------------------------
    # Actions (bound to keyboard shortcuts)
    # -------------------------------------------------------------------------

    def action_switch_ingest(self) -> None:
        """Switch to ingest tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "ingest"
        self.current_context = "ingest"
        logger.debug("Switched to ingest context")

    def action_switch_query(self) -> None:
        """Switch to query tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "query"
        self.current_context = "query"
        logger.debug("Switched to query context")

    def action_switch_settings(self) -> None:
        """Switch to settings tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "settings"
        self.current_context = "settings"
        logger.debug("Switched to settings context")

    def action_show_help(self) -> None:
        """Show help modal."""
        from lsm.gui.shell.tui.screens.help import HelpScreen
        self.push_screen(HelpScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        logger.info("User requested quit")
        self.exit()

    # -------------------------------------------------------------------------
    # Tab Change Handler
    # -------------------------------------------------------------------------

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Handle tab activation to update context."""
        tab_id = event.tab.id
        if tab_id:
            # Remove the "-tab" suffix if present
            context = tab_id.replace("-tab", "")
            if context in ("ingest", "query", "settings"):
                self.current_context = context
                logger.debug(f"Tab activated: {context}")

    # -------------------------------------------------------------------------
    # Public API for screens/widgets
    # -------------------------------------------------------------------------

    @property
    def ingest_provider(self):
        """Get the ingest provider (may be None if not initialized)."""
        return self._ingest_provider

    @property
    def query_embedder(self):
        """Get the query embedder (may be None if not initialized)."""
        return self._query_embedder

    @property
    def query_provider(self):
        """Get the query provider (may be None if not initialized)."""
        return self._query_provider

    @property
    def query_state(self):
        """Get the query session state (may be None if not initialized)."""
        return self._query_state

    def update_cost(self, amount: float) -> None:
        """
        Update the total cost counter.

        Args:
            amount: Cost amount to add
        """
        self.total_cost += amount

    def update_chunk_count(self, count: int) -> None:
        """
        Update the chunk count display.

        Args:
            count: New chunk count
        """
        self.chunk_count = count


def run_tui(config: "LSMConfig") -> int:
    """
    Run the TUI application.

    Args:
        config: LSM configuration

    Returns:
        Exit code (0 for success)
    """
    app = LSMApp(config)
    app.run()
    return 0

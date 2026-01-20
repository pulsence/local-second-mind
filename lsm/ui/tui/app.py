"""
Main TUI Application for LSM.

Provides a rich terminal interface using Textual with:
- Tabbed interface for Ingest/Query/Settings
- Keyboard shortcuts for navigation
- Reactive status display with StatusBar widget
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, TYPE_CHECKING
import logging
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, TabbedContent, TabPane, RichLog
from textual.reactive import reactive

from lsm.logging import get_logger
from lsm.ui.tui.widgets.status import StatusBar

if TYPE_CHECKING:
    from lsm.config.models import LSMConfig

logger = get_logger(__name__)

# Type alias for context
ContextType = Literal["ingest", "query", "settings", "remote"]


class LSMApp(App):
    """
    Local Second Mind TUI Application.

    A rich terminal interface for document ingestion and semantic querying.
    """

    TITLE = "Local Second Mind"
    SUB_TITLE = "Personal Knowledge Management"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+n", "switch_ingest", "Ingest", show=False),
        Binding("ctrl+q", "switch_query", "Query", show=False),
        Binding("ctrl+r", "switch_remote", "Remote", show=False),
        Binding("ctrl+s", "switch_settings", "Settings", show=False),
        Binding("ctrl+p", "switch_remote", "Remote", show=False),
        Binding("f1", "show_help", "Help", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
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

        # Lazy-loaded providers
        self._ingest_provider = None
        self._query_embedder = None
        self._query_provider = None
        self._query_state = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header(show_clock=True, icon="")

        with TabbedContent(initial="query"):
            with TabPane("Query (^q)", id="query"):
                # Import here to avoid circular imports
                from lsm.ui.tui.screens.query import QueryScreen
                yield QueryScreen(id="query-screen")

            with TabPane("Ingest (^n)", id="ingest"):
                from lsm.ui.tui.screens.ingest import IngestScreen
                yield IngestScreen(id="ingest-screen")

            with TabPane("Remote (^r)", id="remote"):
                from lsm.ui.tui.screens.remote import RemoteScreen
                yield RemoteScreen(id="remote-screen")

            with TabPane("Settings (^s)", id="settings"):
                from lsm.ui.tui.screens.settings import SettingsScreen
                yield SettingsScreen(id="settings-screen")

        # Status bar showing mode, chunks, cost
        yield StatusBar(id="main-status-bar")

        yield Footer()

    async def on_mount(self) -> None:
        """Handle application mount - initialize providers."""
        logger.info("LSM TUI application mounted")
        try:
            tabbed_content = self.query_one(TabbedContent)
            tabbed_content.active = "query"
            self.current_context = "query"
        except Exception:
            pass
        self._setup_tui_logging()

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

    def _setup_tui_logging(self) -> None:
        """Route LSM logs to the query log panel when running in TUI."""
        root_logger = logging.getLogger("lsm")
        root_logger.propagate = False

        streams_to_remove = {sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__}

        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in streams_to_remove:
                root_logger.removeHandler(handler)

        root_global = logging.getLogger()
        root_global.setLevel(root_logger.level)
        for handler in list(root_global.handlers):
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in streams_to_remove:
                root_global.removeHandler(handler)

        app = self
        if not hasattr(self, "_tui_log_buffer"):
            from collections import deque
            self._tui_log_buffer = deque(maxlen=500)
        if not hasattr(self, "_tui_stdout"):
            self._tui_stdout = sys.stdout
            self._tui_stderr = sys.stderr

            class _TUIStream:
                def __init__(self, app_instance: "LSMApp") -> None:
                    self._app = app_instance

                def write(self, message: str) -> int:
                    if not message:
                        return 0
                    self._app._tui_log_buffer.append(message.rstrip("\n"))

                    def update() -> None:
                        try:
                            self._app._write_tui_log(message.rstrip())
                        except Exception:
                            pass

                    self._app.call_from_thread(update)
                    return len(message)

                def flush(self) -> None:
                    return None

            sys.stdout = _TUIStream(self)
            sys.stderr = _TUIStream(self)

        class _TUILogHandler(logging.Handler):
            def __init__(self, app_instance: "LSMApp") -> None:
                super().__init__()
                self._app = app_instance
                self._tui_handler = True

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    message = self.format(record)
                    self._app._tui_log_buffer.append(message)

                    def update() -> None:
                        try:
                            self._app._write_tui_log(message)
                        except Exception:
                            pass

                    self._app.call_from_thread(update)
                except Exception:
                    pass

        if not any(getattr(h, "_tui_handler", False) for h in root_logger.handlers):
            handler = _TUILogHandler(app)
            handler.setLevel(root_logger.level)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

        if not any(getattr(h, "_tui_handler", False) for h in root_global.handlers):
            handler = _TUILogHandler(app)
            handler.setLevel(root_logger.level)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            root_global.addHandler(handler)

    def _write_tui_log(self, message: str) -> None:
        """Write log messages to available TUI log widgets."""
        for selector in ("#query-log", "#remote-log"):
            try:
                log_widget = self.query_one(selector, RichLog)
                log_widget.write(f"{message}\n")
            except Exception:
                continue

    # -------------------------------------------------------------------------
    # Provider Initialization (lazy, async-friendly)
    # -------------------------------------------------------------------------

    async def _async_init_ingest_context(self) -> None:
        """Initialize ingest context asynchronously."""
        import asyncio

        if self._ingest_provider is not None:
            return

        logger.info("Initializing ingest context")
        from lsm.vectordb import create_vectordb_provider

        try:
            # Run sync provider creation in thread to not block UI
            self._ingest_provider = await asyncio.to_thread(
                create_vectordb_provider, self.config.vectordb
            )
            logger.info("Ingest context initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ingest context: {e}")
            raise

    async def _async_init_query_context(self) -> None:
        """Initialize query context asynchronously."""
        import asyncio

        if self._query_provider is not None:
            return

        logger.info("Initializing query context")
        from lsm.query.retrieval import init_embedder
        from lsm.query.session import SessionState
        from lsm.query.cost_tracking import CostTracker
        from lsm.vectordb import create_vectordb_provider

        try:
            # Initialize embedder (can be slow)
            self._query_embedder = await asyncio.to_thread(
                init_embedder,
                self.config.embed_model,
                device=self.config.device,
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
            self._query_provider = await asyncio.to_thread(
                create_vectordb_provider, self.config.vectordb
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

    def action_switch_remote(self) -> None:
        """Switch to remote tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "remote"
        self.current_context = "remote"
        logger.debug("Switched to remote context")

    def _activate_settings_tab(self, tab_id: str) -> None:
        """Switch to a settings sub-tab when settings is active."""
        if self.current_context != "settings":
            return
        try:
            settings_screen = self.query_one("#settings-screen")
            tabs = settings_screen.query_one("#settings-tabs", TabbedContent)
            tabs.active = tab_id
        except Exception:
            return

    def action_settings_tab_1(self) -> None:
        """Switch to the settings configuration tab."""
        self._activate_settings_tab("settings-config")

    def action_settings_tab_2(self) -> None:
        """Switch to the settings ingest tab."""
        self._activate_settings_tab("settings-ingest")

    def action_settings_tab_3(self) -> None:
        """Switch to the settings query tab."""
        self._activate_settings_tab("settings-query")

    def action_settings_tab_4(self) -> None:
        """Switch to the settings mode tab."""
        self._activate_settings_tab("settings-mode")

    def action_settings_tab_5(self) -> None:
        """Switch to the settings vector DB tab."""
        self._activate_settings_tab("settings-vdb")

    def action_settings_tab_6(self) -> None:
        """Switch to the settings LLM providers tab."""
        self._activate_settings_tab("settings-llm")

    def action_show_help(self) -> None:
        """Show help modal."""
        from lsm.ui.tui.screens.help import HelpScreen
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
            if context in ("ingest", "query", "settings", "remote"):
                self.current_context = context
                logger.debug(f"Tab activated: {context}")

    # -------------------------------------------------------------------------
    # Watch methods for StatusBar sync
    # -------------------------------------------------------------------------

    def watch_current_mode(self, mode: str) -> None:
        """Sync mode changes to StatusBar."""
        try:
            status_bar = self.query_one("#main-status-bar", StatusBar)
            status_bar.mode = mode
        except Exception:
            pass

    def watch_chunk_count(self, count: int) -> None:
        """Sync chunk count changes to StatusBar."""
        try:
            status_bar = self.query_one("#main-status-bar", StatusBar)
            status_bar.chunk_count = count
        except Exception:
            pass

    def watch_total_cost(self, cost: float) -> None:
        """Sync cost changes to StatusBar."""
        try:
            status_bar = self.query_one("#main-status-bar", StatusBar)
            status_bar.total_cost = cost
        except Exception:
            pass

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

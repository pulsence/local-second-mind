"""
Main TUI Application for LSM.

Provides a rich terminal interface using Textual with:
- Tabbed interface for Ingest/Query/Settings
- Keyboard shortcuts for navigation
- Reactive status display with StatusBar widget
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Literal, TYPE_CHECKING, cast
import logging
import sys
import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual import events
from textual.widgets import Header, Footer, TabbedContent, TabPane, RichLog
from textual.reactive import reactive
from textual.timer import Timer
from textual.message import Message

from lsm.logging import get_logger
from lsm.ui.tui.state import AppState
from lsm.ui.tui.widgets.status import StatusBar

if TYPE_CHECKING:
    from lsm.config.models import LSMConfig
    from lsm.ui.shell.commands.agents import AgentRuntimeUIEvent

logger = get_logger(__name__)

# Type alias for context
ContextType = Literal["ingest", "query", "settings", "remote", "agents"]
DensityMode = Literal["auto", "compact", "comfortable"]
EffectiveDensity = Literal["compact", "comfortable"]


class TUILogEvent(Message):
    """Queued log line from background logging threads."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()


class AgentRuntimeEvent(Message):
    """Queued runtime event from AgentRuntimeManager worker threads."""

    def __init__(self, event: "AgentRuntimeUIEvent") -> None:
        self.event = event
        super().__init__()


class LSMApp(App):
    """
    Local Second Mind TUI Application.

    A rich terminal interface for document ingestion and semantic querying.
    """

    TITLE = "Local Second Mind"
    SUB_TITLE = "Personal Knowledge Management"

    CSS_PATH = [
        "styles/base.tcss",
        "styles/widgets.tcss",
        "styles/query.tcss",
        "styles/ingest.tcss",
        "styles/settings.tcss",
        "styles/remote.tcss",
        "styles/agents.tcss",
    ]

    _DENSITY_AUTO_COMPACT_MAX_WIDTH = 100
    _DENSITY_AUTO_COMPACT_MAX_HEIGHT = 32
    _DENSITY_AUTO_COMFORTABLE_MIN_WIDTH = 106
    _DENSITY_AUTO_COMFORTABLE_MIN_HEIGHT = 34
    _DENSITY_NARROW_MAX_WIDTH = 100
    _DENSITY_RESIZE_DEBOUNCE_SECONDS = 0.15
    _NOTIFY_TIMEOUT_SECONDS = 5.0
    _NOTIFY_ERROR_TIMEOUT_SECONDS = 10.0

    BINDINGS = [
        Binding("ctrl+n", "switch_ingest", "Ingest", show=False),
        Binding("ctrl+q", "switch_query", "Query", show=False),
        Binding("ctrl+r", "switch_remote", "Remote", show=False),
        Binding("ctrl+g", "switch_agents", "Agents", show=False),
        Binding("ctrl+s", "switch_settings", "Settings", show=False),
        Binding("ctrl+p", "switch_remote", "Remote", show=False),
        Binding("f1", "show_help", "Help", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+z", "quit", "Quit", show=False),
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

        configured_density = cast(
            str,
            getattr(getattr(self.config, "global_settings", None), "tui_density_mode", "auto"),
        ).strip().lower()
        if configured_density not in {"auto", "compact", "comfortable"}:
            configured_density = "auto"
        self._density_mode: DensityMode = cast(DensityMode, configured_density)
        self._effective_density: EffectiveDensity = "comfortable"
        self._pending_resize_dimensions: tuple[int, int] | None = None
        self._density_resize_timer: Timer | None = None
        self._ui_thread_id: int = threading.get_ident()
        self._strict_ui_thread_checks: bool = False
        self.ui_state = AppState(active_context="query", density_mode=self._density_mode)

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header(show_clock=True, icon="")

        with TabbedContent(initial="query", id="main-tabs"):
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

            with TabPane("Agents (^g)", id="agents"):
                from lsm.ui.tui.screens.agents import AgentsScreen
                yield AgentsScreen(id="agents-screen")

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
            self._set_active_context("query")
        except Exception:
            pass
        width, height = self._terminal_size()
        self._update_responsive_classes(width, height)
        self._apply_density_mode(force=True)
        self._setup_tui_logging()
        self._bind_agent_runtime_events()
        logger.info("Query context initialization is lazy and will run on first query.")

    def _setup_tui_logging(self) -> None:
        """Route LSM logs to the query log panel when running in TUI."""
        root_logger = logging.getLogger("lsm")
        root_logger.propagate = False

        streams_to_remove = {sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__}

        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in streams_to_remove:
                root_logger.removeHandler(handler)

        app = self
        if not hasattr(self, "_tui_log_buffer"):
            from collections import deque
            self._tui_log_buffer = deque(maxlen=500)

        class _TUILogHandler(logging.Handler):
            def __init__(self, app_instance: "LSMApp") -> None:
                super().__init__()
                self._app = app_instance
                self._tui_handler = True

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    message = self.format(record)
                    self._app._tui_log_buffer.append(message)
                    self._app.post_ui_message(
                        TUILogEvent(message),
                        source="logging",
                    )
                except Exception:
                    pass

        if not any(getattr(h, "_tui_handler", False) for h in root_logger.handlers):
            handler = _TUILogHandler(app)
            handler.setLevel(root_logger.level)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

    def _write_tui_log(self, message: str) -> None:
        """Write log messages to available TUI log widgets."""
        self.assert_ui_thread(action="_write_tui_log")
        context = getattr(self, "current_context", "query")
        selector = "#remote-log" if context == "remote" else "#query-log"
        try:
            log_widget = self.query_one(selector, RichLog)
            log_widget.write(f"{message}\n")
        except Exception:
            return

    def on_tui_log_event(self, message: TUILogEvent) -> None:
        """Handle queued log events on the UI thread."""
        self._write_tui_log(message.message)

    def post_ui_message(self, message: Message, *, source: str = "unknown") -> None:
        """Queue a Textual message to this app from any thread."""
        _ = source
        self.run_on_ui_thread(lambda: self.post_message(message))

    def run_on_ui_thread(self, callback) -> None:
        """Run a callback on the app UI thread from either same or worker thread."""
        try:
            self.call_from_thread(callback)
        except RuntimeError as exc:
            # Textual raises this when already on the app thread.
            if "must run in a different thread" in str(exc):
                callback()
                return
            raise

    def assert_ui_thread(self, *, action: str = "") -> bool:
        """
        Detect unsafe off-thread UI mutation attempts.

        Returns:
            True when running on the UI thread.
        """
        current_id = threading.get_ident()
        if current_id == self._ui_thread_id:
            return True
        detail = f" during {action}" if action else ""
        message = (
            "Off-thread UI mutation attempt detected"
            f"{detail}: current_thread={current_id}, ui_thread={self._ui_thread_id}"
        )
        logger.error(message)
        if self._strict_ui_thread_checks:
            raise RuntimeError(message)
        return False

    def _bind_agent_runtime_events(self) -> None:
        """Bind manager runtime events to app-level queued UI messages."""
        try:
            from lsm.ui.shell.commands.agents import get_agent_runtime_manager

            manager = get_agent_runtime_manager()
            binder = getattr(manager, "set_ui_event_sink", None)
            if callable(binder):
                binder(self._on_agent_runtime_event_from_any_thread)
        except Exception:
            logger.exception("Failed to bind agent runtime UI event sink")

    def _unbind_agent_runtime_events(self) -> None:
        """Detach manager runtime event sink during app shutdown."""
        try:
            from lsm.ui.shell.commands.agents import get_agent_runtime_manager

            manager = get_agent_runtime_manager()
            binder = getattr(manager, "set_ui_event_sink", None)
            if callable(binder):
                binder(None)
        except Exception:
            logger.exception("Failed to unbind agent runtime UI event sink")

    def _on_agent_runtime_event_from_any_thread(
        self,
        event: "AgentRuntimeUIEvent",
    ) -> None:
        """Queue runtime events emitted from worker threads."""
        self.post_ui_message(
            AgentRuntimeEvent(event),
            source="agent-runtime",
        )

    def on_agent_runtime_event(self, message: AgentRuntimeEvent) -> None:
        """Forward runtime events to the agents screen via message queue."""
        try:
            from lsm.ui.tui.screens.agents import AgentRuntimeEventMessage

            agents_screen = self.query_one("#agents-screen")
            agents_screen.post_message(AgentRuntimeEventMessage(message.event))
        except Exception:
            return

    # -------------------------------------------------------------------------
    # Density and Responsive Layout
    # -------------------------------------------------------------------------

    @property
    def density_mode(self) -> DensityMode:
        """Configured density mode (auto, compact, comfortable)."""
        return self._density_mode

    @property
    def effective_density(self) -> EffectiveDensity:
        """Effective runtime density (compact or comfortable)."""
        return self._effective_density

    def density_status_text(self) -> str:
        """Return a human-readable status summary for density settings."""
        width, height = self._terminal_size()
        return (
            f"TUI density mode: {self._density_mode}\n"
            f"Active density: {self._effective_density}\n"
            f"Terminal size: {width}x{height}\n"
            "Auto thresholds: compact when width <= 100 or height <= 32."
        )

    def set_density_mode(self, mode: str) -> tuple[bool, str]:
        """
        Set the configured TUI density mode.

        Args:
            mode: One of auto, compact, comfortable.

        Returns:
            Tuple of success flag and status message.
        """
        normalized = mode.strip().lower()
        if normalized not in {"auto", "compact", "comfortable"}:
            return (
                False,
                "Invalid density mode. Use: auto, compact, comfortable.",
            )

        self._density_mode = cast(DensityMode, normalized)
        self.ui_state.set_density_mode(self._density_mode)

        # Keep config in sync so save operations persist the latest mode choice.
        global_settings = getattr(self.config, "global_settings", None)
        if global_settings is not None and hasattr(global_settings, "tui_density_mode"):
            global_settings.tui_density_mode = self._density_mode

        if self._density_mode != "auto":
            if self._density_resize_timer is not None:
                self._density_resize_timer.stop()
                self._density_resize_timer = None
            self._pending_resize_dimensions = None

        self._apply_density_mode(force=True)
        return (True, self.density_status_text())

    def on_resize(self, event: events.Resize) -> None:
        """Handle terminal resize events for responsive layout and auto density."""
        width = int(getattr(event.size, "width", 0))
        height = int(getattr(event.size, "height", 0))
        self._update_responsive_classes(width, height)
        if self._density_mode == "auto":
            self._schedule_auto_density_recalc(width, height)

    def _schedule_auto_density_recalc(self, width: int, height: int) -> None:
        """Debounce auto-density updates while the terminal is being resized."""
        self._pending_resize_dimensions = (width, height)
        if self._density_resize_timer is not None:
            self._density_resize_timer.stop()
        self._density_resize_timer = self.set_timer(
            self._DENSITY_RESIZE_DEBOUNCE_SECONDS,
            self._apply_pending_auto_density,
        )

    def _apply_pending_auto_density(self) -> None:
        """Apply auto density using the most recent pending terminal size."""
        self._density_resize_timer = None
        if self._density_mode != "auto":
            self._pending_resize_dimensions = None
            return
        width, height = self._pending_resize_dimensions or self._terminal_size()
        self._pending_resize_dimensions = None
        self._apply_density_mode(width=width, height=height)

    def _apply_density_mode(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        force: bool = False,
    ) -> None:
        """Compute and apply the effective density class."""
        if width is None or height is None:
            width, height = self._terminal_size()

        if self._density_mode == "auto":
            effective = self._resolve_auto_density(width, height)
        else:
            effective = cast(EffectiveDensity, self._density_mode)

        self._set_effective_density(effective, force=force)

    def _resolve_auto_density(self, width: int, height: int) -> EffectiveDensity:
        """
        Resolve auto density from terminal dimensions with hysteresis.

        Hysteresis behavior:
        - Enter compact when width <= 100 or height <= 32.
        - Exit compact only when width >= 106 and height >= 34.
        """
        if self._effective_density == "compact":
            if (
                width >= self._DENSITY_AUTO_COMFORTABLE_MIN_WIDTH
                and height >= self._DENSITY_AUTO_COMFORTABLE_MIN_HEIGHT
            ):
                return "comfortable"
            return "compact"

        if (
            width <= self._DENSITY_AUTO_COMPACT_MAX_WIDTH
            or height <= self._DENSITY_AUTO_COMPACT_MAX_HEIGHT
        ):
            return "compact"
        return "comfortable"

    def _set_effective_density(self, density: EffectiveDensity, *, force: bool = False) -> None:
        """Apply effective density by toggling root CSS classes."""
        if not force and density == self._effective_density:
            return

        self._effective_density = density
        self.remove_class("density-compact")
        self.remove_class("density-comfortable")
        self.add_class(f"density-{density}")

    def _terminal_size(self) -> tuple[int, int]:
        """Return current terminal width/height with safe defaults."""
        size = getattr(self, "size", None)
        width = int(getattr(size, "width", 0) or 0)
        height = int(getattr(size, "height", 0) or 0)
        if width <= 0 or height <= 0:
            return (
                self._DENSITY_AUTO_COMPACT_MAX_WIDTH + 1,
                self._DENSITY_AUTO_COMPACT_MAX_HEIGHT + 1,
            )
        return (width, height)

    def _update_responsive_classes(self, width: int, _height: int) -> None:
        """Apply responsive breakpoint classes for narrow terminals."""
        if width <= self._DENSITY_NARROW_MAX_WIDTH:
            self.add_class("density-narrow")
        else:
            self.remove_class("density-narrow")

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

    def _set_active_context(self, context: ContextType) -> None:
        """Update active context in both reactive state and typed app state."""
        self.current_context = context
        self.ui_state.set_active_context(context)

    def action_switch_ingest(self) -> None:
        """Switch to ingest tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "ingest"
        self._set_active_context("ingest")
        logger.debug("Switched to ingest context")

    def action_switch_query(self) -> None:
        """Switch to query tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "query"
        self._set_active_context("query")
        logger.debug("Switched to query context")

    def action_switch_settings(self) -> None:
        """Switch to settings tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "settings"
        self._set_active_context("settings")
        self._refresh_settings_screen()
        logger.debug("Switched to settings context")

    def action_switch_remote(self) -> None:
        """Switch to remote tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "remote"
        self._set_active_context("remote")
        logger.debug("Switched to remote context")

    def action_switch_agents(self) -> None:
        """Switch to agents tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = "agents"
        self._set_active_context("agents")
        logger.debug("Switched to agents context")

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
        """Switch to the settings global tab."""
        self._activate_settings_tab("settings-global")

    def action_settings_tab_2(self) -> None:
        """Switch to the settings ingest tab."""
        self._activate_settings_tab("settings-ingest")

    def action_settings_tab_3(self) -> None:
        """Switch to the settings query tab."""
        self._activate_settings_tab("settings-query")

    def action_settings_tab_4(self) -> None:
        """Switch to the settings llm tab."""
        self._activate_settings_tab("settings-llm")

    def action_settings_tab_5(self) -> None:
        """Switch to the settings vector DB tab."""
        self._activate_settings_tab("settings-vdb")

    def action_settings_tab_6(self) -> None:
        """Switch to the settings modes tab."""
        self._activate_settings_tab("settings-modes")

    def action_settings_tab_7(self) -> None:
        """Switch to the settings remote tab."""
        self._activate_settings_tab("settings-remote")

    def action_settings_tab_8(self) -> None:
        """Switch to the settings chats/notes tab."""
        self._activate_settings_tab("settings-chats-notes")

    def action_show_help(self) -> None:
        """Show help modal."""
        from lsm.ui.tui.screens.help import HelpScreen
        self.push_screen(HelpScreen(context=self.current_context))

    def action_quit(self) -> None:
        """Quit the application."""
        logger.info("User requested quit")
        self._unbind_agent_runtime_events()
        self.exit()

    # -------------------------------------------------------------------------
    # Tab Change Handler
    # -------------------------------------------------------------------------

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Handle tab activation to update context."""
        # Ignore nested TabbedContent events (for example, settings sub-tabs).
        event_tabs = getattr(event, "tabbed_content", None)
        if event_tabs is not None and getattr(event_tabs, "id", None) != "main-tabs":
            try:
                root_tabs = self.query_one("#main-tabs", TabbedContent)
            except Exception:
                root_tabs = None
            if root_tabs is None or event_tabs is not root_tabs:
                return

        tab_id = event.tab.id
        if tab_id:
            # Remove the "-tab" suffix if present
            context = tab_id.replace("-tab", "")
            if context in ("ingest", "query", "settings", "remote", "agents"):
                self._set_active_context(cast(ContextType, context))
                if context == "settings":
                    self._refresh_settings_screen()
                logger.debug(f"Tab activated: {context}")

    def _refresh_settings_screen(self) -> None:
        """Refresh settings fields from the active in-memory config."""
        try:
            settings_screen = self.query_one("#settings-screen")
            refresh = getattr(settings_screen, "refresh_from_config", None)
            if callable(refresh):
                refresh()
        except Exception:
            return

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

    def watch_current_context(self, context: ContextType) -> None:
        """Keep typed app state context synchronized with reactive context changes."""
        self.ui_state.set_active_context(context)

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

    @property
    def selected_agent_id(self) -> Optional[str]:
        """Get currently selected agent id for cross-screen interactions."""
        return self.ui_state.selected_agent_id

    def set_selected_agent_id(self, agent_id: Optional[str]) -> None:
        """Set currently selected agent id for cross-screen interactions."""
        self.ui_state.set_selected_agent_id(agent_id)

    def notify_event(
        self,
        message: str,
        *,
        severity: Literal["info", "warning", "error"] = "info",
        timeout: Optional[float] = None,
    ) -> None:
        """Emit and record a UI notification event."""
        self.assert_ui_thread(action="notify_event")
        effective_timeout = timeout
        if effective_timeout is None:
            effective_timeout = (
                self._NOTIFY_ERROR_TIMEOUT_SECONDS
                if severity == "error"
                else self._NOTIFY_TIMEOUT_SECONDS
            )
        self.ui_state.push_notification(message, severity=severity)
        self.notify(
            message,
            severity=severity,
            timeout=effective_timeout,
        )

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

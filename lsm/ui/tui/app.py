"""
Main TUI Application for LSM.

Provides a rich terminal interface using Textual with:
- Tabbed interface for Ingest/Query/Settings
- Keyboard shortcuts for navigation
- Reactive status display with StatusBar widget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal, TYPE_CHECKING, cast, Callable
import logging
import sys
import threading
import time
import traceback

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual import events
from textual.widgets import Header, Footer, TabbedContent, TabPane, RichLog
from textual.reactive import reactive
from textual.timer import Timer
from textual.message import Message

from lsm.logging import get_logger, format_exception_summary, exception_exc_info
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


@dataclass
class _ManagedWorker:
    """Lifecycle metadata for an app-managed worker object."""

    owner: str
    key: str
    worker: Any
    timeout_s: float
    started_at: float = field(default_factory=time.monotonic)


@dataclass
class _ManagedTimer:
    """Lifecycle metadata for an app-managed timer object."""

    owner: str
    key: str
    timer: Timer
    started_at: float = field(default_factory=time.monotonic)


@dataclass
class _StartupMilestone:
    """Single startup timing measurement."""

    name: str
    elapsed_ms: float


@dataclass
class StartupTimeline:
    """Collects startup timing milestones for performance tracking.

    Records named milestones with elapsed time from timeline creation.
    Used to instrument TUI startup and enforce performance budgets.
    """

    _start_time: float = field(default_factory=time.monotonic)
    _milestones: list[_StartupMilestone] = field(default_factory=list)

    def mark(self, name: str) -> None:
        """Record a named milestone with current elapsed time."""
        elapsed = (time.monotonic() - self._start_time) * 1000
        self._milestones.append(_StartupMilestone(name=name, elapsed_ms=elapsed))

    def elapsed_ms(self, milestone_name: str) -> float | None:
        """Return elapsed time for a named milestone, or None if not found."""
        for m in self._milestones:
            if m.name == milestone_name:
                return m.elapsed_ms
        return None

    @property
    def milestones(self) -> list[_StartupMilestone]:
        """Return a copy of recorded milestones."""
        return list(self._milestones)

    def total_ms(self) -> float:
        """Return total elapsed time since timeline creation."""
        return (time.monotonic() - self._start_time) * 1000


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
    _DEFAULT_WORKER_TIMEOUT_SECONDS = 3.0
    _UI_ERROR_SUMMARY_MAX_LENGTH = 180

    BINDINGS = [
        Binding("ctrl+n", "switch_ingest", "Ingest", show=False),
        Binding("ctrl+q", "switch_query", "Query", show=False),
        Binding("ctrl+r", "switch_remote", "Remote", show=False),
        Binding("ctrl+g", "switch_agents", "Agents", show=False),
        Binding("ctrl+s", "switch_settings", "Settings", show=False),
        Binding("ctrl+p", "switch_remote", "Remote", show=False),
        Binding("f1", "show_help", "Help", show=False),
        Binding("f12", "return_safe_screen", "Safe Screen", show=False),
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
        self._startup_timeline = StartupTimeline()
        self._startup_timeline.mark("init_start")
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
        self._managed_workers: dict[tuple[str, str], _ManagedWorker] = {}
        self._managed_workers_lock = threading.RLock()
        self._managed_timers: dict[tuple[str, str], _ManagedTimer] = {}
        self._managed_timers_lock = threading.RLock()
        self._recovering_ui_error: bool = False
        self._ui_error_count: int = 0
        self._last_ui_error_summary: str = ""
        self.ui_state = AppState(active_context="query", density_mode=self._density_mode)
        self._agent_runtime_bound: bool = False
        self._startup_timeline.mark("init_complete")

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
                yield SettingsScreen(id="settings-screen", disabled=True)

        # Status bar showing mode, chunks, cost
        yield StatusBar(id="main-status-bar")

        yield Footer()

    async def on_mount(self) -> None:
        """Handle application mount - initialize providers."""
        self._startup_timeline.mark("mount_start")
        logger.info("LSM TUI application mounted")
        try:
            tabbed_content = self.query_one("#main-tabs", TabbedContent)
            tabbed_content.active = "query"
            self._set_active_context("query")
        except Exception:
            pass
        width, height = self._terminal_size()
        self._update_responsive_classes(width, height)
        self._apply_density_mode(force=True)
        self._startup_timeline.mark("query_interactive")
        self._setup_tui_logging()
        self._startup_timeline.mark("tui_logging_ready")
        self._schedule_background_init()
        self._startup_timeline.mark("mount_complete")
        logger.info(
            "Startup milestones: %s",
            [(m.name, f"{m.elapsed_ms:.0f}ms") for m in self._startup_timeline.milestones],
        )

    def _schedule_background_init(self) -> None:
        """Defer non-critical startup initialization past the first render."""

        def _background_init() -> None:
            try:
                self._bind_agent_runtime_events()
                self._startup_timeline.mark("agent_runtime_bound")
            except Exception:
                logger.exception("Background agent runtime binding failed")

            self._preload_ml_stack()

            self._startup_timeline.mark("background_init_complete")
            logger.info("Background initialization complete")

        def _start_background_thread() -> None:
            thread = threading.Thread(
                target=_background_init, name="bg-init", daemon=True,
            )
            thread.start()
            self.register_managed_worker(
                owner="app", key="bg-init", worker=thread, timeout_s=120.0,
            )

        self.call_after_refresh(_start_background_thread)

    def _preload_ml_stack(self) -> None:
        """Preload sentence_transformers and embedding model in background.

        The library imports (~60s) cause brief GIL gaps (max ~700ms) from
        C extension init in torch, but the UI stays responsive for 99%+ of
        the time.  The model load itself (~1s) is GIL-friendly with zero
        gaps over 50ms.  Once loaded, first query starts instantly.
        """
        # Phase 1: import the library (GIL gaps from C extensions, but not a lockup)
        try:
            logger.info("Background: importing embedding framework...")
            self._startup_timeline.mark("ml_import_start")
            from lsm.query.retrieval import _import_sentence_transformer
            SentenceTransformer, import_error = _import_sentence_transformer()
            self._startup_timeline.mark("ml_import_complete")
            if SentenceTransformer is None:
                logger.warning(
                    "Embedding framework not available: %s",
                    import_error,
                )
                return
            logger.info("Background: embedding framework imported")
        except Exception:
            logger.exception("Background: failed to import embedding framework")
            return

        # Phase 2: load the model (mostly C/IO, GIL-friendly)
        try:
            model_name = self.config.embed_model
            device = self.config.device
            logger.info(
                "Background: loading embedding model %s on %s...",
                model_name,
                device,
            )
            self._startup_timeline.mark("ml_model_load_start")
            model = SentenceTransformer(model_name, device=device)
            self._startup_timeline.mark("ml_model_load_complete")
            self._query_embedder = model
            logger.info("Background: embedding model ready")
        except Exception:
            logger.exception("Background: failed to load embedding model")

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
                    # Avoid infinite recursion — never log from inside the log handler.
                    # Write to stderr so diagnostic messages are not silently lost.
                    try:
                        import sys as _sys
                        _sys.stderr.write(
                            f"[TUILogHandler] failed to deliver: {record.getMessage()}\n"
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

    def on_tuilog_event(self, message: TUILogEvent) -> None:
        """Handle queued log events on the UI thread."""
        self._write_tui_log(message.message)

    def post_ui_message(self, message: Message, *, source: str = "unknown") -> None:
        """Queue a Textual message to this app from any thread.

        Textual's ``post_message`` is already thread-safe (uses
        ``call_soon_threadsafe`` when called from a non-app thread),
        so we call it directly rather than blocking on ``call_from_thread``.
        """
        _ = source
        self.post_message(message)

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

    def _normalize_worker_token(self, value: str, *, fallback: str) -> str:
        normalized = str(value or "").strip()
        return normalized or fallback

    def _normalize_worker_timeout(self, timeout_s: Optional[float]) -> float:
        try:
            parsed = (
                self._DEFAULT_WORKER_TIMEOUT_SECONDS
                if timeout_s is None
                else float(timeout_s)
            )
        except (TypeError, ValueError):
            parsed = self._DEFAULT_WORKER_TIMEOUT_SECONDS
        return max(0.05, parsed)

    def _worker_kind(self, worker: Any) -> str:
        if isinstance(worker, threading.Thread):
            return "thread"
        if callable(getattr(worker, "cancel", None)):
            return "cancelable"
        return "unknown"

    def register_managed_worker(
        self,
        *,
        owner: str,
        key: str,
        worker: Any,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Register a worker object for lifecycle management."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="worker")
        record = _ManagedWorker(
            owner=owner_token,
            key=key_token,
            worker=worker,
            timeout_s=self._normalize_worker_timeout(timeout_s),
        )
        with self._managed_workers_lock:
            self._managed_workers[(owner_token, key_token)] = record

    def register_managed_timer(
        self,
        *,
        owner: str,
        key: str,
        timer: Timer,
    ) -> None:
        """Register a timer object for lifecycle management."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="timer")
        record = _ManagedTimer(
            owner=owner_token,
            key=key_token,
            timer=timer,
        )
        with self._managed_timers_lock:
            self._managed_timers[(owner_token, key_token)] = record

    def clear_managed_worker(self, *, owner: str, key: str) -> None:
        """Remove a worker from lifecycle tracking without cancelling it."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="worker")
        with self._managed_workers_lock:
            self._managed_workers.pop((owner_token, key_token), None)

    def clear_managed_timer(self, *, owner: str, key: str) -> None:
        """Remove a timer from lifecycle tracking without stopping it."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="timer")
        with self._managed_timers_lock:
            self._managed_timers.pop((owner_token, key_token), None)

    def start_managed_worker(
        self,
        *,
        owner: str,
        key: str,
        start: Callable[[], Any],
        timeout_s: Optional[float] = None,
        cancel_existing: bool = True,
    ) -> Any:
        """Start and register a worker with standardized cancellation semantics."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="worker")
        timeout_value = self._normalize_worker_timeout(timeout_s)
        if cancel_existing:
            self.cancel_managed_worker(
                owner=owner_token,
                key=key_token,
                reason="replaced",
                timeout_s=timeout_value,
            )
        worker = start()
        if worker is None:
            return None
        self.register_managed_worker(
            owner=owner_token,
            key=key_token,
            worker=worker,
            timeout_s=timeout_value,
        )
        return worker

    def start_managed_timer(
        self,
        *,
        owner: str,
        key: str,
        start: Callable[[], Timer | None],
        restart: bool = True,
    ) -> Timer | None:
        """Start and register a timer with idempotent restart semantics."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="timer")
        if restart:
            self.stop_managed_timer(
                owner=owner_token,
                key=key_token,
                reason="replaced",
            )
        timer = start()
        if timer is None:
            return None
        self.register_managed_timer(
            owner=owner_token,
            key=key_token,
            timer=timer,
        )
        return timer

    def cancel_managed_worker(
        self,
        *,
        owner: str,
        key: str,
        reason: str = "",
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Cancel and join a managed worker, returning True when it terminates."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="worker")
        with self._managed_workers_lock:
            record = self._managed_workers.pop((owner_token, key_token), None)
        if record is None:
            return True
        timeout_value = self._normalize_worker_timeout(
            record.timeout_s if timeout_s is None else timeout_s
        )
        worker = record.worker
        kind = self._worker_kind(worker)
        detail = f" ({reason})" if reason else ""

        cancel_fn = getattr(worker, "cancel", None)
        if callable(cancel_fn):
            try:
                cancel_fn()
            except Exception:
                logger.exception(
                    "Failed to cancel worker %s:%s%s",
                    owner_token,
                    key_token,
                    detail,
                )

        if kind != "thread":
            # Async/textual workers complete cooperatively after cancel(); avoid blocking UI thread.
            return True

        try:
            worker.join(timeout=timeout_value)
        except Exception:
            logger.exception(
                "Failed while joining worker %s:%s%s",
                owner_token,
                key_token,
                detail,
            )
            return False
        if worker.is_alive():
            logger.warning(
                "Timed out waiting %.2fs for worker %s:%s%s to exit.",
                timeout_value,
                owner_token,
                key_token,
                detail,
            )
            return False
        return True

    def stop_managed_timer(
        self,
        *,
        owner: str,
        key: str,
        reason: str = "",
    ) -> bool:
        """Stop a managed timer, returning True when stop succeeds."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        key_token = self._normalize_worker_token(key, fallback="timer")
        with self._managed_timers_lock:
            record = self._managed_timers.pop((owner_token, key_token), None)
        if record is None:
            return True
        detail = f" ({reason})" if reason else ""
        try:
            record.timer.stop()
            return True
        except Exception:
            logger.exception(
                "Failed to stop timer %s:%s%s",
                owner_token,
                key_token,
                detail,
            )
            return False

    def cancel_managed_workers_for_owner(
        self,
        *,
        owner: str,
        reason: str = "",
        timeout_s: Optional[float] = None,
    ) -> dict[str, bool]:
        """Cancel all workers registered for a screen/owner."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        with self._managed_workers_lock:
            keys = [key for (worker_owner, key) in self._managed_workers if worker_owner == owner_token]
        results: dict[str, bool] = {}
        for key in keys:
            results[key] = self.cancel_managed_worker(
                owner=owner_token,
                key=key,
                reason=reason,
                timeout_s=timeout_s,
            )
        return results

    def stop_managed_timers_for_owner(
        self,
        *,
        owner: str,
        reason: str = "",
    ) -> dict[str, bool]:
        """Stop all timers registered for a screen/owner."""
        owner_token = self._normalize_worker_token(owner, fallback="app")
        with self._managed_timers_lock:
            keys = [key for (timer_owner, key) in self._managed_timers if timer_owner == owner_token]
        results: dict[str, bool] = {}
        for key in keys:
            results[key] = self.stop_managed_timer(
                owner=owner_token,
                key=key,
                reason=reason,
            )
        return results

    def shutdown_managed_workers(
        self,
        *,
        reason: str = "",
        timeout_s: Optional[float] = None,
    ) -> dict[str, bool]:
        """Cancel all tracked workers during app shutdown."""
        with self._managed_workers_lock:
            worker_keys = list(self._managed_workers.keys())
        results: dict[str, bool] = {}
        for owner_token, key_token in worker_keys:
            results[f"{owner_token}:{key_token}"] = self.cancel_managed_worker(
                owner=owner_token,
                key=key_token,
                reason=reason,
                timeout_s=timeout_s,
            )
        return results

    def shutdown_managed_timers(
        self,
        *,
        reason: str = "",
    ) -> dict[str, bool]:
        """Stop all tracked timers during app shutdown."""
        with self._managed_timers_lock:
            timer_keys = list(self._managed_timers.keys())
        results: dict[str, bool] = {}
        for owner_token, key_token in timer_keys:
            results[f"{owner_token}:{key_token}"] = self.stop_managed_timer(
                owner=owner_token,
                key=key_token,
                reason=reason,
            )
        return results

    def _bind_agent_runtime_events(self) -> None:
        """Bind manager runtime events to app-level queued UI messages."""
        try:
            from lsm.ui.shell.commands.agents import get_agent_runtime_manager

            manager = get_agent_runtime_manager()
            binder = getattr(manager, "set_ui_event_sink", None)
            if callable(binder):
                binder(self._on_agent_runtime_event_from_any_thread)
                self._agent_runtime_bound = True
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
    # UI Error Boundary
    # -------------------------------------------------------------------------

    def _is_recoverable_ui_exception(self, error: Exception) -> bool:
        """Return True when an exception originated from a TUI screen module."""
        tb = error.__traceback__
        if tb is None:
            return False
        try:
            frames = traceback.extract_tb(tb)
        except Exception:
            return False
        for frame in frames:
            filename = str(getattr(frame, "filename", "")).replace("\\", "/").lower()
            if "lsm/ui/tui/screens/" in filename:
                return True
        return False

    def _set_safe_query_context(self) -> None:
        """Switch app focus back to the Query tab/context."""
        try:
            tabbed_content = self.query_one("#main-tabs", TabbedContent)
            tabbed_content.active = "query"
        except Exception:
            logger.exception("Failed to activate safe Query tab during UI error recovery.")
        self._set_active_context("query")

    def _present_recoverable_ui_error(self, error: Exception) -> None:
        """Render recoverable UI error state and keep app running."""
        self._ui_error_count += 1
        error_id = self._ui_error_count
        summary = format_exception_summary(
            error,
            max_length=self._UI_ERROR_SUMMARY_MAX_LENGTH,
        )
        self._last_ui_error_summary = summary
        logger.error(
            "Recoverable UI screen exception #%s",
            error_id,
            exc_info=exception_exc_info(error),
        )

        user_message = (
            f"Recovered from screen error #{error_id}. "
            "Press F12 to return to Query."
        )
        try:
            self.notify_event(
                user_message,
                severity="error",
            )
        except Exception:
            logger.exception("Failed to emit recoverable UI error notification.")

        try:
            from lsm.ui.tui.screens.help import UIErrorRecoveryScreen

            self.push_screen(
                UIErrorRecoveryScreen(
                    error_id=error_id,
                    summary=summary,
                )
            )
        except Exception:
            logger.exception("Failed to display UI error recovery panel.")
            self._set_safe_query_context()

    def _handle_exception(self, error: Exception) -> None:
        """Handle unhandled exceptions with recoverable screen-level boundary."""
        if self._recovering_ui_error:
            super()._handle_exception(error)
            return
        if self._is_recoverable_ui_exception(error):
            self._recovering_ui_error = True
            try:
                self._present_recoverable_ui_error(error)
            finally:
                self._recovering_ui_error = False
            return
        super()._handle_exception(error)

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
            self.stop_managed_timer(
                owner="app",
                key="density-resize",
                reason="density-mode-change",
            )
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
        self._density_resize_timer = self.start_managed_timer(
            owner="app",
            key="density-resize",
            start=lambda: self.set_timer(
                self._DENSITY_RESIZE_DEBOUNCE_SECONDS,
                self._apply_pending_auto_density,
            ),
            restart=True,
        )

    def _apply_pending_auto_density(self) -> None:
        """Apply auto density using the most recent pending terminal size."""
        self.clear_managed_timer(owner="app", key="density-resize")
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
            # Initialize embedder — skip if already preloaded by background init
            if self._query_embedder is None:
                self._query_embedder = await asyncio.to_thread(
                    init_embedder,
                    self.config.embed_model,
                    device=self.config.device,
                )
            else:
                logger.info("Using background-preloaded embedder")

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
        tabbed_content = self.query_one("#main-tabs", TabbedContent)
        tabbed_content.active = "ingest"
        self._set_active_context("ingest")
        logger.debug("Switched to ingest context")

    def action_switch_query(self) -> None:
        """Switch to query tab."""
        tabbed_content = self.query_one("#main-tabs", TabbedContent)
        tabbed_content.active = "query"
        self._set_active_context("query")
        logger.debug("Switched to query context")

    def action_switch_settings(self) -> None:
        """Switch to settings tab."""
        tabbed_content = self.query_one("#main-tabs", TabbedContent)
        tabbed_content.active = "settings"
        self._set_active_context("settings")
        self._refresh_settings_screen()
        logger.debug("Switched to settings context")

    def action_switch_remote(self) -> None:
        """Switch to remote tab."""
        tabbed_content = self.query_one("#main-tabs", TabbedContent)
        tabbed_content.active = "remote"
        self._set_active_context("remote")
        logger.debug("Switched to remote context")

    def action_switch_agents(self) -> None:
        """Switch to agents tab."""
        tabbed_content = self.query_one("#main-tabs", TabbedContent)
        tabbed_content.active = "agents"
        self._set_active_context("agents")
        self._trigger_agents_deferred_init()
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

    def action_return_safe_screen(self) -> None:
        """Return to the safe Query screen after recoverable UI issues."""
        self._set_safe_query_context()
        try:
            self.notify_event(
                "Returned to Query screen.",
                severity="warning",
            )
        except Exception:
            pass

    def action_show_help(self) -> None:
        """Show help modal."""
        from lsm.ui.tui.screens.help import HelpScreen
        self.push_screen(HelpScreen(context=self.current_context))

    def action_quit(self) -> None:
        """Quit the application, prompting if settings have unsaved changes."""
        if self._settings_has_unsaved_changes():
            self.notify(
                "Settings have unsaved changes. Use 'save' or 'discard' in Settings, then quit again.",
                severity="warning",
                timeout=self._NOTIFY_ERROR_TIMEOUT_SECONDS,
            )
            return
        self._force_quit()

    def _force_quit(self) -> None:
        """Unconditionally quit the application."""
        logger.info("User requested quit")
        self.shutdown_managed_timers(reason="app-quit")
        self.shutdown_managed_workers(reason="app-quit")
        self._unbind_agent_runtime_events()
        self.exit()

    def _settings_has_unsaved_changes(self) -> bool:
        """Check if the settings screen has unsaved draft changes."""
        try:
            settings_screen = self.query_one("#settings-screen")
            return bool(getattr(settings_screen, "has_unsaved_changes", False))
        except Exception:
            return False

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
                if context == "agents":
                    self._trigger_agents_deferred_init()
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

    def _trigger_agents_deferred_init(self) -> None:
        """Trigger deferred initialization on the agents screen."""
        try:
            agents_screen = self.query_one("#agents-screen")
            init_fn = getattr(agents_screen, "_ensure_deferred_init", None)
            if callable(init_fn):
                init_fn()
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

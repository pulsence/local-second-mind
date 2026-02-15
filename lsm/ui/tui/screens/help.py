"""Help screen modal for LSM TUI."""

from __future__ import annotations

from typing import Literal, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from lsm import __version__
from lsm.logging import get_logger

logger = get_logger(__name__)

ContextType = Literal["query", "ingest", "remote", "agents", "settings"]

_CONTEXT_LABELS: dict[ContextType, str] = {
    "query": "Query",
    "ingest": "Ingest",
    "remote": "Remote",
    "agents": "Agents",
    "settings": "Settings",
}

_GLOBAL_SHORTCUTS: tuple[str, ...] = (
    "Switch to Query tab (Ctrl+Q)",
    "Switch to Ingest tab (Ctrl+N)",
    "Switch to Remote tab (Ctrl+R or Ctrl+P)",
    "Switch to Agents tab (Ctrl+G)",
    "Switch to Settings tab (Ctrl+S)",
    "Show help (F1)",
    "Return to safe Query screen (F12)",
    "Quit application (Ctrl+C / Ctrl+D / Ctrl+Z)",
)

_CONTEXT_COMMANDS: dict[ContextType, tuple[str, ...]] = {
    "query": (
        "/mode <name>                  Switch query mode",
        "/model                        Show current model selections",
        "/show S#                      Show cited chunk details",
        "/expand S#                    Expand citation text (Ctrl+E)",
        "/open S#                      Open source file (Ctrl+O)",
        "/agent <subcommand>           Manage agent runs and schedules",
        "/memory <subcommand>          Manage memory candidates",
        "/ui density [mode]            View/set layout density",
    ),
    "ingest": (
        "/build [--force]              Run ingest pipeline (Ctrl+B)",
        "/tag [--max N]                Run AI tagging (Ctrl+T)",
        "/stats                        Show collection statistics (Ctrl+Shift+R)",
        "/explore [query]              Browse indexed files",
        "/search <query>               Search metadata",
        "/wipe                         Clear collection (confirmation required)",
    ),
    "remote": (
        "Select provider + enter query  Run remote search (Enter/Search)",
        "Refresh providers              Reload configured provider list",
        "Set Weight                     Update provider blend weight",
        "/remote-providers              See provider summary from Query tab",
        "/remote-search <provider> ...  Run provider search from Query tab",
    ),
    "agents": (
        "Start selected agent           Launch agent run",
        "Status/Pause/Resume/Stop       Control selected run",
        "Approve pending action (F8)    Approve current permission request",
        "Approve for session (F9)       Allow tool for the session",
        "Deny request (F10)             Deny pending permission request",
        "Send interaction reply (F11)   Reply to clarification/feedback request",
        "Prev/Next running agent        F6 / F7",
    ),
    "settings": (
        "set <key> <value>              Update a config value",
        "unset <key>                    Remove a value from draft config",
        "delete <key>                   Delete list/dict entry from draft config",
        "reset [key]                    Reset tab or key to persisted values",
        "default <key>                  Reset key to model default",
        "save                           Persist draft config to disk",
        "discard                        Discard all unsaved changes",
        "discard tab                    Discard current tab unsaved changes",
    ),
}

_CONTEXT_SHORTCUTS: dict[ContextType, tuple[str, ...]] = {
    "query": (
        "Submit input (Enter)",
        "Expand selected citation (Ctrl+E)",
        "Open selected source (Ctrl+O)",
        "Refresh logs (Ctrl+Shift+R)",
    ),
    "ingest": (
        "Run build (Ctrl+B)",
        "Run tagging (Ctrl+T)",
        "Refresh stats (Ctrl+Shift+R)",
        "Clear command input (Escape)",
    ),
    "remote": (
        "Next field (Tab)",
        "Previous field (Shift+Tab)",
        "Run search (Ctrl+Enter or Enter in query field)",
        "Refresh providers (Ctrl+Shift+R)",
    ),
    "agents": (
        "Previous running agent (F6)",
        "Next running agent (F7)",
        "Approve interaction (F8)",
        "Approve interaction for session (F9)",
        "Deny interaction (F10)",
        "Reply to interaction (F11)",
        "Refresh running agents (Ctrl+Shift+R)",
        "Show agent log (Ctrl+L)",
        "Show agent status (Ctrl+I)",
    ),
    "settings": (
        "Global tab (F2)",
        "Ingest tab (F3)",
        "Query tab (F4)",
        "LLM tab (F5)",
        "Vector DB tab (F6)",
        "Modes tab (F7)",
        "Remote tab (F8)",
        "Chats/Notes tab (F9)",
    ),
}

_ALL_COMMANDS: tuple[str, ...] = (
    "Query: /mode /model /models /providers /provider-status /show /expand /open",
    "Query: /set /clear /load /context /costs /budget /cost-estimate /export-citations",
    "Query: /remote-providers /remote-search /remote-search-all /note /notes /agent /memory /ui",
    "",
    "Ingest: /info /stats /explore /show /search /build /tag /tags /wipe",
    "Ingest: /vectordb-providers /vectordb-status",
    "",
    "Global: /help /exit /quit",
)

_WHATS_NEW_VERSION = f"v{__version__}"
_WHATS_NEW_SECTION_TITLE = f"WHAT'S NEW IN {_WHATS_NEW_VERSION}"

_WHATS_NEW: tuple[str, ...] = (
    f"Version: {_WHATS_NEW_VERSION}",
    "Interactive agent approvals and replies directly in the Agents screen.",
    "Live agent log streaming with actor-labeled entries and tool summaries.",
    "Multi-agent controls with running-agent selection and keyboard shortcuts.",
    "Schedule controls plus runtime schedule status surfaced in the TUI.",
)


class HelpScreen(ModalScreen):
    """Context-aware help modal screen."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    def __init__(self, context: str = "query") -> None:
        super().__init__()
        self.context: ContextType = self._normalize_context(context)
        self._show_all_commands = False

    @staticmethod
    def _normalize_context(context: str) -> ContextType:
        normalized = str(context or "").strip().lower()
        if normalized in _CONTEXT_LABELS:
            return cast(ContextType, normalized)
        return "query"

    def _context_label(self) -> str:
        return _CONTEXT_LABELS[self.context]

    @staticmethod
    def _format_lines(lines: tuple[str, ...]) -> str:
        return "\n".join(lines)

    def _global_shortcuts_text(self) -> str:
        return self._format_lines(_GLOBAL_SHORTCUTS)

    def _context_commands_text(self) -> str:
        return self._format_lines(_CONTEXT_COMMANDS[self.context])

    def _context_shortcuts_text(self) -> str:
        return self._format_lines(_CONTEXT_SHORTCUTS[self.context])

    def _all_commands_text(self) -> str:
        return self._format_lines(_ALL_COMMANDS)

    def _whats_new_text(self) -> str:
        return self._format_lines(_WHATS_NEW)

    def compose(self) -> ComposeResult:
        """Compose the help modal layout."""
        context_label = self._context_label()
        with Container(classes="help-modal"):
            with ScrollableContainer(classes="help-content"):
                yield Static(
                    f"Local Second Mind - Help ({context_label})",
                    classes="help-title",
                )

                yield Static(f"CONTEXT: {context_label}", classes="help-section")

                yield Static("GLOBAL SHORTCUTS", classes="help-section")
                yield Static(self._global_shortcuts_text(), markup=False)

                yield Static(f"{context_label.upper()} COMMANDS", classes="help-section")
                yield Static(self._context_commands_text(), markup=False)

                yield Static(f"{context_label.upper()} SHORTCUTS", classes="help-section")
                yield Static(self._context_shortcuts_text(), markup=False)

                yield Static("ALL COMMANDS", classes="help-section")
                yield Button("Show All Commands", id="help-toggle-all-commands")
                yield Static(
                    self._all_commands_text(),
                    id="help-all-commands",
                    markup=False,
                )

                yield Static(_WHATS_NEW_SECTION_TITLE, classes="help-section")
                yield Static(self._whats_new_text(), markup=False)

                yield Button("Close", id="close-help", variant="primary")

    def on_mount(self) -> None:
        self._sync_all_commands_visibility()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close-help":
            self.dismiss()
            return
        if event.button.id == "help-toggle-all-commands":
            self._show_all_commands = not self._show_all_commands
            self._sync_all_commands_visibility()

    def _sync_all_commands_visibility(self) -> None:
        try:
            all_commands_widget = self.query_one("#help-all-commands", Static)
            toggle_button = self.query_one("#help-toggle-all-commands", Button)
        except Exception:
            return

        if self._show_all_commands:
            all_commands_widget.styles.display = "block"
            toggle_button.label = "Hide All Commands"
            return

        all_commands_widget.styles.display = "none"
        toggle_button.label = "Show All Commands"

    def action_dismiss(self) -> None:
        """Dismiss the help screen."""
        self.dismiss()


class UIErrorRecoveryScreen(ModalScreen):
    """Recoverable UI error boundary panel."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("f12", "return_safe_screen", "Return To Query", show=True),
    ]

    def __init__(
        self,
        *,
        error_id: int,
        summary: str,
    ) -> None:
        super().__init__()
        self.error_id = max(1, int(error_id))
        self.summary = str(summary or "").strip() or "Unknown UI error."

    def compose(self) -> ComposeResult:
        """Compose the error recovery modal layout."""
        with Container(classes="help-modal"):
            with ScrollableContainer(classes="help-content"):
                yield Static("Recoverable UI Error", classes="help-title")
                yield Static(
                    "A screen error was detected and the app recovered without exiting.",
                    markup=False,
                )
                yield Static(
                    "Return to Query to continue working. "
                    "You can reopen Help (F1) if needed.",
                    markup=False,
                )
                yield Static(
                    f"Error #{self.error_id}: {self.summary}",
                    classes="help-section",
                    markup=False,
                )
                yield Button(
                    "Return to Query",
                    id="ui-error-return-safe",
                    variant="primary",
                )
                yield Button(
                    "Dismiss",
                    id="ui-error-dismiss",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle recovery panel actions."""
        if event.button.id == "ui-error-return-safe":
            self.action_return_safe_screen()
            return
        if event.button.id == "ui-error-dismiss":
            self.dismiss()

    def action_return_safe_screen(self) -> None:
        """Return to safe query context and close the recovery panel."""
        app_obj = getattr(self, "app", None)
        handler = getattr(app_obj, "action_return_safe_screen", None)
        if callable(handler):
            try:
                handler()
            except Exception:
                logger.exception("Failed to return to safe screen from UI error panel.")
        self.dismiss()

    def action_dismiss(self) -> None:
        """Dismiss the recovery screen."""
        self.dismiss()

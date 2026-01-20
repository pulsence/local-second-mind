"""
Help screen modal for LSM TUI.

Displays keyboard shortcuts and command reference.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, Button
from textual.binding import Binding

from lsm.logging import get_logger

logger = get_logger(__name__)


class HelpScreen(ModalScreen):
    """
    Help modal screen.

    Displays keyboard shortcuts and command reference.
    Press Escape or click Close to dismiss.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    def compose(self) -> ComposeResult:
        """Compose the help modal layout."""
        with Container(classes="help-modal"):
            with Vertical(classes="help-content"):
                yield Static("Local Second Mind - Help", classes="help-title")

                yield Static(
                    "KEYBOARD SHORTCUTS",
                    classes="help-section",
                )

                yield Static(
                    "Ctrl+I      Switch to Ingest tab\n"
                    "Ctrl+Q      Switch to Query tab\n"
                    "Ctrl+S      Switch to Settings tab\n"
                    "Ctrl+P      Switch to Remote tab\n"
                    "F1          Show this help\n"
                    "Ctrl+C      Quit application\n"
                    "Ctrl+Shift+R Refresh (Query/Ingest)\n"
                    "Tab         Next element\n"
                    "Shift+Tab   Previous element\n"
                    "Escape      Close modal / Clear input",
                )

                yield Static(
                    "\nQUERY COMMANDS (TUI)",
                    classes="help-section",
                )

                yield Static(
                    "/help                 Show full command list\n"
                    "/mode                 Show current mode\n"
                    "/mode <name>          Switch mode\n"
                    "/show S#              Show citation details\n"
                    "/expand S#            Expand citation text\n"
                    "/open S#              Open source file\n"
                    "/costs                Show session costs\n"
                    "/debug                Show debug info\n"
                    "/exit                 Quit TUI",
                )

                yield Static(
                    "\nINGEST COMMANDS (TUI)",
                    classes="help-section",
                )

                yield Static(
                    "/build [--force]  Run ingest pipeline\n"
                    "/tag [--max N]    Run AI tagging\n"
                    "/stats            Show statistics\n"
                    "/explore [query]  Browse files\n"
                    "/show <path>      Show file chunks\n"
                    "/search <query>   Search metadata\n"
                    "/wipe             Clear collection\n"
                    "/exit             Quit TUI",
                )

                yield Static(
                    "\nINGEST SHORTCUTS",
                    classes="help-section",
                )

                yield Static(
                    "Ctrl+B      Run build\n"
                    "Ctrl+T      Run tagging\n"
                    "Ctrl+Shift+R Refresh stats",
                )

                yield Button("Close", id="close-help", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "close-help":
            self.dismiss()

    def action_dismiss(self) -> None:
        """Dismiss the help screen."""
        self.dismiss()

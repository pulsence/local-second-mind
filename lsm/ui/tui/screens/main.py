"""
Main screen layout for LSM TUI.

Provides the primary tabbed interface container.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Static

from lsm.logging import get_logger

logger = get_logger(__name__)


class MainScreen(Screen):
    """
    Main screen with tabbed interface.

    This screen is the primary container for the TUI,
    hosting the Query, Ingest, and Settings tabs.
    """

    def compose(self) -> ComposeResult:
        """Compose the main screen layout."""
        yield Container(
            Static("Welcome to Local Second Mind", classes="welcome-text"),
            id="main-container",
            classes="screen-container",
        )

    def on_mount(self) -> None:
        """Handle screen mount."""
        logger.debug("Main screen mounted")

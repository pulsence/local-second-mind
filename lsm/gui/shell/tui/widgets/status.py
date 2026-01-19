"""
Status bar widget for LSM TUI.

Displays:
- Current mode indicator
- Provider health status
- Token/cost counter
- Chunk count
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.widget import Widget
from textual.reactive import reactive

from lsm.gui.shell.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class StatusBar(Widget):
    """
    Status bar widget for displaying session information.

    Shows:
    - Current query mode
    - Chunk count in collection
    - Session cost
    - Provider status
    """

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $primary-darken-1;
        padding: 0 1;
    }

    StatusBar > Horizontal {
        width: 100%;
        height: 1;
    }

    .status-item {
        margin-right: 3;
    }

    .status-label {
        color: $text-muted;
    }

    .status-value {
        color: $text;
        text-style: bold;
    }

    .status-mode {
        color: $success;
    }

    .status-cost {
        color: $warning;
    }

    .status-chunks {
        color: $secondary;
    }
    """

    # Reactive properties
    mode: reactive[str] = reactive("grounded")
    chunk_count: reactive[int] = reactive(0)
    total_cost: reactive[float] = reactive(0.0)
    provider_status: reactive[str] = reactive("ready")

    def compose(self) -> ComposeResult:
        """Compose the status bar layout."""
        with Horizontal():
            yield Static(
                f"Mode: {self.mode}",
                id="status-mode",
                classes="status-item status-mode",
            )
            yield Static(
                f"Chunks: {self.chunk_count:,}",
                id="status-chunks",
                classes="status-item status-chunks",
            )
            yield Static(
                f"Cost: ${self.total_cost:.4f}",
                id="status-cost",
                classes="status-item status-cost",
            )
            yield Static(
                f"Status: {self.provider_status}",
                id="status-provider",
                classes="status-item",
            )

    def watch_mode(self, mode: str) -> None:
        """React to mode changes."""
        try:
            mode_widget = self.query_one("#status-mode", Static)
            mode_widget.update(f"Mode: {mode}")
        except Exception:
            pass

    def watch_chunk_count(self, count: int) -> None:
        """React to chunk count changes."""
        try:
            chunks_widget = self.query_one("#status-chunks", Static)
            chunks_widget.update(f"Chunks: {count:,}")
        except Exception:
            pass

    def watch_total_cost(self, cost: float) -> None:
        """React to cost changes."""
        try:
            cost_widget = self.query_one("#status-cost", Static)
            cost_widget.update(f"Cost: ${cost:.4f}")
        except Exception:
            pass

    def watch_provider_status(self, status: str) -> None:
        """React to provider status changes."""
        try:
            status_widget = self.query_one("#status-provider", Static)
            status_widget.update(f"Status: {status}")
        except Exception:
            pass

    def update_from_app(self) -> None:
        """Update status from app reactive properties."""
        app = self.app
        if hasattr(app, 'current_mode'):
            self.mode = app.current_mode
        if hasattr(app, 'chunk_count'):
            self.chunk_count = app.chunk_count
        if hasattr(app, 'total_cost'):
            self.total_cost = app.total_cost

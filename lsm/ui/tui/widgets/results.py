"""
Results panel widget for LSM TUI.

Provides a scrollable list of query results with:
- Citation labels and expansion
- Source path display
- Syntax highlighting for code blocks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message

from lsm.logging import get_logger

if TYPE_CHECKING:
    from lsm.query.session import Candidate

logger = get_logger(__name__)


class CitationSelected(Message):
    """Message sent when a citation is selected."""

    def __init__(self, index: int, candidate: "Candidate") -> None:
        self.index = index
        self.candidate = candidate
        super().__init__()


class CitationExpanded(Message):
    """Message sent when a citation is expanded."""

    def __init__(self, index: int, candidate: "Candidate") -> None:
        self.index = index
        self.candidate = candidate
        super().__init__()


class ResultItem(Widget):
    """
    Individual result item widget.

    Displays a single citation with source path and chunk preview.
    """

    DEFAULT_CSS = """
    ResultItem {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: round $primary-darken-2;
    }

    ResultItem:hover {
        background: $surface-lighten-1;
        border: round $primary;
    }

    ResultItem.-selected {
        background: $primary-darken-2;
        border: round $primary;
    }
    """

    is_selected: reactive[bool] = reactive(False)
    is_expanded: reactive[bool] = reactive(False)

    def __init__(
        self,
        index: int,
        candidate: "Candidate",
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize a result item.

        Args:
            index: Citation index (1-based)
            candidate: The candidate result
        """
        super().__init__(*args, **kwargs)
        self.index = index
        self.candidate = candidate

    def compose(self) -> ComposeResult:
        """Compose the result item layout."""
        meta = self.candidate.meta or {}
        source_path = meta.get("source_path", "unknown")
        chunk_index = meta.get("chunk_index", "N/A")
        distance = self.candidate.distance
        distance_label = f"{distance:.4f}" if distance is not None else "n/a"

        # Citation header
        yield Static(
            f"[S{self.index}] {source_path}",
            classes="citation-label",
        )

        # Metadata line
        yield Static(
            f"chunk {chunk_index} | distance: {distance_label}",
            classes="source-path",
        )

        # Chunk text (only when expanded)
        if self.is_expanded:
            text = (self.candidate.text or "").strip()
            yield Static(text, classes="chunk-text")

    def watch_is_selected(self, selected: bool) -> None:
        """React to selection state changes."""
        self.set_class(selected, "-selected")

    def watch_is_expanded(self, expanded: bool) -> None:
        """React to expansion state changes."""
        # Re-compose to show full or truncated text
        self.refresh(recompose=True)

    def on_click(self) -> None:
        """Handle click on result item."""
        self.is_selected = not self.is_selected
        if self.is_selected:
            self.post_message(CitationSelected(self.index, self.candidate))


class ResultsPanel(Widget):
    """
    Scrollable results panel widget.

    Displays query results as a list of expandable citations.
    """

    DEFAULT_CSS = """
    ResultsPanel {
        height: 100%;
        background: $surface;
        border: round $primary-darken-1;
        padding: 1;
    }

    ResultsPanel:focus {
        border: round $primary;
    }
    """

    # Reactive state
    selected_index: reactive[Optional[int]] = reactive(None)

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the results panel."""
        super().__init__(*args, **kwargs)
        self._candidates: List["Candidate"] = []
        self._response_text: str = ""

    def compose(self) -> ComposeResult:
        """Compose the results panel layout."""
        with ScrollableContainer(id="results-scroll"):
            # Response text area
            yield Static(
                self._response_text or "Enter a query to see results.",
                id="response-text",
                markup=False,
            )

        # Results list hidden by default (citations still accessible via commands)

    def update_results(
        self,
        response: str,
        candidates: List["Candidate"],
    ) -> None:
        """
        Update the results display.

        Args:
            response: The LLM response text
            candidates: List of candidate results
        """
        self._response_text = response
        self._candidates = candidates
        self.selected_index = None
        self.refresh(recompose=True)

    def clear_results(self) -> None:
        """Clear all results."""
        self._response_text = ""
        self._candidates = []
        self.selected_index = None
        self.refresh(recompose=True)

    def select_citation(self, index: int) -> None:
        """
        Select a citation by index.

        Args:
            index: Citation index (1-based)
        """
        if 1 <= index <= len(self._candidates):
            self.selected_index = index

            # Update all result items
            for i, candidate in enumerate(self._candidates, start=1):
                result_item = self.query_one(f"#result-{i}", ResultItem)
                result_item.is_selected = (i == index)

    def expand_citation(self, index: int) -> None:
        """
        Expand a citation to show full text.

        Args:
            index: Citation index (1-based)
        """
        if 1 <= index <= len(self._candidates):
            result_item = self.query_one(f"#result-{index}", ResultItem)
            result_item.is_expanded = not result_item.is_expanded

    def get_candidate(self, index: int) -> Optional["Candidate"]:
        """
        Get a candidate by index.

        Args:
            index: Citation index (1-based)

        Returns:
            The candidate, or None if index is invalid
        """
        if 1 <= index <= len(self._candidates):
            return self._candidates[index - 1]
        return None

    def on_citation_selected(self, message: CitationSelected) -> None:
        """Handle citation selection."""
        self.selected_index = message.index
        logger.debug(f"Citation selected: S{message.index}")

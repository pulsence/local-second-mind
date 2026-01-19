"""Tests for TUI widget modules."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

# Skip tests if textual is not available
pytest.importorskip("textual")

from textual.widget import Widget


class TestWidgetImports:
    """Tests for widget module imports."""

    def test_import_results_panel(self):
        """Should be able to import ResultsPanel."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        assert ResultsPanel is not None

    def test_import_result_item(self):
        """Should be able to import ResultItem."""
        from lsm.gui.shell.tui.widgets.results import ResultItem
        assert ResultItem is not None

    def test_import_command_input(self):
        """Should be able to import CommandInput."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        assert CommandInput is not None

    def test_import_status_bar(self):
        """Should be able to import StatusBar."""
        from lsm.gui.shell.tui.widgets.status import StatusBar
        assert StatusBar is not None


class TestResultsPanel:
    """Tests for ResultsPanel widget."""

    def test_is_widget(self):
        """ResultsPanel should be a Widget subclass."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        assert issubclass(ResultsPanel, Widget)

    def test_has_selected_index(self):
        """ResultsPanel should have selected_index reactive."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        panel = ResultsPanel()
        assert hasattr(panel, 'selected_index')

    def test_update_results_method(self):
        """ResultsPanel should have update_results method."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        panel = ResultsPanel()
        assert hasattr(panel, 'update_results')
        assert callable(panel.update_results)

    def test_clear_results_method(self):
        """ResultsPanel should have clear_results method."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        panel = ResultsPanel()
        assert hasattr(panel, 'clear_results')
        assert callable(panel.clear_results)

    def test_select_citation_method(self):
        """ResultsPanel should have select_citation method."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        panel = ResultsPanel()
        assert hasattr(panel, 'select_citation')
        assert callable(panel.select_citation)

    def test_expand_citation_method(self):
        """ResultsPanel should have expand_citation method."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        panel = ResultsPanel()
        assert hasattr(panel, 'expand_citation')
        assert callable(panel.expand_citation)

    def test_get_candidate_method(self):
        """ResultsPanel should have get_candidate method."""
        from lsm.gui.shell.tui.widgets.results import ResultsPanel
        panel = ResultsPanel()
        assert hasattr(panel, 'get_candidate')
        assert callable(panel.get_candidate)


class TestResultItem:
    """Tests for ResultItem widget."""

    def test_is_widget(self):
        """ResultItem should be a Widget subclass."""
        from lsm.gui.shell.tui.widgets.results import ResultItem
        assert issubclass(ResultItem, Widget)

    def test_requires_index_and_candidate(self):
        """ResultItem should require index and candidate."""
        from lsm.gui.shell.tui.widgets.results import ResultItem

        # Create mock candidate
        mock_candidate = Mock()
        mock_candidate.meta = {"source_path": "test.txt", "chunk_index": 0}
        mock_candidate.text = "Test text"
        mock_candidate.distance = 0.5

        item = ResultItem(1, mock_candidate)
        assert item.index == 1
        assert item.candidate is mock_candidate

    def test_has_reactive_properties(self):
        """ResultItem should have reactive properties."""
        from lsm.gui.shell.tui.widgets.results import ResultItem

        mock_candidate = Mock()
        mock_candidate.meta = {}
        mock_candidate.text = ""
        mock_candidate.distance = 0.0

        item = ResultItem(1, mock_candidate)
        assert hasattr(item, 'is_selected')
        assert hasattr(item, 'is_expanded')


class TestCommandInput:
    """Tests for CommandInput widget."""

    def test_is_widget(self):
        """CommandInput should be a Widget subclass."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        assert issubclass(CommandInput, Widget)

    def test_has_bindings(self):
        """CommandInput should have keyboard bindings."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        assert hasattr(CommandInput, 'BINDINGS')
        assert len(CommandInput.BINDINGS) > 0

    def test_has_history_index(self):
        """CommandInput should have history_index reactive."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        widget = CommandInput()
        assert hasattr(widget, 'history_index')

    def test_default_placeholder(self):
        """CommandInput should accept placeholder text."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        widget = CommandInput(placeholder="Test placeholder")
        assert widget._placeholder == "Test placeholder"

    def test_accepts_completer(self):
        """CommandInput should accept a completer function."""
        from lsm.gui.shell.tui.widgets.input import CommandInput

        def my_completer(text: str):
            return ["/help", "/exit"]

        widget = CommandInput(completer=my_completer)
        assert widget._completer is my_completer

    def test_add_to_history_method(self):
        """CommandInput should have add_to_history method."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        widget = CommandInput()
        assert hasattr(widget, 'add_to_history')
        assert callable(widget.add_to_history)

    def test_add_to_history_stores_commands(self):
        """add_to_history should store commands."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        widget = CommandInput()

        widget.add_to_history("/help")
        assert "/help" in widget._history

        widget.add_to_history("/mode grounded")
        assert "/mode grounded" in widget._history

    def test_add_to_history_ignores_empty(self):
        """add_to_history should ignore empty strings."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        widget = CommandInput()

        widget.add_to_history("")
        widget.add_to_history("   ")
        assert len(widget._history) == 0

    def test_add_to_history_ignores_duplicates(self):
        """add_to_history should ignore consecutive duplicates."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        widget = CommandInput()

        widget.add_to_history("/help")
        widget.add_to_history("/help")
        assert widget._history.count("/help") == 1

    def test_clear_method(self):
        """CommandInput should have clear method."""
        from lsm.gui.shell.tui.widgets.input import CommandInput
        widget = CommandInput()
        assert hasattr(widget, 'clear')
        assert callable(widget.clear)

    def test_common_prefix_static_method(self):
        """CommandInput should have _common_prefix static method."""
        from lsm.gui.shell.tui.widgets.input import CommandInput

        assert CommandInput._common_prefix([]) == ""
        assert CommandInput._common_prefix(["test"]) == "test"
        assert CommandInput._common_prefix(["/help", "/history"]) == "/h"
        # Note: /model starts with /mode, so common prefix is /mode
        assert CommandInput._common_prefix(["/mode", "/model"]) == "/mode"
        assert CommandInput._common_prefix(["/mod", "/model"]) == "/mod"
        assert CommandInput._common_prefix(["abc", "def"]) == ""


class TestStatusBar:
    """Tests for StatusBar widget."""

    def test_is_widget(self):
        """StatusBar should be a Widget subclass."""
        from lsm.gui.shell.tui.widgets.status import StatusBar
        assert issubclass(StatusBar, Widget)

    def test_has_reactive_properties(self):
        """StatusBar should have reactive properties."""
        from lsm.gui.shell.tui.widgets.status import StatusBar
        widget = StatusBar()
        assert hasattr(widget, 'mode')
        assert hasattr(widget, 'chunk_count')
        assert hasattr(widget, 'total_cost')
        assert hasattr(widget, 'provider_status')

    def test_default_values(self):
        """StatusBar should have sensible default values."""
        from lsm.gui.shell.tui.widgets.status import StatusBar
        widget = StatusBar()
        assert widget.mode == "grounded"
        assert widget.chunk_count == 0
        assert widget.total_cost == 0.0
        assert widget.provider_status == "ready"

    def test_update_from_app_method(self):
        """StatusBar should have update_from_app method."""
        from lsm.gui.shell.tui.widgets.status import StatusBar
        widget = StatusBar()
        assert hasattr(widget, 'update_from_app')
        assert callable(widget.update_from_app)


class TestMessages:
    """Tests for widget messages."""

    def test_citation_selected_message(self):
        """CitationSelected message should carry index and candidate."""
        from lsm.gui.shell.tui.widgets.results import CitationSelected

        mock_candidate = Mock()
        message = CitationSelected(1, mock_candidate)

        assert message.index == 1
        assert message.candidate is mock_candidate

    def test_citation_expanded_message(self):
        """CitationExpanded message should carry index and candidate."""
        from lsm.gui.shell.tui.widgets.results import CitationExpanded

        mock_candidate = Mock()
        message = CitationExpanded(2, mock_candidate)

        assert message.index == 2
        assert message.candidate is mock_candidate

    def test_command_submitted_message(self):
        """CommandSubmitted message should carry command."""
        from lsm.gui.shell.tui.widgets.input import CommandSubmitted

        message = CommandSubmitted("/help")
        assert message.command == "/help"

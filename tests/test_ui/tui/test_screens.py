"""Tests for TUI screen modules."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

# Skip tests if textual is not available
pytest.importorskip("textual")

from textual.widget import Widget
from textual.screen import Screen, ModalScreen


class TestScreenImports:
    """Tests for screen module imports."""

    def test_import_query_screen(self):
        """Should be able to import QueryScreen."""
        from lsm.ui.tui.screens.query import QueryScreen
        assert QueryScreen is not None

    def test_import_ingest_screen(self):
        """Should be able to import IngestScreen."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        assert IngestScreen is not None

    def test_import_settings_screen(self):
        """Should be able to import SettingsScreen."""
        from lsm.ui.tui.screens.settings import SettingsScreen
        assert SettingsScreen is not None

    def test_import_help_screen(self):
        """Should be able to import HelpScreen."""
        from lsm.ui.tui.screens.help import HelpScreen
        assert HelpScreen is not None

    def test_import_main_screen(self):
        """Should be able to import MainScreen."""
        from lsm.ui.tui.screens.main import MainScreen
        assert MainScreen is not None

    def test_import_agents_screen(self):
        """Should be able to import AgentsScreen."""
        from lsm.ui.tui.screens.agents import AgentsScreen
        assert AgentsScreen is not None


class TestQueryScreen:
    """Tests for QueryScreen class."""

    def test_is_widget(self):
        """QueryScreen should be a Widget subclass."""
        from lsm.ui.tui.screens.query import QueryScreen
        assert issubclass(QueryScreen, Widget)

    def test_has_bindings(self):
        """QueryScreen should have keyboard bindings."""
        from lsm.ui.tui.screens.query import QueryScreen
        assert hasattr(QueryScreen, 'BINDINGS')
        assert len(QueryScreen.BINDINGS) > 0

    def test_has_reactive_properties(self):
        """QueryScreen should have reactive properties."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, 'is_loading')
        assert hasattr(screen, 'selected_citation')


class TestIngestScreen:
    """Tests for IngestScreen class."""

    def test_is_widget(self):
        """IngestScreen should be a Widget subclass."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        assert issubclass(IngestScreen, Widget)

    def test_has_bindings(self):
        """IngestScreen should have keyboard bindings."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        assert hasattr(IngestScreen, 'BINDINGS')
        assert len(IngestScreen.BINDINGS) > 0

    def test_has_reactive_properties(self):
        """IngestScreen should have reactive properties."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        screen = IngestScreen()
        assert hasattr(screen, 'is_building')
        assert hasattr(screen, 'build_progress')
        assert hasattr(screen, 'chunk_count')
        assert hasattr(screen, 'file_count')


class TestSettingsScreen:
    """Tests for SettingsScreen class."""

    def test_is_widget(self):
        """SettingsScreen should be a Widget subclass."""
        from lsm.ui.tui.screens.settings import SettingsScreen
        assert issubclass(SettingsScreen, Widget)

    def test_has_reactive_properties(self):
        """SettingsScreen should have reactive properties."""
        from lsm.ui.tui.screens.settings import SettingsScreen
        screen = SettingsScreen()
        assert hasattr(screen, 'current_mode')


class TestHelpScreen:
    """Tests for HelpScreen class."""

    def test_is_modal_screen(self):
        """HelpScreen should be a ModalScreen subclass."""
        from lsm.ui.tui.screens.help import HelpScreen
        assert issubclass(HelpScreen, ModalScreen)

    def test_has_dismiss_binding(self):
        """HelpScreen should have escape binding to dismiss."""
        from lsm.ui.tui.screens.help import HelpScreen
        assert hasattr(HelpScreen, 'BINDINGS')
        bindings = HelpScreen.BINDINGS
        escape_bindings = [b for b in bindings if b.key == "escape"]
        assert len(escape_bindings) > 0

    def test_defaults_to_query_context(self):
        """HelpScreen defaults to query context."""
        from lsm.ui.tui.screens.help import HelpScreen
        screen = HelpScreen()
        assert screen.context == "query"

    def test_invalid_context_falls_back_to_query(self):
        """Unknown context values should map to query."""
        from lsm.ui.tui.screens.help import HelpScreen
        screen = HelpScreen(context="unknown")
        assert screen.context == "query"

    def test_context_commands_are_filtered(self):
        """Context command text should include only the active context commands."""
        from lsm.ui.tui.screens.help import HelpScreen

        agents_help = HelpScreen(context="agents")
        agents_text = agents_help._context_commands_text()
        assert "Start selected agent" in agents_text
        assert "/build [--force]" not in agents_text

        ingest_help = HelpScreen(context="ingest")
        ingest_text = ingest_help._context_commands_text()
        assert "/build [--force]" in ingest_text
        assert "Approve pending action" not in ingest_text

    def test_all_commands_and_whats_new_sections_present(self):
        """All-commands and current-version feature text should include interaction highlights."""
        from lsm import __version__
        from lsm.ui.tui.screens import help as help_module
        from lsm.ui.tui.screens.help import HelpScreen

        screen = HelpScreen(context="query")
        all_commands = screen._all_commands_text()
        whats_new = screen._whats_new_text()

        assert "Query: /mode" in all_commands
        assert "Ingest: /info" in all_commands
        assert f"Version: v{__version__}" in whats_new
        assert help_module._WHATS_NEW_SECTION_TITLE == f"WHAT'S NEW IN v{__version__}"
        assert "Interactive agent approvals and replies" in whats_new


class TestMainScreen:
    """Tests for MainScreen class."""

    def test_is_screen(self):
        """MainScreen should be a Screen subclass."""
        from lsm.ui.tui.screens.main import MainScreen
        assert issubclass(MainScreen, Screen)


class TestQueryScreenIntegration:
    """Integration tests for QueryScreen with widgets."""

    def test_has_completer(self):
        """QueryScreen should have a completer for autocomplete."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, '_completer')
        assert callable(screen._completer)

    def test_has_get_candidates(self):
        """QueryScreen should have _get_candidates method."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, '_get_candidates')
        assert callable(screen._get_candidates)

    def test_get_candidates_returns_none_initially(self):
        """_get_candidates should return None when no candidates."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        result = screen._get_candidates()
        assert result is None

    def test_has_show_message_method(self):
        """QueryScreen should have _show_message method."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, '_show_message')
        assert callable(screen._show_message)

    def test_has_show_citation_method(self):
        """QueryScreen should have _show_citation method."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, '_show_citation')
        assert callable(screen._show_citation)

    def test_has_expand_citation_method(self):
        """QueryScreen should have _expand_citation method."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, '_expand_citation')
        assert callable(screen._expand_citation)

    def test_has_run_query_method(self):
        """QueryScreen should have _run_query method."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, '_run_query')

    def test_has_sync_query_method(self):
        """QueryScreen should have _sync_query method."""
        from lsm.ui.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, '_sync_query')
        assert callable(screen._sync_query)


class TestIngestScreenIntegration:
    """Integration tests for IngestScreen with widgets."""

    def test_has_completer(self):
        """IngestScreen should have a completer for autocomplete."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        screen = IngestScreen()
        assert hasattr(screen, '_completer')
        assert callable(screen._completer)

    def test_has_process_command_method(self):
        """IngestScreen should have _process_command method."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        screen = IngestScreen()
        assert hasattr(screen, '_process_command')

    def test_has_run_ingest_command_method(self):
        """IngestScreen should have _run_ingest_command method."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        screen = IngestScreen()
        assert hasattr(screen, '_run_ingest_command')

    def test_has_refresh_stats_method(self):
        """IngestScreen should have _refresh_stats method."""
        from lsm.ui.tui.screens.ingest import IngestScreen
        screen = IngestScreen()
        assert hasattr(screen, '_refresh_stats')

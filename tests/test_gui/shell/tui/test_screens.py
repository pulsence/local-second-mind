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
        from lsm.gui.shell.tui.screens.query import QueryScreen
        assert QueryScreen is not None

    def test_import_ingest_screen(self):
        """Should be able to import IngestScreen."""
        from lsm.gui.shell.tui.screens.ingest import IngestScreen
        assert IngestScreen is not None

    def test_import_settings_screen(self):
        """Should be able to import SettingsScreen."""
        from lsm.gui.shell.tui.screens.settings import SettingsScreen
        assert SettingsScreen is not None

    def test_import_help_screen(self):
        """Should be able to import HelpScreen."""
        from lsm.gui.shell.tui.screens.help import HelpScreen
        assert HelpScreen is not None

    def test_import_main_screen(self):
        """Should be able to import MainScreen."""
        from lsm.gui.shell.tui.screens.main import MainScreen
        assert MainScreen is not None


class TestQueryScreen:
    """Tests for QueryScreen class."""

    def test_is_widget(self):
        """QueryScreen should be a Widget subclass."""
        from lsm.gui.shell.tui.screens.query import QueryScreen
        assert issubclass(QueryScreen, Widget)

    def test_has_bindings(self):
        """QueryScreen should have keyboard bindings."""
        from lsm.gui.shell.tui.screens.query import QueryScreen
        assert hasattr(QueryScreen, 'BINDINGS')
        assert len(QueryScreen.BINDINGS) > 0

    def test_has_reactive_properties(self):
        """QueryScreen should have reactive properties."""
        from lsm.gui.shell.tui.screens.query import QueryScreen
        screen = QueryScreen()
        assert hasattr(screen, 'is_loading')
        assert hasattr(screen, 'selected_citation')


class TestIngestScreen:
    """Tests for IngestScreen class."""

    def test_is_widget(self):
        """IngestScreen should be a Widget subclass."""
        from lsm.gui.shell.tui.screens.ingest import IngestScreen
        assert issubclass(IngestScreen, Widget)

    def test_has_bindings(self):
        """IngestScreen should have keyboard bindings."""
        from lsm.gui.shell.tui.screens.ingest import IngestScreen
        assert hasattr(IngestScreen, 'BINDINGS')
        assert len(IngestScreen.BINDINGS) > 0

    def test_has_reactive_properties(self):
        """IngestScreen should have reactive properties."""
        from lsm.gui.shell.tui.screens.ingest import IngestScreen
        screen = IngestScreen()
        assert hasattr(screen, 'is_building')
        assert hasattr(screen, 'build_progress')
        assert hasattr(screen, 'chunk_count')
        assert hasattr(screen, 'file_count')


class TestSettingsScreen:
    """Tests for SettingsScreen class."""

    def test_is_widget(self):
        """SettingsScreen should be a Widget subclass."""
        from lsm.gui.shell.tui.screens.settings import SettingsScreen
        assert issubclass(SettingsScreen, Widget)

    def test_has_reactive_properties(self):
        """SettingsScreen should have reactive properties."""
        from lsm.gui.shell.tui.screens.settings import SettingsScreen
        screen = SettingsScreen()
        assert hasattr(screen, 'current_mode')


class TestHelpScreen:
    """Tests for HelpScreen class."""

    def test_is_modal_screen(self):
        """HelpScreen should be a ModalScreen subclass."""
        from lsm.gui.shell.tui.screens.help import HelpScreen
        assert issubclass(HelpScreen, ModalScreen)

    def test_has_dismiss_binding(self):
        """HelpScreen should have escape binding to dismiss."""
        from lsm.gui.shell.tui.screens.help import HelpScreen
        assert hasattr(HelpScreen, 'BINDINGS')
        bindings = HelpScreen.BINDINGS
        escape_bindings = [b for b in bindings if b.key == "escape"]
        assert len(escape_bindings) > 0


class TestMainScreen:
    """Tests for MainScreen class."""

    def test_is_screen(self):
        """MainScreen should be a Screen subclass."""
        from lsm.gui.shell.tui.screens.main import MainScreen
        assert issubclass(MainScreen, Screen)

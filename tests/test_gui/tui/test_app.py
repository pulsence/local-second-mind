"""Tests for the main TUI application."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Skip tests if textual is not available
pytest.importorskip("textual")

from textual.app import App


class TestLSMAppImport:
    """Tests for LSMApp module imports."""

    def test_import_lsm_app(self):
        """Should be able to import LSMApp."""
        from lsm.ui.tui.app import LSMApp
        assert LSMApp is not None

    def test_import_run_tui(self):
        """Should be able to import run_tui function."""
        from lsm.ui.tui.app import run_tui
        assert callable(run_tui)

    def test_lsm_app_is_textual_app(self):
        """LSMApp should be a Textual App subclass."""
        from lsm.ui.tui.app import LSMApp
        assert issubclass(LSMApp, App)


class TestLSMAppInit:
    """Tests for LSMApp initialization."""

    def test_app_requires_config(self):
        """LSMApp should require a config parameter."""
        from lsm.ui.tui.app import LSMApp

        # Create a mock config
        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.vectordb.provider = "chromadb"
        mock_config.embed_model = "test-model"
        mock_config.device = "cpu"
        mock_config.collection = "test"
        mock_config.persist_dir = "/tmp/test"
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"
        mock_config.llm = Mock()
        mock_config.llm.get_query_config = Mock(return_value=Mock(model="test"))

        app = LSMApp(mock_config)
        assert app.config is mock_config

    def test_app_has_bindings(self):
        """LSMApp should have keyboard bindings defined."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)
        assert len(app.BINDINGS) > 0

    def test_app_has_css_path(self):
        """LSMApp should have CSS path defined."""
        from lsm.ui.tui.app import LSMApp
        assert LSMApp.CSS_PATH == "styles.tcss"

    def test_app_has_title(self):
        """LSMApp should have title defined."""
        from lsm.ui.tui.app import LSMApp
        assert LSMApp.TITLE == "Local Second Mind"

    def test_app_reactive_properties(self):
        """LSMApp should have reactive properties."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)

        # Check reactive properties are accessible
        assert hasattr(app, 'current_context')
        assert hasattr(app, 'chunk_count')
        assert hasattr(app, 'current_mode')
        assert hasattr(app, 'total_cost')


class TestLSMAppProviders:
    """Tests for LSMApp provider management."""

    def test_providers_initially_none(self):
        """Providers should be None before initialization."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)

        assert app._ingest_provider is None
        assert app._query_embedder is None
        assert app._query_provider is None
        assert app._query_state is None

    def test_provider_properties(self):
        """Provider properties should return internal values."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)

        # Properties should return None initially
        assert app.ingest_provider is None
        assert app.query_embedder is None
        assert app.query_provider is None
        assert app.query_state is None


class TestLSMAppMethods:
    """Tests for LSMApp public methods."""

    def test_update_cost(self):
        """update_cost should add to total_cost."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)
        initial_cost = app.total_cost

        app.update_cost(0.5)
        assert app.total_cost == initial_cost + 0.5

        app.update_cost(0.25)
        assert app.total_cost == initial_cost + 0.75

    def test_update_chunk_count(self):
        """update_chunk_count should set chunk_count."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)

        app.update_chunk_count(1000)
        assert app.chunk_count == 1000

        app.update_chunk_count(5000)
        assert app.chunk_count == 5000


class TestLSMAppActions:
    """Tests for LSMApp action methods."""

    def test_action_methods_exist(self):
        """Action methods should be defined."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)

        assert hasattr(app, 'action_switch_ingest')
        assert hasattr(app, 'action_switch_query')
        assert hasattr(app, 'action_switch_settings')
        assert hasattr(app, 'action_show_help')
        assert hasattr(app, 'action_quit')

        assert callable(app.action_switch_ingest)
        assert callable(app.action_switch_query)
        assert callable(app.action_switch_settings)
        assert callable(app.action_show_help)
        assert callable(app.action_quit)


class TestLSMAppStatusBarIntegration:
    """Tests for LSMApp StatusBar integration."""

    def test_has_watch_methods(self):
        """LSMApp should have watch methods for StatusBar sync."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)

        assert hasattr(app, 'watch_current_mode')
        assert hasattr(app, 'watch_chunk_count')
        assert hasattr(app, 'watch_total_cost')
        assert callable(app.watch_current_mode)
        assert callable(app.watch_chunk_count)
        assert callable(app.watch_total_cost)

    def test_watch_methods_handle_missing_statusbar(self):
        """Watch methods should handle missing StatusBar gracefully."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)

        # These should not raise even when StatusBar is not mounted
        app.watch_current_mode("insight")
        app.watch_chunk_count(100)
        app.watch_total_cost(1.50)

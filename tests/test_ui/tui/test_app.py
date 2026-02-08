"""Tests for the main TUI application."""

from __future__ import annotations

import asyncio
import importlib
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

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
        keys = {binding.key for binding in app.BINDINGS}
        assert "ctrl+z" in keys

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

    def test_run_on_ui_thread_falls_back_on_same_thread_error(self):
        """run_on_ui_thread should execute callback directly on same-thread error."""
        from lsm.ui.tui.app import LSMApp

        mock_config = Mock()
        mock_config.vectordb = Mock()
        mock_config.query = Mock()
        mock_config.query.mode = "grounded"

        app = LSMApp(mock_config)
        seen = {"called": False}

        def _cb():
            seen["called"] = True

        def _raise(_fn):
            raise RuntimeError("The `call_from_thread` method must run in a different thread from the app.")

        app.call_from_thread = _raise  # type: ignore[assignment]
        app.run_on_ui_thread(_cb)
        assert seen["called"] is True


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


def _build_config(tmp_path: Path, provider: str = "chromadb"):
    cfg = SimpleNamespace()
    cfg.vectordb = SimpleNamespace(provider=provider)
    cfg.embed_model = "mini-model"
    cfg.device = "cpu"
    cfg.collection = "kb"
    cfg.persist_dir = str(tmp_path / "chroma")
    cfg.query = SimpleNamespace(mode="grounded")
    cfg.llm = SimpleNamespace(get_query_config=lambda: SimpleNamespace(model="gpt-test"))
    return cfg


class TestLSMAppBehavior:
    def test_switch_actions_and_tab_activation(self, tmp_path: Path):
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(_build_config(tmp_path))
        tabs = SimpleNamespace(active="")
        app.query_one = lambda selector, _cls=None: tabs if selector else tabs  # type: ignore[assignment]

        app.action_switch_ingest()
        assert app.current_context == "ingest"
        app.action_switch_query()
        assert app.current_context == "query"
        app.action_switch_remote()
        assert app.current_context == "remote"
        app.action_switch_settings()
        assert app.current_context == "settings"

        event = SimpleNamespace(tab=SimpleNamespace(id="query-tab"))
        app.on_tabbed_content_tab_activated(event)
        assert app.current_context == "query"

        event_bad = SimpleNamespace(tab=SimpleNamespace(id="unknown-tab"))
        app.on_tabbed_content_tab_activated(event_bad)
        assert app.current_context == "query"

    def test_activate_settings_subtabs_and_actions(self, tmp_path: Path):
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(_build_config(tmp_path))
        tabs = SimpleNamespace(active="")
        settings_screen = SimpleNamespace(query_one=lambda *_args, **_kwargs: tabs)

        app.query_one = lambda selector, _cls=None: settings_screen  # type: ignore[assignment]
        app.current_context = "settings"

        app.action_settings_tab_1()
        assert tabs.active == "settings-global"
        app.action_settings_tab_6()
        assert tabs.active == "settings-modes"

        app.current_context = "query"
        tabs.active = ""
        app.action_settings_tab_2()
        assert tabs.active == ""

    def test_show_help_and_quit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(_build_config(tmp_path))
        pushed = {}
        monkeypatch.setattr(app, "push_screen", lambda screen: pushed.setdefault("screen", screen))
        monkeypatch.setattr(app, "exit", lambda: pushed.setdefault("quit", True))

        app.action_show_help()
        assert pushed["screen"] is not None
        app.action_quit()
        assert pushed["quit"] is True

    def test_watch_methods_update_status_bar(self, tmp_path: Path):
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(_build_config(tmp_path))
        status = SimpleNamespace(mode="", chunk_count=0, total_cost=0.0)
        app.query_one = lambda *_args, **_kwargs: status  # type: ignore[assignment]

        app.watch_current_mode("insight")
        app.watch_chunk_count(42)
        app.watch_total_cost(1.25)

        assert status.mode == "insight"
        assert status.chunk_count == 42
        assert status.total_cost == 1.25

    def test_on_mount_success_and_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(_build_config(tmp_path))
        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_args, **_kwargs: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_async_init_query_context", lambda: asyncio.sleep(0))

        asyncio.run(app.on_mount())
        assert app.current_context == "query"
        assert tabs.active == "query"

        app2 = LSMApp(_build_config(tmp_path))
        app2.query_one = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no tabs"))  # type: ignore[assignment]
        monkeypatch.setattr(app2, "_setup_tui_logging", lambda: None)

        async def _fail():
            raise RuntimeError("broken")

        monkeypatch.setattr(app2, "_async_init_query_context", _fail)
        notices = []
        monkeypatch.setattr(app2, "notify", lambda msg, **kwargs: notices.append((msg, kwargs)))
        asyncio.run(app2.on_mount())
        assert notices and "Query context unavailable" in notices[0][0]

    def test_async_init_ingest_context(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(_build_config(tmp_path))

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", _to_thread)
        monkeypatch.setattr("lsm.vectordb.create_vectordb_provider", lambda cfg: "provider")
        asyncio.run(app._async_init_ingest_context())
        assert app.ingest_provider == "provider"

        # Already initialized is a no-op.
        asyncio.run(app._async_init_ingest_context())
        assert app.ingest_provider == "provider"

    def test_async_init_query_context_missing_chroma_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(_build_config(tmp_path, provider="chromadb"))

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", _to_thread)
        retrieval_mod = importlib.import_module("lsm.query.retrieval")
        monkeypatch.setattr(retrieval_mod, "init_embedder", lambda *_args, **_kwargs: "embed")

        with pytest.raises(FileNotFoundError):
            asyncio.run(app._async_init_query_context())

    def test_async_init_query_context_success_and_empty_warning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from lsm.ui.tui.app import LSMApp

        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir(parents=True)
        cfg = _build_config(tmp_path, provider="chromadb")
        cfg.persist_dir = str(persist_dir)
        app = LSMApp(cfg)

        class _Provider:
            def count(self):
                return 0

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", _to_thread)
        retrieval_mod = importlib.import_module("lsm.query.retrieval")
        cost_mod = importlib.import_module("lsm.query.cost_tracking")
        session_mod = importlib.import_module("lsm.query.session")
        monkeypatch.setattr(retrieval_mod, "init_embedder", lambda *_args, **_kwargs: "embedder")
        monkeypatch.setattr("lsm.vectordb.create_vectordb_provider", lambda *_args, **_kwargs: _Provider())
        monkeypatch.setattr(cost_mod, "CostTracker", lambda: "costs")
        monkeypatch.setattr(session_mod, "SessionState", lambda model, cost_tracker: SimpleNamespace(model=model, cost_tracker=cost_tracker))
        notices = []
        monkeypatch.setattr(app, "notify", lambda msg, **kwargs: notices.append((msg, kwargs)))

        asyncio.run(app._async_init_query_context())
        assert app.query_embedder == "embedder"
        assert app.query_provider is not None
        assert app.query_state.model == "gpt-test"
        assert app.current_mode == "grounded"
        assert app.chunk_count == 0
        assert notices and "is empty" in notices[0][0]

    def test_run_tui_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from lsm.ui.tui.app import run_tui, LSMApp

        monkeypatch.setattr(LSMApp, "run", lambda self: None)
        assert run_tui(_build_config(tmp_path)) == 0

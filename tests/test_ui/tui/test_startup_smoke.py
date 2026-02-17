"""Startup smoke tests for the TUI application.

Verifies that LSMApp can launch through init/compose/mount lifecycle
without crashing and reaches the Query home screen as the default context.
"""

from __future__ import annotations

import asyncio

import pytest
from types import SimpleNamespace

pytest.importorskip("textual")

from tests.test_ui.tui.fixtures import create_startup_mock_config


class TestStartupSmoke:
    """Verify LSMApp launches to Query home without exceptions."""

    def test_init_no_exception(self) -> None:
        """LSMApp.__init__ completes without raising."""
        from lsm.ui.tui.app import LSMApp

        cfg = create_startup_mock_config()
        app = LSMApp(cfg)
        assert app.config is cfg
        assert app._query_provider is None
        assert app._ingest_provider is None

    def test_compose_references_expected_screens(self) -> None:
        """compose() method references all 5 screen classes and expected tab IDs."""
        import inspect
        from lsm.ui.tui.app import LSMApp

        source = inspect.getsource(LSMApp.compose)

        # All 5 screen classes must be referenced
        for screen in ("QueryScreen", "IngestScreen", "RemoteScreen",
                        "AgentsScreen", "SettingsScreen"):
            assert screen in source, f"{screen} not found in compose()"

        # All 5 tab pane IDs must be present
        for tab_id in ("query", "ingest", "remote", "agents", "settings"):
            assert f'id="{tab_id}"' in source or f"id='{tab_id}'" in source, (
                f"tab id '{tab_id}' not found in compose()"
            )

        # Core UI structure widgets
        assert "Header" in source
        assert "Footer" in source
        assert "StatusBar" in source
        assert "TabbedContent" in source

    def test_on_mount_reaches_query_context(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """on_mount sets active context to query without crashing."""
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(create_startup_mock_config())
        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_a, **_kw: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_schedule_background_init", lambda: None)

        asyncio.run(app.on_mount())

        assert app.current_context == "query"
        assert app.ui_state.active_context == "query"
        assert tabs.active == "query"

    def test_on_mount_defers_agent_binding(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """on_mount does not call _bind_agent_runtime_events directly."""
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(create_startup_mock_config())
        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_a, **_kw: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)

        bind_called = {"count": 0}
        original_bind = app._bind_agent_runtime_events

        def _tracking_bind() -> None:
            bind_called["count"] += 1
            original_bind()

        monkeypatch.setattr(app, "_bind_agent_runtime_events", _tracking_bind)

        # Capture what _schedule_background_init defers
        deferred = []
        monkeypatch.setattr(app, "call_after_refresh", lambda fn: deferred.append(fn))

        asyncio.run(app.on_mount())

        # _bind_agent_runtime_events should NOT have been called during on_mount
        assert bind_called["count"] == 0
        # But it should have been scheduled for deferred execution
        assert len(deferred) == 1

    def test_on_mount_survives_missing_tabbed_content(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """on_mount handles missing TabbedContent gracefully."""
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(create_startup_mock_config())

        def _raise(*_a, **_kw):
            raise RuntimeError("no tabs")

        app.query_one = _raise  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_schedule_background_init", lambda: None)

        # Existing try/except in on_mount should swallow this
        asyncio.run(app.on_mount())
        # current_context still gets set via _set_active_context even if
        # the physical tab activation fails — verify no crash
        assert hasattr(app, "current_context")

    def test_lazy_providers_not_initialized_at_startup(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Providers remain None after init and mount — they are lazy."""
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(create_startup_mock_config())
        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_a, **_kw: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_schedule_background_init", lambda: None)

        asyncio.run(app.on_mount())

        assert app._query_provider is None
        assert app._query_embedder is None
        assert app._ingest_provider is None


class TestScreenImportSmoke:
    """Verify all screen classes can be imported without error."""

    def test_all_screen_classes_importable(self) -> None:
        """All TUI screen classes can be imported."""
        from lsm.ui.tui.screens.query import QueryScreen
        from lsm.ui.tui.screens.ingest import IngestScreen
        from lsm.ui.tui.screens.remote import RemoteScreen
        from lsm.ui.tui.screens.agents import AgentsScreen
        from lsm.ui.tui.screens.settings import SettingsScreen

        assert QueryScreen is not None
        assert IngestScreen is not None
        assert RemoteScreen is not None
        assert AgentsScreen is not None
        assert SettingsScreen is not None

    def test_query_screen_compose_has_command_input(self) -> None:
        """QueryScreen compose() references query-command-input widget ID."""
        import inspect
        from lsm.ui.tui.screens.query import QueryScreen

        source = inspect.getsource(QueryScreen.compose)
        assert "query-command-input" in source

    def test_all_screen_imports_under_budget(self) -> None:
        """Importing all 5 screen modules completes within 1s total."""
        import time

        start = time.monotonic()
        from lsm.ui.tui.screens.query import QueryScreen  # noqa: F811
        from lsm.ui.tui.screens.ingest import IngestScreen  # noqa: F811
        from lsm.ui.tui.screens.remote import RemoteScreen  # noqa: F811
        from lsm.ui.tui.screens.agents import AgentsScreen  # noqa: F811
        from lsm.ui.tui.screens.settings import SettingsScreen  # noqa: F811
        elapsed_ms = (time.monotonic() - start) * 1000

        assert QueryScreen is not None
        assert elapsed_ms < 1000, (
            f"Importing all 5 screen modules took {elapsed_ms:.0f}ms, "
            f"budget is 1000ms"
        )

    def test_retrieval_module_has_lazy_ml_imports(self) -> None:
        """retrieval module must not expose SentenceTransformer at module level.

        This verifies the lazy-import refactoring: the heavy ML stack is only
        imported inside function bodies, not at module scope. Earlier tests in
        the suite may have imported sentence_transformers into sys.modules, so
        we verify structural laziness rather than sys.modules state.
        """
        import lsm.query.retrieval as retrieval

        # SentenceTransformer should NOT be a module-level attribute
        assert not hasattr(retrieval, "SentenceTransformer"), (
            "SentenceTransformer is a module-level attribute — lazy import broken"
        )
        # _import_sentence_transformer helper should exist
        assert hasattr(retrieval, "_import_sentence_transformer"), (
            "_import_sentence_transformer helper is missing"
        )

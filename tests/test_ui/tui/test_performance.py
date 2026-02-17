"""Startup performance budget and timeline tests for the TUI application.

Verifies that startup timing instrumentation works correctly and that
the app reaches Query/Home interactive state within the performance budget.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest
from types import SimpleNamespace

pytest.importorskip("textual")

from tests.test_ui.tui.fixtures import create_startup_mock_config


# Budget configurable via env for CI environments with different characteristics
_QUERY_INTERACTIVE_BUDGET_MS = float(
    os.environ.get("LSM_TEST_STARTUP_BUDGET_MS", "1000"),
)


class TestStartupTimeline:
    """Unit tests for the StartupTimeline dataclass."""

    def test_mark_records_milestones(self) -> None:
        """Milestones are recorded with increasing elapsed_ms."""
        from lsm.ui.tui.app import StartupTimeline

        timeline = StartupTimeline()
        timeline.mark("a")
        time.sleep(0.005)
        timeline.mark("b")

        assert len(timeline.milestones) == 2
        assert timeline.milestones[0].name == "a"
        assert timeline.milestones[1].name == "b"
        assert timeline.milestones[0].elapsed_ms <= timeline.milestones[1].elapsed_ms

    def test_elapsed_ms_returns_none_for_unknown(self) -> None:
        """elapsed_ms returns None for an unrecorded milestone name."""
        from lsm.ui.tui.app import StartupTimeline

        timeline = StartupTimeline()
        timeline.mark("a")
        assert timeline.elapsed_ms("a") is not None
        assert timeline.elapsed_ms("nonexistent") is None

    def test_total_ms_increases(self) -> None:
        """total_ms grows over time."""
        from lsm.ui.tui.app import StartupTimeline

        timeline = StartupTimeline()
        t1 = timeline.total_ms()
        time.sleep(0.005)
        t2 = timeline.total_ms()
        assert t2 > t1

    def test_milestones_returns_copy(self) -> None:
        """milestones property returns a copy, not the internal list."""
        from lsm.ui.tui.app import StartupTimeline

        timeline = StartupTimeline()
        timeline.mark("x")
        ms = timeline.milestones
        ms.clear()
        assert len(timeline.milestones) == 1


class TestStartupPerformanceBudget:
    """Startup timing budget enforcement."""

    def test_query_interactive_under_budget(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LSMApp init + on_mount reaches query_interactive under budget.

        Note: This tests init + on_mount path (without compose). For the full
        compose import-chain test, see test_compose_import_chain_under_budget.
        """
        from lsm.ui.tui.app import LSMApp

        cfg = create_startup_mock_config()

        start = time.monotonic()
        app = LSMApp(cfg)

        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_a, **_kw: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_schedule_background_init", lambda: None)

        asyncio.run(app.on_mount())
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < _QUERY_INTERACTIVE_BUDGET_MS, (
            f"Startup took {elapsed_ms:.0f}ms, budget is "
            f"{_QUERY_INTERACTIVE_BUDGET_MS:.0f}ms"
        )

    def test_compose_import_chain_under_budget(self) -> None:
        """Importing QueryScreen (the compose path) completes within budget.

        This catches the real-world startup chain: compose() imports
        QueryScreen which transitively imports query modules. With lazy
        ML imports, this should be fast.

        In compose(), screens are imported sequentially. By the time
        QueryScreen is imported, earlier screens (IngestScreen) have already
        warmed up shared dependencies (Textual widgets, LSM providers, HTTP
        stack, remote providers). This test pre-loads IngestScreen to
        simulate that shared-dependency warmup, then measures only the
        incremental cost unique to QueryScreen.
        """
        # Pre-load app + an earlier screen to warm shared dependencies
        # (Textual widgets, lsm.remote, HTTP stack, etc.)
        from lsm.ui.tui.app import LSMApp  # noqa: F401
        from lsm.ui.tui.screens.ingest import IngestScreen  # noqa: F401

        start = time.monotonic()
        from lsm.ui.tui.screens.query import QueryScreen  # noqa: F811
        elapsed_ms = (time.monotonic() - start) * 1000

        assert QueryScreen is not None
        # Budget: incremental QueryScreen import should be well under 1s
        assert elapsed_ms < _QUERY_INTERACTIVE_BUDGET_MS, (
            f"QueryScreen import took {elapsed_ms:.0f}ms, budget is "
            f"{_QUERY_INTERACTIVE_BUDGET_MS:.0f}ms"
        )

    def test_sentence_transformers_not_required_by_screen_import(self) -> None:
        """Importing QueryScreen must NOT require sentence_transformers.

        This is the key regression test: the lazy-import refactoring must
        prevent the heavy ML stack from being a required side-effect of
        importing screen modules. We verify by checking the retrieval module
        does not have SentenceTransformer as a module-level attribute.
        """
        import lsm.query.retrieval as retrieval

        # If lazy imports are working, SentenceTransformer should NOT be
        # a module-level attribute (it's now inside _import_sentence_transformer)
        assert not hasattr(retrieval, "SentenceTransformer"), (
            "SentenceTransformer is a module-level attribute of retrieval — "
            "the lazy-import refactoring is broken"
        )
        # The old _sentence_transformer_import_error should also be gone
        assert not hasattr(retrieval, "_sentence_transformer_import_error"), (
            "_sentence_transformer_import_error still exists as module-level attribute"
        )

    def test_startup_milestones_recorded(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After lifecycle, timeline has all expected milestones."""
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(create_startup_mock_config())
        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_a, **_kw: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_schedule_background_init", lambda: None)

        asyncio.run(app.on_mount())

        timeline = app._startup_timeline
        expected = [
            "init_start", "init_complete",
            "mount_start", "query_interactive",
            "tui_logging_ready", "mount_complete",
        ]
        for name in expected:
            assert timeline.elapsed_ms(name) is not None, (
                f"Missing milestone: {name}"
            )

    def test_milestone_ordering(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Milestones occur in expected chronological order."""
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(create_startup_mock_config())
        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_a, **_kw: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_schedule_background_init", lambda: None)

        asyncio.run(app.on_mount())

        timeline = app._startup_timeline
        ordered = [
            "init_start", "init_complete",
            "mount_start", "query_interactive",
            "tui_logging_ready", "mount_complete",
        ]
        prev_ms = -1.0
        for name in ordered:
            ms = timeline.elapsed_ms(name)
            assert ms is not None
            assert ms >= prev_ms, (
                f"Milestone {name} ({ms:.1f}ms) should come after "
                f"previous ({prev_ms:.1f}ms)"
            )
            prev_ms = ms

    def test_background_init_deferred(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_bind_agent_runtime_events is deferred, not called during on_mount."""
        from lsm.ui.tui.app import LSMApp

        app = LSMApp(create_startup_mock_config())
        tabs = SimpleNamespace(active="")
        app.query_one = lambda *_a, **_kw: tabs  # type: ignore[assignment]
        monkeypatch.setattr(app, "_setup_tui_logging", lambda: None)
        monkeypatch.setattr(app, "_preload_ml_stack", lambda: None)

        bind_called = {"count": 0}

        def _tracking_bind() -> None:
            bind_called["count"] += 1

        monkeypatch.setattr(app, "_bind_agent_runtime_events", _tracking_bind)

        deferred = []
        monkeypatch.setattr(app, "call_after_refresh", lambda fn: deferred.append(fn))

        asyncio.run(app.on_mount())

        # Not called during mount
        assert bind_called["count"] == 0
        # Scheduled for after-refresh execution
        assert len(deferred) == 1

        # Execute the deferred callback — starts background thread
        deferred[0]()
        # Wait for the background thread to complete
        import time
        for _ in range(50):
            if bind_called["count"] > 0:
                break
            time.sleep(0.05)
        assert bind_called["count"] == 1


class TestAgentsDeferredInit:
    """Tests for AgentsScreen deferred initialization."""

    def test_on_mount_minimal(self) -> None:
        """After on_mount, deferred init has not run yet."""
        from lsm.ui.tui.screens.agents import AgentsScreen

        screen = AgentsScreen(id="agents-screen")
        # Provide minimal app mock
        fake_app = SimpleNamespace(
            start_managed_worker=lambda **kw: None,
            cancel_managed_workers_for_owner=lambda **kw: {},
            stop_managed_timers_for_owner=lambda **kw: {},
            start_managed_timer=lambda **kw: None,
        )
        screen._test_app = fake_app  # type: ignore[attr-defined]

        # Override app property for test
        original_app = type(screen).app
        type(screen).app = property(lambda self: self._test_app)  # type: ignore[assignment]
        try:
            # Stub query_one to avoid Textual context errors
            screen.query_one = lambda *a, **kw: SimpleNamespace(  # type: ignore[assignment]
                update=lambda *a: None,
                clear=lambda: None,
                clear_columns=lambda: None,
                add_columns=lambda *a: None,
                add_column=lambda *a, **kw: None,
            )
            screen.on_mount()
            assert screen._deferred_init_done is False
        finally:
            type(screen).app = original_app  # type: ignore[assignment]

    def test_ensure_deferred_init_runs_once(self) -> None:
        """_ensure_deferred_init runs full init on first call, no-op on second."""
        from lsm.ui.tui.screens.agents import AgentsScreen

        screen = AgentsScreen(id="agents-screen")
        screen._deferred_init_done = False

        call_count = {"n": 0}

        def _tracking_refresh():
            call_count["n"] += 1

        # Stub all the methods that _ensure_deferred_init calls
        screen._refresh_agent_options = _tracking_refresh  # type: ignore[assignment]
        screen._refresh_running_agents = _tracking_refresh  # type: ignore[assignment]
        screen._refresh_interaction_panel = _tracking_refresh  # type: ignore[assignment]
        screen._initialize_meta_controls = _tracking_refresh  # type: ignore[assignment]
        screen._refresh_meta_panel = _tracking_refresh  # type: ignore[assignment]
        screen._initialize_schedule_controls = _tracking_refresh  # type: ignore[assignment]
        screen._refresh_schedule_entries = _tracking_refresh  # type: ignore[assignment]
        screen._refresh_memory_candidates = _tracking_refresh  # type: ignore[assignment]
        screen._restart_running_refresh_timer = _tracking_refresh  # type: ignore[assignment]
        screen._restart_interaction_poll_timer = _tracking_refresh  # type: ignore[assignment]
        screen._restart_log_stream_timer = _tracking_refresh  # type: ignore[assignment]

        screen._ensure_deferred_init()
        first_count = call_count["n"]
        assert first_count == 11  # 8 refresh/init + 3 timer restarts
        assert screen._deferred_init_done is True

        # Second call is a no-op
        screen._ensure_deferred_init()
        assert call_count["n"] == first_count

    def test_deferred_init_triggered_on_runtime_event(self) -> None:
        """Runtime event triggers deferred init before handling."""
        from lsm.ui.tui.screens.agents import AgentsScreen

        screen = AgentsScreen(id="agents-screen")
        screen._deferred_init_done = False

        init_called = {"count": 0}
        original_ensure = screen._ensure_deferred_init

        def _tracking_ensure():
            init_called["count"] += 1
            # Set flag without running actual init methods
            screen._deferred_init_done = True

        screen._ensure_deferred_init = _tracking_ensure  # type: ignore[assignment]

        # Stub the refresh methods called by the event handler
        screen._refresh_running_agents = lambda: None  # type: ignore[assignment]
        screen._refresh_interaction_panel = lambda: None  # type: ignore[assignment]
        screen._refresh_meta_panel = lambda: None  # type: ignore[assignment]

        # Simulate a runtime event
        event = SimpleNamespace(event=SimpleNamespace(event_type="run_started"))
        screen.on_agent_runtime_event_message(event)

        assert init_called["count"] == 1

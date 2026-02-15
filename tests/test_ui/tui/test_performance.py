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
        """LSMApp init + compose + on_mount reaches query_interactive under budget."""
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

        # Execute the deferred callback
        deferred[0]()
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

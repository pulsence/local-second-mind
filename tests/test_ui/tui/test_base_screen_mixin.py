"""Tests for ManagedScreenMixin shared lifecycle helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest

from lsm.ui.tui.screens.base import ManagedScreenMixin


class _FakeApp:
    """Minimal app double with managed lifecycle API stubs."""

    def __init__(self) -> None:
        self.started_workers: list[dict] = []
        self.registered_workers: list[dict] = []
        self.cleared_workers: list[str] = []
        self.cancelled_owners: list[str] = []
        self.started_timers: list[dict] = []
        self.stopped_timers: list[dict] = []
        self.stopped_timer_owners: list[str] = []
        self._cancel_results: dict = {}

    def start_managed_worker(self, *, owner, key, timeout_s, start):
        self.started_workers.append({"owner": owner, "key": key, "timeout_s": timeout_s})
        start()

    def register_managed_worker(self, *, owner, key, worker, timeout_s):
        self.registered_workers.append({"owner": owner, "key": key, "timeout_s": timeout_s})

    def clear_managed_worker(self, *, owner, key):
        self.cleared_workers.append(key)

    def cancel_managed_workers_for_owner(self, *, owner, reason):
        self.cancelled_owners.append(owner)
        return self._cancel_results

    def start_managed_timer(self, *, owner, key, start, restart):
        self.started_timers.append({"owner": owner, "key": key})
        return start()

    def stop_managed_timer(self, *, owner, key, reason):
        self.stopped_timers.append({"owner": owner, "key": key})

    def stop_managed_timers_for_owner(self, *, owner, reason):
        self.stopped_timer_owners.append(owner)


class _FakeScreen(ManagedScreenMixin):
    """Test double combining mixin with minimal widget-like interface."""

    def __init__(self, app=None, widget_id: str = "") -> None:
        self._test_app = app
        self.id = widget_id
        self.run_worker_calls: list[tuple] = []
        self.set_interval_calls: list[tuple] = []

    @property
    def app(self):
        return self._test_app

    def run_worker(self, coro, exclusive=False):
        self.run_worker_calls.append((coro, exclusive))
        if hasattr(coro, "close"):
            coro.close()

    def set_interval(self, interval, callback):
        self.set_interval_calls.append((interval, callback))
        return SimpleNamespace(stop=lambda: None)


class TestWorkerOwnerToken:
    def test_uses_widget_id_when_set(self) -> None:
        screen = _FakeScreen(widget_id="my-screen")
        assert screen._worker_owner_token() == "my-screen"

    def test_falls_back_to_class_name(self) -> None:
        screen = _FakeScreen(widget_id="")
        assert screen._worker_owner_token() == "_FakeScreen"


class TestStartManagedWorker:
    def test_delegates_to_app(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app, widget_id="query")

        screen._start_managed_worker(
            worker_key="test-work",
            work_factory=lambda: _noop_coro(),
            timeout_s=5.0,
            exclusive=True,
        )

        assert len(app.started_workers) == 1
        assert app.started_workers[0]["key"] == "test-work"
        assert app.started_workers[0]["owner"] == "query"

    def test_falls_back_without_app(self) -> None:
        screen = _FakeScreen(app=None)
        screen._start_managed_worker(
            worker_key="test",
            work_factory=lambda: _noop_coro(),
            timeout_s=1.0,
            exclusive=False,
        )
        assert len(screen.run_worker_calls) == 1


class TestRegisterManagedWorker:
    def test_delegates_to_app(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app)
        screen._register_managed_worker(key="stop", worker=object(), timeout_s=10.0)
        assert len(app.registered_workers) == 1

    def test_noop_without_app(self) -> None:
        screen = _FakeScreen(app=None)
        screen._register_managed_worker(key="stop", worker=object(), timeout_s=10.0)


class TestClearManagedWorker:
    def test_delegates_to_app(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app)
        screen._clear_managed_worker("stop")
        assert "stop" in app.cleared_workers


class TestCancelManagedWorkers:
    def test_delegates_to_app(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app, widget_id="ingest")
        screen._cancel_managed_workers(reason="unmount")
        assert "ingest" in app.cancelled_owners


class TestTimerLifecycle:
    def test_start_managed_timer(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app, widget_id="agents")
        result = screen._start_managed_timer(
            key="poll",
            interval_seconds=2.0,
            callback=lambda: None,
        )
        assert len(app.started_timers) == 1
        assert app.started_timers[0]["key"] == "poll"
        assert result is not None

    def test_start_timer_fallback(self) -> None:
        screen = _FakeScreen(app=None)
        result = screen._start_timer(interval_seconds=1.0, callback=lambda: None)
        assert result is not None
        assert len(screen.set_interval_calls) == 1

    def test_stop_managed_timer(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app)
        screen._stop_managed_timer(
            key="poll",
            timer=SimpleNamespace(stop=lambda: None),
            reason="cleanup",
        )
        assert len(app.stopped_timers) == 1

    def test_cancel_managed_timers(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app, widget_id="query")
        screen._cancel_managed_timers(reason="quit")
        assert "query" in app.stopped_timer_owners

    def test_stop_managed_timers_alias(self) -> None:
        app = _FakeApp()
        screen = _FakeScreen(app=app, widget_id="agents")
        screen._stop_managed_timers(reason="quit")
        assert "agents" in app.stopped_timer_owners

    def test_stop_timer_noop_on_none(self) -> None:
        screen = _FakeScreen()
        screen._stop_timer(None)  # Should not raise


async def _noop_coro():
    pass

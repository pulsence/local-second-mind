from __future__ import annotations

from lsm.ui.tui.state import AppState


def test_defaults_and_snapshot() -> None:
    state = AppState()

    snap = state.snapshot()

    assert snap.active_context == "query"
    assert snap.density_mode == "auto"
    assert snap.selected_agent_id is None
    assert snap.notifications == ()


def test_context_density_and_selected_agent_updates() -> None:
    state = AppState()

    state.set_active_context("settings")
    state.set_density_mode("compact")
    state.set_selected_agent_id("agent-123")

    assert state.active_context == "settings"
    assert state.density_mode == "compact"
    assert state.selected_agent_id == "agent-123"


def test_notifications_push_and_drain() -> None:
    state = AppState()

    first = state.push_notification("Saved", severity="info")
    second = state.push_notification("Permission required", severity="warning")

    assert first.message == "Saved"
    assert second.severity == "warning"
    assert len(state.notifications) == 2

    drained = state.drain_notifications()
    assert len(drained) == 2
    assert state.notifications == ()


def test_clear_notifications() -> None:
    state = AppState()

    state.push_notification("A")
    state.push_notification("B")
    state.clear_notifications()

    assert state.notifications == ()

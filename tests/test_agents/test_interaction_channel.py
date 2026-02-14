from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time

import pytest

_INTERACTION_PATH = (
    Path(__file__).resolve().parents[2] / "lsm" / "agents" / "interaction.py"
)
_INTERACTION_SPEC = importlib.util.spec_from_file_location(
    "lsm_interaction_module",
    _INTERACTION_PATH,
)
if _INTERACTION_SPEC is None or _INTERACTION_SPEC.loader is None:
    raise RuntimeError(f"Unable to load interaction module at {_INTERACTION_PATH}")
_INTERACTION_MODULE = importlib.util.module_from_spec(_INTERACTION_SPEC)
sys.modules[_INTERACTION_SPEC.name] = _INTERACTION_MODULE
_INTERACTION_SPEC.loader.exec_module(_INTERACTION_MODULE)

InteractionChannel = _INTERACTION_MODULE.InteractionChannel
InteractionRequest = _INTERACTION_MODULE.InteractionRequest
InteractionResponse = _INTERACTION_MODULE.InteractionResponse


def _permission_request(request_id: str, tool_name: str = "write_file") -> InteractionRequest:
    return InteractionRequest(
        request_id=request_id,
        request_type="permission",
        tool_name=tool_name,
        risk_level="writes_workspace",
        reason="Tool requires user permission",
        args_summary='{"path": "notes/out.md"}',
        prompt="Allow write_file?",
    )


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_channel_blocks_until_response() -> None:
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    request = _permission_request("req-1")
    holder: dict[str, InteractionResponse] = {}

    def _worker() -> None:
        holder["response"] = channel.post_request(request)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)

    pending = channel.get_pending_request()
    assert pending is not None
    assert pending.request_id == "req-1"

    channel.post_response(
        InteractionResponse(
            request_id="req-1",
            decision="approve",
            user_message="Approved",
        )
    )
    thread.join(timeout=1.0)

    assert "response" in holder
    assert holder["response"].decision == "approve"
    assert holder["response"].user_message == "Approved"
    assert channel.has_pending() is False


def test_channel_timeout_denies_by_default() -> None:
    channel = InteractionChannel(timeout_seconds=0.05, timeout_action="deny")
    with pytest.raises(PermissionError, match="timed out"):
        channel.post_request(_permission_request("req-timeout-deny"))
    assert channel.has_pending() is False


def test_channel_timeout_can_auto_approve() -> None:
    channel = InteractionChannel(timeout_seconds=0.05, timeout_action="approve")
    response = channel.post_request(_permission_request("req-timeout-approve"))
    assert response.request_id == "req-timeout-approve"
    assert response.decision == "approve"
    assert "timed out" in response.user_message.lower()
    assert channel.has_pending() is False


def test_channel_records_approve_session_response() -> None:
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    first_request = _permission_request("req-first", tool_name="write_file")
    holder: dict[str, InteractionResponse] = {}

    def _worker() -> None:
        holder["response"] = channel.post_request(first_request)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)
    channel.post_response(
        InteractionResponse(
            request_id="req-first",
            decision="approve_session",
            user_message="Approve for this run",
        )
    )
    thread.join(timeout=1.0)

    assert holder["response"].decision == "approve_session"
    assert channel.has_session_approval("write_file")

    second_request = _permission_request("req-second", tool_name="write_file")
    second_response = channel.post_request(second_request)
    assert second_response.decision == "approve_session"
    assert second_response.request_id == "req-second"
    assert channel.has_pending() is False


def test_channel_cancel_pending_unblocks_waiter() -> None:
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    request = _permission_request("req-cancel")
    holder: dict[str, InteractionResponse] = {}

    def _worker() -> None:
        holder["response"] = channel.post_request(request)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)

    cancelled = channel.cancel_pending("Agent stopped")
    thread.join(timeout=1.0)

    assert cancelled is True
    assert holder["response"].decision == "deny"
    assert "agent stopped" in holder["response"].user_message.lower()
    assert channel.has_pending() is False


def test_channel_shutdown_cancels_waiters_and_rejects_new_requests() -> None:
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    request = _permission_request("req-shutdown")
    holder: dict[str, InteractionResponse] = {}

    def _worker() -> None:
        holder["response"] = channel.post_request(request)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)

    channel.shutdown("App shutting down")
    thread.join(timeout=1.0)

    assert holder["response"].decision == "deny"
    assert "shutting down" in holder["response"].user_message.lower()
    assert channel.has_pending() is False

    with pytest.raises(RuntimeError, match="shutting down"):
        channel.post_request(_permission_request("req-after-shutdown"))


def test_channel_rejects_second_pending_request() -> None:
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    holder: dict[str, InteractionResponse] = {}

    def _worker() -> None:
        holder["response"] = channel.post_request(_permission_request("req-primary"))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)

    with pytest.raises(RuntimeError, match="another is pending"):
        channel.post_request(_permission_request("req-secondary"))

    channel.cancel_pending("done")
    thread.join(timeout=1.0)

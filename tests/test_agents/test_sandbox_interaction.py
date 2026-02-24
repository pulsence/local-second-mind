from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from lsm.agents.base import AgentStatus
from lsm.agents.harness import AgentHarness
from lsm.agents.interaction import InteractionChannel, InteractionResponse
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class WriteTool(BaseTool):
    name = "write_file"
    description = "Write file tool."
    risk_level = "writes_workspace"
    requires_permission = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def execute(self, args: dict) -> str:
        _ = args
        return "write-ok"


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "query": {"mode": "grounded"},
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 5000,
            "max_iterations": 3,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {"write_file": True},
                "require_permission_by_risk": {},
                "execution_mode": "local_only",
                "force_docker": False,
                "tool_llm_assignments": {},
            },
            "interaction": {
                "timeout_seconds": 1,
                "timeout_action": "deny",
            },
            "agent_configs": {},
        },
    }


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_sandbox_interaction_approval_flow(tmp_path: Path) -> None:
    tool = WriteTool()
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    sandbox = ToolSandbox(
        build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json").agents.sandbox,
        interaction_channel=channel,
    )
    args = {"path": str(tmp_path / "out.md"), "content": "ok"}
    holder: dict[str, object] = {}

    def _worker() -> None:
        try:
            holder["result"] = sandbox.execute(tool, args)
        except Exception as exc:  # pragma: no cover - debug guard
            holder["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)

    pending = channel.get_pending_request()
    assert pending is not None
    assert pending.request_type == "permission"
    assert pending.tool_name == "write_file"

    channel.post_response(
        InteractionResponse(
            request_id=pending.request_id,
            decision="approve",
            user_message="ok",
        )
    )
    thread.join(timeout=1.0)
    assert holder.get("result") == "write-ok"
    assert "error" not in holder


def test_sandbox_interaction_denial_flow(tmp_path: Path) -> None:
    tool = WriteTool()
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    sandbox = ToolSandbox(
        build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json").agents.sandbox,
        interaction_channel=channel,
    )
    args = {"path": str(tmp_path / "out.md"), "content": "ok"}
    holder: dict[str, object] = {}

    def _worker() -> None:
        try:
            sandbox.execute(tool, args)
        except Exception as exc:
            holder["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)

    pending = channel.get_pending_request()
    assert pending is not None
    channel.post_response(
        InteractionResponse(
            request_id=pending.request_id,
            decision="deny",
            user_message="Denied by user",
        )
    )
    thread.join(timeout=1.0)
    assert isinstance(holder.get("error"), PermissionError)
    assert "Denied by user" in str(holder["error"])


def test_sandbox_interaction_timeout_and_auto_approve(tmp_path: Path) -> None:
    tool = WriteTool()
    cfg = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json").agents.sandbox
    args = {"path": str(tmp_path / "out.md"), "content": "ok"}

    deny_channel = InteractionChannel(timeout_seconds=0.05, timeout_action="deny")
    deny_sandbox = ToolSandbox(cfg, interaction_channel=deny_channel)
    with pytest.raises(PermissionError, match="timed out"):
        deny_sandbox.execute(tool, args)

    approve_channel = InteractionChannel(timeout_seconds=0.05, timeout_action="approve")
    approve_sandbox = ToolSandbox(cfg, interaction_channel=approve_channel)
    assert approve_sandbox.execute(tool, args) == "write-ok"


def test_sandbox_interaction_session_approval_cache(tmp_path: Path) -> None:
    tool = WriteTool()
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    sandbox = ToolSandbox(
        build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json").agents.sandbox,
        interaction_channel=channel,
    )
    args = {"path": str(tmp_path / "out.md"), "content": "ok"}

    holder: dict[str, object] = {}

    def _worker() -> None:
        holder["result"] = sandbox.execute(tool, args)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)
    pending = channel.get_pending_request()
    assert pending is not None
    channel.post_response(
        InteractionResponse(
            request_id=pending.request_id,
            decision="approve_session",
            user_message="Approve for session",
        )
    )
    thread.join(timeout=1.0)
    assert holder.get("result") == "write-ok"

    # Second execution should bypass prompting for the same tool.
    assert sandbox.execute(tool, args) == "write-ok"
    assert channel.has_pending() is False


def test_sandbox_interaction_cancel_pending_unblocks(tmp_path: Path) -> None:
    tool = WriteTool()
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    sandbox = ToolSandbox(
        build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json").agents.sandbox,
        interaction_channel=channel,
    )
    args = {"path": str(tmp_path / "out.md"), "content": "ok"}
    holder: dict[str, object] = {}

    def _worker() -> None:
        try:
            sandbox.execute(tool, args)
        except Exception as exc:
            holder["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)
    assert channel.cancel_pending("Agent stopped")
    thread.join(timeout=1.0)
    assert isinstance(holder.get("error"), PermissionError)
    assert "Agent stopped" in str(holder["error"])


def test_sandbox_interaction_shutdown_unblocks_and_rejects_new_waits(tmp_path: Path) -> None:
    tool = WriteTool()
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    sandbox = ToolSandbox(
        build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json").agents.sandbox,
        interaction_channel=channel,
    )
    args = {"path": str(tmp_path / "out.md"), "content": "ok"}
    holder: dict[str, object] = {}

    def _worker() -> None:
        try:
            sandbox.execute(tool, args)
        except Exception as exc:
            holder["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    assert _wait_until(channel.has_pending)

    channel.shutdown("App shutting down")
    thread.join(timeout=1.0)
    assert isinstance(holder.get("error"), PermissionError)
    assert "shutting down" in str(holder["error"]).lower()

    with pytest.raises(PermissionError, match="shut down|shutting down"):
        sandbox.execute(tool, args)


def test_harness_waiting_status_and_stop_cancellation(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self._responses = [
                json.dumps(
                    {
                        "response": "Need write permission",
                        "action": "write_file",
                        "action_arguments": {
                            "path": str(tmp_path / "out.md"),
                            "content": "hello",
                        },
                    }
                ),
                json.dumps({"response": "Done", "action": "DONE", "action_arguments": {}}),
            ]

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            return self._responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(WriteTool())
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    sandbox = ToolSandbox(config.agents.sandbox, interaction_channel=channel)
    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        sandbox,
        agent_name="wait-status",
        interaction_channel=channel,
    )
    thread = harness.start_background(
        AgentContext(messages=[{"role": "user", "content": "test"}])
    )
    assert _wait_until(channel.has_pending)
    assert harness.state.status == AgentStatus.WAITING_USER

    pending = channel.get_pending_request()
    assert pending is not None
    channel.post_response(
        InteractionResponse(
            request_id=pending.request_id,
            decision="approve",
            user_message="Approved",
        )
    )
    thread.join(timeout=2.0)
    assert harness.state.status == AgentStatus.COMPLETED

    # Verify stop path cancels pending interactions and exits as completed.
    second_harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        ToolSandbox(config.agents.sandbox, interaction_channel=channel),
        agent_name="wait-stop",
        interaction_channel=channel,
    )
    second_thread = second_harness.start_background(
        AgentContext(messages=[{"role": "user", "content": "test"}])
    )
    assert _wait_until(channel.has_pending)
    second_harness.stop()
    second_thread.join(timeout=2.0)
    assert channel.has_pending() is False
    assert second_harness.state.status == AgentStatus.COMPLETED

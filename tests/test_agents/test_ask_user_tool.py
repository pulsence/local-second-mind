from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsm.agents.base import AgentStatus
from lsm.agents.harness import AgentHarness
from lsm.agents.interaction import InteractionChannel, InteractionResponse
from lsm.agents.models import AgentContext
from lsm.agents.tools.ask_user import AskUserTool
from lsm.agents.tools.base import ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw
from lsm.config.models import InteractionConfig


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "path": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "path": str(tmp_path / ".chroma"),
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
                "require_user_permission": {},
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


def test_ask_user_tool_requires_bound_harness() -> None:
    tool = AskUserTool()
    with pytest.raises(RuntimeError, match="not bound"):
        tool.execute({"prompt": "Need detail?"})


def test_ask_user_tool_posts_clarification_request_and_returns_reply() -> None:
    class StubHarness:
        def __init__(self) -> None:
            self.requests = []

        def request_interaction(self, request):
            self.requests.append(request)
            return InteractionResponse(
                request_id=request.request_id,
                decision="reply",
                user_message="Use markdown bullets.",
            )

    harness = StubHarness()
    tool = AskUserTool()
    tool.bind_harness(harness)

    result = tool.execute(
        {
            "prompt": "How should I format the answer?",
            "context": "Output format was ambiguous.",
        }
    )
    assert result == "Use markdown bullets."
    assert len(harness.requests) == 1
    request = harness.requests[0]
    assert request.request_type == "clarification"
    assert request.prompt == "How should I format the answer?"
    assert request.reason == "Output format was ambiguous."
    args = json.loads(request.args_summary or "{}")
    assert args["prompt"] == "How should I format the answer?"
    assert args["context"] == "Output format was ambiguous."


def test_ask_user_tool_denial_raises_permission_error() -> None:
    class StubHarness:
        def request_interaction(self, request):
            return InteractionResponse(
                request_id=request.request_id,
                decision="deny",
                user_message="No clarification allowed.",
            )

    tool = AskUserTool()
    tool.bind_harness(StubHarness())
    with pytest.raises(PermissionError, match="No clarification allowed"):
        tool.execute({"prompt": "Can I ask a follow-up?"})


def test_ask_user_tool_auto_continue_skips_prompt() -> None:
    class StubHarness:
        def __init__(self) -> None:
            self.agent_config = SimpleNamespace(
                interaction=InteractionConfig(auto_continue=True)
            )

        def request_interaction(self, request):
            raise AssertionError("auto-continue should not prompt for interaction")

    tool = AskUserTool()
    tool.bind_harness(StubHarness())
    assert tool.execute({"prompt": "Proceed?"}) == "Continue with your best judgment."


def test_harness_handles_ask_user_clarification_and_waiting_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self._responses = [
                json.dumps(
                    {
                        "response": "Need clarification",
                        "action": "ask_user",
                        "action_arguments": {
                            "prompt": "What output format should I use?",
                            "context": "Need explicit formatting preference.",
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
    channel = InteractionChannel(timeout_seconds=1, timeout_action="deny")
    registry = ToolRegistry()
    registry.register(AskUserTool())
    sandbox = ToolSandbox(config.agents.sandbox, interaction_channel=channel)
    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        sandbox,
        agent_name="ask-user",
        tool_allowlist={"read_file"},
        interaction_channel=channel,
    )
    thread = harness.start_background(
        AgentContext(messages=[{"role": "user", "content": "Do the task"}])
    )
    assert _wait_until(channel.has_pending)
    assert harness.state.status == AgentStatus.WAITING_USER

    pending = channel.get_pending_request()
    assert pending is not None
    assert pending.request_type == "clarification"
    assert pending.prompt == "What output format should I use?"
    assert pending.reason == "Need explicit formatting preference."

    channel.post_response(
        InteractionResponse(
            request_id=pending.request_id,
            decision="reply",
            user_message="Use concise bullet points.",
        )
    )
    thread.join(timeout=2.0)
    assert harness.state.status == AgentStatus.COMPLETED
    assert harness.context is not None
    assert [item["name"] for item in harness.context.tool_definitions] == ["ask_user"]
    assert any(
        entry.actor == "tool"
        and entry.action == "ask_user"
        and "concise bullet points" in entry.content
        for entry in harness.state.log_entries
    )

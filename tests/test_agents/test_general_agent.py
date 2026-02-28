from __future__ import annotations

import inspect
import json
from pathlib import Path

from lsm.agents.base import AgentStatus
from lsm.agents.models import AgentContext
from lsm.agents.phase import PhaseResult
from lsm.agents.productivity.general import GeneralAgent
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: dict) -> str:
        return str(args.get("text", ""))


class RestrictedTool(BaseTool):
    name = "restricted"
    description = "Requires permission."
    requires_permission = True
    input_schema = {
        "type": "object",
        "properties": {"payload": {"type": "string"}},
        "required": ["payload"],
    }

    def execute(self, args: dict) -> str:
        return str(args.get("payload", ""))


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
            "max_tokens_budget": 8000,
            "max_iterations": 3,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def test_general_agent_tool_allowlist_defaults() -> None:
    assert "read_file" in GeneralAgent.tool_allowlist
    assert "write_file" in GeneralAgent.tool_allowlist
    assert "bash" not in GeneralAgent.tool_allowlist
    assert "powershell" not in GeneralAgent.tool_allowlist


def test_general_agent_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.productivity.general as general_module

    source = inspect.getsource(general_module)
    assert "AgentHarness(" not in source, (
        "GeneralAgent must not directly instantiate AgentHarness; use _run_phase() instead"
    )


def test_general_agent_passes_system_prompt(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {"system_prompt": None}

    def fake_run_phase(self_agent, system_prompt: str = "", user_message: str = "", **kwargs):
        captured["system_prompt"] = system_prompt
        return PhaseResult(final_text="", tool_calls=[], stop_reason="done")

    monkeypatch.setattr("lsm.agents.base.BaseAgent._run_phase", fake_run_phase)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgent(config.llm, registry, sandbox, config.agents)

    agent.run(AgentContext(messages=[{"role": "user", "content": "Summarize"}]))

    assert captured["system_prompt"] == agent._system_prompt()


def test_general_agent_run_phase_called_with_topic_as_user_message(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict = {"user_message": None}

    def fake_run_phase(self_agent, system_prompt: str = "", user_message: str = "", **kwargs):
        captured["user_message"] = user_message
        return PhaseResult(final_text="Done", tool_calls=[], stop_reason="done")

    monkeypatch.setattr("lsm.agents.base.BaseAgent._run_phase", fake_run_phase)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgent(config.llm, registry, sandbox, config.agents)

    agent.run(AgentContext(messages=[{"role": "user", "content": "Plan my tasks"}]))

    assert captured["user_message"] == "Plan my tasks"


def test_general_agent_writes_summary(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self._responses = [
                json.dumps(
                    {
                        "response": "Calling echo",
                        "action": "echo",
                        "action_arguments": {"text": "hello"},
                    }
                ),
                json.dumps(
                    {
                        "response": "Done",
                        "action": "DONE",
                        "action_arguments": {},
                    }
                ),
            ]

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            return self._responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgent(config.llm, registry, sandbox, config.agents)
    agent.tool_allowlist = set(agent.tool_allowlist or set()) | {"echo"}

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Plan tasks"}]))
    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.summary_path.exists()

    summary_text = agent.last_result.summary_path.read_text(encoding="utf-8")
    assert "## Artifacts" in summary_text
    assert "## Tools Used" in summary_text


def test_general_agent_respects_iteration_guardrail(monkeypatch, tmp_path: Path) -> None:
    call_count = {"count": 0}

    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            call_count["count"] += 1
            return json.dumps(
                {
                    "response": "Echo once",
                    "action": "echo",
                    "action_arguments": {"text": "iteration"},
                }
            )

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    raw = _base_raw(tmp_path)
    raw["agents"]["max_iterations"] = 1  # global limit applied to this agent
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgent(config.llm, registry, sandbox, config.agents)
    agent.tool_allowlist = set(agent.tool_allowlist or set()) | {"echo"}

    agent.run(AgentContext(messages=[{"role": "user", "content": "Iterate once"}]))
    assert agent.last_result is not None
    # max_iterations=1 means only one LLM call before the loop ends
    assert call_count["count"] == 1


def test_general_agent_records_permission_denial_and_completes(
    monkeypatch, tmp_path: Path
) -> None:
    """With run_bounded(), permission denial continues the loop; agent completes gracefully."""

    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self._responses = [
                json.dumps(
                    {
                        "response": "Try restricted tool",
                        "action": "restricted",
                        "action_arguments": {"payload": "secret"},
                    }
                ),
                json.dumps(
                    {
                        "response": "Done.",
                        "action": "DONE",
                        "action_arguments": {},
                    }
                ),
            ]

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            return self._responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(RestrictedTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgent(config.llm, registry, sandbox, config.agents)
    # "restricted" is NOT in tool_allowlist, so it will be denied by the harness

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Restricted"}]))
    assert state.status.value == "completed"
    assert agent.last_result is not None
    summary_text = agent.last_result.summary_path.read_text(encoding="utf-8")
    assert "Status: completed" in summary_text

from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.base import AgentState, AgentStatus
from lsm.agents.models import AgentContext
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


def test_general_agent_passes_system_prompt(monkeypatch, tmp_path: Path) -> None:
    captured = {"prompt": None}

    class HarnessStub:
        def __init__(self, *args, **kwargs):
            captured["prompt"] = kwargs.get("system_prompt")
            self.state = AgentState()

        def run(self, context: AgentContext) -> AgentState:
            _ = context
            self.state.set_status(AgentStatus.COMPLETED)
            return self.state

        def get_state_path(self) -> None:
            return None

    monkeypatch.setattr("lsm.agents.harness.AgentHarness", HarnessStub)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgent(config.llm, registry, sandbox, config.agents)
    agent.tool_allowlist = set(agent.tool_allowlist or set()) | {"echo"}
    agent.run(AgentContext(messages=[{"role": "user", "content": "Summarize"}]))

    assert captured["prompt"] == agent._system_prompt()


def test_general_agent_writes_summary_and_run_summary(monkeypatch, tmp_path: Path) -> None:
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
    assert agent.last_result.run_summary_path is not None
    assert agent.last_result.run_summary_path.exists()

    summary_text = agent.last_result.summary_path.read_text(encoding="utf-8")
    assert "## Artifacts" in summary_text
    assert str(agent.last_result.run_summary_path) in summary_text


def test_general_agent_respects_iteration_guardrail(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self._responses = [
                json.dumps(
                    {
                        "response": "Echo once",
                        "action": "echo",
                        "action_arguments": {"text": "iteration"},
                    }
                ),
                json.dumps(
                    {
                        "response": "Echo again",
                        "action": "echo",
                        "action_arguments": {"text": "extra"},
                    }
                ),
            ]

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            return self._responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    raw = _base_raw(tmp_path)
    raw["agents"]["agent_configs"] = {"general": {"max_iterations": 1}}
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgent(config.llm, registry, sandbox, config.agents)
    agent.tool_allowlist = set(agent.tool_allowlist or set()) | {"restricted"}

    agent.run(AgentContext(messages=[{"role": "user", "content": "Iterate once"}]))
    assert agent.last_result is not None
    run_summary_path = agent.last_result.run_summary_path
    assert run_summary_path is not None

    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert summary["token_usage"]["iterations"] == 1


def test_general_agent_marks_failure_on_permission_denial(monkeypatch, tmp_path: Path) -> None:
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

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Restricted"}]))
    assert state.status.value == "failed"
    assert agent.last_result is not None
    summary_text = agent.last_result.summary_path.read_text(encoding="utf-8")
    assert "Status: failed" in summary_text

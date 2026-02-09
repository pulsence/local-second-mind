from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: dict) -> str:
        return str(args["text"])


class BigOutputTool(BaseTool):
    name = "big_output"
    description = "Large output."
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> str:
        _ = args
        return "x" * (AgentHarness._MAX_TOOL_OUTPUT_CHARS + 50)


def _base_raw(tmp_path: Path, *, max_iterations: int = 5, max_tokens_budget: int = 5000) -> dict:
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
            "max_tokens_budget": max_tokens_budget,
            "max_iterations": max_iterations,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "require_permission_by_risk": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def test_security_resources_iteration_cap_enforced(monkeypatch, tmp_path: Path) -> None:
    class LoopProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return json.dumps(
                {
                    "response": "loop",
                    "action": "echo",
                    "action_arguments": {"text": "ok"},
                }
            )

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: LoopProvider())
    config = build_config_from_raw(
        _base_raw(tmp_path, max_iterations=3, max_tokens_budget=5000),
        tmp_path / "config.json",
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    harness = AgentHarness(config.agents, registry, config.llm, ToolSandbox(config.agents.sandbox))
    state = harness.run(AgentContext(messages=[{"role": "user", "content": "go"}]))

    assert state.status.value == "completed"
    assert harness.context is not None
    assert harness.context.budget_tracking["iterations"] == 3
    assert len([entry for entry in state.log_entries if entry.actor == "tool"]) == 3


def test_security_resources_token_budget_exhaustion_stops_execution(monkeypatch, tmp_path: Path) -> None:
    class BudgetProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return json.dumps(
                {
                    "response": "run",
                    "action": "echo",
                    "action_arguments": {"text": "x" * 2000},
                }
            )

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: BudgetProvider())
    config = build_config_from_raw(
        _base_raw(tmp_path, max_iterations=10, max_tokens_budget=50),
        tmp_path / "config.json",
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    harness = AgentHarness(config.agents, registry, config.llm, ToolSandbox(config.agents.sandbox))
    state = harness.run(AgentContext(messages=[{"role": "user", "content": "budget"}]))

    assert state.status.value == "completed"
    assert any("budget exhaustion" in entry.content for entry in state.log_entries)


def test_security_resources_budget_tracking_accuracy(monkeypatch, tmp_path: Path) -> None:
    payload = json.dumps({"response": "ok", "action": "DONE", "action_arguments": {}})

    class DoneProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return payload

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: DoneProvider())
    config = build_config_from_raw(
        _base_raw(tmp_path, max_iterations=3, max_tokens_budget=5000),
        tmp_path / "config.json",
    )
    harness = AgentHarness(
        config.agents,
        ToolRegistry(),
        config.llm,
        ToolSandbox(config.agents.sandbox),
    )
    harness.run(AgentContext(messages=[{"role": "user", "content": "done"}]))
    assert harness.context is not None
    assert harness.context.budget_tracking["tokens_used"] == max(1, len(payload) // 4)


def test_security_resources_large_tool_output_is_truncated(monkeypatch, tmp_path: Path) -> None:
    class Provider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.responses = [
                json.dumps({"response": "step", "action": "big_output", "action_arguments": {}}),
                json.dumps({"response": "done", "action": "DONE", "action_arguments": {}}),
            ]

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return self.responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: Provider())
    config = build_config_from_raw(
        _base_raw(tmp_path, max_iterations=5, max_tokens_budget=100000),
        tmp_path / "config.json",
    )
    registry = ToolRegistry()
    registry.register(BigOutputTool())
    harness = AgentHarness(config.agents, registry, config.llm, ToolSandbox(config.agents.sandbox))
    state = harness.run(AgentContext(messages=[{"role": "user", "content": "truncate"}]))

    tool_logs = [entry for entry in state.log_entries if entry.actor == "tool"]
    assert len(tool_logs) == 1
    assert "[TRUNCATED 50 chars]" in tool_logs[0].content
    assert len(tool_logs[0].content) <= AgentHarness._MAX_TOOL_OUTPUT_CHARS + 64


def test_security_resources_graceful_stop_on_mid_iteration_budget_hit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class Provider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            self.calls += 1
            return json.dumps({"response": "step", "action": "big_output", "action_arguments": {}})

    provider = Provider()
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    config = build_config_from_raw(
        _base_raw(tmp_path, max_iterations=5, max_tokens_budget=20),
        tmp_path / "config.json",
    )
    registry = ToolRegistry()
    registry.register(BigOutputTool())
    harness = AgentHarness(config.agents, registry, config.llm, ToolSandbox(config.agents.sandbox))
    state = harness.run(AgentContext(messages=[{"role": "user", "content": "budget-hit"}]))

    assert state.status.value == "completed"
    assert provider.calls == 1
    assert any("budget exhaustion" in entry.content for entry in state.log_entries)

from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class AllowedTool(BaseTool):
    name = "allowed"
    description = "Allowed tool."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, args: dict) -> str:
        self.calls += 1
        return str(args.get("text", ""))


class BlockedTool(BaseTool):
    name = "blocked"
    description = "Blocked tool."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, args: dict) -> str:
        self.calls += 1
        return str(args.get("text", ""))


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
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def test_harness_allowlist_filters_llm_tool_context_and_blocks_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.last_tools_context = ""

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, mode, kwargs
            self.last_tools_context = str(context)
            return json.dumps(
                {
                    "response": "Try blocked tool",
                    "action": "blocked",
                    "action_arguments": {"text": "x"},
                }
            )

    provider = FakeProvider()
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    allowed = AllowedTool()
    blocked = BlockedTool()
    registry.register(allowed)
    registry.register(blocked)

    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        ToolSandbox(config.agents.sandbox),
        agent_name="allowlist",
        tool_allowlist={"allowed"},
    )
    state = harness.run(AgentContext(messages=[{"role": "user", "content": "do work"}]))

    assert state.status.value == "failed"
    assert allowed.calls == 0
    assert blocked.calls == 0
    assert any("not allowed" in entry.content for entry in state.log_entries)

    context_payload = json.loads(provider.last_tools_context)
    assert isinstance(context_payload, list)
    assert [item["name"] for item in context_payload] == ["allowed"]

    state_path = harness.get_state_path()
    assert state_path is not None
    summary_path = state_path.parent / "run_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["run_outcome"] == "failed"
    assert summary["tools_used"] == {}
    assert summary["approvals_denials"]["approvals"] == 0
    assert summary["approvals_denials"]["denials"] == 1
    assert summary["approvals_denials"]["by_tool"]["blocked"]["denials"] == 1


def test_harness_creates_workspace_and_persists_context_path(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return json.dumps({"response": "Done", "action": "DONE", "action_arguments": {}})

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(AllowedTool())
    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        ToolSandbox(config.agents.sandbox),
        agent_name="workspace",
    )

    state = harness.run(AgentContext(messages=[{"role": "user", "content": "done"}]))
    assert state.status.value == "completed"

    workspace = harness.get_workspace_path()
    assert workspace is not None
    assert workspace.exists()
    assert workspace.name == "workspace"
    assert workspace.parent.name.startswith("workspace_")

    assert harness.context is not None
    assert harness.context.run_workspace == str(workspace)

    state_path = harness.get_state_path()
    assert state_path is not None
    assert state_path.exists()
    assert state_path.parent == workspace.parent
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["context"]["run_workspace"] == str(workspace)

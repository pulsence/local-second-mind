from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.assistants.assistant import AssistantAgent
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class MemoryPutStubTool(BaseTool):
    name = "memory_put"
    description = "Stub memory tool."
    requires_permission = False
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "object"},
            "type": {"type": "string"},
            "scope": {"type": "string"},
            "tags": {"type": "array"},
            "rationale": {"type": "string"},
        },
        "required": ["key", "value"],
    }

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, args: dict) -> str:
        self.calls.append(dict(args))
        return json.dumps({"status": "pending", "key": args.get("key")})


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


def _write_run_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_assistant_agent_summarizes_runs(tmp_path: Path) -> None:
    agents_root = tmp_path / "Agents"
    summary_a = agents_root / "general" / "logs" / "general_20240101" / "run_summary.json"
    summary_b = agents_root / "writing" / "logs" / "writing_20240102" / "run_summary.json"

    _write_run_summary(
        summary_a,
        {
            "agent_name": "general",
            "topic": "Alpha",
            "status": "completed",
            "tools_used": {"read_file": 2},
            "approvals_denials": {"denials": 0},
            "artifacts_created": ["alpha.txt"],
            "token_usage": {"iterations": 2},
        },
    )
    _write_run_summary(
        summary_b,
        {
            "agent_name": "writing",
            "topic": "Beta",
            "status": "failed",
            "tools_used": {"write_file": 1},
            "approvals_denials": {"denials": 1},
            "artifacts_created": ["beta.md"],
            "token_usage": {"iterations": 1},
        },
    )

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    memory_tool = MemoryPutStubTool()
    registry.register(memory_tool)
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = AssistantAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Summarize runs"}]))
    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.summary_path.exists()
    assert agent.last_result.summary_json_path.exists()

    summary_text = agent.last_result.summary_path.read_text(encoding="utf-8")
    assert "Total runs: 2" in summary_text
    assert "failed" in summary_text.lower()

    summary_payload = json.loads(
        agent.last_result.summary_json_path.read_text(encoding="utf-8")
    )
    assert summary_payload["run_count"] == 2
    assert memory_tool.calls

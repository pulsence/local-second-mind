from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext
from lsm.agents.tools import AwaitAgentTool, CollectArtifactsTool, SpawnAgentTool, ToolSandbox
from lsm.agents.tools.base import ToolRegistry
from lsm.config.loader import build_config_from_raw


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
            "max_iterations": 4,
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
            "agent_configs": {},
        },
    }


def test_meta_system_tools_spawn_await_and_collect(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            return json.dumps(
                {
                    "response": "Done",
                    "action": "DONE",
                    "action_arguments": {},
                }
            )

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(SpawnAgentTool())
    registry.register(AwaitAgentTool())
    registry.register(CollectArtifactsTool())

    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        ToolSandbox(config.agents.sandbox),
        agent_name="meta",
    )
    harness.run(AgentContext(messages=[{"role": "user", "content": "plan only"}]))

    spawn_payload = json.loads(
        registry.lookup("spawn_agent").execute(
            {
                "name": "research",
                "params": {"topic": "Sub-agent topic"},
            }
        )
    )
    assert spawn_payload["status"] == "running"
    assert spawn_payload["agent_name"] == "research"
    agent_id = spawn_payload["agent_id"]
    assert agent_id.startswith("research_")

    await_payload = json.loads(
        registry.lookup("await_agent").execute(
            {
                "agent_id": agent_id,
                "timeout_seconds": 2,
            }
        )
    )
    assert await_payload["done"] is True
    assert await_payload["status"] == "completed"
    assert await_payload["state_path"]

    collect_payload = json.loads(
        registry.lookup("collect_artifacts").execute(
            {
                "agent_id": agent_id,
                "pattern": "*run_summary.json",
            }
        )
    )
    assert collect_payload["agent_id"] == agent_id
    assert collect_payload["pattern"] == "*run_summary.json"
    assert any(path.endswith("run_summary.json") for path in collect_payload["artifacts"])


def test_spawn_tool_requires_bound_harness() -> None:
    tool = SpawnAgentTool()
    try:
        tool.execute({"name": "research"})
        assert False, "Expected RuntimeError for unbound spawn tool"
    except RuntimeError as exc:
        assert "not bound" in str(exc)

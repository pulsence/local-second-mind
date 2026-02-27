from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.base import AgentStatus, BaseAgent
from lsm.agents.factory import AgentRegistry
from lsm.agents.meta import MetaAgent
from lsm.agents.models import AgentContext
from lsm.agents.phase import PhaseResult
from lsm.agents.tools.base import ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


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
            "max_tokens_budget": 20000,
            "max_iterations": 6,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {"meta": {"max_iterations": 4}},
        },
    }


def _patch_meta_execution(monkeypatch) -> None:
    def _fake_run_phase(self, *args, **kwargs):
        _ = args, kwargs
        return PhaseResult(final_text="# Final Result\n\nConsolidated sub-agent output.\n", tool_calls=[], stop_reason="end_turn")

    monkeypatch.setattr(MetaAgent, "_run_phase", _fake_run_phase)

    class FakeChildAgent(BaseAgent):
        def __init__(self, name: str, agent_config) -> None:
            super().__init__(name=name, description=f"Fake {name} agent")
            self.agent_config = agent_config

        def run(self, initial_context: AgentContext):
            output_dir = Path(str(initial_context.run_workspace or self.agent_config.agents_folder))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.name}_output.md"
            output_path.write_text("output", encoding="utf-8")
            self.state.add_artifact(str(output_path))
            self.state.set_status(AgentStatus.COMPLETED)
            return self.state

    def _fake_create_agent(name, llm_registry, tool_registry, sandbox, agent_config):
        _ = llm_registry, tool_registry, sandbox
        return FakeChildAgent(str(name), agent_config)

    monkeypatch.setattr("lsm.agents.factory.create_agent", _fake_create_agent)


def test_general_meta_agent_selects_assistant_tasks(monkeypatch, tmp_path: Path) -> None:
    _patch_meta_execution(monkeypatch)
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(
            messages=[
                {
                    "role": "user",
                    "content": "Summarize my email and calendar for today",
                }
            ]
        )
    )
    assert state.status.value == "completed"
    assert agent.last_task_graph is not None
    names = [task.agent_name for task in agent.last_task_graph.tasks]
    assert "email_assistant" in names
    assert "calendar_assistant" in names
    assert "synthesis" in names

    assert agent.last_result is not None
    assert agent.last_result.final_result_path is not None
    assert agent.last_result.final_result_path.exists()


def test_general_meta_agent_tool_allowlist_metadata() -> None:
    entry = AgentRegistry().get_entry("meta")
    assert entry is not None
    assert entry.tool_allowlist is not None
    assert "spawn_agent" in entry.tool_allowlist
    assert "collect_artifacts" in entry.tool_allowlist


def test_general_meta_agent_system_prompt() -> None:
    assert MetaAgent.system_prompt
    prompt_lower = MetaAgent.system_prompt.lower()
    assert "meta" in prompt_lower
    assert "spawn_agent" in prompt_lower
    assert "final_result" in prompt_lower

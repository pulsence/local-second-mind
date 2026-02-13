from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.factory import AgentRegistry, create_agent
from lsm.agents.meta import MetaAgent
from lsm.agents.models import AgentContext
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


def test_meta_agent_builds_default_dependency_order(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(
            messages=[
                {
                    "role": "user",
                    "content": "Draft a theology brief on Aquinas and the Eucharist",
                }
            ]
        )
    )
    assert state.status.value == "completed"
    assert agent.last_task_graph is not None
    assert [task.agent_name for task in agent.last_task_graph.topological_sort()] == [
        "research",
        "writing",
        "synthesis",
    ]
    assert agent.last_execution_order == ["research_1", "writing_1", "synthesis_1"]
    assert agent.last_task_graph.is_done() is True
    assert any("Planned task graph" in entry.content for entry in state.log_entries)


def test_meta_agent_accepts_structured_goal_json(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(config.llm, registry, sandbox, config.agents)

    structured_goal = json.dumps(
        {
            "goal": "Weekly literature digest",
            "tasks": [
                {
                    "id": "research_a",
                    "agent_name": "research",
                    "params": {"topic": "weekly digest"},
                    "expected_artifacts": ["research_outline.md"],
                },
                {
                    "id": "synthesis_a",
                    "agent_name": "synthesis",
                    "params": {"topic": "weekly digest"},
                    "depends_on": ["research_a"],
                    "expected_artifacts": ["synthesis.md"],
                },
            ],
        }
    )
    state = agent.run(AgentContext(messages=[{"role": "user", "content": structured_goal}]))

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.goal == "Weekly literature digest"
    assert [task.id for task in agent.last_result.task_graph.tasks] == [
        "research_a",
        "synthesis_a",
    ]
    assert agent.last_execution_order == ["research_a", "synthesis_a"]


def test_meta_agent_uses_curator_pipeline_for_curation_goals(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(messages=[{"role": "user", "content": "Curate and deduplicate notes"}])
    )

    assert state.status.value == "completed"
    assert agent.last_task_graph is not None
    assert len(agent.last_task_graph.tasks) == 1
    assert agent.last_task_graph.tasks[0].agent_name == "curator"
    assert agent.last_execution_order == ["curator_1"]


def test_agent_factory_registers_meta_agent(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)

    built = create_agent("meta", config.llm, registry, sandbox, config.agents)
    assert isinstance(built, MetaAgent)
    assert "meta" in AgentRegistry().list_agents()


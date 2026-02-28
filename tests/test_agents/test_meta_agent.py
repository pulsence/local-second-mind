from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from lsm.agents.base import AgentStatus, BaseAgent
from lsm.agents.factory import AgentRegistry, create_agent
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


def _make_fake_run_phase(final_text: str = "# Final Result\n\nConsolidated sub-agent output.\n"):
    def _fake_run_phase(self, *args, **kwargs):
        _ = args, kwargs
        return PhaseResult(final_text=final_text, tool_calls=[], stop_reason="end_turn")
    return _fake_run_phase


def _patch_meta_execution(monkeypatch) -> None:
    monkeypatch.setattr(MetaAgent, "_run_phase", _make_fake_run_phase())

    class FakeChildAgent(BaseAgent):
        def __init__(self, name: str, agent_config) -> None:
            super().__init__(name=name, description=f"Fake {name} agent")
            self.agent_config = agent_config

        def run(self, initial_context: AgentContext):
            run_workspace = str(initial_context.run_workspace or "").strip()
            output_dir = (
                Path(run_workspace)
                if run_workspace
                else Path(self.agent_config.agents_folder)
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.name}_output.md"
            output_path.write_text(
                f"# {self.name.title()} Output\n\n"
                f"Topic: {initial_context.messages[-1]['content']}\n",
                encoding="utf-8",
            )
            self.state.add_artifact(str(output_path))
            self._log(f"Created fake artifact at {output_path}")
            self.state.set_status(AgentStatus.COMPLETED)
            return self.state

    def _fake_create_agent(
        name,
        llm_registry,
        tool_registry,
        sandbox,
        agent_config,
        lsm_config=None,
    ):
        _ = llm_registry, tool_registry, sandbox, lsm_config
        return FakeChildAgent(str(name), agent_config)

    monkeypatch.setattr("lsm.agents.factory.create_agent", _fake_create_agent)


def test_meta_agent_builds_default_dependency_order(monkeypatch, tmp_path: Path) -> None:
    _patch_meta_execution(monkeypatch)
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )

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


def test_meta_agent_accepts_structured_goal_json(monkeypatch, tmp_path: Path) -> None:
    _patch_meta_execution(monkeypatch)
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )

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


def test_meta_agent_uses_curator_pipeline_for_curation_goals(monkeypatch, tmp_path: Path) -> None:
    _patch_meta_execution(monkeypatch)
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )

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
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)

    built = create_agent(
        "meta",
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )
    assert isinstance(built, MetaAgent)
    assert "meta" in AgentRegistry().list_agents()


def test_meta_agent_has_no_tokens_used_attribute(monkeypatch, tmp_path: Path) -> None:
    _patch_meta_execution(monkeypatch)
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = MetaAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )

    agent.run(
        AgentContext(
            messages=[{"role": "user", "content": "Curate and deduplicate notes"}]
        )
    )

    with pytest.raises(AttributeError):
        getattr(agent, "_tokens_used")


# ---------------------------------------------------------------------------
# Source-inspection tests
# ---------------------------------------------------------------------------


def test_meta_agent_does_not_call_create_provider() -> None:
    import lsm.agents.meta.meta as meta_module

    source = inspect.getsource(meta_module)
    assert "create_provider" not in source


def test_meta_agent_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.meta.meta as meta_module

    source = inspect.getsource(meta_module)
    assert "AgentHarness(" not in source

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from lsm.agents.base import AgentStatus, BaseAgent
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
            "max_concurrent": 2,
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


def _patch_meta_dependencies(monkeypatch, concurrency_state, sleep_s: float = 0.05) -> None:
    def _fake_run_phase(self, *args, **kwargs):
        _ = args, kwargs
        return PhaseResult(final_text="# Final Result\n\nConsolidated sub-agent output.\n", tool_calls=[], stop_reason="end_turn")

    monkeypatch.setattr(MetaAgent, "_run_phase", _fake_run_phase)

    class FakeChildAgent(BaseAgent):
        def __init__(self, name: str, agent_config) -> None:
            super().__init__(name=name, description=f"Fake {name} agent")
            self.agent_config = agent_config

        def run(self, initial_context: AgentContext):
            with concurrency_state["lock"]:
                concurrency_state["current"] += 1
                concurrency_state["max"] = max(
                    concurrency_state["max"], concurrency_state["current"]
                )
            time.sleep(sleep_s)
            run_workspace = str(initial_context.run_workspace or "").strip()
            output_dir = Path(run_workspace) if run_workspace else Path(self.agent_config.agents_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.name}_artifact.md"
            output_path.write_text(f"# {self.name} artifact\n", encoding="utf-8")
            self.state.add_artifact(str(output_path))
            with concurrency_state["lock"]:
                concurrency_state["current"] -= 1
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


def test_meta_agent_runs_parallel_groups_concurrently(monkeypatch, tmp_path: Path) -> None:
    concurrency_state = {"current": 0, "max": 0, "lock": threading.Lock()}
    _patch_meta_dependencies(monkeypatch, concurrency_state, sleep_s=0.1)

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
            "goal": "Parallel run",
            "tasks": [
                {"id": "t1", "agent_name": "research", "parallel_group": "g1"},
                {"id": "t2", "agent_name": "writing", "parallel_group": "g1"},
                {
                    "id": "t3",
                    "agent_name": "synthesis",
                    "parallel_group": "g2",
                    "depends_on": ["t1", "t2"],
                },
            ],
        }
    )

    state = agent.run(AgentContext(messages=[{"role": "user", "content": structured_goal}]))
    assert state.status.value == "completed"
    assert concurrency_state["max"] >= 2


def test_meta_agent_respects_max_concurrent(monkeypatch, tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"]["max_concurrent"] = 1
    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.agents is not None

    concurrency_state = {"current": 0, "max": 0, "lock": threading.Lock()}
    _patch_meta_dependencies(monkeypatch, concurrency_state, sleep_s=0.05)

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
            "goal": "Serialized run",
            "tasks": [
                {"id": "t1", "agent_name": "research", "parallel_group": "g1"},
                {"id": "t2", "agent_name": "writing", "parallel_group": "g1"},
            ],
        }
    )
    agent.run(AgentContext(messages=[{"role": "user", "content": structured_goal}]))
    assert concurrency_state["max"] == 1


def test_meta_agent_enforces_sandbox_monotonicity(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"]["sandbox"]["allowed_write_paths"] = [str(tmp_path / "allowed")]
    config = build_config_from_raw(raw, tmp_path / "config.json")
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

    run_root = tmp_path / "outside"
    shared_workspace = tmp_path / "outside" / "workspace"
    sub_dir = tmp_path / "outside" / "sub"
    run_root.mkdir(parents=True, exist_ok=True)
    shared_workspace.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent._build_child_sandbox(sub_dir=sub_dir, shared_workspace=shared_workspace)
        assert False, "Expected PermissionError for sandbox monotonicity"
    except PermissionError:
        assert True


def test_meta_agent_merges_artifacts_deterministically(monkeypatch, tmp_path: Path) -> None:
    concurrency_state = {"current": 0, "max": 0, "lock": threading.Lock()}
    _patch_meta_dependencies(monkeypatch, concurrency_state, sleep_s=0.01)

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
            "goal": "Order artifacts",
            "tasks": [
                {"id": "t1", "agent_name": "writing", "parallel_group": "g1"},
                {"id": "t2", "agent_name": "research", "parallel_group": "g1"},
            ],
        }
    )
    agent.run(AgentContext(messages=[{"role": "user", "content": structured_goal}]))

    artifact_paths = [path for path in agent.state.artifacts if path.endswith("_artifact.md")]
    assert artifact_paths
    # Deterministic order by agent name then task order -> research before writing.
    assert artifact_paths[0].endswith("research_artifact.md")
    assert artifact_paths[1].endswith("writing_artifact.md")

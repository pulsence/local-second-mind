from __future__ import annotations

from pathlib import Path

import pytest

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


def _patch_meta_execution(monkeypatch) -> list[ToolSandbox]:
    captured_sandboxes: list[ToolSandbox] = []

    def _fake_run_phase(self, *args, **kwargs):
        _ = args, kwargs
        return PhaseResult(final_text="# Final Result\n\nSynthetic final result.\n", tool_calls=[], stop_reason="end_turn")

    monkeypatch.setattr(MetaAgent, "_run_phase", _fake_run_phase)

    class FakeChildAgent(BaseAgent):
        def __init__(self, name: str, agent_config) -> None:
            super().__init__(name=name, description=f"Fake {name} agent")
            self.agent_config = agent_config

        def run(self, initial_context: AgentContext):
            out_dir = Path(
                str(initial_context.run_workspace or self.agent_config.agents_folder)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"{self.name}_artifact.md"
            output.write_text(
                f"# Artifact\n\nAgent: {self.name}\n",
                encoding="utf-8",
            )
            self.state.add_artifact(str(output))
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
        _ = llm_registry, tool_registry, lsm_config
        captured_sandboxes.append(sandbox)
        return FakeChildAgent(str(name), agent_config)

    monkeypatch.setattr("lsm.agents.factory.create_agent", _fake_create_agent)
    return captured_sandboxes


def test_meta_agent_creates_shared_workspace_layout(monkeypatch, tmp_path: Path) -> None:
    captured_sandboxes = _patch_meta_execution(monkeypatch)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    agent = MetaAgent(
        config.llm,
        ToolRegistry(),
        ToolSandbox(config.agents.sandbox),
        config.agents,
        lsm_config=config,
    )

    state = agent.run(
        AgentContext(
            messages=[
                {
                    "role": "user",
                    "content": "Create a multi-step theology brief",
                }
            ]
        )
    )

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.run_root is not None
    run_root = agent.last_result.run_root
    shared_workspace = run_root / "workspace"

    assert run_root.exists()
    assert run_root.name.startswith("meta_")
    assert shared_workspace.exists()
    assert (run_root / "sub_agents").exists()
    assert (run_root / "sub_agents" / "research_001").exists()
    assert (run_root / "sub_agents" / "writing_001").exists()
    assert (run_root / "sub_agents" / "synthesis_001").exists()

    final_result = run_root / "final_result.md"
    meta_log = run_root / "meta_log.md"
    assert final_result.exists()
    assert meta_log.exists()

    sub_artifacts = list((run_root / "sub_agents").rglob("*_artifact.md"))
    assert len(sub_artifacts) >= 3

    expected_dirs = [
        run_root / "sub_agents" / "research_001",
        run_root / "sub_agents" / "writing_001",
        run_root / "sub_agents" / "synthesis_001",
    ]
    assert len(captured_sandboxes) == len(expected_dirs)
    for sandbox, sub_dir in zip(captured_sandboxes, expected_dirs):
        sandbox.check_read_path(shared_workspace)
        sandbox.check_read_path(sub_dir)
        sandbox.check_write_path(sub_dir)
        with pytest.raises(PermissionError):
            sandbox.check_write_path(shared_workspace)

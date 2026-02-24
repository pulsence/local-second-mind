from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.base import AgentStatus, BaseAgent
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


class _FakeChildAgent(BaseAgent):
    def __init__(self, name: str, agent_config) -> None:
        super().__init__(name=name, description=f"Fake {name} agent")
        self.agent_config = agent_config

    def run(self, initial_context: AgentContext):
        out_dir = Path(str(initial_context.run_workspace or self.agent_config.agents_folder))
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{self.name}.md"
        output.write_text(
            f"# {self.name.title()}\n\nTopic:\n{initial_context.messages[-1]['content']}\n",
            encoding="utf-8",
        )
        self.state.add_artifact(str(output))
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state


def _patch_child_factory(monkeypatch) -> None:
    def _fake_create_agent(name, llm_registry, tool_registry, sandbox, agent_config):
        _ = llm_registry, tool_registry, sandbox
        return _FakeChildAgent(str(name), agent_config)

    monkeypatch.setattr("lsm.agents.factory.create_agent", _fake_create_agent)


def test_meta_agent_writes_synthesized_final_result(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return "# Final Synthesis\n\nConsolidated cross-agent output.\n"

    _patch_child_factory(monkeypatch)
    monkeypatch.setattr(
        "lsm.agents.meta.meta.create_provider",
        lambda cfg: FakeProvider(),
    )

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    agent = MetaAgent(
        config.llm,
        ToolRegistry(),
        ToolSandbox(config.agents.sandbox),
        config.agents,
    )

    goal = json.dumps(
        {
            "goal": "Weekly theology digest",
            "tasks": [
                {
                    "id": "research_a",
                    "agent_name": "research",
                    "params": {"topic": "digest sources"},
                    "expected_artifacts": ["research_outline.md"],
                },
                {
                    "id": "synthesis_a",
                    "agent_name": "synthesis",
                    "params": {"topic": "digest synthesis"},
                    "depends_on": ["research_a"],
                    "expected_artifacts": ["synthesis.md"],
                },
            ],
        }
    )
    state = agent.run(AgentContext(messages=[{"role": "user", "content": goal}]))

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.final_result_path is not None
    assert agent.last_result.meta_log_path is not None

    final_text = agent.last_result.final_result_path.read_text(encoding="utf-8")
    log_text = agent.last_result.meta_log_path.read_text(encoding="utf-8")
    assert "# Final Synthesis" in final_text
    assert "Consolidated cross-agent output." in final_text
    assert "Task `research_a`" in log_text
    assert "Task `synthesis_a`" in log_text


def test_meta_agent_falls_back_when_synthesis_provider_fails(monkeypatch, tmp_path: Path) -> None:
    _patch_child_factory(monkeypatch)

    def _raise_provider(_cfg):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("lsm.agents.meta.meta.create_provider", _raise_provider)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    agent = MetaAgent(
        config.llm,
        ToolRegistry(),
        ToolSandbox(config.agents.sandbox),
        config.agents,
    )

    state = agent.run(
        AgentContext(messages=[{"role": "user", "content": "Generate synthesis fallback"}])
    )

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.final_result_path is not None

    final_text = agent.last_result.final_result_path.read_text(encoding="utf-8")
    assert "# Final Result:" in final_text
    assert "## Task Outputs" in final_text
    assert "research_1" in final_text

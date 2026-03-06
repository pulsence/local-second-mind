from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.factory import AgentRegistry
from lsm.agents.models import AgentContext
from lsm.agents.tools import create_default_tool_registry
from lsm.agents.tools.sandbox import ToolSandbox

from tests.test_agents.execution_smoke_utils import (
    FakeCalendarProvider,
    FakeEmailProvider,
    FakeNewsProvider,
    ScriptedHarnessProvider,
    build_pipeline,
    build_smoke_config,
    make_memory_store,
)


def test_all_registered_agents_execute_real_runtime_paths(monkeypatch, tmp_path: Path) -> None:
    config = build_smoke_config(tmp_path)
    memory_store = make_memory_store(tmp_path)
    try:
        pipeline = build_pipeline(config)
        tool_registry = create_default_tool_registry(
            config,
            collection=pipeline.db,
            embedder=pipeline.embedder,
            memory_store=memory_store,
            pipeline=pipeline,
        )

        monkeypatch.setattr(
            "lsm.agents.harness.create_provider",
            lambda llm_config: ScriptedHarnessProvider(),
        )

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        manuscript_path = docs_dir / "manuscript.md"
        manuscript_path.write_text(
            "# Draft\n\n## Section One\n\nOriginal text.\n\n## Section Two\n\nMore text.\n",
            encoding="utf-8",
        )
        (docs_dir / "aquinas.md").write_text(
            "# Aquinas\n\nSin is a voluntary act contrary to reason.\n",
            encoding="utf-8",
        )
        (docs_dir / "language.md").write_text(
            "# Etymology\n\nHamartia can mean missing the mark.\n",
            encoding="utf-8",
        )

        agents_root = Path(config.agents.agents_folder)
        agents_root.mkdir(parents=True, exist_ok=True)
        sample_summary_dir = agents_root / "bootstrap"
        sample_summary_dir.mkdir(parents=True, exist_ok=True)
        (sample_summary_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "agent_name": "bootstrap",
                    "status": "completed",
                    "topic": "Bootstrap summary",
                    "tools_used": {"query_and_synthesize": 1},
                    "artifacts_created": [],
                    "approvals_denials": {"denials": 0},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        config.agents.agent_configs = {
            "calendar_assistant": {"provider_instance": FakeCalendarProvider()},
            "curator": {"scope_path": str(docs_dir)},
            "email_assistant": {"provider_instance": FakeEmailProvider()},
            "meta": {"execution_mode": "plan_only", "enable_final_synthesis": False},
            "assistant_meta": {"execution_mode": "plan_only", "enable_final_synthesis": False},
            "news_assistant": {"provider_instances": [FakeNewsProvider()]},
        }

        payloads = {
            "assistant": "Summarize recent agent activity.",
            "assistant_meta": "Review assistant output quality.",
            "calendar_assistant": {"action": "summary", "query": "meetings"},
            "curator": "Curate theology notes",
            "email_assistant": {"action": "summary", "query": "outline"},
            "general": "Review the workspace.",
            "librarian": "Build an idea graph for sin.",
            "manuscript_editor": str(manuscript_path),
            "meta": "Plan a research and writing pass on sin.",
            "news_assistant": {"topics": ["theology"], "query": "sin"},
            "research": "What is sin according to Thomas Aquinas?",
            "synthesis": "Synthesize the notes on sin.",
            "writing": "Write a grounded overview of sin.",
        }

        registry = AgentRegistry()
        executed: set[str] = set()

        for name in registry.list_agents():
            executed.add(name)
            sandbox = ToolSandbox(config.agents.sandbox)
            agent = registry.create(
                name,
                config.llm,
                tool_registry,
                sandbox,
                config.agents,
                config,
            )
            content = payloads[name]
            state = agent.run(AgentContext(messages=[{"role": "user", "content": content}]))
            assert state.status.value == "completed", name
            assert state.artifacts, name

            if name == "research":
                messages = [entry.content for entry in state.log_entries]
                assert any(msg.startswith("Research iteration 1 - ") for msg in messages)
                assert all("\\n" not in msg for msg in messages)

        assert executed == set(registry.list_agents())
    finally:
        memory_store.close()

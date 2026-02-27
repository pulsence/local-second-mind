from __future__ import annotations

import inspect
from pathlib import Path

from lsm.agents.models import AgentContext
from lsm.agents.productivity.manuscript_editor import ManuscriptEditorAgent
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


def test_manuscript_editor_revises_sections_and_writes_logs(tmp_path: Path) -> None:
    manuscript = (
        "# Chapter One\n\n"
        "Intro line with trailing spaces.   \n\n"
        "## Section A\n\n"
        "Some content with whitespace.  \n"
    )
    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text(manuscript, encoding="utf-8")

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = ManuscriptEditorAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(messages=[{"role": "user", "content": str(manuscript_path)}])
    )

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()
    assert agent.last_result.revision_log_path.exists()
    assert agent.last_result.log_path.exists()

    revision_log = agent.last_result.revision_log_path.read_text(encoding="utf-8")
    assert "Revision Log" in revision_log
    assert "Chapter One" in revision_log
    assert "Round 1" in revision_log

    output_text = agent.last_result.output_path.read_text(encoding="utf-8")
    assert output_text != manuscript
    assert any(line.endswith("  ") for line in manuscript.splitlines())
    assert not any(line.endswith("  ") for line in output_text.splitlines())


def test_manuscript_editor_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.productivity.manuscript_editor as editor_module

    source = inspect.getsource(editor_module)
    assert "AgentHarness(" not in source, (
        "ManuscriptEditorAgent must not directly instantiate AgentHarness; use _run_phase() instead"
    )

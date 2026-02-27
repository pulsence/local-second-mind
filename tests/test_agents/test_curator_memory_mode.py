from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.academic import CuratorAgent
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
            "max_iterations": 4,
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


def test_curator_memory_mode_generates_candidates_from_run_summaries(
    tmp_path: Path,
) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    agents_folder = config.agents.agents_folder
    agents_folder.mkdir(parents=True, exist_ok=True)

    summary_a = {
        "tools_used": {"query_knowledge_base": 2, "read_file": 1},
        "constraints": ["avoid speculation", "must include citations"],
        "approvals_denials": {
            "by_tool": {
                "query_knowledge_base": {"approvals": 3, "denials": 0},
                "write_file": {"approvals": 0, "denials": 2},
            }
        },
    }
    summary_b = {
        "tools_used": {"query_knowledge_base": 2, "read_file": 2},
        "constraints": ["avoid speculation"],
        "approvals_denials": {
            "by_tool": {
                "query_knowledge_base": {"approvals": 2, "denials": 0},
                "write_file": {"approvals": 0, "denials": 1},
            }
        },
    }

    (agents_folder / "research_1").mkdir(parents=True, exist_ok=True)
    (agents_folder / "research_2").mkdir(parents=True, exist_ok=True)
    (agents_folder / "research_1" / "run_summary.json").write_text(
        json.dumps(summary_a, indent=2),
        encoding="utf-8",
    )
    (agents_folder / "research_2" / "run_summary.json").write_text(
        json.dumps(summary_b, indent=2),
        encoding="utf-8",
    )

    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = CuratorAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(
            messages=[
                {"role": "user", "content": "distill recurring patterns --mode memory"}
            ]
        )
    )

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()
    assert agent.last_result.output_path.name == "memory_candidates.md"

    candidates_json_path = agent.last_result.output_path.parent / "memory_candidates.json"
    assert candidates_json_path.exists()
    candidates = json.loads(candidates_json_path.read_text(encoding="utf-8"))
    assert isinstance(candidates, list)

    keys = {item["key"] for item in candidates}
    assert "preferred_tool_query_knowledge_base" in keys
    assert "preferred_tool_read_file" in keys
    assert "constraint_avoid_speculation" in keys
    assert "permission_guardrail_write_file" in keys
    assert "trusted_tool_query_knowledge_base" in keys

    markdown = agent.last_result.output_path.read_text(encoding="utf-8")
    assert "# Memory Candidates: distill recurring patterns" in markdown
    assert "## Candidates" in markdown

    assert str(agent.last_result.output_path) in state.artifacts
    assert str(candidates_json_path) in state.artifacts
    assert agent.last_result.log_path.exists()

from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import ToolRegistry
from lsm.agents.tools.read_file import ReadFileTool
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.agents.tools.write_file import WriteFileTool
from lsm.agents.workspace import ensure_agent_workspace
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
            "max_iterations": 5,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {"write_file": False},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def test_harness_creates_agent_workspace_layout(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            return json.dumps({"response": "Done", "action": "DONE", "action_arguments": {}})

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    harness = AgentHarness(
        config.agents,
        ToolRegistry(),
        config.llm,
        ToolSandbox(config.agents.sandbox),
        agent_name="workspace",
    )

    state = harness.run(AgentContext(messages=[{"role": "user", "content": "done"}]))
    assert state.status.value == "completed"

    agent_root = config.agents.agents_folder / "workspace"
    assert (agent_root / "logs").exists()
    assert (agent_root / "artifacts").exists()
    assert (agent_root / "memory").exists()


def test_file_tools_default_to_agent_workspace(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    sandbox = ToolSandbox(config.agents.sandbox)
    agent_root = ensure_agent_workspace(
        "writer",
        config.agents.agents_folder,
        sandbox=sandbox,
    )

    write_tool = WriteFileTool()
    sandbox.execute(write_tool, {"path": "output.txt", "content": "hello"})
    expected_path = agent_root / "output.txt"
    assert expected_path.exists()
    assert expected_path.read_text(encoding="utf-8") == "hello"

    read_tool = ReadFileTool()
    assert sandbox.execute(read_tool, {"path": "output.txt"}) == "hello"

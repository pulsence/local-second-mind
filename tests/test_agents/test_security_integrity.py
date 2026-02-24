from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext
from lsm.agents.tools import WriteFileTool
from lsm.agents.tools.base import ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw
from lsm.config.models.agents import SandboxConfig


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
            "max_tokens_budget": 5000,
            "max_iterations": 5,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {"write_file": False},
                "require_permission_by_risk": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def test_security_integrity_write_only_within_allowed_paths(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_write_paths=[allowed],
            require_user_permission={"write_file": False},
        )
    )
    tool = WriteFileTool()
    with pytest.raises(PermissionError, match="not allowed for write"):
        sandbox.execute(
            tool,
            {"path": str(outside / "out.txt"), "content": "nope"},
        )


def test_security_integrity_parent_directory_creation_is_safe(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    nested_path = allowed / "a" / "b" / "note.txt"

    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_write_paths=[allowed],
            require_user_permission={"write_file": False},
        )
    )
    tool = WriteFileTool()
    sandbox.execute(tool, {"path": str(nested_path), "content": "hello"})

    assert nested_path.exists()
    assert nested_path.read_text(encoding="utf-8") == "hello"


def test_security_integrity_overwrite_requires_permission(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "note.txt"
    target.write_text("original", encoding="utf-8")

    sandbox = ToolSandbox(SandboxConfig(allowed_write_paths=[allowed]))
    tool = WriteFileTool()
    with pytest.raises(PermissionError, match="requires user permission"):
        sandbox.execute(tool, {"path": str(target), "content": "overwrite"})


def test_security_integrity_artifacts_tracked_in_agent_state(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"

    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "response": "write",
                        "action": "write_file",
                        "action_arguments": {"path": str(target), "content": "artifact"},
                    }
                ),
                json.dumps({"response": "done", "action": "DONE", "action_arguments": {}}),
            ]

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
            return self.responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    harness = AgentHarness(config.agents, registry, config.llm, ToolSandbox(config.agents.sandbox))

    state = harness.run(AgentContext(messages=[{"role": "user", "content": "write"}]))
    expected_artifact = str(target.resolve())
    assert expected_artifact in state.artifacts
    assert target.exists()

    state_path = harness.get_state_path()
    assert state_path is not None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert expected_artifact in payload.get("artifacts", [])

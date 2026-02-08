from __future__ import annotations

from pathlib import Path

import pytest

from lsm.config.loader import build_config_from_raw, config_to_raw
from lsm.config.models.agents import AgentConfig, SandboxConfig


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {
            "global_folder": str(tmp_path / "lsm-global"),
        },
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "test_collection",
        },
        "llms": {
            "providers": [
                {"provider_name": "openai", "api_key": "test-key"}
            ],
            "services": {
                "query": {
                    "provider": "openai",
                    "model": "gpt-5.2",
                }
            },
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "test_collection",
        },
        "query": {"mode": "grounded"},
    }


def test_agent_config_validate_happy_path() -> None:
    cfg = AgentConfig(
        enabled=True,
        max_tokens_budget=1000,
        max_iterations=5,
        context_window_strategy="compact",
        sandbox=SandboxConfig(
            allowed_read_paths=[Path(".")],
            allowed_write_paths=[Path(".")],
            allow_url_access=False,
        ),
    )
    cfg.validate()


def test_agent_config_validate_rejects_bad_context_strategy() -> None:
    cfg = AgentConfig(context_window_strategy="wide")
    with pytest.raises(ValueError, match="context_window_strategy"):
        cfg.validate()


def test_sandbox_config_validate_rejects_empty_read_path() -> None:
    cfg = SandboxConfig(allowed_read_paths=[Path(" ")])
    with pytest.raises(ValueError, match="allowed_read_paths"):
        cfg.validate()


def test_build_config_reads_agents_section(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {
        "enabled": True,
        "agents_folder": "Agents",
        "max_tokens_budget": 12000,
        "max_iterations": 18,
        "context_window_strategy": "fresh",
        "sandbox": {
            "allowed_read_paths": [str(tmp_path)],
            "allowed_write_paths": [str(tmp_path / "out")],
            "allow_url_access": True,
            "require_user_permission": {"write_file": True},
            "tool_llm_assignments": {"query_remote": "decomposition"},
        },
        "agent_configs": {
            "research": {"max_iterations": 10}
        },
    }

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.agents is not None
    assert config.agents.enabled is True
    assert config.agents.max_tokens_budget == 12000
    assert config.agents.max_iterations == 18
    assert config.agents.context_window_strategy == "fresh"
    assert config.agents.sandbox.allow_url_access is True
    assert config.agents.sandbox.require_user_permission["write_file"] is True
    assert config.agents.sandbox.tool_llm_assignments["query_remote"] == "decomposition"
    assert config.agents.agent_configs["research"]["max_iterations"] == 10


def test_build_config_resolves_agents_folder_relative_to_config_path(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {"enabled": True, "agents_folder": "Agents"}
    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.agents is not None
    assert config.agents.agents_folder == (tmp_path / "Agents").resolve()


def test_config_to_raw_includes_agents_section(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {
        "enabled": True,
        "agents_folder": "Agents",
        "max_tokens_budget": 999,
        "max_iterations": 3,
        "context_window_strategy": "compact",
        "sandbox": {
            "allowed_read_paths": [str(tmp_path)],
            "allowed_write_paths": [str(tmp_path / "out")],
            "allow_url_access": False,
            "require_user_permission": {"write_file": True},
            "tool_llm_assignments": {"query_llm": "default"},
        },
        "agent_configs": {"research": {"enabled": True}},
    }
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert "agents" in serialized
    assert serialized["agents"]["enabled"] is True
    assert serialized["agents"]["max_tokens_budget"] == 999
    assert serialized["agents"]["max_iterations"] == 3
    assert serialized["agents"]["context_window_strategy"] == "compact"
    assert serialized["agents"]["sandbox"]["require_user_permission"]["write_file"] is True
    assert serialized["agents"]["agent_configs"]["research"]["enabled"] is True

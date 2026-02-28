from __future__ import annotations

from pathlib import Path

import pytest

from lsm.config.loader import build_config_from_raw, config_to_raw
from lsm.config.models.agents import (
    AgentConfig,
    InteractionConfig,
    MemoryConfig,
    SandboxConfig,
)


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {
            "global_folder": str(tmp_path / "lsm-global"),
        },
        "ingest": {
            "roots": [str(tmp_path / "docs")],
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
            "provider": "sqlite",
            "path": str(tmp_path / "data"),
            "collection": "test_collection",
        },
        "query": {"mode": "grounded"},
    }


def test_agent_config_validate_happy_path() -> None:
    cfg = AgentConfig(
        enabled=True,
        max_tokens_budget=1000,
        max_iterations=5,
        max_concurrent=3,
        log_stream_queue_limit=256,
        context_window_strategy="compact",
        sandbox=SandboxConfig(
            allowed_read_paths=[Path(".")],
            allowed_write_paths=[Path(".")],
            allow_url_access=False,
        ),
        interaction=InteractionConfig(
            timeout_seconds=60,
            timeout_action="deny",
            auto_continue=True,
            acknowledged_timeout_seconds=0,
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


def test_sandbox_config_validate_rejects_empty_command_allowlist_entry() -> None:
    cfg = SandboxConfig(command_allowlist=["", "ls"])
    with pytest.raises(ValueError, match="command_allowlist"):
        cfg.validate()


def test_build_config_reads_agents_section(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {
        "enabled": True,
        "agents_folder": "Agents",
        "max_tokens_budget": 12000,
        "max_iterations": 18,
        "max_concurrent": 4,
        "log_stream_queue_limit": 333,
        "context_window_strategy": "fresh",
        "sandbox": {
            "allowed_read_paths": [str(tmp_path)],
            "allowed_write_paths": [str(tmp_path / "out")],
            "allow_url_access": True,
            "require_user_permission": {"write_file": True},
            "require_permission_by_risk": {"network": True},
            "execution_mode": "prefer_docker",
            "force_docker": True,
            "limits": {
                "timeout_s_default": 15,
                "max_stdout_kb": 128,
                "max_file_write_mb": 4,
            },
            "docker": {"enabled": True, "image": "test-image"},
            "wsl2": {"enabled": True, "distro": "Ubuntu"},
            "command_allowlist": ["ls", "git"],
            "command_denylist": ["rm"],
            "tool_llm_assignments": {"query_arxiv": "decomposition"},
        },
        "memory": {
            "enabled": True,
            "storage_backend": "sqlite",
            "ttl_project_fact_days": 120,
            "ttl_task_state_days": 14,
            "ttl_cache_hours": 12,
        },
        "interaction": {
            "timeout_seconds": 42,
            "timeout_action": "approve",
            "auto_continue": True,
            "acknowledged_timeout_seconds": 120,
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
    assert config.agents.max_concurrent == 4
    assert config.agents.log_stream_queue_limit == 333
    assert config.agents.context_window_strategy == "fresh"
    assert config.agents.sandbox.allow_url_access is True
    assert config.agents.sandbox.require_user_permission["write_file"] is True
    assert config.agents.sandbox.require_permission_by_risk["network"] is True
    assert config.agents.sandbox.execution_mode == "prefer_docker"
    assert config.agents.sandbox.force_docker is True
    assert config.agents.sandbox.limits["timeout_s_default"] == 15
    assert config.agents.sandbox.limits["max_stdout_kb"] == 128
    assert config.agents.sandbox.limits["max_file_write_mb"] == 4
    assert config.agents.sandbox.docker["enabled"] is True
    assert config.agents.sandbox.docker["image"] == "test-image"
    assert config.agents.sandbox.wsl2["enabled"] is True
    assert config.agents.sandbox.wsl2["distro"] == "Ubuntu"
    assert config.agents.sandbox.command_allowlist == ["ls", "git"]
    assert config.agents.sandbox.command_denylist == ["rm"]
    assert config.agents.sandbox.tool_llm_assignments["query_arxiv"] == "decomposition"
    assert config.agents.memory.enabled is True
    assert config.agents.memory.storage_backend == "sqlite"
    assert config.agents.memory.ttl_project_fact_days == 120
    assert config.agents.memory.ttl_task_state_days == 14
    assert config.agents.memory.ttl_cache_hours == 12
    assert config.agents.interaction.timeout_seconds == 42
    assert config.agents.interaction.timeout_action == "approve"
    assert config.agents.interaction.auto_continue is True
    assert config.agents.interaction.acknowledged_timeout_seconds == 120
    assert config.agents.agent_configs["research"]["max_iterations"] == 10


def test_build_config_resolves_agents_folder_relative_to_global_folder(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {"enabled": True, "agents_folder": "Agents"}
    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.agents is not None
    assert config.agents.agents_folder == (tmp_path / "lsm-global" / "Agents").resolve()


def test_build_config_rejects_legacy_memory_sqlite_path(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {
        "enabled": True,
        "memory": {
            "enabled": True,
            "storage_backend": "sqlite",
            "sqlite_path": "memory.sqlite3",
        },
    }
    with pytest.raises(ValueError, match="Unsupported legacy memory field 'sqlite_path'"):
        build_config_from_raw(raw, tmp_path / "config.json")


def test_config_to_raw_includes_agents_section(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {
        "enabled": True,
        "agents_folder": "Agents",
        "max_tokens_budget": 999,
        "max_iterations": 3,
        "max_concurrent": 2,
        "log_stream_queue_limit": 77,
        "context_window_strategy": "compact",
        "sandbox": {
            "allowed_read_paths": [str(tmp_path)],
            "allowed_write_paths": [str(tmp_path / "out")],
            "allow_url_access": False,
            "require_user_permission": {"write_file": True},
            "require_permission_by_risk": {"writes_workspace": True},
            "execution_mode": "local_only",
            "force_docker": False,
            "limits": {"timeout_s_default": 22, "max_stdout_kb": 33, "max_file_write_mb": 2},
            "docker": {"enabled": False, "mem_limit_mb": 1024},
            "wsl2": {"enabled": False},
            "command_allowlist": ["cat"],
            "command_denylist": ["powershell.exe"],
            "tool_llm_assignments": {"query_llm": "default"},
        },
        "memory": {
            "enabled": True,
            "storage_backend": "sqlite",
            "postgres_connection_string": None,
            "postgres_table_prefix": "agent_memory",
            "ttl_project_fact_days": 90,
            "ttl_task_state_days": 7,
            "ttl_cache_hours": 24,
        },
        "interaction": {
            "timeout_seconds": 120,
            "timeout_action": "deny",
            "auto_continue": True,
            "acknowledged_timeout_seconds": 60,
        },
        "agent_configs": {"research": {"enabled": True}},
    }
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert "agents" in serialized
    assert serialized["agents"]["enabled"] is True
    assert serialized["agents"]["max_tokens_budget"] == 999
    assert serialized["agents"]["max_iterations"] == 3
    assert serialized["agents"]["max_concurrent"] == 2
    assert serialized["agents"]["log_stream_queue_limit"] == 77
    assert serialized["agents"]["context_window_strategy"] == "compact"
    assert serialized["agents"]["sandbox"]["require_user_permission"]["write_file"] is True
    assert serialized["agents"]["sandbox"]["require_permission_by_risk"]["writes_workspace"] is True
    assert serialized["agents"]["sandbox"]["execution_mode"] == "local_only"
    assert serialized["agents"]["sandbox"]["force_docker"] is False
    assert serialized["agents"]["sandbox"]["limits"]["timeout_s_default"] == 22
    assert serialized["agents"]["sandbox"]["limits"]["max_stdout_kb"] == 33
    assert serialized["agents"]["sandbox"]["limits"]["max_file_write_mb"] == 2
    assert serialized["agents"]["sandbox"]["docker"]["enabled"] is False
    assert serialized["agents"]["sandbox"]["docker"]["mem_limit_mb"] == 1024
    assert serialized["agents"]["sandbox"]["wsl2"]["enabled"] is False
    assert serialized["agents"]["sandbox"]["command_allowlist"] == ["cat"]
    assert serialized["agents"]["sandbox"]["command_denylist"] == ["powershell.exe"]
    assert serialized["agents"]["memory"]["enabled"] is True
    assert serialized["agents"]["memory"]["storage_backend"] == "sqlite"
    assert serialized["agents"]["memory"]["ttl_project_fact_days"] == 90
    assert serialized["agents"]["memory"]["ttl_task_state_days"] == 7
    assert serialized["agents"]["memory"]["ttl_cache_hours"] == 24
    assert serialized["agents"]["interaction"]["timeout_seconds"] == 120
    assert serialized["agents"]["interaction"]["timeout_action"] == "deny"
    assert serialized["agents"]["interaction"]["auto_continue"] is True
    assert serialized["agents"]["interaction"]["acknowledged_timeout_seconds"] == 60
    assert serialized["agents"]["agent_configs"]["research"]["enabled"] is True


def test_sandbox_config_validate_rejects_unknown_risk_level() -> None:
    cfg = SandboxConfig(require_permission_by_risk={"unknown": True})
    with pytest.raises(ValueError, match="require_permission_by_risk"):
        cfg.validate()


def test_sandbox_config_validate_rejects_unknown_execution_mode() -> None:
    cfg = SandboxConfig(execution_mode="invalid")
    with pytest.raises(ValueError, match="execution_mode"):
        cfg.validate()


def test_sandbox_config_validate_rejects_invalid_limits() -> None:
    cfg = SandboxConfig(limits={"timeout_s_default": 0, "max_stdout_kb": 0, "max_file_write_mb": 0})
    with pytest.raises(ValueError, match="timeout_s_default"):
        cfg.validate()


def test_memory_config_validate_rejects_unknown_backend() -> None:
    cfg = MemoryConfig(storage_backend="unknown")
    with pytest.raises(ValueError, match="storage_backend"):
        cfg.validate()


def test_memory_config_validate_rejects_non_positive_ttl_values() -> None:
    cfg = MemoryConfig(ttl_project_fact_days=0)
    with pytest.raises(ValueError, match="ttl_project_fact_days"):
        cfg.validate()

    cfg = MemoryConfig(ttl_task_state_days=0)
    with pytest.raises(ValueError, match="ttl_task_state_days"):
        cfg.validate()

    cfg = MemoryConfig(ttl_cache_hours=0)
    with pytest.raises(ValueError, match="ttl_cache_hours"):
        cfg.validate()


def test_interaction_config_validate_rejects_invalid_timeout_action() -> None:
    cfg = InteractionConfig(timeout_seconds=60, timeout_action="invalid")
    with pytest.raises(ValueError, match="timeout_action"):
        cfg.validate()


def test_agent_config_validate_rejects_non_positive_max_concurrent() -> None:
    cfg = AgentConfig(max_concurrent=0)
    with pytest.raises(ValueError, match="max_concurrent"):
        cfg.validate()


def test_agent_config_validate_rejects_non_positive_log_stream_queue_limit() -> None:
    cfg = AgentConfig(log_stream_queue_limit=0)
    with pytest.raises(ValueError, match="log_stream_queue_limit"):
        cfg.validate()


def test_interaction_config_defaults_acknowledged_timeout_to_zero() -> None:
    cfg = InteractionConfig()
    assert cfg.acknowledged_timeout_seconds == 0


def test_interaction_config_validate_rejects_negative_acknowledged_timeout() -> None:
    cfg = InteractionConfig(acknowledged_timeout_seconds=-1)
    with pytest.raises(ValueError, match="acknowledged_timeout_seconds"):
        cfg.validate()

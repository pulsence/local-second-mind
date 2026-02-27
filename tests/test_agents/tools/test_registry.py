"""
Tests for create_default_tool_registry().
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lsm.agents.tools import create_default_tool_registry
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
            "max_tokens_budget": 5000,
            "max_iterations": 10,
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


class _FakeCollection:
    pass


class _FakeEmbedder:
    pass


def test_registry_contains_query_knowledge_base(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = create_default_tool_registry(
        config,
        collection=_FakeCollection(),
        embedder=_FakeEmbedder(),
    )
    tool_names = {tool.name for tool in registry.list_tools()}
    assert "query_knowledge_base" in tool_names


def test_registry_does_not_contain_query_embeddings(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = create_default_tool_registry(
        config,
        collection=_FakeCollection(),
        embedder=_FakeEmbedder(),
    )
    tool_names = {tool.name for tool in registry.list_tools()}
    assert "query_embeddings" not in tool_names


def test_query_embeddings_file_does_not_exist() -> None:
    from pathlib import Path as _Path
    import lsm.agents.tools as _tools_pkg

    tools_dir = _Path(_tools_pkg.__file__).parent
    assert not (tools_dir / "query_embeddings.py").exists(), (
        "query_embeddings.py must be deleted — it is no longer a supported tool"
    )


def test_registry_contains_per_source_query_tools(tmp_path: Path) -> None:
    """Each configured remote source gets its own query_<name> tool in the registry."""
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {"name": "test_source", "type": "web_search", "api_key": "test-key", "max_results": 5},
        {"name": "other_source", "type": "wikipedia", "max_results": 5},
    ]
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = create_default_tool_registry(config)
    tool_names = {tool.name for tool in registry.list_tools()}
    assert "query_test_source" in tool_names, "Expected query_test_source in registry"
    assert "query_other_source" in tool_names, "Expected query_other_source in registry"


def test_registry_does_not_contain_generic_query_remote(tmp_path: Path) -> None:
    """The single generic 'query_remote' tool must not exist — only per-source tools."""
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {"name": "test_source", "type": "web_search", "api_key": "test-key", "max_results": 5},
    ]
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = create_default_tool_registry(config)
    tool_names = {tool.name for tool in registry.list_tools()}
    assert "query_remote" not in tool_names, (
        "query_remote must not exist as a generic single tool; "
        "use per-source query_<name> tools instead"
    )

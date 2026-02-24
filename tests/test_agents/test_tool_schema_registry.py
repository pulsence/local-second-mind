from __future__ import annotations

from pathlib import Path

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
    }


def test_tool_registry_includes_advanced_tool_definitions(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = create_default_tool_registry(config)
    definitions = {definition["name"]: definition for definition in registry.list_definitions()}

    for name in ("find_file", "find_section", "edit_file"):
        assert name in definitions
        schema = definitions[name]["input_schema"]
        assert schema["type"] == "object"

    find_file_required = set(definitions["find_file"]["input_schema"].get("required", []))
    assert "path" in find_file_required

    find_section_required = set(definitions["find_section"]["input_schema"].get("required", []))
    assert "section" in find_section_required

    edit_file_required = set(definitions["edit_file"]["input_schema"].get("required", []))
    assert {"path", "start_hash", "end_hash", "new_content"}.issubset(edit_file_required)

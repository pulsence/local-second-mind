from pathlib import Path

import pytest

from lsm.config.loader import (
    parse_config_text,
    build_config_from_raw,
    build_llm_provider_config,
    build_source_policy_config,
    config_to_raw,
)
from lsm.config.models import RemoteProviderRef


def _base_raw(tmp_path: Path) -> dict:
    return {
        "roots": [str(tmp_path / "docs")],
        "llms": [
            {
                "provider_name": "openai",
                "api_key": "test-key",
                "query": {
                    "model": "gpt-5.2",
                    "temperature": 0.3,
                    "max_tokens": 256,
                },
            }
        ],
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "test_collection",
        },
        "query": {"mode": "grounded"},
    }


def test_parse_config_text_json() -> None:
    parsed = parse_config_text('{"a": 1}', "config.json")
    assert parsed["a"] == 1


def test_parse_config_text_yaml() -> None:
    parsed = parse_config_text("a: 1\nb: 2\n", "config.yaml")
    assert parsed["a"] == 1
    assert parsed["b"] == 2


def test_build_config_uses_vectordb_fields_without_top_level_fallback(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["persist_dir"] = str(tmp_path / "legacy_dir")
    raw["collection"] = "legacy_collection"

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.ingest.persist_dir.name == ".chroma"
    assert config.ingest.collection == "test_collection"
    assert config.vectordb.persist_dir.name == ".chroma"
    assert config.vectordb.collection == "test_collection"


def test_build_source_policy_does_not_accept_remote_provides_alias() -> None:
    policy = build_source_policy_config(
        {
            "remote": {
                "enabled": True,
                "remote_provides": ["wikipedia"],
            }
        }
    )
    assert policy.remote.remote_providers is None


def test_build_source_policy_accepts_weighted_provider_objects() -> None:
    policy = build_source_policy_config(
        {
            "remote": {
                "enabled": True,
                "remote_providers": [{"source": "arxiv", "weight": 0.9}],
            }
        }
    )
    assert isinstance(policy.remote.remote_providers[0], RemoteProviderRef)
    assert policy.remote.remote_providers[0].source == "arxiv"
    assert policy.remote.remote_providers[0].weight == 0.9


def test_build_llm_provider_rejects_legacy_provider_fields() -> None:
    with pytest.raises(ValueError, match="Legacy llms\\[\\] fields"):
        build_llm_provider_config(
            {
                "provider_name": "openai",
                "model": "gpt-5.2",
                "query": {"model": "gpt-5.2"},
            }
        )


def test_config_to_raw_omits_legacy_top_level_vectordb_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert "persist_dir" not in serialized
    assert "collection" not in serialized
    assert serialized["vectordb"]["persist_dir"].endswith(".chroma")
    assert serialized["vectordb"]["collection"] == "test_collection"


def test_build_config_supports_global_folder(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["global_folder"] = str(tmp_path / "lsm-global")
    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.global_folder == (tmp_path / "lsm-global").resolve()
    assert config.global_folder.exists()
    assert (config.global_folder / "Notes").exists()
    assert (config.global_folder / "Chats").exists()


def test_config_to_raw_includes_global_folder(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["global_folder"] = str(tmp_path / "lsm-global")
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert serialized["global_folder"] == str((tmp_path / "lsm-global").resolve())


def test_notes_config_is_global_not_mode_scoped(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["notes"] = {"enabled": False, "dir": "research_notes"}
    raw["modes"] = [
        {
            "name": "research",
            "synthesis_style": "grounded",
            "source_policy": {},
            "notes": {"enabled": True, "dir": "ignored_notes"},
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert config.notes.enabled is False
    assert config.notes.dir == "research_notes"
    assert "notes" in serialized
    assert serialized["notes"]["dir"] == "research_notes"
    assert "notes" not in serialized["modes"][0]

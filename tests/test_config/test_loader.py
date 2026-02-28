import os
from pathlib import Path

import pytest

from lsm.config.loader import (
    parse_config_text,
    build_config_from_raw,
    build_global_config,
    build_llm_provider_config,
    build_source_policy_config,
    config_to_raw,
    build_remote_providers_registry,
    build_modes_registry,
    build_llm_config,
    save_config_to_file,
    load_config_from_file,
    load_raw_config,
)
from lsm.config.models import GlobalConfig, RemoteProviderRef


@pytest.fixture(autouse=True)
def _restore_env_after_test():
    original = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original)


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
                    "temperature": 0.3,
                    "max_tokens": 256,
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


def test_parse_config_text_json() -> None:
    parsed = parse_config_text('{"a": 1}', "config.json")
    assert parsed["a"] == 1


def test_parse_config_text_yaml() -> None:
    parsed = parse_config_text("a: 1\nb: 2\n", "config.yaml")
    assert parsed["a"] == 1
    assert parsed["b"] == 2


def test_build_global_config_defaults() -> None:
    cfg = build_global_config({})
    assert cfg.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.device == "cpu"
    assert cfg.batch_size == 32
    assert cfg.tui_density_mode == "auto"


def test_build_global_config_custom_values() -> None:
    cfg = build_global_config({
        "global": {
            "global_folder": "/tmp/lsm",
            "embed_model": "custom-model",
            "device": "cuda:0",
            "batch_size": 64,
            "tui_density_mode": "compact",
        }
    })
    assert cfg.embed_model == "custom-model"
    assert cfg.device == "cuda:0"
    assert cfg.batch_size == 64
    assert cfg.tui_density_mode == "compact"
    assert isinstance(cfg.global_folder, Path)


def test_build_global_config_reads_mcp_servers() -> None:
    cfg = build_global_config(
        {
            "global": {
                "mcp_servers": [
                    {
                        "name": "demo",
                        "command": "demo-server",
                        "args": ["--flag"],
                        "env": {"DEMO": "1"},
                    }
                ]
            }
        }
    )
    assert len(cfg.mcp_servers) == 1
    server = cfg.mcp_servers[0]
    assert server.name == "demo"
    assert server.command == "demo-server"
    assert server.args == ["--flag"]
    assert server.env == {"DEMO": "1"}


def test_build_llm_config_loads_tiers(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["llms"]["tiers"] = {
        "quick": {"provider": "openai", "model": "gpt-5-nano"},
    }

    registry = build_llm_config(raw)
    assert "quick" in registry.tiers
    assert registry.tiers["quick"].model == "gpt-5-nano"


def test_build_config_reads_ingest_from_nested_section(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["chunk_size"] = 500
    raw["ingest"]["chunk_overlap"] = 50
    raw["ingest"]["enable_ocr"] = True

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.ingest.chunk_size == 500
    assert config.ingest.chunk_overlap == 50
    assert config.ingest.enable_ocr is True


def test_build_config_reads_max_heading_depth_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["max_heading_depth"] = 2
    raw["ingest"]["roots"] = [
        {
            "path": str(tmp_path / "docs"),
            "max_heading_depth": 3,
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.ingest.max_heading_depth == 2
    assert config.ingest.roots[0].max_heading_depth == 3


def test_config_to_raw_includes_max_heading_depth_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["max_heading_depth"] = 2
    raw["ingest"]["roots"] = [
        {
            "path": str(tmp_path / "docs"),
            "max_heading_depth": 4,
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert serialized["ingest"]["max_heading_depth"] == 2
    assert serialized["ingest"]["roots"][0]["max_heading_depth"] == 4


def test_build_config_reads_intelligent_heading_depth(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["intelligent_heading_depth"] = True

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.ingest.intelligent_heading_depth is True


def test_config_to_raw_includes_intelligent_heading_depth(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["intelligent_heading_depth"] = True

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert serialized["ingest"]["intelligent_heading_depth"] is True


def test_build_config_reads_global_settings(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["global"]["embed_model"] = "custom-model"
    raw["global"]["device"] = "cuda:0"
    raw["global"]["batch_size"] = 128
    raw["global"]["tui_density_mode"] = "comfortable"

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.global_settings.embed_model == "custom-model"
    assert config.global_settings.device == "cuda:0"
    assert config.global_settings.batch_size == 128
    assert config.global_settings.tui_density_mode == "comfortable"
    # Shortcut properties delegate to global_settings
    assert config.embed_model == "custom-model"
    assert config.device == "cuda:0"
    assert config.batch_size == 128


def test_build_config_uses_vectordb_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.vectordb.path.name == "data"
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


def test_config_to_raw_uses_global_and_ingest_sections(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    # Global section exists with expected fields
    assert "global" in serialized
    assert "embed_model" in serialized["global"]
    assert "device" in serialized["global"]
    assert "batch_size" in serialized["global"]
    assert "global_folder" in serialized["global"]
    assert "tui_density_mode" in serialized["global"]

    # Ingest section exists with expected fields
    assert "ingest" in serialized
    assert "roots" in serialized["ingest"]
    assert "chunk_size" in serialized["ingest"]
    assert "manifest" not in serialized["ingest"]
    assert "chroma_flush_interval" not in serialized["ingest"]
    assert "enable_versioning" not in serialized["ingest"]

    # No flat top-level ingest/global fields
    assert "roots" not in serialized
    assert "embed_model" not in serialized
    assert "device" not in serialized
    assert "batch_size" not in serialized
    assert "global_folder" not in serialized
    assert "chunk_size" not in serialized

    # vectordb still present
    assert serialized["vectordb"]["path"].endswith("data")
    assert serialized["vectordb"]["collection"] == "test_collection"


def test_build_config_paths_resolve_relative_to_global_folder(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)
    config_path = project_dir / "config.json"
    raw = _base_raw(tmp_path)
    raw["global"]["global_folder"] = "lsm-global"
    raw["vectordb"]["path"] = "db"

    config = build_config_from_raw(raw, config_path)

    expected_global = (project_dir / "lsm-global").resolve()
    assert config.global_folder == expected_global
    assert config.vectordb.path == (expected_global / "db").resolve()


def test_build_config_rejects_legacy_ingest_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["manifest"] = str(tmp_path / ".ingest" / "manifest.json")

    with pytest.raises(ValueError, match="Unsupported legacy ingest field 'manifest'"):
        build_config_from_raw(raw, tmp_path / "config.json")


def test_build_config_rejects_legacy_vectordb_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["vectordb"]["persist_dir"] = str(tmp_path / ".chroma")

    with pytest.raises(ValueError, match="Unsupported legacy vectordb field 'persist_dir'"):
        build_config_from_raw(raw, tmp_path / "config.json")


def test_build_config_supports_global_folder(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["global"]["global_folder"] = str(tmp_path / "lsm-global")
    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.global_folder == (tmp_path / "lsm-global").resolve()
    assert config.global_folder.exists()
    assert (config.global_folder / "Notes").exists()
    assert (config.global_folder / "Chats").exists()


def test_config_to_raw_includes_global_folder(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["global"]["global_folder"] = str(tmp_path / "lsm-global")
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert serialized["global"]["global_folder"] == str((tmp_path / "lsm-global").resolve())


def test_config_to_raw_includes_mcp_servers(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["global"]["mcp_servers"] = [
        {
            "name": "demo",
            "command": "demo-server",
            "args": ["--flag"],
            "env": {"DEMO": "1"},
        }
    ]
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    mcp = serialized["global"]["mcp_servers"]
    assert isinstance(mcp, list)
    assert mcp[0]["name"] == "demo"
    assert mcp[0]["command"] == "demo-server"


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


def test_mode_chats_overrides_roundtrip(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["modes"] = [
        {
            "name": "research",
            "synthesis_style": "grounded",
            "source_policy": {},
            "chats": {"auto_save": False, "dir": "Chats/Research"},
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    mode_cfg = config.modes["research"]
    assert mode_cfg.chats is not None
    assert mode_cfg.chats.auto_save is False
    assert mode_cfg.chats.dir == "Chats/Research"
    assert serialized["modes"][0]["chats"]["auto_save"] is False
    assert serialized["modes"][0]["chats"]["dir"] == "Chats/Research"


def test_parse_config_text_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported config format"):
        parse_config_text("a: 1", "config.txt")


def test_build_llm_config_requires_non_empty_list() -> None:
    with pytest.raises(ValueError, match="must include 'llms'"):
        build_llm_config({})


def test_build_modes_registry_invalid_type_returns_none() -> None:
    assert build_modes_registry({"modes": {"name": "bad"}}) is None


def test_build_modes_registry_skips_invalid_entries() -> None:
    modes = build_modes_registry(
        {
            "modes": [
                {"name": "good", "synthesis_style": "grounded", "source_policy": {}},
                {"name": "good", "synthesis_style": "grounded", "source_policy": {}},
                {"bad": "entry"},
                "not-a-dict",
            ]
        }
    )
    assert modes is not None
    assert list(modes.keys()) == ["good"]


def test_build_remote_providers_registry_handles_invalid_entries() -> None:
    providers = build_remote_providers_registry(
        {
            "remote_providers": [
                {"name": "ok", "type": "wikipedia"},
                {"name": "", "type": "wikipedia"},
                {"name": "missing-type"},
                "nope",
            ]
        }
    )
    assert providers is not None
    assert len(providers) == 1
    assert providers[0].name == "ok"


def test_build_remote_providers_registry_invalid_section_type() -> None:
    assert build_remote_providers_registry({"remote_providers": "bad"}) is None


def test_build_remote_provider_config_reads_cache_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {
            "name": "wiki",
            "type": "wikipedia",
            "cache_results": True,
            "cache_ttl": 7200,
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.remote_providers is not None
    provider = config.remote_providers[0]
    assert provider.cache_results is True
    assert provider.cache_ttl == 7200


def test_build_config_reads_remote_provider_chains(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {"name": "openalex", "type": "openalex"},
        {"name": "crossref", "type": "crossref"},
    ]
    raw["remote_provider_chains"] = [
        {
            "name": "Research Digest",
            "agent_description": "Use for DOI enrichment",
            "links": [
                {"source": "openalex"},
                {"source": "crossref", "map": ["doi:doi"]},
            ],
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.remote_provider_chains is not None
    assert len(config.remote_provider_chains) == 1
    chain = config.remote_provider_chains[0]
    assert chain.name == "Research Digest"
    assert chain.links[1].map == ["doi:doi"]


def test_build_config_reads_remote_preconfigured_chains(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote"] = {"chains": ["scholarly_discovery"]}
    raw["remote_providers"] = [
        {"name": "openalex", "type": "openalex"},
        {"name": "crossref", "type": "crossref"},
        {"name": "unpaywall", "type": "unpaywall"},
        {"name": "core", "type": "core", "api_key": "test"},
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.remote_provider_chains is not None
    assert any(chain.name == "scholarly_discovery" for chain in config.remote_provider_chains)


def test_config_to_raw_includes_remote_provider_chains(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {"name": "openalex", "type": "openalex"},
        {"name": "crossref", "type": "crossref"},
    ]
    raw["remote_provider_chains"] = [
        {
            "name": "Research Digest",
            "agent_description": "Use for DOI enrichment",
            "links": [
                {"source": "openalex"},
                {"source": "crossref", "map": ["doi:doi"]},
            ],
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)
    assert serialized["remote_provider_chains"] is not None
    assert serialized["remote_provider_chains"][0]["name"] == "Research Digest"
    assert serialized["remote_provider_chains"][0]["links"][1]["map"] == ["doi:doi"]


def test_config_to_raw_includes_remote_chains(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote"] = {"chains": ["scholarly_discovery"]}
    raw["remote_providers"] = [
        {"name": "openalex", "type": "openalex"},
        {"name": "crossref", "type": "crossref"},
        {"name": "unpaywall", "type": "unpaywall"},
        {"name": "core", "type": "core", "api_key": "test"},
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)
    assert serialized["remote"] == {"chains": ["scholarly_discovery"]}


def test_build_config_skips_chain_with_unknown_provider(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [{"name": "openalex", "type": "openalex"}]
    raw["remote_provider_chains"] = [
        {
            "name": "Bad Chain",
            "links": [
                {"source": "openalex"},
                {"source": "crossref", "map": ["doi:doi"]},
            ],
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.remote_provider_chains is None


def test_config_to_raw_includes_remote_cache_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {
            "name": "wiki",
            "type": "wikipedia",
            "cache_results": True,
            "cache_ttl": 7200,
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)
    assert serialized["remote_providers"][0]["cache_results"] is True
    assert serialized["remote_providers"][0]["cache_ttl"] == 7200


def test_remote_provider_extra_fields_roundtrip(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {
            "name": "openalex",
            "type": "openalex",
            "email": "you@example.com",
            "year_from": 2020,
            "open_access_only": True,
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    provider = config.remote_providers[0]
    assert provider.extra["email"] == "you@example.com"
    assert provider.extra["year_from"] == 2020
    assert provider.extra["open_access_only"] is True

    serialized = config_to_raw(config)
    assert serialized["remote_providers"][0]["email"] == "you@example.com"
    assert serialized["remote_providers"][0]["year_from"] == 2020
    assert serialized["remote_providers"][0]["open_access_only"] is True


def test_remote_provider_oauth_roundtrip(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {
            "name": "gmail",
            "type": "gmail",
            "oauth": {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "scopes": ["scope.one", "scope.two"],
                "redirect_uri": "http://localhost:9999/callback",
                "refresh_buffer_seconds": 120,
            },
        }
    ]

    config = build_config_from_raw(raw, tmp_path / "config.json")
    provider = config.remote_providers[0]
    assert provider.oauth is not None
    assert provider.oauth.client_id == "client-id"
    assert provider.oauth.scopes == ["scope.one", "scope.two"]
    assert provider.oauth.redirect_uri == "http://localhost:9999/callback"
    assert provider.oauth.refresh_buffer_seconds == 120

    serialized = config_to_raw(config)
    oauth = serialized["remote_providers"][0]["oauth"]
    assert oauth["client_id"] == "client-id"
    assert oauth["client_secret"] == "client-secret"
    assert oauth["scopes"] == ["scope.one", "scope.two"]
    assert oauth["redirect_uri"] == "http://localhost:9999/callback"
    assert oauth["refresh_buffer_seconds"] == 120


def test_save_and_load_config_json_roundtrip(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    out_path = tmp_path / "saved.json"

    save_config_to_file(config, out_path)
    loaded = load_config_from_file(out_path)
    assert loaded.vectordb.collection == "test_collection"
    assert loaded.embed_model == config.embed_model


def test_save_and_load_config_yaml_roundtrip(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.yaml")
    out_path = tmp_path / "saved.yaml"

    save_config_to_file(config, out_path)
    loaded = load_config_from_file(out_path)
    assert loaded.vectordb.collection == "test_collection"


def test_save_config_rejects_unknown_suffix(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    with pytest.raises(ValueError, match="Unsupported config format"):
        save_config_to_file(config, tmp_path / "saved.ini")


def test_load_raw_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_config(tmp_path / "missing.json")


def test_build_ingest_config_requires_roots_in_ingest_section(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    del raw["ingest"]["roots"]
    with pytest.raises(ValueError, match="ingest section must include 'roots'"):
        build_config_from_raw(raw, tmp_path / "config.json")


def test_build_config_reads_language_detection_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["enable_language_detection"] = True
    raw["ingest"]["enable_translation"] = True
    raw["ingest"]["translation_target"] = "fr"

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.ingest.enable_language_detection is True
    assert config.ingest.enable_translation is True
    assert config.ingest.translation_target == "fr"


def test_build_config_language_detection_defaults_false(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.ingest.enable_language_detection is False
    assert config.ingest.enable_translation is False
    assert config.ingest.translation_target == "en"


def test_config_to_raw_includes_language_translation_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["ingest"]["enable_language_detection"] = True
    raw["ingest"]["enable_translation"] = True
    raw["ingest"]["translation_target"] = "de"

    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert serialized["ingest"]["enable_language_detection"] is True
    assert serialized["ingest"]["enable_translation"] is True
    assert serialized["ingest"]["translation_target"] == "de"


def test_build_global_config_reads_embedding_dimension(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["global"]["embedding_dimension"] = 768

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.global_settings.embedding_dimension == 768
    assert config.embedding_dimension == 768


def test_build_global_config_embedding_dimension_null(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    # Default model is known, so dimension auto-detected

    config = build_config_from_raw(raw, tmp_path / "config.json")

    assert config.embedding_dimension == 384  # all-MiniLM-L6-v2


def test_config_to_raw_includes_embedding_dimension(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    serialized = config_to_raw(config)

    assert "embedding_dimension" in serialized["global"]


def test_build_query_config_reads_cache_and_chat_fields(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["query"]["chat_mode"] = "chat"
    raw["query"]["enable_query_cache"] = True
    raw["query"]["query_cache_ttl"] = 120
    raw["query"]["query_cache_size"] = 9
    raw["query"]["enable_llm_server_cache"] = True

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.query.chat_mode == "chat"
    assert config.query.enable_query_cache is True
    assert config.query.query_cache_ttl == 120
    assert config.query.query_cache_size == 9
    assert config.query.enable_llm_server_cache is True


def test_build_chats_config_and_serialization_defaults(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["chats"] = {"enabled": True, "dir": "Chats", "auto_save": False, "format": "markdown"}

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.chats.enabled is True
    assert config.chats.auto_save is False
    serialized = config_to_raw(config)
    assert "chats" in serialized
    assert serialized["chats"]["dir"] == "Chats"

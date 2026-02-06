"""
Tests for query mode and global notes configuration.
"""

from pathlib import Path

from lsm.config.models import (
    FeatureLLMConfig,
    IngestConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LSMConfig,
    LocalSourcePolicy,
    ModeConfig,
    ModelKnowledgePolicy,
    NotesConfig,
    QueryConfig,
    RemoteSourcePolicy,
    SourcePolicyConfig,
    VectorDBConfig,
)


def _make_config(mode: str, modes: dict | None = None, notes: NotesConfig | None = None) -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(
            roots=[Path("/test")],
            manifest=Path("/tmp/manifest.json"),
        ),
        query=QueryConfig(mode=mode),
        llm=LLMRegistryConfig(
            llms=[
                LLMProviderConfig(
                    provider_name="openai",
                    api_key="test",
                    query=FeatureLLMConfig(model="gpt-5.2"),
                ),
            ]
        ),
        vectordb=VectorDBConfig(
            persist_dir=Path("/tmp/.chroma"),
            collection="test",
        ),
        modes=modes,
        notes=notes or NotesConfig(),
        config_path=Path("/tmp/config.json"),
    )


class TestModeConfig:
    def test_mode_config_defaults(self):
        mode = ModeConfig()
        assert mode.synthesis_style == "grounded"
        assert mode.source_policy is not None

    def test_mode_config_custom_values(self):
        source_policy = SourcePolicyConfig(
            local=LocalSourcePolicy(k=15, min_relevance=0.3),
            remote=RemoteSourcePolicy(enabled=True),
            model_knowledge=ModelKnowledgePolicy(enabled=True),
        )
        mode = ModeConfig(synthesis_style="insight", source_policy=source_policy)
        assert mode.synthesis_style == "insight"
        assert mode.source_policy.local.k == 15


class TestNotesConfig:
    def test_notes_config_defaults(self):
        notes = NotesConfig()
        assert notes.enabled is True
        assert notes.dir == "notes"
        assert notes.template == "default"
        assert notes.filename_format == "timestamp"
        assert notes.integration == "none"
        assert notes.wikilinks is False
        assert notes.backlinks is False
        assert notes.include_tags is False

    def test_notes_config_custom_values(self):
        notes = NotesConfig(
            enabled=True,
            dir="research_notes",
            template="research",
            filename_format="query_slug",
            integration="obsidian",
            wikilinks=True,
            backlinks=True,
            include_tags=True,
        )
        assert notes.dir == "research_notes"
        assert notes.template == "research"
        assert notes.filename_format == "query_slug"
        assert notes.integration == "obsidian"
        assert notes.wikilinks is True
        assert notes.backlinks is True
        assert notes.include_tags is True


class TestBuiltInModes:
    def test_grounded_mode_config(self):
        config = _make_config("grounded")
        mode = config.get_mode_config()
        assert mode.synthesis_style == "grounded"
        assert mode.source_policy.remote.enabled is False
        assert mode.source_policy.model_knowledge.enabled is False

    def test_insight_mode_config(self):
        config = _make_config("insight")
        mode = config.get_mode_config()
        assert mode.synthesis_style == "insight"
        assert mode.source_policy.model_knowledge.enabled is True

    def test_hybrid_mode_config(self):
        config = _make_config("hybrid")
        mode = config.get_mode_config()
        assert mode.synthesis_style == "grounded"
        assert mode.source_policy.remote.enabled is True
        assert mode.source_policy.model_knowledge.enabled is True


class TestCustomModes:
    def test_custom_mode_registration(self):
        custom_mode = ModeConfig(
            synthesis_style="grounded",
            source_policy=SourcePolicyConfig(
                local=LocalSourcePolicy(k=20, min_relevance=0.15),
                remote=RemoteSourcePolicy(enabled=True, max_results=15, remote_providers=["arxiv"]),
                model_knowledge=ModelKnowledgePolicy(enabled=True, require_label=False),
            ),
        )
        config = _make_config("my_custom_mode", modes={"my_custom_mode": custom_mode})
        mode = config.get_mode_config()
        assert mode.source_policy.local.k == 20
        assert mode.source_policy.remote.max_results == 15
        assert mode.source_policy.remote.remote_providers == ["arxiv"]

    def test_mode_fallback_to_default(self):
        config = _make_config("nonexistent_mode")
        config.validate()
        mode = config.get_mode_config()
        assert mode.synthesis_style == "grounded"


class TestGlobalNotesOnConfig:
    def test_notes_are_global(self):
        config = _make_config("grounded", notes=NotesConfig(enabled=False, dir="research_notes"))
        assert config.notes.enabled is False
        assert config.notes.dir == "research_notes"

    def test_notes_default_enabled(self):
        config = _make_config("grounded")
        assert config.notes.enabled is True

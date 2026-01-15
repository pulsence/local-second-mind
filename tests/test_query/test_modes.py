"""
Tests for query mode system.
"""

import pytest
from pathlib import Path

from lsm.config.models import (
    LSMConfig,
    ModeConfig,
    SourcePolicyConfig,
    LocalSourcePolicy,
    RemoteSourcePolicy,
    ModelKnowledgePolicy,
    NotesConfig,
    IngestConfig,
    QueryConfig,
    VectorDBConfig,
    LLMRegistryConfig,
    LLMProviderConfig,
    FeatureLLMConfig,
)


def _make_config(mode: str, modes: dict | None = None) -> LSMConfig:
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
        config_path=Path("/tmp/config.json"),
    )


class TestModeConfig:
    """Tests for ModeConfig dataclass."""

    def test_mode_config_defaults(self):
        """Test ModeConfig has reasonable defaults."""
        mode = ModeConfig()

        assert mode.synthesis_style == "grounded"
        assert mode.source_policy is not None
        assert mode.notes is not None

    def test_mode_config_custom_values(self):
        """Test creating ModeConfig with custom values."""
        source_policy = SourcePolicyConfig(
            local=LocalSourcePolicy(k=15, min_relevance=0.3),
            remote=RemoteSourcePolicy(enabled=True),
            model_knowledge=ModelKnowledgePolicy(enabled=True)
        )
        notes = NotesConfig(enabled=True, dir="my_notes")

        mode = ModeConfig(
            synthesis_style="insight",
            source_policy=source_policy,
            notes=notes
        )

        assert mode.synthesis_style == "insight"
        assert mode.source_policy.local.k == 15
        assert mode.notes.dir == "my_notes"


class TestSourcePolicyConfig:
    """Tests for SourcePolicyConfig and related classes."""

    def test_local_source_policy_defaults(self):
        """Test LocalSourcePolicy defaults."""
        policy = LocalSourcePolicy()

        assert policy.enabled is True
        assert policy.k == 12
        assert policy.k_rerank == 6
        assert policy.min_relevance == 0.25

    def test_remote_source_policy_defaults(self):
        """Test RemoteSourcePolicy defaults."""
        policy = RemoteSourcePolicy()

        assert policy.enabled is False
        assert policy.rank_strategy == "weighted"
        assert policy.max_results == 5

    def test_model_knowledge_policy_defaults(self):
        """Test ModelKnowledgePolicy defaults."""
        policy = ModelKnowledgePolicy()

        assert policy.enabled is False
        assert policy.require_label is True

    def test_source_policy_config_composition(self):
        """Test SourcePolicyConfig composes all policy types."""
        source_policy = SourcePolicyConfig(
            local=LocalSourcePolicy(k=20),
            remote=RemoteSourcePolicy(enabled=True, max_results=10),
            model_knowledge=ModelKnowledgePolicy(enabled=True)
        )

        assert source_policy.local.k == 20
        assert source_policy.remote.enabled is True
        assert source_policy.remote.max_results == 10
        assert source_policy.model_knowledge.enabled is True


class TestNotesConfig:
    """Tests for NotesConfig."""

    def test_notes_config_defaults(self):
        """Test NotesConfig defaults."""
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
        """Test NotesConfig with custom values."""
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

        assert notes.enabled is True
        assert notes.dir == "research_notes"
        assert notes.template == "research"
        assert notes.filename_format == "query_slug"
        assert notes.integration == "obsidian"
        assert notes.wikilinks is True
        assert notes.backlinks is True
        assert notes.include_tags is True


class TestBuiltInModes:
    """Tests for built-in query modes."""

    def test_grounded_mode_config(self):
        """Test built-in grounded mode configuration."""
        config = _make_config("grounded")

        mode = config.get_mode_config()

        assert mode.synthesis_style == "grounded"
        assert mode.source_policy.local is not None
        assert mode.source_policy.remote.enabled is False
        assert mode.source_policy.model_knowledge.enabled is False

    def test_insight_mode_config(self):
        """Test built-in insight mode configuration."""
        config = _make_config("insight")

        mode = config.get_mode_config()

        assert mode.synthesis_style == "insight"
        assert mode.source_policy.local is not None
        # Insight mode should allow some flexibility
        assert mode.source_policy.model_knowledge.enabled is False  # Still conservative by default

    def test_hybrid_mode_config(self):
        """Test built-in hybrid mode configuration."""
        config = _make_config("hybrid")

        mode = config.get_mode_config()

        assert mode.synthesis_style == "grounded"
        assert mode.source_policy.local is not None
        assert mode.source_policy.remote.enabled is True
        assert mode.source_policy.model_knowledge.enabled is True


class TestCustomModes:
    """Tests for custom user-defined modes."""

    def test_custom_mode_registration(self):
        """Test registering a custom mode."""
        custom_mode = ModeConfig(
            synthesis_style="grounded",
            source_policy=SourcePolicyConfig(
                local=LocalSourcePolicy(k=20, min_relevance=0.15),
                remote=RemoteSourcePolicy(enabled=True, max_results=15),
                model_knowledge=ModelKnowledgePolicy(enabled=True, require_label=False)
            ),
            notes=NotesConfig(enabled=True, dir="custom_notes")
        )

        config = _make_config("my_custom_mode", modes={"my_custom_mode": custom_mode})

        mode = config.get_mode_config()

        assert mode.synthesis_style == "grounded"
        assert mode.source_policy.local.k == 20
        assert mode.source_policy.remote.max_results == 15
        assert mode.notes.dir == "custom_notes"

    def test_mode_fallback_to_default(self):
        """Test mode falls back to default when not found."""
        config = _make_config("nonexistent_mode")

        # Should fall back to grounded mode
        mode = config.get_mode_config()

        assert mode.synthesis_style == "grounded"


class TestModeSourceBlending:
    """Tests for source blending in different modes."""

    def test_grounded_mode_uses_only_local(self):
        """Test grounded mode uses only local sources."""
        config = _make_config("grounded")

        mode = config.get_mode_config()
        policy = mode.source_policy

        assert policy.local is not None
        assert policy.remote.enabled is False
        assert policy.model_knowledge.enabled is False

    def test_hybrid_mode_blends_all_sources(self):
        """Test hybrid mode enables all source types."""
        config = _make_config("hybrid")

        mode = config.get_mode_config()
        policy = mode.source_policy

        assert policy.local is not None
        assert policy.remote.enabled is True
        assert policy.model_knowledge.enabled is True

    def test_custom_source_blend(self):
        """Test custom source blending configuration."""
        custom_mode = ModeConfig(
            synthesis_style="grounded",
            source_policy=SourcePolicyConfig(
                local=LocalSourcePolicy(k=10),
                remote=RemoteSourcePolicy(enabled=True, max_results=5),
                model_knowledge=ModelKnowledgePolicy(enabled=False)
            )
        )

        config = _make_config("custom", modes={"custom": custom_mode})

        mode = config.get_mode_config()
        policy = mode.source_policy

        # Local + Remote, but not model knowledge
        assert policy.local is not None
        assert policy.remote.enabled is True
        assert policy.model_knowledge.enabled is False


class TestModeWithNotes:
    """Tests for modes with notes enabled."""

    def test_mode_with_notes_enabled(self):
        """Test mode with notes enabled."""
        mode_with_notes = ModeConfig(
            synthesis_style="grounded",
            source_policy=SourcePolicyConfig(),
            notes=NotesConfig(
                enabled=True,
                dir="research_notes",
                filename_format="query_slug"
            )
        )

        config = _make_config("research", modes={"research": mode_with_notes})

        mode = config.get_mode_config()

        assert mode.notes.enabled is True
        assert mode.notes.dir == "research_notes"
        assert mode.notes.filename_format == "query_slug"

    def test_mode_notes_enabled_by_default(self):
        """Test notes are enabled by default."""
        config = _make_config("grounded")

        mode = config.get_mode_config()

        assert mode.notes.enabled is True

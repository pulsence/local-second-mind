"""Tests for embedding model validation and well-known models dictionary."""

from __future__ import annotations

import pytest

from lsm.config.models.constants import DEFAULT_EMBED_MODEL, WELL_KNOWN_EMBED_MODELS
from lsm.config.models.global_config import GlobalConfig


class TestWellKnownModels:
    """Tests for the WELL_KNOWN_EMBED_MODELS dictionary."""

    def test_is_non_empty(self) -> None:
        assert len(WELL_KNOWN_EMBED_MODELS) > 0

    def test_all_values_are_positive_ints(self) -> None:
        for model, dim in WELL_KNOWN_EMBED_MODELS.items():
            assert isinstance(dim, int), f"{model} dimension is not int: {type(dim)}"
            assert dim > 0, f"{model} has non-positive dimension: {dim}"

    def test_default_model_is_present(self) -> None:
        assert DEFAULT_EMBED_MODEL in WELL_KNOWN_EMBED_MODELS

    def test_default_model_dimension(self) -> None:
        assert WELL_KNOWN_EMBED_MODELS[DEFAULT_EMBED_MODEL] == 384

    def test_all_keys_are_strings(self) -> None:
        for key in WELL_KNOWN_EMBED_MODELS:
            assert isinstance(key, str)


class TestGlobalConfigAutoDetection:
    """Tests for embedding dimension auto-detection in GlobalConfig."""

    def test_auto_detects_known_model(self) -> None:
        config = GlobalConfig(embed_model="sentence-transformers/all-MiniLM-L6-v2")
        assert config.embedding_dimension == 384

    def test_auto_detects_large_model(self) -> None:
        config = GlobalConfig(embed_model="intfloat/e5-large-v2")
        assert config.embedding_dimension == 1024

    def test_auto_detects_multilingual_model(self) -> None:
        config = GlobalConfig(
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        assert config.embedding_dimension == 384

    def test_auto_detects_768_dim_model(self) -> None:
        config = GlobalConfig(embed_model="sentence-transformers/all-mpnet-base-v2")
        assert config.embedding_dimension == 768


class TestGlobalConfigUnknownModel:
    """Tests for unknown model handling."""

    def test_unknown_model_leaves_dimension_none(self) -> None:
        config = GlobalConfig(embed_model="some/unknown-model")
        assert config.embedding_dimension is None

    def test_custom_model_no_auto_detect(self) -> None:
        config = GlobalConfig(embed_model="my-custom-fine-tuned-model")
        assert config.embedding_dimension is None


class TestGlobalConfigExplicitDimension:
    """Tests for explicit embedding_dimension configuration."""

    def test_explicit_dimension_honored(self) -> None:
        config = GlobalConfig(
            embed_model="some/unknown-model",
            embedding_dimension=512,
        )
        assert config.embedding_dimension == 512

    def test_explicit_dimension_overrides_auto_detect(self) -> None:
        config = GlobalConfig(
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=512,
        )
        # Explicit value should be kept, not overridden by auto-detect
        assert config.embedding_dimension == 512

    def test_explicit_none_triggers_auto_detect(self) -> None:
        config = GlobalConfig(
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=None,
        )
        assert config.embedding_dimension == 384


class TestGlobalConfigValidation:
    """Tests for embedding dimension validation."""

    def test_negative_dimension_rejected(self) -> None:
        config = GlobalConfig(
            embed_model="some/model",
            embedding_dimension=-1,
        )
        with pytest.raises(ValueError, match="embedding_dimension"):
            config.validate()

    def test_zero_dimension_rejected(self) -> None:
        config = GlobalConfig(
            embed_model="some/model",
            embedding_dimension=0,
        )
        with pytest.raises(ValueError, match="embedding_dimension"):
            config.validate()

    def test_positive_dimension_accepted(self) -> None:
        config = GlobalConfig(
            embed_model="some/model",
            embedding_dimension=768,
        )
        config.validate()  # Should not raise

    def test_none_dimension_accepted(self) -> None:
        config = GlobalConfig(embed_model="some/unknown-model")
        config.validate()  # Should not raise

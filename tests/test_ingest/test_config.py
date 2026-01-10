"""
Tests for lsm.ingest.config module.

Tests configuration loading, normalization, and defaults.
"""

import pytest
import json
import yaml
from pathlib import Path
from lsm.ingest.config import (
    load_config,
    normalize_config,
    DEFAULT_EXTS,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_json_config(self, sample_config_file):
        """Test loading a JSON config file."""
        config = load_config(sample_config_file)

        assert isinstance(config, dict)
        assert "roots" in config
        assert "collection" in config

    def test_load_yaml_config(self, tmp_path, sample_config_dict):
        """Test loading a YAML config file."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(sample_config_dict))

        config = load_config(yaml_path)

        assert isinstance(config, dict)
        assert "roots" in config
        assert config["collection"] == "test_kb"

    def test_load_yml_extension(self, tmp_path, sample_config_dict):
        """Test loading a .yml file."""
        yml_path = tmp_path / "config.yml"
        yml_path.write_text(yaml.dump(sample_config_dict))

        config = load_config(yml_path)

        assert isinstance(config, dict)

    def test_load_nonexistent_config(self, tmp_path):
        """Test loading a config file that doesn't exist."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_config(nonexistent)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            load_config(invalid_json)

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            load_config(invalid_yaml)

    def test_load_empty_json(self, tmp_path):
        """Test loading an empty JSON file."""
        empty_json = tmp_path / "empty.json"
        empty_json.write_text("{}")

        config = load_config(empty_json)

        assert config == {}

    def test_load_dotenv_called(self, tmp_path, mocker):
        """Test that load_dotenv is called."""
        # Mock load_dotenv to verify it's called
        mock_load_dotenv = mocker.patch("lsm.ingest.config.load_dotenv")

        config_path = tmp_path / "config.json"
        config_path.write_text('{"roots": ["/tmp"]}')

        load_config(config_path)

        # Verify load_dotenv was called
        mock_load_dotenv.assert_called_once()


class TestNormalizeConfig:
    """Tests for normalize_config function."""

    def test_normalize_config_basic(self, sample_config_file, sample_config_dict):
        """Test basic config normalization."""
        normalized = normalize_config(sample_config_dict, sample_config_file)

        assert isinstance(normalized, dict)
        assert "roots" in normalized
        assert "collection" in normalized
        assert "exts" in normalized

    def test_normalize_config_applies_defaults(self, tmp_path):
        """Test that defaults are applied."""
        minimal_config = {"roots": [str(tmp_path)]}
        config_path = tmp_path / "config.json"

        normalized = normalize_config(minimal_config, config_path)

        # Should have defaults
        assert normalized["collection"] == DEFAULT_COLLECTION
        assert normalized["embed_model"] == DEFAULT_EMBED_MODEL
        assert normalized["device"] == "cpu"
        assert normalized["batch_size"] == 32

    def test_normalize_config_missing_roots(self, tmp_path):
        """Test normalization fails without roots."""
        config_path = tmp_path / "config.json"
        invalid_config = {"collection": "test"}

        with pytest.raises(ValueError, match="roots"):
            normalize_config(invalid_config, config_path)

    def test_normalize_config_empty_roots(self, tmp_path):
        """Test normalization fails with empty roots list."""
        config_path = tmp_path / "config.json"
        invalid_config = {"roots": []}

        with pytest.raises(ValueError, match="roots"):
            normalize_config(invalid_config, config_path)

    def test_normalize_config_converts_paths(self, tmp_path):
        """Test that paths are converted to Path objects."""
        config = {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": ".chroma",
            "manifest": ".ingest/manifest.json"
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # Roots should be list of Path objects
        assert all(isinstance(r, Path) for r in normalized["roots"])

        # persist_dir and manifest should be resolved Paths
        assert isinstance(normalized["persist_dir"], Path)
        assert isinstance(normalized["manifest"], Path)

    def test_normalize_config_resolves_relative_paths(self, tmp_path):
        """Test that relative paths are resolved relative to config file."""
        config = {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": ".chroma"
        }
        config_path = tmp_path / "subdir" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        normalized = normalize_config(config, config_path)

        # persist_dir should be resolved relative to config file location
        expected_persist = (tmp_path / "subdir" / ".chroma").resolve()
        assert normalized["persist_dir"] == expected_persist

    def test_normalize_config_extensions_merge(self, tmp_path):
        """Test that extensions are merged with defaults."""
        config = {
            "roots": [str(tmp_path)],
            "extensions": [".custom"]
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # Should include both default and custom extensions
        assert ".txt" in normalized["exts"]  # Default
        assert ".custom" in normalized["exts"]  # Custom

    def test_normalize_config_extensions_override(self, tmp_path):
        """Test that override_extensions replaces defaults."""
        config = {
            "roots": [str(tmp_path)],
            "extensions": [".custom"],
            "override_extensions": True
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # Should only have custom extensions
        assert ".custom" in normalized["exts"]
        assert ".txt" not in normalized["exts"]  # Default should be excluded

    def test_normalize_config_exclude_dirs_merge(self, tmp_path):
        """Test that exclude_dirs are merged with defaults."""
        config = {
            "roots": [str(tmp_path)],
            "exclude_dirs": ["custom_exclude"]
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # Should include both default and custom excludes
        assert ".git" in normalized["exclude_dirs"]  # Default
        assert "custom_exclude" in normalized["exclude_dirs"]  # Custom

    def test_normalize_config_exclude_dirs_override(self, tmp_path):
        """Test that override_excludes replaces defaults."""
        config = {
            "roots": [str(tmp_path)],
            "exclude_dirs": ["custom_exclude"],
            "override_excludes": True
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # Should only have custom excludes
        assert "custom_exclude" in normalized["exclude_dirs"]
        assert ".git" not in normalized["exclude_dirs"]  # Default excluded

    def test_normalize_config_normalizes_extensions(self, tmp_path):
        """Test that extensions are normalized (lowercased, dot-prefixed)."""
        config = {
            "roots": [str(tmp_path)],
            "extensions": ["TXT", "MD", "pdf", ".docx"],  # Mixed formats
            "override_extensions": True
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # All should be lowercase with leading dot
        assert ".txt" in normalized["exts"]
        assert ".md" in normalized["exts"]
        assert ".pdf" in normalized["exts"]
        assert ".docx" in normalized["exts"]

        # No uppercase or missing dots
        assert "TXT" not in normalized["exts"]
        assert "pdf" not in normalized["exts"]

    def test_normalize_config_type_conversions(self, tmp_path):
        """Test that values are converted to correct types."""
        config = {
            "roots": [str(tmp_path)],
            "batch_size": "64",  # String instead of int
            "dry_run": "true",  # String instead of bool
            "chroma_flush_interval": "5000"
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # Should be converted to correct types
        assert normalized["batch_size"] == 64
        assert normalized["batch_size"] == "64"  # Python's int() accepts strings
        assert isinstance(normalized["dry_run"], bool)
        assert normalized["chroma_flush_interval"] == 5000


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_extensions_include_common_formats(self):
        """Test that default extensions include common formats."""
        assert ".txt" in DEFAULT_EXTS
        assert ".md" in DEFAULT_EXTS
        assert ".pdf" in DEFAULT_EXTS
        assert ".docx" in DEFAULT_EXTS
        assert ".html" in DEFAULT_EXTS

    def test_default_exclude_dirs_include_common_patterns(self):
        """Test that default excludes include common patterns."""
        assert ".git" in DEFAULT_EXCLUDE_DIRS
        assert "__pycache__" in DEFAULT_EXCLUDE_DIRS
        assert "node_modules" in DEFAULT_EXCLUDE_DIRS
        assert ".venv" in DEFAULT_EXCLUDE_DIRS

    def test_default_collection_name(self):
        """Test default collection name."""
        assert DEFAULT_COLLECTION == "local_kb"

    def test_default_embed_model(self):
        """Test default embedding model."""
        assert "sentence-transformers" in DEFAULT_EMBED_MODEL
        assert "MiniLM" in DEFAULT_EMBED_MODEL


class TestConfigEdgeCases:
    """Tests for edge cases in configuration."""

    def test_config_with_null_values(self, tmp_path):
        """Test config with null values."""
        config = {
            "roots": [str(tmp_path)],
            "extensions": None,
            "exclude_dirs": None
        }
        config_path = tmp_path / "config.json"

        normalized = normalize_config(config, config_path)

        # Should use defaults for null values
        assert len(normalized["exts"]) > 0
        assert len(normalized["exclude_dirs"]) > 0

    def test_config_with_extra_fields(self, tmp_path):
        """Test that extra fields don't cause errors."""
        config = {
            "roots": [str(tmp_path)],
            "extra_field": "extra_value",
            "another_extra": 123
        }
        config_path = tmp_path / "config.json"

        # Should not raise error
        normalized = normalize_config(config, config_path)

        assert "roots" in normalized

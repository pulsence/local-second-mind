"""Tests for partial ingest limits (max_files, max_seconds)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lsm.config.models.ingest import IngestConfig, RootConfig


# ------------------------------------------------------------------
# Config field defaults
# ------------------------------------------------------------------


class TestPartialIngestConfigDefaults:
    def test_max_files_default_none(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)])
        assert cfg.max_files is None

    def test_max_seconds_default_none(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)])
        assert cfg.max_seconds is None

    def test_max_files_set(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_files=10)
        assert cfg.max_files == 10

    def test_max_seconds_set(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_seconds=60)
        assert cfg.max_seconds == 60


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestPartialIngestValidation:
    def test_max_files_zero_rejected(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_files=0)
        with pytest.raises(ValueError, match="max_files"):
            cfg.validate()

    def test_max_files_negative_rejected(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_files=-1)
        with pytest.raises(ValueError, match="max_files"):
            cfg.validate()

    def test_max_seconds_zero_rejected(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_seconds=0)
        with pytest.raises(ValueError, match="max_seconds"):
            cfg.validate()

    def test_max_seconds_negative_rejected(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_seconds=-5)
        with pytest.raises(ValueError, match="max_seconds"):
            cfg.validate()

    def test_max_files_none_accepted(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_files=None)
        cfg.validate()  # should not raise

    def test_max_seconds_none_accepted(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_seconds=None)
        cfg.validate()  # should not raise

    def test_max_files_positive_accepted(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], max_files=1)
        cfg.validate()  # should not raise


# ------------------------------------------------------------------
# Config loader round-trip
# ------------------------------------------------------------------


class TestPartialIngestLoader:
    def test_build_ingest_config_reads_limits(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_ingest_config

        raw = {
            "ingest": {
                "roots": [str(tmp_path)],
                "max_files": 42,
                "max_seconds": 300,
            }
        }
        cfg = build_ingest_config(raw, tmp_path)
        assert cfg.max_files == 42
        assert cfg.max_seconds == 300

    def test_build_ingest_config_defaults_none(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_ingest_config

        raw = {"ingest": {"roots": [str(tmp_path)]}}
        cfg = build_ingest_config(raw, tmp_path)
        assert cfg.max_files is None
        assert cfg.max_seconds is None

    def test_config_to_raw_includes_limits(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_config_from_raw, config_to_raw

        raw = {
            "global": {
                "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "ingest": {
                "roots": [str(tmp_path)],
                "max_files": 10,
                "max_seconds": 120,
            },
            "vectordb": {},
            "llms": {
                "providers": [
                    {"provider_name": "local"},
                ],
                "services": {
                    "default": {"provider": "local", "model": "test"},
                },
            },
            "query": {},
        }
        config = build_config_from_raw(raw, tmp_path)
        out = config_to_raw(config)
        assert out["ingest"]["max_files"] == 10
        assert out["ingest"]["max_seconds"] == 120

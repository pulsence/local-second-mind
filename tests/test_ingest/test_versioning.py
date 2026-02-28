"""Tests for always-on chunk version metadata and manifest version helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lsm.ingest.manifest import get_next_version


class TestVersioningConfig:
    def test_loader_rejects_legacy_enable_versioning_field(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_ingest_config

        raw = {"ingest": {"roots": [str(tmp_path)], "enable_versioning": True}}
        with pytest.raises(ValueError, match="Unsupported legacy ingest field 'enable_versioning'"):
            build_ingest_config(raw, tmp_path)

    def test_config_to_raw_does_not_emit_enable_versioning(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_config_from_raw, config_to_raw

        raw = {
            "global": {"embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
            "ingest": {"roots": [str(tmp_path)]},
            "vectordb": {"provider": "sqlite", "path": str(tmp_path / "data")},
            "llms": {
                "providers": [{"provider_name": "local"}],
                "services": {"default": {"provider": "local", "model": "m"}},
            },
            "query": {},
        }
        config = build_config_from_raw(raw, tmp_path / "config.json")
        out = config_to_raw(config)
        assert "enable_versioning" not in out["ingest"]


class TestGetNextVersion:
    def test_new_file_returns_1(self) -> None:
        assert get_next_version({}, "path/to/file.pdf") == 1

    def test_existing_no_version_field_returns_1(self) -> None:
        manifest = {"file.pdf": {"file_hash": "abc"}}
        assert get_next_version(manifest, "file.pdf") == 1

    def test_increment(self) -> None:
        manifest = {"file.pdf": {"file_hash": "abc", "version": 3}}
        assert get_next_version(manifest, "file.pdf") == 4

    def test_version_zero_increments_to_1(self) -> None:
        manifest = {"file.pdf": {"version": 0}}
        assert get_next_version(manifest, "file.pdf") == 1

    def test_database_next_version(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(str(tmp_path / "lsm.db"))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lsm_manifest (
                source_path TEXT PRIMARY KEY,
                version INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO lsm_manifest(source_path, version) VALUES (?, ?)",
            ("/tmp/file.txt", 4),
        )
        conn.commit()
        assert get_next_version({}, "/tmp/file.txt", connection=conn) == 5
        assert get_next_version({}, "/tmp/new.txt", connection=conn) == 1


class TestVersioningModels:
    def test_parse_result_version_default(self) -> None:
        from lsm.ingest.models import ParseResult

        pr = ParseResult(
            source_path="a.pdf",
            fp=Path("a.pdf"),
            mtime_ns=0,
            size=0,
            file_hash="h",
            chunks=[],
            ext=".pdf",
            had_prev=False,
            ok=True,
        )
        assert pr.version == 1

    def test_write_job_version_default(self) -> None:
        from lsm.ingest.models import WriteJob

        wj = WriteJob(
            source_path="a.pdf",
            fp=Path("a.pdf"),
            mtime_ns=0,
            size=0,
            file_hash="h",
            ext=".pdf",
            chunks=[],
            embeddings=[],
            had_prev=False,
        )
        assert wj.version == 1

"""Tests for chunk version control (enable_versioning)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from lsm.config.models.ingest import IngestConfig, RootConfig
from lsm.ingest.manifest import get_next_version


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


class TestVersioningConfig:
    def test_default_false(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)])
        assert cfg.enable_versioning is False

    def test_enable(self, tmp_path: Path) -> None:
        cfg = IngestConfig(roots=[RootConfig(path=tmp_path)], enable_versioning=True)
        assert cfg.enable_versioning is True

    def test_loader_roundtrip(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_ingest_config

        raw = {"ingest": {"roots": [str(tmp_path)], "enable_versioning": True}}
        cfg = build_ingest_config(raw, tmp_path)
        assert cfg.enable_versioning is True

    def test_config_to_raw(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_config_from_raw, config_to_raw

        raw = {
            "global": {"embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
            "ingest": {"roots": [str(tmp_path)], "enable_versioning": True},
            "vectordb": {},
            "llms": {
                "providers": [{"provider_name": "local"}],
                "services": {"default": {"provider": "local", "model": "m"}},
            },
            "query": {},
        }
        config = build_config_from_raw(raw, tmp_path)
        out = config_to_raw(config)
        assert out["ingest"]["enable_versioning"] is True


# ------------------------------------------------------------------
# Manifest get_next_version
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class TestVersioningModels:
    def test_parse_result_version_default(self) -> None:
        from lsm.ingest.models import ParseResult

        pr = ParseResult(
            source_path="a.pdf", fp=Path("a.pdf"), mtime_ns=0, size=0,
            file_hash="h", chunks=[], ext=".pdf", had_prev=False, ok=True,
        )
        assert pr.version == 1

    def test_write_job_version_default(self) -> None:
        from lsm.ingest.models import WriteJob

        wj = WriteJob(
            source_path="a.pdf", fp=Path("a.pdf"), mtime_ns=0, size=0,
            file_hash="h", ext=".pdf", chunks=[], embeddings=[], had_prev=False,
        )
        assert wj.version == 1


# ------------------------------------------------------------------
# VectorDB provider methods
# ------------------------------------------------------------------


class TestVectorDBProviderVersioningMethods:
    def test_base_update_metadatas_raises(self) -> None:
        from lsm.vectordb.base import BaseVectorDBProvider

        # BaseVectorDBProvider is ABC so can't instantiate; test via subclass
        class _Stub(BaseVectorDBProvider):
            name = "stub"
            def is_available(self): return True
            def add_chunks(self, *a, **kw): pass
            def query(self, *a, **kw): pass
            def delete_by_id(self, *a, **kw): pass
            def delete_by_filter(self, *a, **kw): pass
            def count(self): return 0
            def get_stats(self): return {}
            def optimize(self): return {}
            def health_check(self): return {}

        from lsm.config.models.vectordb import VectorDBConfig
        stub = _Stub(VectorDBConfig())
        with pytest.raises(NotImplementedError):
            stub.update_metadatas(["id1"], [{"k": "v"}])

    def test_base_get_by_filter_raises(self) -> None:
        from lsm.vectordb.base import BaseVectorDBProvider
        from lsm.config.models.vectordb import VectorDBConfig

        class _Stub(BaseVectorDBProvider):
            name = "stub"
            def is_available(self): return True
            def add_chunks(self, *a, **kw): pass
            def query(self, *a, **kw): pass
            def delete_by_id(self, *a, **kw): pass
            def delete_by_filter(self, *a, **kw): pass
            def count(self): return 0
            def get_stats(self): return {}
            def optimize(self): return {}
            def health_check(self): return {}

        stub = _Stub(VectorDBConfig())
        with pytest.raises(NotImplementedError):
            stub.get_by_filter({"source_path": "x"})


# ------------------------------------------------------------------
# Query retrieval where_filter
# ------------------------------------------------------------------


class TestRetrieveCandidatesWhereFilter:
    def test_where_filter_passed_to_provider(self) -> None:
        from lsm.query.retrieval import retrieve_candidates
        from lsm.vectordb.base import BaseVectorDBProvider, VectorDBQueryResult
        from unittest.mock import MagicMock

        mock_provider = MagicMock(spec=BaseVectorDBProvider)
        mock_provider.query.return_value = VectorDBQueryResult(
            ids=["c1"], documents=["text"], metadatas=[{"is_current": True}],
            distances=[0.1],
        )

        candidates = retrieve_candidates(
            mock_provider, [0.0] * 384, k=5, where_filter={"is_current": True},
        )
        assert len(candidates) == 1
        # Verify filter was passed
        mock_provider.query.assert_called_once_with([0.0] * 384, top_k=5, filters={"is_current": True})

    def test_where_filter_none_by_default(self) -> None:
        from lsm.query.retrieval import retrieve_candidates
        from lsm.vectordb.base import BaseVectorDBProvider, VectorDBQueryResult
        from unittest.mock import MagicMock

        mock_provider = MagicMock(spec=BaseVectorDBProvider)
        mock_provider.query.return_value = VectorDBQueryResult(
            ids=[], documents=[], metadatas=[], distances=[],
        )

        retrieve_candidates(mock_provider, [0.0] * 384, k=5)
        mock_provider.query.assert_called_once_with([0.0] * 384, top_k=5, filters=None)

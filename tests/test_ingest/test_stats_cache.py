"""Tests for DB-backed stats caching."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

from lsm.ingest.stats_cache import StatsCache


_SAMPLE_STATS = {
    "total_chunks": 1000,
    "unique_files": 50,
    "file_types": {".pdf": 600, ".md": 300, ".txt": 100},
    "top_files": {"file1.pdf": 80, "file2.md": 60},
    "avg_chunks_per_file": 20.0,
    "max_chunks_per_file": 80,
    "min_chunks_per_file": 2,
    "analyzed_chunks": 1000,
    "analysis_mode": "full",
}


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "runtime"


def _cache(tmp_path: Path, *, max_age_seconds: int = 3600) -> StatsCache:
    return StatsCache(
        db_path=_db_path(tmp_path),
        cache_key="test_collection_stats",
        max_age_seconds=max_age_seconds,
    )


def test_db_cache_load_missing_returns_none(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    assert cache.load() is None


def test_db_cache_save_and_load_roundtrip(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.save(_SAMPLE_STATS, chunk_count=1000)

    loaded = cache.load()
    assert loaded is not None
    assert loaded["stats"]["total_chunks"] == 1000
    assert loaded["chunk_count"] == 1000
    assert "cached_at" in loaded

    db_path = _db_path(tmp_path) / "lsm.db"
    assert db_path.exists()
    assert not (tmp_path / "stats_cache.json").exists()

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM lsm_stats_cache").fetchone()[0]
    conn.close()
    assert count == 1


def test_db_cache_staleness_checks_count_and_age(tmp_path: Path) -> None:
    cache = _cache(tmp_path, max_age_seconds=3600)
    cache.save(_SAMPLE_STATS, chunk_count=1000)
    assert cache.is_stale(1000) is False
    assert cache.is_stale(999) is True

    immediate_stale = _cache(tmp_path, max_age_seconds=0)
    immediate_stale.save(_SAMPLE_STATS, chunk_count=1000)
    time.sleep(0.01)
    assert immediate_stale.is_stale(1000) is True


def test_db_cache_get_if_fresh_and_invalidate(tmp_path: Path) -> None:
    cache = _cache(tmp_path, max_age_seconds=3600)
    cache.save(_SAMPLE_STATS, chunk_count=1000)
    assert cache.get_if_fresh(1000) == _SAMPLE_STATS
    assert cache.get_if_fresh(999) is None

    cache.invalidate()
    assert cache.load() is None


def test_stats_integration_cache_hit_avoids_full_scan(tmp_path: Path) -> None:
    from lsm.ingest.stats import get_collection_stats

    cache = _cache(tmp_path)
    cache.save(_SAMPLE_STATS, chunk_count=1000)

    mock_provider = MagicMock()
    mock_provider.count.return_value = 1000

    stats = get_collection_stats(mock_provider, stats_cache=cache)
    assert stats["total_chunks"] == 1000
    assert stats["unique_files"] == 50
    mock_provider.get.assert_not_called()


def test_stats_integration_recomputes_on_stale_cache(tmp_path: Path) -> None:
    from lsm.ingest.stats import get_collection_stats
    from lsm.vectordb.base import VectorDBGetResult

    cache = _cache(tmp_path)
    cache.save(_SAMPLE_STATS, chunk_count=500)

    mock_provider = MagicMock()
    mock_provider.count.return_value = 2
    mock_provider.get.side_effect = [
        VectorDBGetResult(
            ids=["a", "b"],
            metadatas=[
                {"source_path": "/a.pdf", "ext": ".pdf", "ingested_at": "2026-01-01"},
                {"source_path": "/b.md", "ext": ".md", "ingested_at": "2026-01-02"},
            ],
        ),
        VectorDBGetResult(ids=[], metadatas=[]),
    ]

    stats = get_collection_stats(mock_provider, stats_cache=cache)
    assert stats["total_chunks"] == 2

    fresh = _cache(tmp_path)
    assert fresh.is_stale(2) is False

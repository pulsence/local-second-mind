"""Tests for stats caching (StatsCache)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lsm.ingest.stats_cache import StatsCache


# ------------------------------------------------------------------
# Sample stats for testing
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# load / save
# ------------------------------------------------------------------


class TestStatsCacheLoadSave:
    def test_load_missing_file(self, tmp_path: Path) -> None:
        cache = StatsCache(tmp_path / "nonexistent.json")
        assert cache.load() is None

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        p = tmp_path / "corrupt.json"
        p.write_text("{bad json", encoding="utf-8")
        cache = StatsCache(p)
        assert cache.load() is None

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "stats_cache.json"
        cache = StatsCache(p)
        cache.save(_SAMPLE_STATS, chunk_count=1000)

        loaded = cache.load()
        assert loaded is not None
        assert loaded["stats"]["total_chunks"] == 1000
        assert loaded["chunk_count"] == 1000
        assert "cached_at" in loaded

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "stats_cache.json"
        cache = StatsCache(p)
        cache.save(_SAMPLE_STATS, chunk_count=500)
        assert p.exists()


# ------------------------------------------------------------------
# is_stale
# ------------------------------------------------------------------


class TestStatsCacheIsStaleFresh:
    def test_stale_when_no_file(self, tmp_path: Path) -> None:
        cache = StatsCache(tmp_path / "missing.json")
        assert cache.is_stale(100) is True

    def test_stale_when_count_changed(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        cache = StatsCache(p)
        cache.save(_SAMPLE_STATS, chunk_count=1000)
        assert cache.is_stale(1001) is True

    def test_stale_when_expired(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        cache = StatsCache(p, max_age_seconds=0)
        cache.save(_SAMPLE_STATS, chunk_count=1000)
        # max_age_seconds=0 means immediately stale
        time.sleep(0.01)
        assert cache.is_stale(1000) is True

    def test_fresh_when_count_matches(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        cache = StatsCache(p, max_age_seconds=3600)
        cache.save(_SAMPLE_STATS, chunk_count=1000)
        assert cache.is_stale(1000) is False


# ------------------------------------------------------------------
# get_if_fresh
# ------------------------------------------------------------------


class TestStatsCacheGetIfFresh:
    def test_returns_stats_when_fresh(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        cache = StatsCache(p, max_age_seconds=3600)
        cache.save(_SAMPLE_STATS, chunk_count=1000)

        result = cache.get_if_fresh(1000)
        assert result is not None
        assert result["total_chunks"] == 1000
        assert result["unique_files"] == 50

    def test_returns_none_when_stale(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        cache = StatsCache(p, max_age_seconds=3600)
        cache.save(_SAMPLE_STATS, chunk_count=1000)

        assert cache.get_if_fresh(999) is None

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        cache = StatsCache(tmp_path / "missing.json")
        assert cache.get_if_fresh(0) is None


# ------------------------------------------------------------------
# invalidate
# ------------------------------------------------------------------


class TestStatsCacheInvalidate:
    def test_invalidate_deletes_file(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.json"
        cache = StatsCache(p)
        cache.save(_SAMPLE_STATS, chunk_count=1000)
        assert p.exists()

        cache.invalidate()
        assert not p.exists()

    def test_invalidate_missing_file_no_error(self, tmp_path: Path) -> None:
        cache = StatsCache(tmp_path / "missing.json")
        cache.invalidate()  # should not raise


# ------------------------------------------------------------------
# Integration with get_collection_stats
# ------------------------------------------------------------------


class TestStatsCacheIntegration:
    def test_cache_hit_avoids_full_scan(self, tmp_path: Path) -> None:
        """When cache is fresh, get_collection_stats returns cached stats."""
        from lsm.ingest.stats import get_collection_stats

        # Create a pre-populated cache
        cache_path = tmp_path / "stats_cache.json"
        cache = StatsCache(cache_path)
        cache.save(_SAMPLE_STATS, chunk_count=1000)

        # Mock collection that should NOT be scanned
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1000
        mock_collection.name = "test_collection"

        stats = get_collection_stats(mock_collection, cache_path=cache_path)
        assert stats["total_chunks"] == 1000
        assert stats["unique_files"] == 50
        # iter_collection_metadatas should not have been called
        mock_collection.get.assert_not_called()

    def test_recomputes_on_stale_cache(self, tmp_path: Path) -> None:
        """When cache is stale, get_collection_stats recomputes stats."""
        from lsm.ingest.stats import get_collection_stats

        # Create a cache with different count
        cache_path = tmp_path / "stats_cache.json"
        cache = StatsCache(cache_path)
        cache.save(_SAMPLE_STATS, chunk_count=500)  # different from actual count

        # Mock collection with actual data
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.name = "test_collection"
        mock_collection.get.return_value = {
            "ids": ["a", "b"],
            "metadatas": [
                {"source_path": "/a.pdf", "ext": ".pdf", "ingested_at": "2026-01-01"},
                {"source_path": "/b.md", "ext": ".md", "ingested_at": "2026-01-02"},
            ],
        }

        stats = get_collection_stats(mock_collection, cache_path=cache_path)
        assert stats["total_chunks"] == 2
        # Cache should have been updated
        new_cache = StatsCache(cache_path)
        assert new_cache.is_stale(2) is False

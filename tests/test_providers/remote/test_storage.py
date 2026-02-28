from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from lsm.remote.storage import (
    load_cached_results,
    load_feed_cache,
    save_feed_cache,
    save_results,
)


def test_save_and_load_cached_results_roundtrip_db(tmp_path: Path) -> None:
    vectordb_path = tmp_path / "data"
    results = [
        {
            "title": "Example",
            "url": "https://example.com",
            "snippet": "example snippet",
            "score": 0.9,
            "metadata": {"source": "test"},
        }
    ]

    cache_path = save_results(
        provider_name="web search",
        query="test query",
        results=results,
        global_folder=tmp_path,
        vectordb_path=vectordb_path,
    )

    assert cache_path == vectordb_path / "lsm.db"
    assert cache_path.exists()
    assert not (tmp_path / "Downloads").exists()

    loaded = load_cached_results(
        provider_name="web search",
        query="test query",
        global_folder=tmp_path,
        max_age=3600,
        vectordb_path=vectordb_path,
    )
    assert loaded == results


def test_load_cached_results_returns_none_when_expired_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vectordb_path = tmp_path / "data"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("lsm.remote.storage._now_utc", lambda: now)

    save_results(
        provider_name="wikipedia",
        query="cached query",
        results=[{"title": "Old", "url": "https://old.example"}],
        global_folder=tmp_path,
        vectordb_path=vectordb_path,
        cache_ttl_seconds=60,
    )

    monkeypatch.setattr("lsm.remote.storage._now_utc", lambda: now + timedelta(seconds=120))
    loaded = load_cached_results(
        provider_name="wikipedia",
        query="cached query",
        global_folder=tmp_path,
        max_age=3600,
        vectordb_path=vectordb_path,
    )
    assert loaded is None


def test_feed_cache_roundtrip_returns_stale_entry_with_fresh_false(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vectordb_path = tmp_path / "data"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("lsm.remote.storage._now_utc", lambda: now)

    save_feed_cache(
        feed_url="https://example.com/rss",
        items=[{"identifier": "a1", "title": "Item"}],
        seen_ids=["a1"],
        global_folder=tmp_path,
        vectordb_path=vectordb_path,
        cache_ttl_seconds=60,
    )

    monkeypatch.setattr("lsm.remote.storage._now_utc", lambda: now + timedelta(seconds=120))
    cache = load_feed_cache(
        feed_url="https://example.com/rss",
        global_folder=tmp_path,
        max_age=3600,
        vectordb_path=vectordb_path,
    )

    assert cache is not None
    assert cache.feed_url == "https://example.com/rss"
    assert cache.seen_ids == ["a1"]
    assert cache.items == [{"identifier": "a1", "title": "Item"}]
    assert cache.fresh is False

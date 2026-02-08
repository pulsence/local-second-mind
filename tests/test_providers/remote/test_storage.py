from __future__ import annotations

from pathlib import Path

from lsm.remote.storage import load_cached_results, save_results


def test_save_and_load_cached_results_roundtrip(tmp_path: Path) -> None:
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
    )

    assert cache_path.exists()
    loaded = load_cached_results(
        provider_name="web search",
        query="test query",
        global_folder=tmp_path,
        max_age=3600,
    )
    assert loaded == results


def test_load_cached_results_returns_none_when_expired(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results = [{"title": "Old", "url": "https://old.example"}]

    monkeypatch.setattr("lsm.remote.storage.time.time", lambda: 1000)
    save_results(
        provider_name="wikipedia",
        query="cached query",
        results=results,
        global_folder=tmp_path,
    )

    monkeypatch.setattr("lsm.remote.storage.time.time", lambda: 2000)
    loaded = load_cached_results(
        provider_name="wikipedia",
        query="cached query",
        global_folder=tmp_path,
        max_age=100,
    )
    assert loaded is None

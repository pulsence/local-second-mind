"""Shared helpers for live remote provider tests."""

from __future__ import annotations

from lsm.remote.base import RemoteResult


def assert_empty_query_returns_empty(provider) -> None:
    assert provider.search("", max_results=3) == []
    assert provider.search("   ", max_results=3) == []


def assert_non_empty_results(results: list[RemoteResult], min_results: int = 1) -> None:
    assert isinstance(results, list)
    assert len(results) >= min_results
    first = results[0]
    assert isinstance(first, RemoteResult)
    assert isinstance(first.title, str) and first.title.strip()
    assert isinstance(first.url, str) and first.url.strip()
    assert isinstance(first.snippet, str)
    assert isinstance(first.score, float)
    assert isinstance(first.metadata, dict)

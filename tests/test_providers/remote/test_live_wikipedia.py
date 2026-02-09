"""Live tests for Wikipedia remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.wikipedia import WikipediaProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_wikipedia_search() -> None:
    provider = WikipediaProvider(
        {
            "user_agent": "LocalSecondMindTest/1.0 (integration tests)",
            "min_interval_seconds": 0.0,
            "timeout": 20,
        }
    )
    results = provider.search("retrieval augmented generation", max_results=3)

    assert_non_empty_results(results)
    assert results[0].metadata.get("citation")
    assert "language" in results[0].metadata
    assert_empty_query_returns_empty(provider)

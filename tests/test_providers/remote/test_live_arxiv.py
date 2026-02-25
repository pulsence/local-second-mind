"""Live tests for arXiv remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.academic.arxiv import ArXivProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_arxiv_search() -> None:
    provider = ArXivProvider(
        {
            "user_agent": "LocalSecondMindTest/1.0 (integration tests)",
            "min_interval_seconds": 0.0,
            "timeout": 20,
        }
    )
    results = provider.search("title:retrieval augmented generation", max_results=3)

    assert_non_empty_results(results)
    assert "arxiv_id" in results[0].metadata
    assert isinstance(results[0].metadata.get("categories"), list)
    assert_empty_query_returns_empty(provider)

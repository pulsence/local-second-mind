"""Live tests for Semantic Scholar remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.semantic_scholar import SemanticScholarProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_semantic_scholar_search(test_config) -> None:
    provider = SemanticScholarProvider(
        {
            "api_key": test_config.semantic_scholar_api_key,
            "min_interval_seconds": 0.0,
            "timeout": 20,
        }
    )
    results = provider.search("retrieval augmented generation", max_results=3)

    assert_non_empty_results(results)
    assert "paper_id" in results[0].metadata
    assert "citation_count" in results[0].metadata
    assert_empty_query_returns_empty(provider)

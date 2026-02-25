"""Live tests for Brave Search remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.web.brave import BraveSearchProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_brave_search(test_config) -> None:
    if not test_config.brave_api_key:
        pytest.skip("LSM_TEST_BRAVE_API_KEY is required for Brave live tests")

    provider = BraveSearchProvider(
        {
            "api_key": test_config.brave_api_key,
            "timeout": 20,
        }
    )
    results = provider.search("local first retrieval augmented generation", max_results=3)

    assert_non_empty_results(results)
    assert results[0].url.startswith("http")
    assert "rank" in results[0].metadata
    assert_empty_query_returns_empty(provider)

"""Live tests for OpenAlex remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.academic.openalex import OpenAlexProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_openalex_search() -> None:
    provider = OpenAlexProvider(
        {
            "min_interval_seconds": 0.0,
            "timeout": 20,
        }
    )
    results = provider.search("retrieval augmented generation", max_results=3)

    assert_non_empty_results(results)
    assert "openalex_id" in results[0].metadata
    assert "topics" in results[0].metadata
    assert_empty_query_returns_empty(provider)

"""Live tests for PhilPapers remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.philpapers import PhilPapersProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_philpapers_search() -> None:
    provider = PhilPapersProvider(
        {
            "min_interval_seconds": 0.0,
            "timeout": 25,
        }
    )
    results = provider.search("epistemology knowledge justification", max_results=3)

    assert_non_empty_results(results)
    assert "source" in results[0].metadata
    assert "citation" in results[0].metadata
    assert_empty_query_returns_empty(provider)

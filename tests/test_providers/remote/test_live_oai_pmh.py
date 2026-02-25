"""Live tests for generic OAI-PMH remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_oai_pmh_search() -> None:
    provider = OAIPMHProvider(
        {
            "repository": "zenodo",
            "min_interval_seconds": 0.0,
            "timeout": 30,
        }
    )
    results = provider.search("data", max_results=3)

    assert_non_empty_results(results)
    assert "repository" in results[0].metadata
    assert "oai_identifier" in results[0].metadata
    assert_empty_query_returns_empty(provider)

"""Live tests for CORE remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.academic.core import COREProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_core_search(test_config) -> None:
    if not test_config.core_api_key:
        pytest.skip("LSM_TEST_CORE_API_KEY is required for CORE live tests")

    provider = COREProvider(
        {
            "api_key": test_config.core_api_key,
            "min_interval_seconds": 0.0,
            "timeout": 25,
        }
    )
    results = provider.search("knowledge graph retrieval", max_results=3)

    assert_non_empty_results(results)
    assert "core_id" in results[0].metadata
    assert "citation" in results[0].metadata
    assert_empty_query_returns_empty(provider)

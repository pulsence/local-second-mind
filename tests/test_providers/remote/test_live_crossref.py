"""Live tests for Crossref remote provider."""

from __future__ import annotations

import pytest

from lsm.remote.providers.crossref import CrossrefProvider
from tests.test_providers.remote.live_checks import (
    assert_empty_query_returns_empty,
    assert_non_empty_results,
)


pytestmark = [pytest.mark.live, pytest.mark.live_remote]


def test_live_crossref_search_and_doi_lookup() -> None:
    provider = CrossrefProvider(
        {
            "min_interval_seconds": 0.0,
            "timeout": 20,
        }
    )

    keyword_results = provider.search("retrieval augmented generation", max_results=3)
    assert_non_empty_results(keyword_results)
    assert "doi" in keyword_results[0].metadata

    doi_results = provider.search("doi:10.1038/nphys1170", max_results=1)
    assert_non_empty_results(doi_results)
    doi = (doi_results[0].metadata.get("doi") or "").lower()
    assert "10.1038/nphys1170" in doi

    assert_empty_query_returns_empty(provider)

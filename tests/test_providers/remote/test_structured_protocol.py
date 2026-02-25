from __future__ import annotations

from lsm.remote.base import RemoteResult
from lsm.remote.providers.arxiv import ArXivProvider
from lsm.remote.providers.brave import BraveSearchProvider
from lsm.remote.providers.core import COREProvider
from lsm.remote.providers.crossref import CrossrefProvider
from lsm.remote.providers.ixtheo import IxTheoProvider
from lsm.remote.providers.oai_pmh import OAIPMHProvider
from lsm.remote.providers.openalex import OpenAlexProvider
from lsm.remote.providers.philpapers import PhilPapersProvider
from lsm.remote.providers.semantic_scholar import SemanticScholarProvider
from lsm.remote.providers.wikipedia import WikipediaProvider
from lsm.remote.providers.rss import RSSProvider


PROVIDER_CLASSES = [
    BraveSearchProvider,
    WikipediaProvider,
    ArXivProvider,
    SemanticScholarProvider,
    COREProvider,
    PhilPapersProvider,
    IxTheoProvider,
    OpenAlexProvider,
    CrossrefProvider,
    OAIPMHProvider,
    RSSProvider,
]


def test_all_providers_expose_structured_protocol_methods() -> None:
    for provider_cls in PROVIDER_CLASSES:
        provider = provider_cls({})
        input_fields = provider.get_input_fields()
        output_fields = provider.get_output_fields()
        description = provider.get_description()

        assert isinstance(input_fields, list)
        assert input_fields
        assert any(field.get("name") == "query" for field in input_fields)

        assert isinstance(output_fields, list)
        assert output_fields
        assert any(field.get("name") == "url" for field in output_fields)

        assert isinstance(description, str)
        assert description.strip()


def test_search_structured_builds_query_and_normalizes_output() -> None:
    provider = BraveSearchProvider({})
    provider.search = lambda query, max_results=5: [  # type: ignore[assignment]
        RemoteResult(
            title="Example",
            url="https://example.com",
            snippet=f"query={query}",
            score=0.9,
            metadata={"doi": "10.1000/xyz", "authors": ["Doe"], "year": 2024},
        )
    ]

    results = provider.search_structured(
        {
            "title": "Vector Search",
            "author": "Jane Doe",
            "keywords": ["retrieval", "embeddings"],
            "year": 2024,
            "doi": "10.1000/xyz",
        },
        max_results=1,
    )

    assert len(results) == 1
    item = results[0]
    assert item["url"] == "https://example.com"
    assert item["title"] == "Example"
    assert item["description"]
    assert item["doi"] == "10.1000/xyz"
    assert item["authors"] == ["Doe"]
    assert item["year"] == 2024

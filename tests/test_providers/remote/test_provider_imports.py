from __future__ import annotations

from lsm.remote.factory import create_remote_provider, get_registered_providers
from lsm.remote.providers.academic.arxiv import ArXivProvider as NewArXivProvider
from lsm.remote.providers.academic.openalex import OpenAlexProvider as NewOpenAlexProvider
from lsm.remote.providers.news.rss import RSSProvider as NewRSSProvider
from lsm.remote.providers.web.wikipedia import WikipediaProvider as NewWikipediaProvider
from lsm.remote.providers.arxiv import ArXivProvider as LegacyArXivProvider
from lsm.remote.providers.openalex import OpenAlexProvider as LegacyOpenAlexProvider
from lsm.remote.providers.rss import RSSProvider as LegacyRSSProvider
from lsm.remote.providers.wikipedia import WikipediaProvider as LegacyWikipediaProvider


def test_legacy_provider_modules_alias_new_locations() -> None:
    assert LegacyArXivProvider is NewArXivProvider
    assert LegacyOpenAlexProvider is NewOpenAlexProvider
    assert LegacyRSSProvider is NewRSSProvider
    assert LegacyWikipediaProvider is NewWikipediaProvider


def test_factory_registration_points_to_new_modules() -> None:
    registry = get_registered_providers()
    assert registry["arxiv"].__module__ == "lsm.remote.providers.academic.arxiv"
    assert registry["openalex"].__module__ == "lsm.remote.providers.academic.openalex"
    assert registry["rss"].__module__ == "lsm.remote.providers.news.rss"
    assert registry["wikipedia"].__module__ == "lsm.remote.providers.web.wikipedia"


def test_config_reference_resolution_for_provider_types() -> None:
    provider = create_remote_provider("wikipedia", {"type": "wikipedia"})
    assert provider.__class__ is NewWikipediaProvider

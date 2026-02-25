"""News and RSS providers."""

from __future__ import annotations

__all__ = [
    "RSSProvider",
    "NYTimesProvider",
    "GuardianProvider",
    "GDELTProvider",
    "NewsAPIProvider",
]

from lsm.remote.providers.news.rss import RSSProvider
from lsm.remote.providers.news.nytimes import NYTimesProvider
from lsm.remote.providers.news.guardian import GuardianProvider
from lsm.remote.providers.news.gdelt import GDELTProvider
from lsm.remote.providers.news.newsapi import NewsAPIProvider

"""LSM Remote Providers - individual provider implementations.

This module re-exports remote providers from domain subpackages:
- academic: scholarly and metadata sources
- cultural: archives and cultural heritage datasets
- news: news APIs and RSS
- web: web search and encyclopedias

Backward-compatible import modules live in this package to preserve legacy paths.
"""

from __future__ import annotations

__all__ = [
    "BraveSearchProvider",
    "WikipediaProvider",
    "ArXivProvider",
    "SemanticScholarProvider",
    "COREProvider",
    "BaseOAIProvider",
    "OAIPMHProvider",
    "OAIPMHClient",
    "OAIRecord",
    "KNOWN_REPOSITORIES",
    "RSSProvider",
    "PhilPapersProvider",
    "IxTheoProvider",
    "OpenAlexProvider",
    "CrossrefProvider",
]

from lsm.remote.providers.web.brave import BraveSearchProvider
from lsm.remote.providers.web.wikipedia import WikipediaProvider
from lsm.remote.providers.academic.arxiv import ArXivProvider
from lsm.remote.providers.academic.semantic_scholar import SemanticScholarProvider
from lsm.remote.providers.academic.core import COREProvider
from lsm.remote.providers.base_oai import BaseOAIProvider
from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider, OAIPMHClient, OAIRecord, KNOWN_REPOSITORIES
from lsm.remote.providers.news.rss import RSSProvider
from lsm.remote.providers.academic.philpapers import PhilPapersProvider
from lsm.remote.providers.academic.ixtheo import IxTheoProvider
from lsm.remote.providers.academic.openalex import OpenAlexProvider
from lsm.remote.providers.academic.crossref import CrossrefProvider

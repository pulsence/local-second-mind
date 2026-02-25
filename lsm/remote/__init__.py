"""LSM Remote module - remote source providers.

This module contains providers for fetching information from remote sources:
- base: Abstract base class for remote providers
- factory: Provider factory and registry
- providers: Individual provider implementations (Brave, Wikipedia, arXiv, etc.)
"""

from __future__ import annotations

__all__ = [
    # Base classes
    "BaseRemoteProvider",
    "RemoteResult",
    "BaseOAIProvider",
    # Factory functions
    "create_remote_provider",
    "register_remote_provider",
    "get_registered_providers",
    "get_remote_provider",
    "RemoteProviderFactory",
    "RemoteProviderChain",
    # Storage helpers
    "save_results",
    "load_cached_results",
    "save_feed_cache",
    "load_feed_cache",
    "FeedCache",
    # Provider implementations
    "BraveSearchProvider",
    "WikipediaProvider",
    "ArXivProvider",
    "SemanticScholarProvider",
    "COREProvider",
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

# Base classes
from lsm.remote.base import BaseRemoteProvider, RemoteResult

# Factory
from lsm.remote.factory import (
    create_remote_provider,
    register_remote_provider,
    get_registered_providers,
    get_remote_provider,
    RemoteProviderFactory,
)
from lsm.remote.chain import RemoteProviderChain
from lsm.remote.storage import (
    save_results,
    load_cached_results,
    save_feed_cache,
    load_feed_cache,
    FeedCache,
)

# Provider implementations
from lsm.remote.providers.brave import BraveSearchProvider
from lsm.remote.providers.wikipedia import WikipediaProvider
from lsm.remote.providers.arxiv import ArXivProvider
from lsm.remote.providers.semantic_scholar import SemanticScholarProvider
from lsm.remote.providers.core import COREProvider
from lsm.remote.providers.base_oai import BaseOAIProvider
from lsm.remote.providers.oai_pmh import OAIPMHProvider, OAIPMHClient, OAIRecord, KNOWN_REPOSITORIES
from lsm.remote.providers.rss import RSSProvider
from lsm.remote.providers.philpapers import PhilPapersProvider
from lsm.remote.providers.ixtheo import IxTheoProvider
from lsm.remote.providers.openalex import OpenAlexProvider
from lsm.remote.providers.crossref import CrossrefProvider

# Register providers with the factory
register_remote_provider("web_search", BraveSearchProvider)
register_remote_provider("brave_search", BraveSearchProvider)
register_remote_provider("wikipedia", WikipediaProvider)
register_remote_provider("arxiv", ArXivProvider)
register_remote_provider("semantic_scholar", SemanticScholarProvider)
register_remote_provider("core", COREProvider)
register_remote_provider("philpapers", PhilPapersProvider)
register_remote_provider("ixtheo", IxTheoProvider)
register_remote_provider("openalex", OpenAlexProvider)
register_remote_provider("crossref", CrossrefProvider)
register_remote_provider("oai_pmh", OAIPMHProvider)
register_remote_provider("rss", RSSProvider)

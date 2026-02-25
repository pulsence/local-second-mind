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
    "ScholarlyDiscoveryChain",
    "build_chain",
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
    "UnpaywallProvider",
    "PubMedProvider",
    "SSRNProvider",
    "PhilArchiveProvider",
    "ProjectMUSEProvider",
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
from lsm.remote.chains import ScholarlyDiscoveryChain, build_chain
from lsm.remote.storage import (
    save_results,
    load_cached_results,
    save_feed_cache,
    load_feed_cache,
    FeedCache,
)

# Provider implementations
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
from lsm.remote.providers.academic.unpaywall import UnpaywallProvider
from lsm.remote.providers.academic.pubmed import PubMedProvider
from lsm.remote.providers.academic.ssrn import SSRNProvider
from lsm.remote.providers.academic.philarchive import PhilArchiveProvider
from lsm.remote.providers.academic.project_muse import ProjectMUSEProvider

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
register_remote_provider("unpaywall", UnpaywallProvider)
register_remote_provider("pubmed", PubMedProvider)
register_remote_provider("ssrn", SSRNProvider)
register_remote_provider("philarchive", PhilArchiveProvider)
register_remote_provider("project_muse", ProjectMUSEProvider)
register_remote_provider("oai_pmh", OAIPMHProvider)
register_remote_provider("rss", RSSProvider)

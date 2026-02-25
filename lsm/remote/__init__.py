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
    "OAuth2Client",
    "OAuthToken",
    "OAuthTokenStore",
    "OAuthCallbackServer",
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
    "ArchiveOrgProvider",
    "DPLAProvider",
    "LOCProvider",
    "SmithsonianProvider",
    "MetProvider",
    "RijksmuseumProvider",
    "IIIFProvider",
    "WikidataProvider",
    "NYTimesProvider",
    "GuardianProvider",
    "GDELTProvider",
    "NewsAPIProvider",
    "PerseusCTSProvider",
    "GmailProvider",
    "MicrosoftGraphMailProvider",
    "IMAPProvider",
    "EmailMessage",
    "EmailDraft",
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
from lsm.remote.oauth import OAuth2Client, OAuthToken, OAuthTokenStore, OAuthCallbackServer
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
from lsm.remote.providers.cultural.archive_org import ArchiveOrgProvider
from lsm.remote.providers.cultural.dpla import DPLAProvider
from lsm.remote.providers.cultural.loc import LOCProvider
from lsm.remote.providers.cultural.smithsonian import SmithsonianProvider
from lsm.remote.providers.cultural.met import MetProvider
from lsm.remote.providers.cultural.rijksmuseum import RijksmuseumProvider
from lsm.remote.providers.cultural.iiif import IIIFProvider
from lsm.remote.providers.cultural.wikidata import WikidataProvider
from lsm.remote.providers.news.nytimes import NYTimesProvider
from lsm.remote.providers.news.guardian import GuardianProvider
from lsm.remote.providers.news.gdelt import GDELTProvider
from lsm.remote.providers.news.newsapi import NewsAPIProvider
from lsm.remote.providers.cultural.perseus_cts import PerseusCTSProvider
from lsm.remote.providers.communication.gmail import GmailProvider
from lsm.remote.providers.communication.microsoft_graph_mail import MicrosoftGraphMailProvider
from lsm.remote.providers.communication.imap import IMAPProvider
from lsm.remote.providers.communication.models import EmailMessage, EmailDraft

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
register_remote_provider("archive_org", ArchiveOrgProvider)
register_remote_provider("dpla", DPLAProvider)
register_remote_provider("loc", LOCProvider)
register_remote_provider("smithsonian", SmithsonianProvider)
register_remote_provider("met", MetProvider)
register_remote_provider("rijksmuseum", RijksmuseumProvider)
register_remote_provider("iiif", IIIFProvider)
register_remote_provider("wikidata", WikidataProvider)
register_remote_provider("nytimes", NYTimesProvider)
register_remote_provider("guardian", GuardianProvider)
register_remote_provider("gdelt", GDELTProvider)
register_remote_provider("newsapi", NewsAPIProvider)
register_remote_provider("perseus_cts", PerseusCTSProvider)
register_remote_provider("gmail", GmailProvider)
register_remote_provider("microsoft_graph_mail", MicrosoftGraphMailProvider)
register_remote_provider("imap", IMAPProvider)
register_remote_provider("oai_pmh", OAIPMHProvider)
register_remote_provider("rss", RSSProvider)

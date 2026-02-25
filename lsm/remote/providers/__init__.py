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

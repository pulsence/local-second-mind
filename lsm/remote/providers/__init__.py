"""LSM Remote Providers - individual provider implementations.

This module contains all remote source provider implementations:
- brave: Brave Search API
- wikipedia: Wikipedia API
- arxiv: arXiv OAI-PMH
- semantic_scholar: Semantic Scholar API
- core: CORE academic repository
- oai_pmh: Generic OAI-PMH harvester
- philpapers: PhilPapers philosophy database
- ixtheo: IxTheo theology/philosophy database
- openalex: OpenAlex research metadata
- crossref: Crossref bibliographic data
"""

from __future__ import annotations

__all__ = [
    "BraveSearchProvider",
    "WikipediaProvider",
    "ArXivProvider",
    "SemanticScholarProvider",
    "COREProvider",
    "OAIPMHProvider",
    "OAIPMHClient",
    "OAIRecord",
    "KNOWN_REPOSITORIES",
    "PhilPapersProvider",
    "IxTheoProvider",
    "OpenAlexProvider",
    "CrossrefProvider",
]

from lsm.remote.providers.brave import BraveSearchProvider
from lsm.remote.providers.wikipedia import WikipediaProvider
from lsm.remote.providers.arxiv import ArXivProvider
from lsm.remote.providers.semantic_scholar import SemanticScholarProvider
from lsm.remote.providers.core import COREProvider
from lsm.remote.providers.oai_pmh import OAIPMHProvider, OAIPMHClient, OAIRecord, KNOWN_REPOSITORIES
from lsm.remote.providers.philpapers import PhilPapersProvider
from lsm.remote.providers.ixtheo import IxTheoProvider
from lsm.remote.providers.openalex import OpenAlexProvider
from lsm.remote.providers.crossref import CrossrefProvider

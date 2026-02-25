"""Academic remote providers."""

from __future__ import annotations

__all__ = [
    "ArXivProvider",
    "SemanticScholarProvider",
    "COREProvider",
    "CrossrefProvider",
    "OpenAlexProvider",
    "PhilPapersProvider",
    "IxTheoProvider",
    "OAIPMHProvider",
    "OAIPMHClient",
    "OAIRecord",
    "KNOWN_REPOSITORIES",
]

from lsm.remote.providers.academic.arxiv import ArXivProvider
from lsm.remote.providers.academic.semantic_scholar import SemanticScholarProvider
from lsm.remote.providers.academic.core import COREProvider
from lsm.remote.providers.academic.crossref import CrossrefProvider
from lsm.remote.providers.academic.openalex import OpenAlexProvider
from lsm.remote.providers.academic.philpapers import PhilPapersProvider
from lsm.remote.providers.academic.ixtheo import IxTheoProvider
from lsm.remote.providers.academic.oai_pmh import (
    OAIPMHProvider,
    OAIPMHClient,
    OAIRecord,
    KNOWN_REPOSITORIES,
)

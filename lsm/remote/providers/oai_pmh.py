"""Backward-compatible import for OAI-PMH provider."""

from __future__ import annotations

from lsm.remote.providers.academic.oai_pmh import (
    OAIPMHProvider,
    OAIPMHClient,
    OAIRecord,
    KNOWN_REPOSITORIES,
    OAIPMHError,
    DublinCoreParser,
    MARCParser,
    DataciteParser,
    METADATA_PARSERS,
    OAIRepository,
)

__all__ = [
    "OAIPMHProvider",
    "OAIPMHClient",
    "OAIRecord",
    "KNOWN_REPOSITORIES",
    "OAIPMHError",
    "DublinCoreParser",
    "MARCParser",
    "DataciteParser",
    "METADATA_PARSERS",
    "OAIRepository",
]

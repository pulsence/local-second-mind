"""
OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting) provider.

DEPRECATED: This module is deprecated in favor of lsm.remote.providers.oai_pmh.
This module re-exports from lsm.remote.providers.oai_pmh for backward compatibility.
"""

from __future__ import annotations

# Re-export everything from the new location
from lsm.remote.providers.oai_pmh import (
    OAIPMHProvider,
    OAIPMHClient,
    OAIPMHError,
    OAIRecord,
    OAIRepository,
    DublinCoreParser,
    MARCParser,
    DataciteParser,
    KNOWN_REPOSITORIES,
    METADATA_PARSERS,
)

__all__ = [
    "OAIPMHProvider",
    "OAIPMHClient",
    "OAIPMHError",
    "OAIRecord",
    "OAIRepository",
    "DublinCoreParser",
    "MARCParser",
    "DataciteParser",
    "KNOWN_REPOSITORIES",
    "METADATA_PARSERS",
]

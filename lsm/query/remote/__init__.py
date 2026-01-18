"""
Remote source providers for query augmentation.

DEPRECATED: This module is deprecated in favor of lsm.remote.
All functionality has been moved to lsm.remote.

This module re-exports from lsm.remote for backward compatibility.
"""

# Re-export everything from the new location
from lsm.remote import (
    # Base classes
    BaseRemoteProvider,
    RemoteResult,
    # Factory functions
    create_remote_provider,
    register_remote_provider,
    get_registered_providers,
    # Provider implementations
    BraveSearchProvider,
    WikipediaProvider,
    ArXivProvider,
    SemanticScholarProvider,
    COREProvider,
    OAIPMHProvider,
    OAIPMHClient,
    OAIRecord,
    KNOWN_REPOSITORIES,
    PhilPapersProvider,
    IxTheoProvider,
    OpenAlexProvider,
    CrossrefProvider,
)

__all__ = [
    "BaseRemoteProvider",
    "RemoteResult",
    "create_remote_provider",
    "register_remote_provider",
    "get_registered_providers",
    "BraveSearchProvider",
    "WikipediaProvider",
    "ArXivProvider",
    "SemanticScholarProvider",
    "COREProvider",
    "PhilPapersProvider",
    "IxTheoProvider",
    "OpenAlexProvider",
    "CrossrefProvider",
    "OAIPMHProvider",
    "OAIPMHClient",
    "OAIRecord",
    "KNOWN_REPOSITORIES",
]

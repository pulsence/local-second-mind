"""
Remote source providers for query augmentation.

Provides interfaces and implementations for fetching information from
remote sources (web search, APIs, etc.) to augment local knowledge.
"""

from .base import BaseRemoteProvider, RemoteResult
from .factory import create_remote_provider, register_remote_provider

# Import and register built-in providers
from .brave import BraveSearchProvider
from .wikipedia import WikipediaProvider
from .arxiv import ArXivProvider
from .semantic_scholar import SemanticScholarProvider
from .core import COREProvider
from .philpapers import PhilPapersProvider
from .ixtheo import IxTheoProvider
from .openalex import OpenAlexProvider
from .crossref import CrossrefProvider

# Register Brave Search provider
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

__all__ = [
    "BaseRemoteProvider",
    "RemoteResult",
    "create_remote_provider",
    "register_remote_provider",
    "BraveSearchProvider",
    "WikipediaProvider",
    "ArXivProvider",
    "SemanticScholarProvider",
    "COREProvider",
    "PhilPapersProvider",
    "IxTheoProvider",
    "OpenAlexProvider",
    "CrossrefProvider",
]

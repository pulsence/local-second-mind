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

# Register Brave Search provider
register_remote_provider("web_search", BraveSearchProvider)
register_remote_provider("brave_search", BraveSearchProvider)
register_remote_provider("wikipedia", WikipediaProvider)
register_remote_provider("arxiv", ArXivProvider)

__all__ = [
    "BaseRemoteProvider",
    "RemoteResult",
    "create_remote_provider",
    "register_remote_provider",
    "BraveSearchProvider",
    "WikipediaProvider",
    "ArXivProvider",
]

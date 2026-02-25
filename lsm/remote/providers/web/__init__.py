"""Web search and general web content providers."""

from __future__ import annotations

__all__ = [
    "BraveSearchProvider",
    "WikipediaProvider",
]

from lsm.remote.providers.web.brave import BraveSearchProvider
from lsm.remote.providers.web.wikipedia import WikipediaProvider

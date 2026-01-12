"""
Base interface for remote source providers.

Defines the contract for fetching information from external sources
like web search, APIs, knowledge graphs, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class RemoteResult:
    """
    A single result from a remote source provider.

    Attributes:
        title: Title or heading of the result
        url: Source URL or identifier
        snippet: Text snippet or summary
        score: Relevance score (0.0-1.0, higher is better)
        metadata: Additional provider-specific metadata
    """

    title: str
    """Title or heading of the result."""

    url: str
    """Source URL or identifier."""

    snippet: str
    """Text snippet or summary."""

    score: float = 1.0
    """Relevance score (0.0-1.0, higher is better)."""

    metadata: Dict[str, Any] = None
    """Additional provider-specific metadata."""

    def __post_init__(self):
        """Initialize metadata dict if None."""
        if self.metadata is None:
            self.metadata = {}


class BaseRemoteProvider(ABC):
    """
    Abstract base class for remote source providers.

    Remote providers fetch information from external sources to augment
    the local knowledge base. Examples include web search, APIs, knowledge
    graphs, news feeds, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the remote provider.

        Args:
            config: Provider-specific configuration (API keys, endpoints, etc.)
        """
        self.config = config

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[RemoteResult]:
        """
        Search the remote source for relevant information.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects

        Raises:
            Exception: If the search fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the provider name.

        Returns:
            Human-readable provider name
        """
        pass

    def validate_config(self) -> None:
        """
        Validate the provider configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Default: no validation required
        pass

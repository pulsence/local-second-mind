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
    """Additional provider-specific metadata (must include a stable source_id)."""

    def __post_init__(self):
        """Initialize metadata dict if None."""
        if self.metadata is None:
            self.metadata = {}
        if not str(self.title or "").strip() and self.url:
            self.title = str(self.url)
        if self.snippet is None or not str(self.snippet).strip():
            fallback = str(self.title or "").strip()
            self.snippet = fallback or str(self.url or "").strip()
        if not str(self.metadata.get("source_id", "")).strip() and self.url:
            self.metadata["source_id"] = str(self.url)


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

    def get_input_fields(self) -> List[Dict[str, Any]]:
        """
        Describe structured input fields accepted by this provider.

        Returns:
            List of field definition dicts with keys:
            - name: field name
            - type: value type
            - description: short field description
            - required: whether field is required
        """
        return [
            {
                "name": "query",
                "type": "string",
                "description": "Free-text search query.",
                "required": True,
            }
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        """
        Describe standard structured output fields for remote results.
        """
        return [
            {"name": "url", "type": "string", "description": "Result URL or identifier"},
            {"name": "title", "type": "string", "description": "Result title"},
            {"name": "description", "type": "string", "description": "Result snippet/summary"},
            {"name": "doi", "type": "string", "description": "Digital Object Identifier when available"},
            {"name": "authors", "type": "array[string]", "description": "Author list when available"},
            {"name": "year", "type": "integer", "description": "Publication year when available"},
            {"name": "score", "type": "number", "description": "Provider relevance score"},
            {"name": "metadata", "type": "object", "description": "Provider-specific metadata"},
        ]

    def get_description(self) -> str:
        """
        Human-readable provider description for LLM tool selection.
        """
        return f"{self.get_name()} remote provider"

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Structured search entrypoint for dict-based inputs.

        Default behavior composes a text query from common fields and delegates
        to `search(query, max_results)`.
        """
        query = self._compose_query_from_input(input_dict)
        if not query:
            return []
        results = self.search(query, max_results=max_results)
        return [self._to_structured_output(result) for result in results]

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[RemoteResult]:
        """
        Search the remote source for relevant information.

        This text-query method remains the canonical provider implementation.
        `search_structured` delegates to this by default.

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

    def _compose_query_from_input(self, input_dict: Dict[str, Any]) -> str:
        """Build a best-effort text query from structured input."""
        if not isinstance(input_dict, dict):
            return str(input_dict or "").strip()

        if input_dict.get("query"):
            return str(input_dict.get("query")).strip()

        parts: List[str] = []
        if input_dict.get("title"):
            parts.append(f'title:"{str(input_dict["title"]).strip()}"')
        if input_dict.get("author"):
            parts.append(f'author:"{str(input_dict["author"]).strip()}"')
        if input_dict.get("doi"):
            parts.append(f'doi:{str(input_dict["doi"]).strip()}')
        if input_dict.get("year"):
            parts.append(str(input_dict["year"]).strip())

        keywords = input_dict.get("keywords")
        if isinstance(keywords, list):
            parts.extend([str(k).strip() for k in keywords if str(k).strip()])
        elif keywords:
            parts.append(str(keywords).strip())

        return " ".join([p for p in parts if p]).strip()

    def _to_structured_output(self, result: RemoteResult) -> Dict[str, Any]:
        """Normalize RemoteResult into structured output dict."""
        metadata = result.metadata or {}
        return {
            "url": result.url,
            "title": result.title,
            "description": result.snippet,
            "doi": metadata.get("doi"),
            "authors": metadata.get("authors"),
            "year": metadata.get("year"),
            "score": result.score,
            "metadata": metadata,
        }


# Aliases for backward compatibility
RemoteProvider = BaseRemoteProvider

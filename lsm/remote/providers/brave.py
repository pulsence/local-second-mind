"""
Brave Search API provider for remote search.

Provides web search capabilities using the Brave Search API.
API documentation: https://brave.com/search/api/
"""

from __future__ import annotations

import os
from typing import List, Dict, Any
import requests

from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.logging import get_logger

logger = get_logger(__name__)


class BraveSearchProvider(BaseRemoteProvider):
    """
    Remote provider using Brave Search API.

    Brave Search provides privacy-focused web search with a simple API.
    Requires a Brave Search API key.
    """

    SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Brave Search provider.

        Args:
            config: Configuration dict with optional keys:
                - api_key: Brave Search API key (or from BRAVE_API_KEY env var)
                - endpoint: Custom API endpoint (default: official Brave endpoint)
                - timeout: Request timeout in seconds (default: 10)
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.get("api_key") or os.getenv("BRAVE_API_KEY")
        self.endpoint = config.get("endpoint") or self.SEARCH_ENDPOINT
        self.timeout = config.get("timeout", 10)
        self.enabled = config.get("enabled", True)
        self.max_results = config.get("max_results", 5)
        self.weight = config.get("weight", 1.0)

    @property
    def name(self) -> str:
        return "brave_search"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def validate_config(self) -> None:
        """Validate Brave Search configuration."""
        if not self.api_key:
            raise ValueError(
                "Brave Search API key required. Set 'api_key' in config or "
                "BRAVE_API_KEY environment variable."
            )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[RemoteResult]:
        """
        Search using Brave Search API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of RemoteResult objects

        Raises:
            Exception: If the API request fails
        """
        logger.debug(f"Brave Search: query='{query}', max_results={max_results}")

        # Prepare request
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        params = {
            "q": query,
            "count": max_results,
            "text_decorations": False,  # No HTML markup in snippets
            "search_lang": "en",
        }

        try:
            # Make API request
            response = requests.get(
                self.endpoint,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            web_results = data.get("web", {}).get("results", [])

            for i, item in enumerate(web_results[:max_results]):
                # Extract fields
                title = item.get("title", "")
                url = item.get("url", "")
                description = item.get("description", "")

                # Brave doesn't provide explicit scores, use rank
                # Score: 1.0 for first result, decreasing to 0.2 for last
                score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))

                # Additional metadata
                metadata = {
                    "rank": i + 1,
                    "age": item.get("age"),
                    "language": item.get("language"),
                    "family_friendly": item.get("family_friendly", True),
                }

                results.append(
                    RemoteResult(
                        title=title,
                        url=url,
                        snippet=description,
                        score=score,
                        metadata=metadata,
                    )
                )

            logger.info(f"Brave Search returned {len(results)} results")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Brave Search API error: {e}")
            return []

        except Exception as e:
            logger.error(f"Brave Search parsing error: {e}")
            return []

    def get_name(self) -> str:
        """Get provider name."""
        return "Brave Search"

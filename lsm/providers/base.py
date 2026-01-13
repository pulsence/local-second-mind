"""
Base LLM provider interface.

Defines the contract that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers (OpenAI, Anthropic, local models) must implement this interface.
    """

    @abstractmethod
    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using the LLM.

        Args:
            question: User's question
            candidates: List of candidate dicts with 'text' and 'metadata'
            k: Number of candidates to return
            **kwargs: Provider-specific options

        Returns:
            Reranked list of candidates (top k)

        Raises:
            Exception: If reranking fails
        """
        pass

    @abstractmethod
    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
        """
        Generate an answer with citations.

        Args:
            question: User's question
            context: Context block with source citations
            mode: Query mode ('grounded' or 'insight')
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            Generated answer text

        Raises:
            Exception: If synthesis fails
        """
        pass

    @abstractmethod
    def generate_tags(
        self,
        text: str,
        num_tags: int = 3,
        existing_tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate relevant tags for a text chunk using LLM.

        Args:
            text: Text content to tag
            num_tags: Number of tags to generate (default: 3)
            existing_tags: Optional list of existing tags to consider for consistency
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            List of generated tag strings

        Raises:
            Exception: If tag generation fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is configured and available.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            Provider name (e.g., 'openai', 'anthropic', 'local')
        """
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """
        Get the current model name.

        Returns:
            Model name (e.g., 'gpt-5.2', 'claude-3-sonnet')
        """
        pass

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Estimate the cost for a request (optional).

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD, or None if not available
        """
        return None  # Default implementation

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.name}/{self.model}"

    def __repr__(self) -> str:
        """Developer representation of the provider."""
        return f"{self.__class__.__name__}(model='{self.model}')"

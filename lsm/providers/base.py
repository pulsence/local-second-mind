"""
Base LLM provider interface.

Defines the contract that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional, Callable, Iterable

from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProviderHealthStats:
    """Track provider call outcomes for health monitoring."""
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_category: Optional[str] = None
    circuit_open_until: Optional[datetime] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_error": self.last_error,
            "last_error_category": self.last_error_category,
            "circuit_open_until": (
                self.circuit_open_until.isoformat() if self.circuit_open_until else None
            ),
        }


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers (OpenAI, Anthropic, local models) must implement this interface.
    """

    _GLOBAL_HEALTH_STATS: Dict[str, ProviderHealthStats] = {}
    CIRCUIT_BREAKER_THRESHOLD = 5
    CIRCUIT_BREAKER_COOLDOWN_SECONDS = 30.0

    def __init__(self) -> None:
        self.last_response_id: Optional[str] = None
        key = f"{self.name}:{self.model}"
        if key not in self._GLOBAL_HEALTH_STATS:
            self._GLOBAL_HEALTH_STATS[key] = ProviderHealthStats()
        self._health_stats = self._GLOBAL_HEALTH_STATS[key]

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
    def stream_synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> Iterable[str]:
        """
        Stream an answer with citations in chunks.

        Args:
            question: User's question
            context: Context block with source citations
            mode: Query mode ('grounded' or 'insight')
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Yields:
            Text chunks as they are generated

        Raises:
            Exception: If streaming fails
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

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        """
        Get pricing for the current model.

        Returns:
            Dict with 'input' and 'output' prices per 1M tokens in USD,
            or None if pricing is not available for this model.
        """
        return None

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Estimate the cost for a request.

        Uses get_model_pricing() to look up per-1M-token rates for the
        current model and calculates the cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD, or None if pricing is not available
        """
        pricing = self.get_model_pricing()
        if pricing is None:
            return None
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def health_check(self) -> Dict[str, Any]:
        """
        Return provider health and recent call stats.

        Returns:
            Dictionary with availability and recent success/failure stats
        """
        return {
            "provider": self.name,
            "model": self.model,
            "available": self.is_available(),
            "status": "available" if self.is_available() else "unavailable",
            "stats": self._health_stats.as_dict(),
        }

    def list_models(self) -> List[str]:
        """
        List models available to the provider.

        Providers that do not support model listing should return an empty list.
        """
        return []

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.name}/{self.model}"

    def __repr__(self) -> str:
        """Developer representation of the provider."""
        return f"{self.__class__.__name__}(model='{self.model}')"

    def _record_success(self, context: Optional[str] = None) -> None:
        """Record a successful provider call."""
        self._health_stats.success_count += 1
        self._health_stats.consecutive_failures = 0
        self._health_stats.last_success = datetime.utcnow()
        self._health_stats.circuit_open_until = None
        if context:
            logger.debug(f"{self.name}/{self.model} call succeeded: {context}")

    def _record_failure(self, error: Exception, context: Optional[str] = None) -> None:
        """Record a failed provider call and log centrally."""
        self._health_stats.failure_count += 1
        self._health_stats.consecutive_failures += 1
        self._health_stats.last_failure = datetime.utcnow()
        self._health_stats.last_error = str(error)
        self._health_stats.last_error_category = self.categorize_error(error)
        if self._health_stats.consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
            self._health_stats.circuit_open_until = datetime.utcnow() + timedelta(
                seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS
            )
        label = f"{self.name}/{self.model}"
        if context:
            label = f"{label} ({context})"
        logger.error(f"Provider error in {label}: {error}")

    def categorize_error(self, error: Exception) -> str:
        """
        Categorize provider exceptions as retryable or fatal.
        """
        if isinstance(error, (TimeoutError, ConnectionError)):
            return "retryable"
        if isinstance(error, (PermissionError, ValueError)):
            return "fatal"

        message = str(error).lower()
        retryable_markers = (
            "timeout",
            "timed out",
            "temporar",
            "unavailable",
            "rate limit",
            "429",
            "503",
            "connection reset",
            "connection aborted",
            "try again",
        )
        for marker in retryable_markers:
            if marker in message:
                return "retryable"
        return "fatal"

    def is_retryable_error(self, error: Exception) -> bool:
        """Return True if an exception should be retried."""
        return self.categorize_error(error) == "retryable"

    def _is_circuit_open(self) -> bool:
        until = self._health_stats.circuit_open_until
        if until is None:
            return False
        if datetime.utcnow() >= until:
            self._health_stats.circuit_open_until = None
            return False
        return True

    def _with_retry(
        self,
        func: Callable[[], Any],
        action: str,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        retry_on: Optional[Callable[[Exception], bool]] = None,
    ) -> Any:
        """
        Execute a provider call with exponential backoff retries.

        Args:
            func: Callable to execute
            action: Short description for logging
            max_attempts: Maximum attempts before raising
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            retry_on: Optional predicate to decide if error is retryable
        """
        if self._is_circuit_open():
            raise RuntimeError(
                f"Circuit breaker open for {self.name}/{self.model}; "
                "recent failures exceeded threshold."
            )

        attempt = 0
        while True:
            try:
                return func()
            except Exception as e:
                attempt += 1
                should_retry = attempt < max_attempts
                checker = retry_on or self.is_retryable_error
                should_retry = should_retry and checker(e)

                if not should_retry:
                    raise

                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                logger.warning(
                    f"{self.name}/{self.model} {action} failed (attempt {attempt}/{max_attempts}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

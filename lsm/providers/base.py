"""
Base LLM provider interface.

Defines the contract that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any, Optional, Callable, Iterable

from lsm.logging import get_logger
from .helpers import (
    RERANK_INSTRUCTIONS,
    RERANK_JSON_SCHEMA,
    TAGS_JSON_SCHEMA,
    format_user_content,
    generate_fallback_answer,
    get_synthesis_instructions,
    get_tag_instructions,
    parse_json_payload,
    parse_ranking_response,
    prepare_candidates_for_rerank,
)

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
    def _send_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs
    ) -> str:
        """
        Send a single non-streaming message to the provider.

        Args:
            system: System instruction content
            user: User prompt content
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            **kwargs: Provider-specific request fields

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def _send_streaming_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs
    ) -> Iterable[str]:
        """
        Send a streaming message to the provider.

        Args:
            system: System instruction content
            user: User prompt content
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            **kwargs: Provider-specific request fields

        Yields:
            Text chunks from the model response
        """
        pass

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

        """
        if not candidates:
            return []

        k = max(1, min(k, len(candidates)))
        payload = {
            "question": question,
            "top_n": k,
            "candidates": prepare_candidates_for_rerank(candidates),
        }
        instructions = RERANK_INSTRUCTIONS.format(k=k)

        try:
            raw = self._send_message(
                system=instructions,
                user=json.dumps(payload),
                temperature=0.2,
                max_tokens=800,
                json_schema=RERANK_JSON_SCHEMA,
                json_schema_name="rerank_response",
                reasoning_effort="low",
                **kwargs,
            )
            data = parse_json_payload(raw)
            ranking = data.get("ranking", []) if isinstance(data, dict) else None
            if not isinstance(ranking, list):
                logger.warning(
                    f"LLM rerank returned invalid response structure "
                    f"({self.name}/{self.model}), "
                    f"falling back to local candidate ordering"
                )
                return candidates[:k]
            chosen = parse_ranking_response(ranking, candidates, k)
            self._record_success("rerank")
            return chosen
        except Exception as e:
            logger.warning(
                f"LLM rerank failed ({self.name}/{self.model}: {e}), "
                f"falling back to local candidate ordering"
            )
            return candidates[:k]

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

        """
        instructions = get_synthesis_instructions(mode)
        user_content = format_user_content(question, context)
        opts = dict(kwargs)
        temperature = opts.pop("temperature", self._default_temperature())
        max_tokens = int(opts.pop("max_tokens", self._default_max_tokens()))

        try:
            answer = self._send_message(
                system=instructions,
                user=user_content,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="medium",
                **opts,
            )
            self._record_success("synthesize")
            return answer
        except Exception as e:
            self._record_failure(e, "synthesize")
            return self._fallback_answer(question, context)

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

        """
        instructions = get_synthesis_instructions(mode)
        user_content = format_user_content(question, context)
        opts = dict(kwargs)
        temperature = opts.pop("temperature", self._default_temperature())
        max_tokens = int(opts.pop("max_tokens", self._default_max_tokens()))

        try:
            emitted = False
            for chunk in self._send_streaming_message(
                system=instructions,
                user=user_content,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="medium",
                **opts,
            ):
                if chunk:
                    emitted = True
                    yield chunk
            self._record_success("stream_synthesize")
            if not emitted:
                logger.warning(f"{self.name}/{self.model} streaming synthesis emitted no text")
        except Exception as e:
            self._record_failure(e, "stream_synthesize")
            raise

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

        """
        instructions = get_tag_instructions(num_tags, existing_tags)
        user_content = f"Text:\n{text[:2000]}"
        opts = dict(kwargs)
        temperature = opts.pop("temperature", self._default_temperature())
        max_tokens = int(min(opts.pop("max_tokens", self._default_max_tokens()), 200))

        try:
            raw = self._send_message(
                system=instructions,
                user=user_content,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=TAGS_JSON_SCHEMA,
                json_schema_name="tags_response",
                reasoning_effort="low",
                **opts,
            )
            data = parse_json_payload(raw)
            tags: Optional[List[Any]] = None
            if isinstance(data, dict) and isinstance(data.get("tags"), list):
                tags = data["tags"]
            elif isinstance(data, list):
                tags = data

            if tags and all(isinstance(tag, str) for tag in tags):
                cleaned = [tag.lower().strip() for tag in tags if tag.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            if isinstance(raw, str) and "{" not in raw and "[" not in raw:
                cleaned = [tag.strip().lower() for tag in raw.split(",") if tag.strip()]
                if cleaned:
                    self._record_success("generate_tags")
                    return cleaned[:num_tags]

            self._record_failure(ValueError("Failed to parse tag response"), "generate_tags")
            return []
        except Exception as e:
            self._record_failure(e, "generate_tags")
            return []

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

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        """Generate a fallback answer when provider requests fail."""
        return generate_fallback_answer(
            question=question,
            context=context,
            provider_name=self.name,
            max_chars=max_chars,
        )

    def _default_temperature(self) -> Optional[float]:
        config = getattr(self, "config", None)
        return getattr(config, "temperature", None)

    def _default_max_tokens(self) -> int:
        config = getattr(self, "config", None)
        return int(getattr(config, "max_tokens", 2000))

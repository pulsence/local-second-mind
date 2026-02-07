"""
Anthropic Claude provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from anthropic import Anthropic

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider
from .helpers import (
    RERANK_INSTRUCTIONS,
    format_user_content,
    generate_fallback_answer,
    get_synthesis_instructions,
    get_tag_instructions,
    parse_json_payload,
    parse_ranking_response,
    prepare_candidates_for_rerank,
)

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Configuration:
        provider: anthropic
        model: claude-3-5-sonnet-20241022
        api_key: ${ANTHROPIC_API_KEY}
        temperature: 0.7
        max_tokens: 2000
    """

    # Per-1M-token pricing in USD (input / output)
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        # Claude 4.x series
        "claude-opus-4.6":          {"input":  5.00, "output": 25.00},
        "claude-opus-4.5":          {"input":  5.00, "output": 25.00},
        "claude-opus-4.1":          {"input": 15.00, "output": 75.00},
        "claude-opus-4":            {"input": 15.00, "output": 75.00},
        "claude-sonnet-4.5":        {"input":  3.00, "output": 15.00},
        "claude-sonnet-4":          {"input":  3.00, "output": 15.00},
        "claude-haiku-4.5":         {"input":  1.00, "output":  5.00},
        # Claude 3.x series (legacy)
        "claude-haiku-3.5":         {"input":  0.80, "output":  4.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022":  {"input": 1.00, "output":  5.00},
        "claude-3-opus-20240229":   {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input":  3.00, "output": 15.00},
        "claude-3-haiku-20240307":  {"input":  0.25, "output":  1.25},
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Optional[Anthropic] = None
        super().__init__()
        logger.debug(f"Initialized Anthropic provider with model: {config.model}")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> List[str]:
        if not self.is_available():
            return []

        try:
            res = self.client.models.list(limit=1000)
            items = getattr(res, "data", None)
            if items is None:
                items = list(res)
            ids: List[str] = []
            for item in items or []:
                model_id = getattr(item, "id", None) or getattr(item, "name", None)
                if isinstance(model_id, str) and model_id:
                    ids.append(model_id)
            ids.sort()
            return ids
        except Exception as e:
            logger.debug(f"Failed to list Anthropic models: {e}")
            return []

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            if not self.is_available():
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it in config or as environment variable."
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _extract_text(self, response: Any) -> str:
        parts = []
        for part in getattr(response, "content", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts).strip()
        return (getattr(response, "content", "") or "").strip()

    def _is_retryable_error(self, error: Exception) -> bool:
        name = error.__class__.__name__
        return name in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
        }

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        k = max(1, min(k, len(candidates)))

        items = prepare_candidates_for_rerank(candidates)
        instructions = RERANK_INSTRUCTIONS.format(k=k)

        payload = {
            "question": question,
            "top_n": k,
            "candidates": items,
        }

        try:
            logger.debug(f"Requesting Claude rerank for {len(candidates)} candidates -> top {k}")

            def _call(system_prompt: str) -> Any:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=400,
                    temperature=0.2,
                    system=system_prompt,
                    messages=[{"role": "user", "content": json.dumps(payload)}],
                )

            resp = self._with_retry(
                lambda: _call(instructions),
                "rerank",
                retry_on=self._is_retryable_error,
            )
            raw = self._extract_text(resp)
            if not raw:
                raise json.JSONDecodeError("Empty rerank response", raw, 0)

            data = parse_json_payload(raw)
            ranking = data.get("ranking", []) if isinstance(data, dict) else None
            if not isinstance(ranking, list):
                self._record_failure(ValueError("Invalid rerank response"), "rerank")
                return candidates[:k]

            chosen = parse_ranking_response(ranking, candidates, k)

            self._record_success("rerank")
            return chosen

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Claude rerank response: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

        except Exception as e:
            logger.warning(f"Claude rerank failed: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
        instructions = get_synthesis_instructions(mode)
        user_content = format_user_content(question, context)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            logger.debug(f"Requesting Claude synthesis in {mode} mode")

            def _call() -> Any:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                )

            resp = self._with_retry(_call, "synthesize", retry_on=self._is_retryable_error)
            answer = self._extract_text(resp)
            self._record_success("synthesize")
            return answer

        except Exception as e:
            logger.error(f"Claude synthesis failed: {e}")
            self._record_failure(e, "synthesize")
            return self._fallback_answer(question, context)

    def generate_tags(
        self,
        text: str,
        num_tags: int = 3,
        existing_tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        instructions = get_tag_instructions(num_tags, existing_tags)

        user_content = f"Text:\n{text[:2000]}"

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

        try:
            logger.debug(f"Requesting Claude tag generation for text of length {len(text)}")

            def _call() -> Any:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                )

            resp = self._with_retry(_call, "generate_tags", retry_on=self._is_retryable_error)
            content = self._extract_text(resp)
            if not content:
                raise json.JSONDecodeError("Empty tag response", content, 0)
            data = parse_json_payload(content)
            tags = data.get("tags", []) if isinstance(data, dict) else None

            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                cleaned = [t.lower().strip() for t in tags if t.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            raise ValueError(f"Unexpected tag structure: {data}")

        except Exception as e:
            logger.error(f"Claude tag generation failed: {e}")
            self._record_failure(e, "generate_tags")
            raise

    def stream_synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ):
        instructions = get_synthesis_instructions(mode)
        user_content = format_user_content(question, context)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            try:
                stream = self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                )
                with stream as s:
                    for text in s.text_stream:
                        if text:
                            yield text
            except AttributeError:
                stream = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                    stream=True,
                )
                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type is None and isinstance(event, dict):
                        event_type = event.get("type")
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta is None and isinstance(event, dict):
                            delta = event.get("delta")
                        text = getattr(delta, "text", None)
                        if text is None and isinstance(delta, dict):
                            text = delta.get("text")
                        if text:
                            yield text

            self._record_success("stream_synthesize")

        except Exception as e:
            logger.error(f"Claude streaming synthesis failed: {e}")
            self._record_failure(e, "stream_synthesize")
            raise

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        """Get pricing for the current Anthropic model."""
        return self.MODEL_PRICING.get(self.model)

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        return generate_fallback_answer(question, context, "Anthropic API", max_chars=max_chars)

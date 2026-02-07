"""
OpenAI LLM provider implementation.

Implements the BaseLLMProvider interface for OpenAI's API.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from openai import OpenAI, OpenAIError, RateLimitError

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider
from .helpers import (
    RERANK_INSTRUCTIONS,
    UnsupportedParamTracker,
    format_user_content,
    generate_fallback_answer,
    get_synthesis_instructions,
    get_tag_instructions,
    parse_json_payload,
    parse_ranking_response,
    prepare_candidates_for_rerank,
)

logger = get_logger(__name__)


_UNSUPPORTED_PARAM_TRACKER = UnsupportedParamTracker()


def _model_supports_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation.

    Uses OpenAI's Responses API for reranking and synthesis.
    """

    # Per-1M-token pricing in USD (input / output)
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        # GPT-5 series
        "gpt-5.2":            {"input":  1.75,  "output":  14.00},
        "gpt-5.1":            {"input":  1.25,  "output":  10.00},
        "gpt-5":              {"input":  1.25,  "output":  10.00},
        "gpt-5-mini":         {"input":  0.25,  "output":   2.00},
        "gpt-5-nano":         {"input":  0.05,  "output":   0.40},
        # GPT-4.1 series
        "gpt-4.1":            {"input":  2.00,  "output":   8.00},
        "gpt-4.1-mini":       {"input":  0.40,  "output":   1.60},
        "gpt-4.1-nano":       {"input":  0.10,  "output":   0.40},
        # GPT-4o series
        "gpt-4o":             {"input":  2.50,  "output":  10.00},
        "gpt-4o-mini":        {"input":  0.15,  "output":   0.60},
        # o-series reasoning models
        "o4-mini":            {"input":  1.10,  "output":   4.40},
        "o3":                 {"input":  2.00,  "output":   8.00},
        "o3-mini":            {"input":  1.10,  "output":   4.40},
        "o1":                 {"input": 15.00,  "output":  60.00},
        "o1-mini":            {"input":  1.10,  "output":   4.40},
        # Legacy models
        "gpt-4-turbo":        {"input": 10.00,  "output":  30.00},
        "gpt-4":              {"input": 30.00,  "output":  60.00},
        "gpt-3.5-turbo":      {"input":  0.50,  "output":   1.50},
    }

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration with API key and model settings
        """
        self.config = config

        # Create OpenAI client
        if config.api_key:
            self.client = OpenAI(api_key=config.api_key)
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var

        super().__init__()
        logger.debug(f"Initialized OpenAI provider with model: {config.model}")

    @property
    def name(self) -> str:
        """Get provider name."""
        return "openai"

    @property
    def model(self) -> str:
        """Get current model."""
        return self.config.model

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        if self.config.api_key:
            return True
        return bool(os.getenv("OPENAI_API_KEY"))

    def list_models(self) -> List[str]:
        """List models available to the current API key."""
        if not self.is_available():
            return []

        try:
            res = self.client.models.list()
            ids: List[str] = []
            for model in getattr(res, "data", []) or []:
                model_id = getattr(model, "id", None)
                if isinstance(model_id, str):
                    ids.append(model_id)
            ids.sort()
            return ids
        except Exception as e:
            logger.debug(f"Failed to list OpenAI models: {e}")
            return []

    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is due to quota/rate limit."""
        if isinstance(error, RateLimitError):
            try:
                data = getattr(error, "response", None)
                if data is not None:
                    j = data.json()
                    return j.get("error", {}).get("code") == "insufficient_quota"
            except Exception:
                pass
            return True
        return isinstance(error, OpenAIError)

    def _is_retryable_error(self, error: Exception) -> bool:
        name = error.__class__.__name__
        return name in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
        }

    def _call_responses(self, request_args: Dict[str, Any], action: str):
        return self._with_retry(
            lambda: self.client.responses.create(**request_args),
            action,
            retry_on=self._is_retryable_error,
        )

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using OpenAI LLM.

        Args:
            question: User's question
            candidates: List of candidate dicts with 'text' and 'metadata'
            k: Number of candidates to return
            **kwargs: Additional options (temperature, etc.)

        Returns:
            Reranked list of candidates (top k)
        """
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
            logger.debug(f"Requesting LLM rerank for {len(candidates)} candidates -> top {k}")

            request_args = {
                "model": self.config.model,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": json.dumps(payload)}],
            }

            if _UNSUPPORTED_PARAM_TRACKER.should_send(self.config.model, "text"):
                request_args["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "rerank_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "ranking": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "index": {"type": "integer"},
                                            "reason": {"type": "string"}
                                        },
                                        "required": ["index", "reason"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["ranking"],
                            "additionalProperties": False
                        }
                    }
                }

            try:
                resp = self._call_responses(request_args, "rerank")
            except Exception as e:
                if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "text"):
                    logger.warning("Model does not support text.format for structured output; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.config.model, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "rerank")
                else:
                    raise

            raw = (resp.output_text or "").strip()
            logger.debug(f"LLM rerank response length: {len(raw)} chars")

            data = parse_json_payload(raw)
            ranking = data.get("ranking", []) if isinstance(data, dict) else None
            if not isinstance(ranking, list):
                logger.warning("Invalid ranking format, falling back to vector order")
                self._record_failure(ValueError("Invalid rerank response"), "rerank")
                return candidates[:k]

            chosen = parse_ranking_response(ranking, candidates, k)

            self._record_success("rerank")
            logger.info(f"LLM reranked {len(candidates)} → {len(chosen)} candidates")
            return chosen

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM rerank response: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

        except Exception as e:
            if self._is_quota_error(e):
                logger.warning(f"LLM rerank quota/rate limit hit: {e}")
            else:
                logger.warning(f"LLM rerank failed: {e}")
            self._record_failure(e, "rerank")
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
            context: Context block with [S1], [S2] citations
            mode: Query mode ('grounded' or 'insight')
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            Generated answer text with citations
        """
        # Build mode-specific instructions
        instructions = get_synthesis_instructions(mode)
        user_content = format_user_content(question, context)

        try:
            logger.debug(f"Requesting synthesis in {mode} mode")

            # Get optional parameters
            temperature = kwargs.get("temperature", self.config.temperature)
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

            request_args = {
                "model": self.config.model,
                "reasoning": {"effort": "medium"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            if (
                temperature is not None
                and _model_supports_temperature(self.config.model)
                and _UNSUPPORTED_PARAM_TRACKER.should_send(self.config.model, "temperature")
            ):
                request_args["temperature"] = temperature

            try:
                resp = self._call_responses(request_args, "synthesize")
            except Exception as e:
                if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                    logger.warning("Model does not support temperature; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.config.model, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "synthesize")
                else:
                    raise

            answer = (resp.output_text or "").strip()
            self._record_success("synthesize")
            logger.info(f"Generated answer ({len(answer)} chars) in {mode} mode")
            return answer

        except Exception as e:
            if self._is_quota_error(e):
                logger.error(f"Synthesis quota/rate limit hit: {e}")
                self._record_failure(e, "synthesize")
                return self._fallback_answer(question, context)
            else:
                logger.error(f"Synthesis failed: {e}")
                self._record_failure(e, "synthesize")
            return self._fallback_answer(question, context)

    def generate_tags(
        self,
        text: str,
        num_tags: int = 3,
        existing_tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate relevant tags for a text chunk using OpenAI LLM.

        Args:
            text: Text content to tag
            num_tags: Number of tags to generate (default: 3)
            existing_tags: Optional list of existing tags to consider for consistency
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            List of generated tag strings

        Raises:
            Exception: If tag generation fails
        """
        instructions = get_tag_instructions(num_tags, existing_tags)

        user_content = f"Text:\n{text[:2000]}"

        try:
            logger.debug(f"Requesting tag generation for text of length {len(text)}")

            # Get optional parameters
            temperature = kwargs.get("temperature", self.config.temperature)
            max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

            request_args = {
                "model": self.config.model,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            if temperature is not None and _UNSUPPORTED_PARAM_TRACKER.should_send(self.config.model, "temperature"):
                request_args["temperature"] = temperature

            # Responses API uses text.format for structured output, not response_format
            if _UNSUPPORTED_PARAM_TRACKER.should_send(self.config.model, "text"):
                request_args["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "tags_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["tags"],
                            "additionalProperties": False
                        }
                    }
                }

            try:
                resp = self._call_responses(request_args, "generate_tags")
            except Exception as e:
                # Handle unsupported parameters
                if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                    logger.warning("Model does not support temperature for tagging; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.config.model, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "generate_tags")
                elif _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "text"):
                    logger.warning("Model does not support text.format for structured output; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.config.model, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "generate_tags")
                else:
                    raise

            # Extract tags from response
            content = (resp.output_text or "").strip()
            logger.debug(f"LLM tagging response length: {len(content)} chars")

            if not content:
                logger.warning("LLM response was empty")
                self._record_failure(ValueError("Empty tag response"), "generate_tags")
                return []

            data = parse_json_payload(content)
            if isinstance(data, dict) and isinstance(data.get("tags"), list):
                tags = data["tags"]
                if tags and all(isinstance(t, str) for t in tags):
                    tags = [t.lower().strip() for t in tags if t.strip()]
                    logger.info(f"Generated {len(tags)} tags for text")
                    self._record_success("generate_tags")
                    return tags[:num_tags]
            elif isinstance(data, list) and all(isinstance(t, str) for t in data):
                tags = [t.lower().strip() for t in data if t.strip()]
                logger.info(f"Generated {len(tags)} tags for text")
                self._record_success("generate_tags")
                return tags[:num_tags]

            # Last resort: treat as comma-separated list (only if no JSON structure found)
            if "{" not in content and "[" not in content:
                tags = [t.strip().lower() for t in content.split(",")]
                tags = [t for t in tags if t]
                if tags:
                    logger.info(f"Generated {len(tags)} tags from comma-separated text")
                    self._record_success("generate_tags")
                    return tags[:num_tags]

            logger.warning(f"Could not parse tags from response: {content!r}")
            self._record_failure(ValueError("Failed to parse tag response"), "generate_tags")
            return []

            logger.error(f"Could not extract tags from LLM response: {content}")
            self._record_failure(ValueError("Empty tag response"), "generate_tags")
            return []

        except Exception as e:
            logger.error(f"Failed to generate tags: {e}")
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

        request_args = {
            "model": self.config.model,
            "reasoning": {"effort": "medium"},
            "instructions": instructions,
            "input": [{"role": "user", "content": user_content}],
            "max_output_tokens": max_tokens,
        }

        if (
            temperature is not None
            and _model_supports_temperature(self.config.model)
            and _UNSUPPORTED_PARAM_TRACKER.should_send(self.config.model, "temperature")
        ):
            request_args["temperature"] = temperature

        try:
            try:
                stream = self.client.responses.create(**request_args, stream=True)
            except Exception as e:
                if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                    logger.warning("Model does not support temperature; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.config.model, "temperature")
                    request_args.pop("temperature", None)
                    stream = self.client.responses.create(**request_args, stream=True)
                else:
                    raise

            emitted = False
            for event in stream:
                event_type = getattr(event, "type", None)
                if event_type is None and isinstance(event, dict):
                    event_type = event.get("type")
                if event_type in {"response.output_text.delta", "response.output_text"}:
                    delta = getattr(event, "delta", None)
                    if delta is None and isinstance(event, dict):
                        delta = event.get("delta")
                    if delta:
                        emitted = True
                        yield delta
                elif event_type == "response.completed":
                    break

            self._record_success("stream_synthesize")
            if not emitted:
                logger.warning("OpenAI stream completed without text output")

        except Exception as e:
            if self._is_quota_error(e):
                logger.error(f"Streaming quota/rate limit hit: {e}")
            else:
                logger.error(f"Streaming synthesis failed: {e}")
            self._record_failure(e, "stream_synthesize")
            raise

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        """
        Generate a fallback answer when API is unavailable.

        Args:
            question: User's question
            context: Context block
            max_chars: Maximum characters to include

        Returns:
            Fallback message with context snippet
        """
        return generate_fallback_answer(question, context, "OpenAI API", max_chars=max_chars)

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        """Get pricing for the current OpenAI model."""
        return self.MODEL_PRICING.get(self.config.model)

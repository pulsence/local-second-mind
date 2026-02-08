"""
Azure OpenAI provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from openai import AzureOpenAI

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


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI provider.

    Configuration:
        provider: azure_openai
        model: gpt-35-turbo
        api_key: ${AZURE_OPENAI_API_KEY}
        endpoint: https://your-resource.openai.azure.com/
        api_version: 2023-05-15
        deployment_name: gpt-35-turbo
    """

    # Azure OpenAI mirrors standard OpenAI per-1M-token pricing (USD)
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
        self.config = config
        self.last_response_id: Optional[str] = None
        self.endpoint = config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = config.api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = (
            config.deployment_name
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or config.model
        )
        self.client = AzureOpenAI(
            api_key=config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )
        super().__init__()
        logger.debug(f"Initialized Azure OpenAI provider with deployment: {self.deployment_name}")

    @property
    def name(self) -> str:
        return "azure_openai"

    @property
    def model(self) -> str:
        return self.deployment_name

    def is_available(self) -> bool:
        has_key = bool(self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"))
        return has_key and bool(self.endpoint) and bool(self.api_version)

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
            request_args = {
                "model": self.deployment_name,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": json.dumps(payload)}],
            }

            if _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "text"):
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
                    logger.warning("Deployment does not support text.format; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "rerank")
                else:
                    raise

            raw = (resp.output_text or "").strip()
            data = parse_json_payload(raw)
            ranking = data.get("ranking", []) if isinstance(data, dict) else None

            if not isinstance(ranking, list):
                self._record_failure(ValueError("Invalid rerank response"), "rerank")
                return candidates[:k]

            chosen = parse_ranking_response(ranking, candidates, k)

            self._record_success("rerank")
            return chosen

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Azure rerank response: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

        except Exception as e:
            logger.warning(f"Azure rerank failed: {e}")
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
            request_args = {
                "model": self.deployment_name,
                "reasoning": {"effort": "medium"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            previous_response_id = kwargs.get("previous_response_id")
            if kwargs.get("enable_server_cache") and previous_response_id:
                request_args["previous_response_id"] = previous_response_id

            prompt_cache_key = kwargs.get("prompt_cache_key")
            if kwargs.get("enable_server_cache") and prompt_cache_key:
                request_args["prompt_cache_key"] = prompt_cache_key

            if (
                temperature is not None
                and _model_supports_temperature(self.deployment_name)
                and _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "temperature")
            ):
                request_args["temperature"] = temperature

            try:
                resp = self._call_responses(request_args, "synthesize")
            except Exception as e:
                if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                    logger.warning("Deployment does not support temperature; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "synthesize")
                elif _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "prompt_cache_key"):
                    logger.warning("Deployment does not support prompt_cache_key; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "prompt_cache_key")
                    request_args.pop("prompt_cache_key", None)
                    resp = self._call_responses(request_args, "synthesize")
                elif _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "previous_response_id"):
                    logger.warning("Deployment does not support previous_response_id; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "previous_response_id")
                    request_args.pop("previous_response_id", None)
                    resp = self._call_responses(request_args, "synthesize")
                else:
                    raise

            answer = (resp.output_text or "").strip()
            self.last_response_id = getattr(resp, "id", None)
            self._record_success("synthesize")
            return answer

        except Exception as e:
            logger.error(f"Azure synthesis failed: {e}")
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

        try:
            temperature = kwargs.get("temperature", self.config.temperature)
            max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

            request_args = {
                "model": self.deployment_name,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            if temperature is not None and _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "temperature"):
                request_args["temperature"] = temperature

            if _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "text"):
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
                if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                    logger.warning("Deployment does not support temperature for tagging; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "generate_tags")
                elif _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "text"):
                    logger.warning("Deployment does not support text.format; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "generate_tags")
                else:
                    raise

            content = (resp.output_text or "").strip()
            data = parse_json_payload(content)
            tags = data.get("tags", []) if isinstance(data, dict) else None

            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                cleaned = [t.lower().strip() for t in tags if t.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            raise ValueError(f"Unexpected tag structure: {data}")

        except Exception as e:
            logger.error(f"Azure tag generation failed: {e}")
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
            "model": self.deployment_name,
            "reasoning": {"effort": "medium"},
            "instructions": instructions,
            "input": [{"role": "user", "content": user_content}],
            "max_output_tokens": max_tokens,
        }

        if (
            temperature is not None
            and _model_supports_temperature(self.deployment_name)
            and _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "temperature")
        ):
            request_args["temperature"] = temperature

        try:
            try:
                stream = self.client.responses.create(**request_args, stream=True)
            except Exception as e:
                if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                    logger.warning("Deployment does not support temperature; retrying without it.")
                    _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "temperature")
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
                logger.warning("Azure stream completed without text output")

        except Exception as e:
            logger.error(f"Azure streaming synthesis failed: {e}")
            self._record_failure(e, "stream_synthesize")
            raise

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        """Get pricing for the current Azure OpenAI deployment."""
        return self.MODEL_PRICING.get(self.deployment_name)

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        return generate_fallback_answer(question, context, "Azure OpenAI", max_chars=max_chars)

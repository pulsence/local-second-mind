"""
Google Gemini provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types as genai_types

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


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider.

    Configuration:
        provider: gemini
        model: gemini-1.5-pro
        api_key: ${GOOGLE_API_KEY}
        temperature: 0.7
        max_tokens: 2000
    """

    # Per-1M-token pricing in USD (input / output) -- Google AI Studio rates
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        # Gemini 2.5 series
        "gemini-2.5-pro":        {"input": 1.25,  "output": 10.00},
        "gemini-2.5-flash":      {"input": 0.30,  "output":  2.50},
        "gemini-2.5-flash-lite": {"input": 0.10,  "output":  0.40},
        # Gemini 2.0 series
        "gemini-2.0-flash":      {"input": 0.10,  "output":  0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output":  0.30},
        # Gemini 1.5 series (legacy)
        "gemini-1.5-pro":        {"input": 1.25,  "output":  5.00},
        "gemini-1.5-flash":      {"input": 0.075, "output":  0.30},
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        self._client: Optional[genai.Client] = None
        self.last_response_id: Optional[str] = None
        super().__init__()
        logger.debug(f"Initialized Gemini provider with model: {config.model}")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> List[str]:
        if not self.is_available():
            return []

        try:
            client = self._ensure_client()
            res = client.models.list()
            items = getattr(res, "page", None)
            if items is None:
                try:
                    items = list(res)
                except TypeError:
                    items = []

            ids: List[str] = []
            for item in items or []:
                name = getattr(item, "name", None) or getattr(item, "display_name", None)
                if not isinstance(name, str) or not name:
                    continue
                if "/" in name:
                    name = name.split("/")[-1]
                ids.append(name)
            ids = sorted(set(ids))
            return ids
        except Exception as e:
            logger.debug(f"Failed to list Gemini models: {e}")
            return []

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not set. "
                    "Set it in config or as environment variable."
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_config(self, temperature: float, max_tokens: int):
        return genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        name = error.__class__.__name__
        return name in {
            "ResourceExhausted",
            "ServiceUnavailable",
            "DeadlineExceeded",
            "InternalServerError",
        }

    def _generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        client = self._ensure_client()

        def _call() -> Any:
            config = self._build_config(temperature, max_tokens)
            return client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

        resp = self._with_retry(_call, "generate_content", retry_on=self._is_retryable_error)
        self.last_response_id = (
            getattr(resp, "response_id", None)
            or getattr(resp, "id", None)
        )
        text = getattr(resp, "text", None)
        if not text:
            return ""
        return text.strip()

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
            prompt = f"{instructions}\n\n{json.dumps(payload)}"
            raw = self._generate(prompt, temperature=0.2, max_tokens=400)
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

        except Exception as e:
            logger.warning(f"Gemini rerank failed: {e}")
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
            prompt = f"{instructions}\n\n{user_content}"
            answer = self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
            self._record_success("synthesize")
            return answer

        except Exception as e:
            logger.error(f"Gemini synthesis failed: {e}")
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
            prompt = f"{instructions}\n\n{user_content}"
            raw = self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
            if not raw:
                raise json.JSONDecodeError("Empty tag response", raw, 0)
            data = parse_json_payload(raw)
            tags = data.get("tags", []) if isinstance(data, dict) else None
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                cleaned = [t.lower().strip() for t in tags if t.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            raise ValueError(f"Unexpected tag structure: {data}")

        except Exception as e:
            logger.error(f"Gemini tag generation failed: {e}")
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
            client = self._ensure_client()
            config = self._build_config(temperature, max_tokens)
            prompt = f"{instructions}\n\n{user_content}"
            stream = client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=config,
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text

            self._record_success("stream_synthesize")

        except Exception as e:
            logger.error(f"Gemini streaming synthesis failed: {e}")
            self._record_failure(e, "stream_synthesize")
            raise

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        """Get pricing for the current Gemini model."""
        return self.MODEL_PRICING.get(self.model)

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        return generate_fallback_answer(question, context, "Gemini API", max_chars=max_chars)

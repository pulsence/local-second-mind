"""
Google Gemini provider implementation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        self._client: Optional[genai.Client] = None
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
            return sorted(set(ids))
        except Exception as e:
            logger.debug(f"Failed to list Gemini models: {e}")
            return []

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY not set. Set it in config or env.")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_config(self, temperature: float, max_tokens: int):
        return genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        return error.__class__.__name__ in {
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
        self.last_response_id = getattr(resp, "response_id", None) or getattr(resp, "id", None)
        text = getattr(resp, "text", None)
        if not text:
            return ""
        return text.strip()

    def _send_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs,
    ) -> str:
        prompt = f"{system}\n\n{user}"
        return self._generate(prompt, temperature=temperature or 0.0, max_tokens=max_tokens)

    def _send_streaming_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs,
    ):
        client = self._ensure_client()
        config = self._build_config(temperature or 0.0, max_tokens)
        prompt = f"{system}\n\n{user}"
        stream = client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        return self.MODEL_PRICING.get(self.model)

"""
Local model provider implementation (Ollama).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class LocalProvider(BaseLLMProvider):
    """Local model provider using Ollama's HTTP API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = (
            config.base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        super().__init__()
        logger.debug(f"Initialized Local provider with model: {config.model}")

    @property
    def name(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return bool(self.base_url)

    def health_check(self) -> Dict[str, Any]:
        data = super().health_check()
        data["base_url"] = self.base_url
        return data

    def _is_retryable_error(self, error: Exception) -> bool:
        return isinstance(error, requests.exceptions.RequestException)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def _chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        def _call() -> Dict[str, Any]:
            return self._post("/api/chat", payload)

        resp = self._with_retry(_call, "chat", retry_on=self._is_retryable_error)
        self.last_response_id = None
        message = resp.get("message", {}) if isinstance(resp, dict) else {}
        return (message.get("content") or "").strip()

    def send_message(
        self,
        input: str,
        instruction: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        previous_response_id: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_retention: Optional[int] = None,
        **kwargs,
    ) -> str:
        _ = previous_response_id, prompt_cache_key, prompt_cache_retention, kwargs
        user = f"{prompt}\n\n{input}" if prompt else input
        return self._chat(
            instruction or "",
            user,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens,
        )

    def _chat_stream(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        url = f"{self.base_url}/api/chat"
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message", {}) if isinstance(data, dict) else {}
            content = message.get("content")
            if content:
                yield content
            if data.get("done"):
                break

    def send_streaming_message(
        self,
        input: str,
        instruction: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        previous_response_id: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_retention: Optional[int] = None,
        **kwargs,
    ):
        _ = previous_response_id, prompt_cache_key, prompt_cache_retention, kwargs
        user = f"{prompt}\n\n{input}" if prompt else input
        yield from self._chat_stream(
            instruction or "",
            user,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens,
        )

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        return {"input": 0.0, "output": 0.0}

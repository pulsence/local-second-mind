"""
Anthropic Claude provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "claude-opus-4.6": {"input": 5.00, "output": 25.00},
        "claude-opus-4.5": {"input": 5.00, "output": 25.00},
        "claude-opus-4.1": {"input": 15.00, "output": 75.00},
        "claude-opus-4": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
        "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
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

    @property
    def supports_function_calling(self) -> bool:
        return True

    @staticmethod
    def _format_tool_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": definition.get("name"),
            "description": definition.get("description"),
            "input_schema": definition.get("input_schema", {}),
        }

    @staticmethod
    def _extract_tool_calls(response: Any) -> tuple[list[Dict[str, Any]], str]:
        tool_calls: list[Dict[str, Any]] = []
        text_parts: list[str] = []
        for part in getattr(response, "content", []) or []:
            part_type = getattr(part, "type", None)
            if part_type is None and isinstance(part, dict):
                part_type = part.get("type")
            if part_type == "tool_use":
                name = getattr(part, "name", None)
                if name is None and isinstance(part, dict):
                    name = part.get("name")
                tool_input = getattr(part, "input", None)
                if tool_input is None and isinstance(part, dict):
                    tool_input = part.get("input")
                tool_calls.append(
                    {
                        "name": str(name or "").strip(),
                        "arguments": tool_input if isinstance(tool_input, dict) else {},
                    }
                )
                continue
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                text_parts.append(text)
        return tool_calls, "\n".join(text_parts).strip()

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
                raise ValueError("ANTHROPIC_API_KEY not set. Set it in config or env.")
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _extract_text(self, response: Any) -> str:
        parts: List[str] = []
        for part in getattr(response, "content", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts).strip()
        return (getattr(response, "content", "") or "").strip()

    def _is_retryable_error(self, error: Exception) -> bool:
        return error.__class__.__name__ in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
        }

    def _send_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs,
    ) -> str:
        system_payload: Any = system
        if kwargs.get("enable_server_cache"):
            system_payload = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        tools_payload = None
        if tools:
            tools_payload = [
                self._format_tool_definition(tool)
                for tool in tools
                if isinstance(tool, dict)
            ]

        tool_choice_payload = None
        if isinstance(tool_choice, str) and tool_choice:
            tool_choice_payload = {"type": tool_choice}
        elif isinstance(tool_choice, dict):
            tool_choice_payload = tool_choice

        def _call() -> Any:
            request_args: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_payload,
                "messages": [{"role": "user", "content": user}],
            }
            if tools_payload is not None:
                request_args["tools"] = tools_payload
            if tool_choice_payload is not None:
                request_args["tool_choice"] = tool_choice_payload
            return self.client.messages.create(**request_args)

        resp = self._with_retry(_call, "send_message", retry_on=self._is_retryable_error)
        self.last_response_id = getattr(resp, "id", None)
        tool_calls, text = self._extract_tool_calls(resp)
        if tool_calls:
            return json.dumps({"response": text, "tool_calls": tool_calls})
        return text or self._extract_text(resp)

    def _send_streaming_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs,
    ):
        try:
            stream = self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            with stream as session:
                for text in session.text_stream:
                    if text:
                        yield text
            return
        except AttributeError:
            pass

        stream = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
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

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        return self.MODEL_PRICING.get(self.model)

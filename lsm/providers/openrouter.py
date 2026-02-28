"""
OpenRouter LLM provider implementation.

Implements provider-specific transport for OpenRouter's OpenAI-compatible API.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider
from .helpers import UnsupportedParamTracker

logger = get_logger(__name__)
_UNSUPPORTED_PARAM_TRACKER = UnsupportedParamTracker()

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _normalize_fallback_models(models: Optional[List[str]]) -> List[str]:
    cleaned: List[str] = []
    for model in models or []:
        value = str(model).strip()
        if value:
            cleaned.append(value)
    return cleaned


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider implementation."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = (
            config.base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or _DEFAULT_BASE_URL
        )
        self.last_usage: Optional[Dict[str, Any]] = None
        self.last_response_model: Optional[str] = None

        headers: Dict[str, str] = {}
        referer = (
            os.getenv("OPENROUTER_APP_URL")
            or os.getenv("OPENROUTER_REFERRER")
            or os.getenv("OPENROUTER_SITE_URL")
        )
        app_name = os.getenv("OPENROUTER_APP_NAME") or os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if app_name:
            headers["X-Title"] = app_name

        client_kwargs: Dict[str, Any] = {
            "api_key": config.api_key or os.getenv("OPENROUTER_API_KEY"),
            "base_url": self.base_url,
        }
        if headers:
            client_kwargs["default_headers"] = headers

        self.client = OpenAI(**client_kwargs)
        super().__init__()
        logger.debug(
            "Initialized OpenRouter provider with model: %s (base_url=%s)",
            config.model,
            self.base_url,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def supports_function_calling(self) -> bool:
        return True

    @staticmethod
    def _format_tool_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": definition.get("name"),
                "description": definition.get("description"),
                "parameters": definition.get("input_schema", {}),
            },
        }

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _extract_tool_calls(self, response: Any) -> list[Dict[str, Any]]:
        tool_calls: list[Dict[str, Any]] = []
        choices = getattr(response, "choices", []) or []
        if not choices:
            return tool_calls
        message = getattr(choices[0], "message", None)
        if message is None and isinstance(choices[0], dict):
            message = choices[0].get("message")
        raw_calls = getattr(message, "tool_calls", None)
        if raw_calls is None and isinstance(message, dict):
            raw_calls = message.get("tool_calls")
        for call in raw_calls or []:
            function = getattr(call, "function", None)
            if function is None and isinstance(call, dict):
                function = call.get("function")
            name = getattr(function, "name", None) if function is not None else None
            if name is None and isinstance(function, dict):
                name = function.get("name")
            arguments = getattr(function, "arguments", None) if function is not None else None
            if arguments is None and isinstance(function, dict):
                arguments = function.get("arguments")
            tool_calls.append(
                {
                    "name": str(name or "").strip(),
                    "arguments": self._parse_tool_arguments(arguments),
                }
            )
        return tool_calls

    def _normalize_usage(self, usage: Any) -> Optional[Dict[str, Any]]:
        if not usage:
            return None
        if isinstance(usage, dict):
            return dict(usage)
        data: Dict[str, Any] = {}
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        ):
            value = getattr(usage, field, None)
            if value is not None:
                data[field] = value
        return data or None

    def _cache_control_block(self) -> Dict[str, Any]:
        return {"type": "ephemeral"}

    def _build_message(
        self, role: str, content: str, *, use_cache: bool
    ) -> Dict[str, Any]:
        if use_cache:
            return {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": self._cache_control_block(),
                    }
                ],
            }
        return {"role": role, "content": content}

    def is_available(self) -> bool:
        return bool(self.config.api_key or os.getenv("OPENROUTER_API_KEY"))

    def list_models(self) -> List[str]:
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
        except Exception as exc:
            logger.debug("Failed to list OpenRouter models: %s", exc)
            return []

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
        if previous_response_id is not None and _UNSUPPORTED_PARAM_TRACKER.should_send(
            self.model, "previous_response_id"
        ):
            logger.debug(
                "OpenRouter model '%s' does not support 'previous_response_id'; ignoring.",
                self.model,
            )
            _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.model, "previous_response_id")
        if prompt_cache_retention is not None and _UNSUPPORTED_PARAM_TRACKER.should_send(
            self.model, "prompt_cache_retention"
        ):
            logger.debug(
                "OpenRouter model '%s' does not support 'prompt_cache_retention'; ignoring.",
                self.model,
            )
            _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.model, "prompt_cache_retention")
        user = f"{prompt}\n\n{input}" if prompt else input
        use_cache = bool(kwargs.get("enable_server_cache") or prompt_cache_key)
        messages: List[Dict[str, Any]] = []
        if instruction:
            messages.append(self._build_message("system", instruction, use_cache=use_cache))
        messages.append(self._build_message("user", user, use_cache=use_cache))

        request_args: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            request_args["temperature"] = temperature
        if prompt_cache_key:
            request_args["extra_headers"] = {"x-prompt-cache-key": prompt_cache_key}

        tools = kwargs.get("tools")
        if tools:
            request_args["tools"] = [
                self._format_tool_definition(tool) for tool in tools if isinstance(tool, dict)
            ]
            tool_choice = kwargs.get("tool_choice")
            if tool_choice:
                request_args["tool_choice"] = tool_choice

        extra_body: Dict[str, Any] = {}
        fallback_models = _normalize_fallback_models(self.config.fallback_models)
        if fallback_models:
            extra_body["models"] = [self.config.model] + fallback_models
            extra_body["route"] = "fallback"
        if kwargs.get("include_usage") or use_cache:
            extra_body["usage"] = {"include": True}
        if extra_body:
            request_args["extra_body"] = extra_body

        resp = self.client.chat.completions.create(**request_args)

        self.last_response_id = getattr(resp, "id", None)
        self.last_response_model = getattr(resp, "model", None)
        self.last_usage = self._normalize_usage(getattr(resp, "usage", None))

        tool_calls = self._extract_tool_calls(resp) if tools else []
        text = ""
        choices = getattr(resp, "choices", []) or []
        if choices:
            message = getattr(choices[0], "message", None)
            if message is None and isinstance(choices[0], dict):
                message = choices[0].get("message")
            content = getattr(message, "content", None) if message is not None else None
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str):
                text = content

        if tool_calls:
            return json.dumps({"response": text.strip(), "tool_calls": tool_calls})
        return text.strip()

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
    ) -> Iterable[str]:
        if previous_response_id is not None and _UNSUPPORTED_PARAM_TRACKER.should_send(
            self.model, "previous_response_id"
        ):
            logger.debug(
                "OpenRouter model '%s' does not support 'previous_response_id'; ignoring.",
                self.model,
            )
            _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.model, "previous_response_id")
        if prompt_cache_retention is not None and _UNSUPPORTED_PARAM_TRACKER.should_send(
            self.model, "prompt_cache_retention"
        ):
            logger.debug(
                "OpenRouter model '%s' does not support 'prompt_cache_retention'; ignoring.",
                self.model,
            )
            _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.model, "prompt_cache_retention")
        user = f"{prompt}\n\n{input}" if prompt else input
        use_cache = bool(kwargs.get("enable_server_cache") or prompt_cache_key)
        messages: List[Dict[str, Any]] = []
        if instruction:
            messages.append(self._build_message("system", instruction, use_cache=use_cache))
        messages.append(self._build_message("user", user, use_cache=use_cache))

        request_args: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if temperature is not None:
            request_args["temperature"] = temperature
        if prompt_cache_key:
            request_args["extra_headers"] = {"x-prompt-cache-key": prompt_cache_key}

        tools = kwargs.get("tools")
        if tools:
            request_args["tools"] = [
                self._format_tool_definition(tool) for tool in tools if isinstance(tool, dict)
            ]
            tool_choice = kwargs.get("tool_choice")
            if tool_choice:
                request_args["tool_choice"] = tool_choice

        extra_body: Dict[str, Any] = {}
        fallback_models = _normalize_fallback_models(self.config.fallback_models)
        if fallback_models:
            extra_body["models"] = [self.config.model] + fallback_models
            extra_body["route"] = "fallback"
        if kwargs.get("include_usage") or use_cache:
            extra_body["usage"] = {"include": True}
        if extra_body:
            request_args["extra_body"] = extra_body

        stream = self.client.chat.completions.create(**request_args)
        for event in stream:
            choices = getattr(event, "choices", []) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
            text = getattr(delta, "content", None) if delta is not None else None
            if text is None and isinstance(delta, dict):
                text = delta.get("content")
            if text:
                yield text

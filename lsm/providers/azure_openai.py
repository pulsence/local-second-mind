"""
Azure OpenAI provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import AzureOpenAI

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider
from .helpers import UnsupportedParamTracker

logger = get_logger(__name__)

_UNSUPPORTED_PARAM_TRACKER = UnsupportedParamTracker()


def _model_supports_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider."""

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "gpt-5.2": {"input": 1.75, "output": 14.00},
        "gpt-5.1": {"input": 1.25, "output": 10.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o4-mini": {"input": 1.10, "output": 4.40},
        "o3": {"input": 2.00, "output": 8.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, config: LLMConfig):
        self.config = config
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
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
            if item_type != "function_call":
                continue
            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")
            arguments = getattr(item, "arguments", None)
            if arguments is None and isinstance(item, dict):
                arguments = item.get("arguments")
            tool_calls.append(
                {
                    "name": str(name or "").strip(),
                    "arguments": self._parse_tool_arguments(arguments),
                }
            )
        return tool_calls

    def is_available(self) -> bool:
        has_key = bool(self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"))
        return has_key and bool(self.endpoint) and bool(self.api_version)

    def _is_retryable_error(self, error: Exception) -> bool:
        return error.__class__.__name__ in {
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

    def _send_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs,
    ) -> str:
        request_args: Dict[str, Any] = {
            "model": self.deployment_name,
            "reasoning": {"effort": kwargs.get("reasoning_effort", "medium")},
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "max_output_tokens": max_tokens,
        }
        tools = kwargs.get("tools")
        if tools:
            request_args["tools"] = [
                self._format_tool_definition(tool) for tool in tools if isinstance(tool, dict)
            ]
            tool_choice = kwargs.get("tool_choice")
            if tool_choice:
                request_args["tool_choice"] = tool_choice

        previous_response_id = kwargs.get("previous_response_id")
        if kwargs.get("enable_server_cache") and previous_response_id:
            request_args["previous_response_id"] = previous_response_id

        prompt_cache_key = kwargs.get("prompt_cache_key")
        if kwargs.get("enable_server_cache") and prompt_cache_key:
            request_args["prompt_cache_key"] = prompt_cache_key

        json_schema = kwargs.get("json_schema")
        if json_schema and _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "text"):
            request_args["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": kwargs.get("json_schema_name", "response"),
                    "strict": True,
                    "schema": json_schema,
                }
            }

        if (
            temperature is not None
            and _model_supports_temperature(self.deployment_name)
            and _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "temperature")
        ):
            request_args["temperature"] = temperature

        try:
            resp = self._call_responses(request_args, "send_message")
        except Exception as e:
            if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "temperature")
                request_args.pop("temperature", None)
                resp = self._call_responses(request_args, "send_message")
            elif _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "text"):
                _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "text")
                request_args.pop("text", None)
                resp = self._call_responses(request_args, "send_message")
            elif _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "prompt_cache_key"):
                _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "prompt_cache_key")
                request_args.pop("prompt_cache_key", None)
                resp = self._call_responses(request_args, "send_message")
            elif _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "previous_response_id"):
                _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "previous_response_id")
                request_args.pop("previous_response_id", None)
                resp = self._call_responses(request_args, "send_message")
            else:
                raise

        self.last_response_id = getattr(resp, "id", None)
        tool_calls = self._extract_tool_calls(resp) if tools else []
        if tool_calls:
            return json.dumps(
                {"response": (resp.output_text or "").strip(), "tool_calls": tool_calls}
            )
        return (resp.output_text or "").strip()

    def _send_streaming_message(
        self,
        system: str,
        user: str,
        temperature: Optional[float],
        max_tokens: int,
        **kwargs,
    ):
        request_args: Dict[str, Any] = {
            "model": self.deployment_name,
            "reasoning": {"effort": kwargs.get("reasoning_effort", "medium")},
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "max_output_tokens": max_tokens,
        }

        if (
            temperature is not None
            and _model_supports_temperature(self.deployment_name)
            and _UNSUPPORTED_PARAM_TRACKER.should_send(self.deployment_name, "temperature")
        ):
            request_args["temperature"] = temperature

        try:
            stream = self.client.responses.create(**request_args, stream=True)
        except Exception as e:
            if _UNSUPPORTED_PARAM_TRACKER.is_unsupported_error(e, "temperature"):
                _UNSUPPORTED_PARAM_TRACKER.mark_unsupported(self.deployment_name, "temperature")
                request_args.pop("temperature", None)
                stream = self.client.responses.create(**request_args, stream=True)
            else:
                raise

        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type is None and isinstance(event, dict):
                event_type = event.get("type")
            if event_type in {"response.output_text.delta", "response.output_text"}:
                delta = getattr(event, "delta", None)
                if delta is None and isinstance(event, dict):
                    delta = event.get("delta")
                if delta:
                    yield delta
            elif event_type == "response.completed":
                break

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        return self.MODEL_PRICING.get(self.deployment_name)

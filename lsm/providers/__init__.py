"""
LLM provider abstractions for Local Second Mind query pipeline.

Provides a unified interface for different LLM providers (OpenAI, Anthropic, local models).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .factory import create_provider, list_available_providers, register_provider

_LAZY_EXPORTS = {
    "BaseLLMProvider": ("lsm.providers.base", "BaseLLMProvider"),
    "OpenAIProvider": ("lsm.providers.openai", "OpenAIProvider"),
    "OpenRouterProvider": ("lsm.providers.openrouter", "OpenRouterProvider"),
    "AnthropicProvider": ("lsm.providers.anthropic", "AnthropicProvider"),
    "LocalProvider": ("lsm.providers.local", "LocalProvider"),
    "GeminiProvider": ("lsm.providers.gemini", "GeminiProvider"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lsm.providers' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()) | set(__all__))


__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "AnthropicProvider",
    "LocalProvider",
    "GeminiProvider",
    "create_provider",
    "list_available_providers",
    "register_provider",
]

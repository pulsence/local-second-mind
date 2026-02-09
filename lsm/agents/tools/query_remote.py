"""
Tool for querying a configured remote provider.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from lsm.config.models import LSMConfig, RemoteProviderConfig
from lsm.remote.factory import create_remote_provider

from .base import BaseTool


class QueryRemoteTool(BaseTool):
    """Query one remote provider with structured input."""

    name = "query_remote"
    description = "Query a configured remote provider with structured input."
    risk_level = "network"
    needs_network = True
    input_schema = {
        "type": "object",
        "properties": {
            "provider": {"type": "string", "description": "Configured remote provider name."},
            "input": {"type": "object", "description": "Structured provider input."},
            "max_results": {"type": "integer", "description": "Result limit override."},
        },
        "required": ["provider", "input"],
    }

    def __init__(self, config: LSMConfig) -> None:
        self.config = config

    def execute(self, args: Dict[str, Any]) -> str:
        provider_name = str(args.get("provider", "")).strip()
        if not provider_name:
            raise ValueError("provider is required")

        provider_input = args.get("input", {})
        if not isinstance(provider_input, dict):
            raise ValueError("input must be an object")

        max_results = int(args.get("max_results", 5))
        provider_cfg = self._find_provider(provider_name)
        provider = create_remote_provider(
            provider_cfg.type,
            self._provider_config_to_dict(provider_cfg),
        )
        results = provider.search_structured(
            provider_input,
            max_results=provider_cfg.max_results or max_results,
        )
        return json.dumps(results, indent=2)

    def _find_provider(self, provider_name: str) -> RemoteProviderConfig:
        for provider in self.config.remote_providers or []:
            if provider.name.lower() == provider_name.lower():
                return provider
        raise ValueError(f"Remote provider is not configured: {provider_name}")

    @staticmethod
    def _provider_config_to_dict(provider_cfg: RemoteProviderConfig) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "type": provider_cfg.type,
            "weight": provider_cfg.weight,
            "api_key": provider_cfg.api_key,
            "endpoint": provider_cfg.endpoint,
            "max_results": provider_cfg.max_results,
            "language": provider_cfg.language,
            "user_agent": provider_cfg.user_agent,
            "timeout": provider_cfg.timeout,
            "min_interval_seconds": provider_cfg.min_interval_seconds,
            "section_limit": provider_cfg.section_limit,
            "snippet_max_chars": provider_cfg.snippet_max_chars,
            "include_disambiguation": provider_cfg.include_disambiguation,
        }
        if provider_cfg.extra:
            config.update(provider_cfg.extra)
        return config

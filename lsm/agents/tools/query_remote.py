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
    """Query one specific remote provider with structured input.

    Each instance is parameterized by a single ``RemoteProviderConfig``.
    The tool name is ``query_<provider_name>`` so each source gets its own
    uniquely-named tool entry in the registry.
    """

    risk_level = "network"
    needs_network = True

    def __init__(self, provider_cfg: RemoteProviderConfig, config: LSMConfig) -> None:
        self._provider_cfg = provider_cfg
        self.config = config
        self.name = f"query_{provider_cfg.name}"
        self.description = (
            f"Query {provider_cfg.name} ({provider_cfg.type}) for structured information."
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "input": {"type": "object", "description": "Structured provider input."},
                "max_results": {"type": "integer", "description": "Result limit override."},
            },
            "required": ["input"],
        }

    def execute(self, args: Dict[str, Any]) -> str:
        provider_input = args.get("input", {})
        if not isinstance(provider_input, dict):
            raise ValueError("input must be an object")

        max_results = int(args.get("max_results", 5))
        provider_config = self._provider_config_to_dict(self._provider_cfg)
        if self.config.global_folder is not None:
            provider_config["global_folder"] = str(self.config.global_folder)
        provider = create_remote_provider(
            self._provider_cfg.type,
            provider_config,
        )
        results = provider.search_structured(
            provider_input,
            max_results=self._provider_cfg.max_results or max_results,
        )
        return json.dumps(results, indent=2)

    @staticmethod
    def _provider_config_to_dict(provider_cfg: RemoteProviderConfig) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "type": provider_cfg.type,
            "weight": provider_cfg.weight,
            "api_key": provider_cfg.api_key,
            "oauth": None,
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
        if provider_cfg.oauth is not None:
            config["oauth"] = {
                "client_id": provider_cfg.oauth.client_id,
                "client_secret": provider_cfg.oauth.client_secret,
                "scopes": list(provider_cfg.oauth.scopes or []),
                "redirect_uri": provider_cfg.oauth.redirect_uri,
                "refresh_buffer_seconds": provider_cfg.oauth.refresh_buffer_seconds,
            }
        else:
            config.pop("oauth", None)
        if provider_cfg.extra:
            config.update(provider_cfg.extra)
        return config

"""
Remote provider chain execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.config.models import LSMConfig, RemoteProviderChainConfig
from lsm.logging import get_logger
from lsm.remote.factory import create_remote_provider

logger = get_logger(__name__)


def _provider_runtime_config(
    provider_cfg: Any,
    *,
    global_folder: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build provider config preserving provider-specific passthrough keys."""
    runtime_config: Dict[str, Any] = {
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
    if global_folder is not None:
        runtime_config["global_folder"] = str(global_folder)
    if getattr(provider_cfg, "extra", None):
        runtime_config.update(provider_cfg.extra)
    return runtime_config


class RemoteProviderChain:
    """
    Execute a configured chain of remote providers.

    Each link can map output fields from the previous link into structured input
    fields for the next link.
    """

    def __init__(
        self,
        config: LSMConfig,
        chain_config: RemoteProviderChainConfig,
    ):
        self.config = config
        self.chain_config = chain_config

    def execute(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute chain links in order and return outputs from the final link.
        """
        current_inputs: List[Dict[str, Any]] = [dict(input_dict or {})]
        current_outputs: List[Dict[str, Any]] = []

        for index, link in enumerate(self.chain_config.links):
            provider_cfg = self._get_provider_config(link.source)
            provider = create_remote_provider(
                provider_cfg.type,
                _provider_runtime_config(provider_cfg, global_folder=self.config.global_folder),
            )

            current_outputs = []
            for source_input in current_inputs:
                if index == 0:
                    provider_input = source_input
                else:
                    provider_input = self._map_fields(source_input, link.map)
                if not provider_input:
                    continue
                try:
                    result = provider.search_structured(
                        provider_input,
                        max_results=provider_cfg.max_results or max_results,
                    )
                    if result:
                        current_outputs.extend(result)
                except Exception as exc:
                    logger.error(
                        f"Remote chain link '{link.source}' failed during execution: {exc}"
                    )

            current_inputs = current_outputs
            if not current_inputs:
                break

        return current_outputs

    def _get_provider_config(self, provider_name: str):
        for provider in self.config.remote_providers or []:
            if provider.name.lower() == provider_name.lower():
                return provider
        raise ValueError(f"Remote provider not configured for chain link: {provider_name}")

    @staticmethod
    def _map_fields(
        source: Dict[str, Any],
        mappings: List[str] | None,
    ) -> Dict[str, Any]:
        if not mappings:
            return dict(source)

        mapped: Dict[str, Any] = {}
        for mapping in mappings:
            src_field, dst_field = [part.strip() for part in mapping.split(":", 1)]
            value = source.get(src_field)
            if value is not None:
                mapped[dst_field] = value
        return mapped

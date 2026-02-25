"""
Preconfigured remote provider chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

from lsm.config.models import ChainLink, LSMConfig, RemoteProviderChainConfig
from lsm.logging import get_logger
from lsm.remote.chain import RemoteProviderChain
from lsm.remote.utils import filename_from_url, sanitize_filename

logger = get_logger(__name__)


class ScholarlyDiscoveryChain(RemoteProviderChain):
    """
    OpenAlex -> Crossref -> Unpaywall -> CORE pipeline with full-text download.
    """

    def execute(self, input_dict: dict, max_results: int = 5) -> list[dict]:
        results = super().execute(input_dict, max_results=max_results)
        self._download_full_text(results)
        return results

    def _download_full_text(self, results: list[dict]) -> None:
        global_folder = self.config.global_folder
        if global_folder is None:
            return
        download_dir = Path(global_folder) / "Downloads" / "scholarly"
        download_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata") or {}
            if metadata.get("downloaded_path"):
                continue
            url = self._pick_full_text_url(metadata)
            if not url:
                continue
            filename = filename_from_url(
                url,
                fallback=metadata.get("source_id")
                or result.get("title")
                or "full_text",
            )
            if "." not in filename:
                if "pdf" in url.lower():
                    filename = f"{filename}.pdf"
            filename = sanitize_filename(filename, fallback="full_text")
            path = download_dir / filename
            if path.exists():
                metadata["downloaded_path"] = str(path)
                result["metadata"] = metadata
                continue
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                path.write_bytes(response.content)
                metadata["downloaded_path"] = str(path)
                result["metadata"] = metadata
            except Exception as exc:
                logger.warning(f"Failed downloading full text from {url}: {exc}")

    @staticmethod
    def _pick_full_text_url(metadata: dict) -> Optional[str]:
        for key in ("pdf_url", "download_url", "full_text_url", "oa_url"):
            value = metadata.get(key)
            if value:
                return str(value)
        return None


@dataclass(frozen=True)
class PreconfiguredChain:
    name: str
    config: RemoteProviderChainConfig
    chain_cls: type[RemoteProviderChain]


def _scholarly_chain_config() -> RemoteProviderChainConfig:
    return RemoteProviderChainConfig(
        name="scholarly_discovery",
        agent_description=(
            "Discover scholarly works, enrich metadata, resolve open-access links, "
            "and retrieve full text."
        ),
        links=[
            ChainLink(source="openalex"),
            ChainLink(source="crossref"),
            ChainLink(source="unpaywall", map=["doi:doi"]),
            ChainLink(source="core", map=["doi:doi"]),
        ],
    )


_PRECONFIGURED = {
    "scholarly_discovery": PreconfiguredChain(
        name="scholarly_discovery",
        config=_scholarly_chain_config(),
        chain_cls=ScholarlyDiscoveryChain,
    )
}


def get_preconfigured_chain_configs(
    enabled: Optional[Iterable[str]],
) -> list[RemoteProviderChainConfig]:
    if not enabled:
        return []
    normalized = {str(name).strip().lower() for name in enabled if str(name).strip()}
    configs: list[RemoteProviderChainConfig] = []
    for key, entry in _PRECONFIGURED.items():
        if key in normalized:
            configs.append(entry.config)
    return configs


def get_preconfigured_chain(name: str) -> Optional[PreconfiguredChain]:
    if not name:
        return None
    return _PRECONFIGURED.get(str(name).strip().lower())


def build_chain(
    config: LSMConfig,
    chain_config: RemoteProviderChainConfig,
) -> RemoteProviderChain:
    entry = get_preconfigured_chain(chain_config.name)
    if entry is None:
        return RemoteProviderChain(config=config, chain_config=chain_config)
    return entry.chain_cls(config=config, chain_config=chain_config)

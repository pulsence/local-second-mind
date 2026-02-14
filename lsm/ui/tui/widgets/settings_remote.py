"""Remote settings tab widget."""

from __future__ import annotations

import re
from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

from lsm.config.models import ChainLink, RemoteProviderChainConfig, RemoteProviderConfig, RemoteProviderRef
from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class RemoteSettingsTab(BaseSettingsTab):
    """Settings view for remote providers and chains."""

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Remote Providers and Chains", classes="settings-section-title"),
            Static("Remote Chains", classes="settings-subsection-title"),
            Vertical(id="settings-remote-chains-list"),
            Horizontal(Button("Add Chain", id="settings-remote-chain-add"), classes="settings-actions"),
            Static("Remote Providers", classes="settings-subsection-title"),
            Vertical(id="settings-remote-providers-list"),
            Horizontal(Button("Add Provider", id="settings-remote-provider-add"), classes="settings-actions"),
            self._save_reset_row("remote"),
            classes="settings-section",
        )

    def refresh_fields(self, config: Any) -> None:
        has_providers_container = self._container_exists("#settings-remote-providers-list")
        has_chains_container = self._container_exists("#settings-remote-chains-list")
        if not has_providers_container and not has_chains_container:
            return

        providers = list(getattr(config, "remote_providers", None) or [])
        if has_providers_container:
            provider_widgets: list[Widget] = []
            if not providers:
                provider_widgets.append(Static("No remote providers configured.", classes="settings-label"))

            for idx, provider in enumerate(providers):
                provider_widgets.append(
                    Container(
                        Static(f"Provider {idx + 1}", classes="settings-subsection-title"),
                        self._field(
                            "Name",
                            f"settings-remote-provider-{idx}-name",
                            value=str(getattr(provider, "name", "")),
                        ),
                        self._field(
                            "Type",
                            f"settings-remote-provider-{idx}-type",
                            value=str(getattr(provider, "type", "")),
                        ),
                        self._field(
                            "Weight",
                            f"settings-remote-provider-{idx}-weight",
                            value=str(getattr(provider, "weight", "")),
                        ),
                        self._field(
                            "API key",
                            f"settings-remote-provider-{idx}-api-key",
                            value=self._format_optional(getattr(provider, "api_key", None)),
                        ),
                        self._field(
                            "Endpoint",
                            f"settings-remote-provider-{idx}-endpoint",
                            value=self._format_optional(getattr(provider, "endpoint", None)),
                        ),
                        self._field(
                            "Max results",
                            f"settings-remote-provider-{idx}-max-results",
                            value=self._format_optional(getattr(provider, "max_results", None)),
                        ),
                        self._field(
                            "Language",
                            f"settings-remote-provider-{idx}-language",
                            value=self._format_optional(getattr(provider, "language", None)),
                        ),
                        self._field(
                            "User agent",
                            f"settings-remote-provider-{idx}-user-agent",
                            value=self._format_optional(getattr(provider, "user_agent", None)),
                        ),
                        self._field(
                            "Timeout (s)",
                            f"settings-remote-provider-{idx}-timeout",
                            value=self._format_optional(getattr(provider, "timeout", None)),
                        ),
                        self._field(
                            "Min interval (s)",
                            f"settings-remote-provider-{idx}-min-interval",
                            value=self._format_optional(getattr(provider, "min_interval_seconds", None)),
                        ),
                        self._field(
                            "Section limit",
                            f"settings-remote-provider-{idx}-section-limit",
                            value=self._format_optional(getattr(provider, "section_limit", None)),
                        ),
                        self._field(
                            "Snippet max chars",
                            f"settings-remote-provider-{idx}-snippet-max-chars",
                            value=self._format_optional(getattr(provider, "snippet_max_chars", None)),
                        ),
                        self._field(
                            "Include disambiguation",
                            f"settings-remote-provider-{idx}-include-disambiguation",
                            value=self._format_optional(getattr(provider, "include_disambiguation", None)),
                        ),
                        self._field(
                            "Cache results",
                            f"settings-remote-provider-{idx}-cache-results",
                            field_type="switch",
                            value=bool(getattr(provider, "cache_results", False)),
                        ),
                        self._field(
                            "Cache TTL",
                            f"settings-remote-provider-{idx}-cache-ttl",
                            value=str(getattr(provider, "cache_ttl", "")),
                        ),
                        Horizontal(
                            Button("Remove Provider", id=f"settings-remote-provider-remove-{idx}"),
                            classes="settings-actions",
                        ),
                        classes="settings-subsection",
                    )
                )

            self._replace_container_children("#settings-remote-providers-list", provider_widgets)

        chains = list(getattr(config, "remote_provider_chains", None) or [])
        if has_chains_container:
            chain_widgets: list[Widget] = []
            if not chains:
                chain_widgets.append(Static("No remote chains configured.", classes="settings-label"))

            for chain_idx, chain in enumerate(chains):
                link_widgets: list[Widget] = []
                for link_idx, link in enumerate(getattr(chain, "links", None) or []):
                    link_widgets.append(
                        Container(
                            Static(f"Link {link_idx + 1}", classes="settings-subsection-title"),
                            self._field(
                                "Source",
                                f"settings-remote-chain-{chain_idx}-link-{link_idx}-source",
                                value=str(getattr(link, "source", "")),
                            ),
                            self._field(
                                "Map (comma-separated output:input)",
                                f"settings-remote-chain-{chain_idx}-link-{link_idx}-map",
                                value=", ".join(getattr(link, "map", None) or []),
                            ),
                            Horizontal(
                                Button(
                                    "Remove Link",
                                    id=f"settings-remote-chain-{chain_idx}-link-remove-{link_idx}",
                                ),
                                classes="settings-actions",
                            ),
                            classes="settings-subsection",
                        )
                    )

                chain_widgets.append(
                    Container(
                        Static(f"Chain {chain_idx + 1}", classes="settings-subsection-title"),
                        self._field(
                            "Name",
                            f"settings-remote-chain-{chain_idx}-name",
                            value=str(getattr(chain, "name", "")),
                        ),
                        self._field(
                            "Agent description",
                            f"settings-remote-chain-{chain_idx}-agent-description",
                            value=str(getattr(chain, "agent_description", "")),
                        ),
                        *link_widgets,
                        Horizontal(
                            Button("Add Link", id=f"settings-remote-chain-{chain_idx}-link-add"),
                            classes="settings-actions",
                        ),
                        Horizontal(
                            Button("Remove Chain", id=f"settings-remote-chain-remove-{chain_idx}"),
                            classes="settings-actions",
                        ),
                        classes="settings-subsection",
                    )
                )

            self._replace_container_children("#settings-remote-chains-list", chain_widgets)

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        text = str(value or "").strip()

        provider_match = re.fullmatch(r"settings-remote-provider-(\d+)-([a-z-]+)", field_id)
        if provider_match:
            idx = int(provider_match.group(1))
            field = provider_match.group(2)
            providers = config.remote_providers or []
            if idx >= len(providers):
                return True
            provider = providers[idx]
            if field == "name":
                if not text:
                    raise ValueError("provider name cannot be empty")
                if provider.name != text:
                    old_name = provider.name
                    provider.name = text
                    self._rename_provider_refs(config, old_name, text)
            elif field == "type":
                if not text:
                    raise ValueError("provider type cannot be empty")
                provider.type = text
            elif field == "weight":
                provider.weight = float(text) if text else 1.0
            elif field == "api-key":
                provider.api_key = text or None
            elif field == "endpoint":
                provider.endpoint = text or None
            elif field == "max-results":
                provider.max_results = int(text) if text else None
            elif field == "language":
                provider.language = text or None
            elif field == "user-agent":
                provider.user_agent = text or None
            elif field == "timeout":
                provider.timeout = int(text) if text else None
            elif field == "min-interval":
                provider.min_interval_seconds = float(text) if text else None
            elif field == "section-limit":
                provider.section_limit = int(text) if text else None
            elif field == "snippet-max-chars":
                provider.snippet_max_chars = int(text) if text else None
            elif field == "include-disambiguation":
                provider.include_disambiguation = self._parse_optional_bool(text)
            elif field == "cache-results":
                provider.cache_results = bool(value)
            elif field == "cache-ttl":
                provider.cache_ttl = int(text) if text else 86400
            return True

        chain_match = re.fullmatch(r"settings-remote-chain-(\d+)-([a-z-]+)", field_id)
        if chain_match:
            idx = int(chain_match.group(1))
            field = chain_match.group(2)
            chains = config.remote_provider_chains or []
            if idx >= len(chains):
                return True
            chain = chains[idx]
            if field == "name":
                if not text:
                    raise ValueError("chain name cannot be empty")
                chain.name = text
            elif field == "agent-description":
                chain.agent_description = text
            return True

        link_match = re.fullmatch(r"settings-remote-chain-(\d+)-link-(\d+)-([a-z-]+)", field_id)
        if link_match:
            chain_idx = int(link_match.group(1))
            link_idx = int(link_match.group(2))
            field = link_match.group(3)
            chains = config.remote_provider_chains or []
            if chain_idx >= len(chains):
                return True
            links = chains[chain_idx].links
            if link_idx >= len(links):
                return True
            link = links[link_idx]
            if field == "source":
                link.source = text
            elif field == "map":
                link.map = self._parse_csv(text)
            return True

        return False

    def handle_button(self, button_id: str, config: Any) -> bool:
        if button_id == "settings-remote-provider-add":
            self._add_provider(config)
            return True
        if button_id == "settings-remote-chain-add":
            self._add_chain(config)
            return True

        provider_remove = re.fullmatch(r"settings-remote-provider-remove-(\d+)", button_id)
        if provider_remove:
            self._remove_provider(config, int(provider_remove.group(1)))
            return True

        chain_remove = re.fullmatch(r"settings-remote-chain-remove-(\d+)", button_id)
        if chain_remove:
            self._remove_chain(config, int(chain_remove.group(1)))
            return True

        add_link = re.fullmatch(r"settings-remote-chain-(\d+)-link-add", button_id)
        if add_link:
            self._add_chain_link(config, int(add_link.group(1)))
            return True

        remove_link = re.fullmatch(r"settings-remote-chain-(\d+)-link-remove-(\d+)", button_id)
        if remove_link:
            self._remove_chain_link(config, int(remove_link.group(1)), int(remove_link.group(2)))
            return True

        return False

    def _add_provider(self, config: Any) -> None:
        if config.remote_providers is None:
            config.remote_providers = []
        existing = {p.name for p in config.remote_providers}
        name = self._next_name(existing, "provider")
        config.remote_providers.append(RemoteProviderConfig(name=name, type="web_search"))
        self.refresh_fields(config)
        self.post_status(f"Added remote provider '{name}'", False)

    def _remove_provider(self, config: Any, idx: int) -> None:
        providers = config.remote_providers or []
        if idx >= len(providers):
            return

        removed = providers.pop(idx).name
        if not providers:
            config.remote_providers = None

        for mode_config in (config.modes or {}).values():
            refs = mode_config.source_policy.remote.remote_providers
            if not refs:
                continue
            filtered = []
            for ref in refs:
                if isinstance(ref, RemoteProviderRef):
                    if ref.source.lower() != removed.lower():
                        filtered.append(ref)
                elif str(ref).lower() != removed.lower():
                    filtered.append(ref)
            mode_config.source_policy.remote.remote_providers = filtered or None

        if config.remote_provider_chains:
            cleaned = []
            for chain in config.remote_provider_chains:
                chain.links = [link for link in chain.links if link.source.lower() != removed.lower()]
                if chain.links:
                    cleaned.append(chain)
            config.remote_provider_chains = cleaned or None

        self.refresh_fields(config)
        self.post_status(f"Removed remote provider '{removed}'", False)

    def _add_chain(self, config: Any) -> None:
        if config.remote_provider_chains is None:
            config.remote_provider_chains = []

        existing = {c.name for c in config.remote_provider_chains}
        name = self._next_name(existing, "chain")

        default_source = ""
        if config.remote_providers:
            default_source = config.remote_providers[0].name

        config.remote_provider_chains.append(
            RemoteProviderChainConfig(name=name, links=[ChainLink(source=default_source)])
        )
        self.refresh_fields(config)
        self.post_status(f"Added remote chain '{name}'", False)

    def _remove_chain(self, config: Any, idx: int) -> None:
        chains = config.remote_provider_chains or []
        if idx >= len(chains):
            return

        removed = chains.pop(idx).name
        if not chains:
            config.remote_provider_chains = None

        self.refresh_fields(config)
        self.post_status(f"Removed remote chain '{removed}'", False)

    def _add_chain_link(self, config: Any, chain_idx: int) -> None:
        chains = config.remote_provider_chains or []
        if chain_idx >= len(chains):
            return

        default_source = ""
        if config.remote_providers:
            default_source = config.remote_providers[0].name

        chains[chain_idx].links.append(ChainLink(source=default_source))
        self.refresh_fields(config)
        self.post_status(f"Added link to chain '{chains[chain_idx].name}'", False)

    def _remove_chain_link(self, config: Any, chain_idx: int, link_idx: int) -> None:
        chains = config.remote_provider_chains or []
        if chain_idx >= len(chains):
            return

        links = chains[chain_idx].links
        if link_idx >= len(links):
            return
        if len(links) <= 1:
            raise ValueError("A chain must keep at least one link.")

        links.pop(link_idx)
        self.refresh_fields(config)
        self.post_status(f"Removed link from chain '{chains[chain_idx].name}'", False)

    def _rename_provider_refs(self, config: Any, old_name: str, new_name: str) -> None:
        for mode_config in (config.modes or {}).values():
            refs = mode_config.source_policy.remote.remote_providers or []
            for ref_idx, ref in enumerate(refs):
                if isinstance(ref, RemoteProviderRef):
                    if ref.source == old_name:
                        ref.source = new_name
                elif str(ref) == old_name:
                    refs[ref_idx] = new_name

        for chain in config.remote_provider_chains or []:
            for link in chain.links:
                if link.source == old_name:
                    link.source = new_name

    def _container_exists(self, selector: str) -> bool:
        try:
            self.query_one(selector)
            return True
        except Exception:
            return False

    @staticmethod
    def _next_name(existing_names: set[str], prefix: str) -> str:
        index = 1
        while True:
            candidate = f"{prefix}_{index}"
            if candidate not in existing_names:
                return candidate
            index += 1

    @staticmethod
    def _parse_optional_bool(value: str) -> Optional[bool]:
        text = (value or "").strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError("expected true/false or blank")

    @staticmethod
    def _parse_csv(text: str) -> Optional[list[str]]:
        values = [item.strip() for item in text.split(",") if item.strip()]
        return values or None

    @staticmethod
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)

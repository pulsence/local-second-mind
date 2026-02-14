"""LLM settings tab widget."""

from __future__ import annotations

import re
from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

from lsm.config.models import LLMProviderConfig, LLMServiceConfig
from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class LLMSettingsTab(BaseSettingsTab):
    """Settings view for LLM providers and services."""

    _LLM_COMMON_SERVICES = (
        "default",
        "query",
        "decomposition",
        "tagging",
        "ranking",
        "translation",
    )

    def compose(self) -> ComposeResult:
        yield Container(
            Static("LLM Providers and Services", classes="settings-section-title"),
            Static("Services", classes="settings-subsection-title"),
            Vertical(id="settings-llm-services-list"),
            Horizontal(Button("Add Service", id="settings-llm-service-add"), classes="settings-actions"),
            Static("Providers", classes="settings-subsection-title"),
            Vertical(id="settings-llm-providers-list"),
            Horizontal(Button("Add Provider", id="settings-llm-provider-add"), classes="settings-actions"),
            self._save_reset_row("llm"),
            classes="settings-section",
        )

    def refresh_fields(self, config: Any) -> None:
        llm_cfg = getattr(config, "llm", None)
        if llm_cfg is None:
            return

        has_providers_container = self._container_exists("#settings-llm-providers-list")
        has_services_container = self._container_exists("#settings-llm-services-list")
        if not has_providers_container and not has_services_container:
            return

        providers = list(getattr(llm_cfg, "providers", []) or [])
        services = getattr(llm_cfg, "services", {}) or {}

        if has_providers_container:
            provider_widgets: list[Widget] = []
            if not providers:
                provider_widgets.append(Static("No providers configured.", classes="settings-label"))

            for idx, provider in enumerate(providers):
                provider_widgets.append(
                    Container(
                        Static(f"Provider {idx + 1}", classes="settings-subsection-title"),
                        self._field(
                            "Name",
                            f"settings-llm-provider-{idx}-name",
                            value=str(getattr(provider, "provider_name", "")),
                        ),
                        self._field(
                            "API key",
                            f"settings-llm-provider-{idx}-api-key",
                            value=self._format_optional(getattr(provider, "api_key", None)),
                        ),
                        self._field(
                            "Base URL",
                            f"settings-llm-provider-{idx}-base-url",
                            value=self._format_optional(getattr(provider, "base_url", None)),
                        ),
                        self._field(
                            "Endpoint",
                            f"settings-llm-provider-{idx}-endpoint",
                            value=self._format_optional(getattr(provider, "endpoint", None)),
                        ),
                        self._field(
                            "API version",
                            f"settings-llm-provider-{idx}-api-version",
                            value=self._format_optional(getattr(provider, "api_version", None)),
                        ),
                        self._field(
                            "Deployment",
                            f"settings-llm-provider-{idx}-deployment-name",
                            value=self._format_optional(getattr(provider, "deployment_name", None)),
                        ),
                        Horizontal(
                            Button("Remove Provider", id=f"settings-llm-provider-remove-{idx}"),
                            classes="settings-actions",
                        ),
                        classes="settings-subsection",
                    )
                )

            self._replace_container_children("#settings-llm-providers-list", provider_widgets)

        if has_services_container:
            service_names = self._ordered_service_names(config)
            service_widgets: list[Widget] = []
            if not service_names:
                service_widgets.append(Static("No services configured.", classes="settings-label"))

            for idx, service_name in enumerate(service_names):
                service = services.get(service_name)
                if service is None:
                    continue
                service_widgets.append(
                    Container(
                        Static(f"Service {idx + 1}", classes="settings-subsection-title"),
                        Static(f"Name: {service_name}", classes="settings-label"),
                        self._field(
                            "Provider",
                            f"settings-llm-service-{idx}-provider",
                            value=str(getattr(service, "provider", "")),
                        ),
                        self._field(
                            "Model",
                            f"settings-llm-service-{idx}-model",
                            value=str(getattr(service, "model", "")),
                        ),
                        self._field(
                            "Temperature",
                            f"settings-llm-service-{idx}-temperature",
                            value=self._format_optional(getattr(service, "temperature", None)),
                        ),
                        self._field(
                            "Max tokens",
                            f"settings-llm-service-{idx}-max-tokens",
                            value=self._format_optional(getattr(service, "max_tokens", None)),
                        ),
                        Horizontal(
                            Button("Remove Service", id=f"settings-llm-service-remove-{idx}"),
                            classes="settings-actions",
                        ),
                        classes="settings-subsection",
                    )
                )

            self._replace_container_children("#settings-llm-services-list", service_widgets)

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        llm_cfg = getattr(config, "llm", None)
        if llm_cfg is None:
            return False

        text = str(value or "").strip()

        provider_match = re.fullmatch(r"settings-llm-provider-(\d+)-([a-z-]+)", field_id)
        if provider_match:
            idx = int(provider_match.group(1))
            field = provider_match.group(2)
            providers = llm_cfg.providers or []
            if idx >= len(providers):
                return True
            provider = providers[idx]
            if field == "name":
                if not text:
                    raise ValueError("provider name cannot be empty")
                if provider.provider_name != text:
                    old_name = provider.provider_name
                    provider.provider_name = text
                    for service in llm_cfg.services.values():
                        if service.provider == old_name:
                            service.provider = text
            elif field == "api-key":
                provider.api_key = text or None
            elif field == "base-url":
                provider.base_url = text or None
            elif field == "endpoint":
                provider.endpoint = text or None
            elif field == "api-version":
                provider.api_version = text or None
            elif field == "deployment-name":
                provider.deployment_name = text or None
            return True

        service_match = re.fullmatch(r"settings-llm-service-(\d+)-([a-z-]+)", field_id)
        if service_match:
            idx = int(service_match.group(1))
            field = service_match.group(2)
            service_names = self._ordered_service_names(config)
            if idx >= len(service_names):
                return True
            service = llm_cfg.services[service_names[idx]]
            if field == "provider":
                service.provider = text
            elif field == "model":
                service.model = text
            elif field == "temperature":
                service.temperature = float(text) if text else None
            elif field == "max-tokens":
                service.max_tokens = int(text) if text else None
            return True

        return False

    def handle_button(self, button_id: str, config: Any) -> bool:
        if button_id == "settings-llm-provider-add":
            self._add_provider(config)
            return True
        if button_id == "settings-llm-service-add":
            self._add_service(config)
            return True

        provider_remove = re.fullmatch(r"settings-llm-provider-remove-(\d+)", button_id)
        if provider_remove:
            self._remove_provider(config, int(provider_remove.group(1)))
            return True

        service_remove = re.fullmatch(r"settings-llm-service-remove-(\d+)", button_id)
        if service_remove:
            self._remove_service(config, int(service_remove.group(1)))
            return True

        return False

    def _add_provider(self, config: Any) -> None:
        llm_cfg = config.llm
        existing = {p.provider_name for p in (llm_cfg.providers or [])}
        name = self._next_name(existing, "provider")
        llm_cfg.providers.append(LLMProviderConfig(provider_name=name))
        if not llm_cfg.services:
            llm_cfg.services["default"] = LLMServiceConfig(provider=name, model="gpt-5.2")
        self.refresh_fields(config)
        self.post_status(f"Added LLM provider '{name}'", False)

    def _remove_provider(self, config: Any, idx: int) -> None:
        llm_cfg = config.llm
        providers = llm_cfg.providers or []
        if idx >= len(providers):
            return
        if len(providers) <= 1:
            raise ValueError("At least one LLM provider is required.")

        removed = providers.pop(idx).provider_name
        fallback_provider = providers[0].provider_name
        for service in llm_cfg.services.values():
            if service.provider == removed:
                service.provider = fallback_provider

        self.refresh_fields(config)
        self.post_status(f"Removed LLM provider '{removed}'", False)

    def _add_service(self, config: Any) -> None:
        llm_cfg = config.llm
        provider_name = llm_cfg.providers[0].provider_name if llm_cfg.providers else ""
        model = "gpt-5.2"
        default_service = llm_cfg.services.get("default")
        if default_service is not None and default_service.model:
            model = default_service.model

        existing = set(llm_cfg.services.keys())
        name = self._next_name(existing, "service")
        llm_cfg.services[name] = LLMServiceConfig(provider=provider_name, model=model)
        self.refresh_fields(config)
        self.post_status(f"Added LLM service '{name}'", False)

    def _remove_service(self, config: Any, idx: int) -> None:
        llm_cfg = config.llm
        service_names = self._ordered_service_names(config)
        if idx >= len(service_names):
            return
        if len(llm_cfg.services) <= 1:
            raise ValueError("At least one LLM service is required.")

        removed = service_names[idx]
        llm_cfg.services.pop(removed, None)
        self.refresh_fields(config)
        self.post_status(f"Removed LLM service '{removed}'", False)

    def _ordered_service_names(self, config: Any) -> list[str]:
        services = getattr(config.llm, "services", {}) or {}
        names = list(services.keys())
        ordered: list[str] = []
        for service_name in self._LLM_COMMON_SERVICES:
            if service_name in services:
                ordered.append(service_name)
        for service_name in names:
            if service_name not in ordered:
                ordered.append(service_name)
        return ordered

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
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)

"""Modes settings tab widget."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from lsm.config.models import RemoteProviderRef
from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class ModesSettingsTab(BaseSettingsTab):
    """Settings view for mode inspection."""

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Mode Browser", classes="settings-section-title"),
            self._select_field("Active mode", "settings-modes-mode"),
            self._field("Synthesis style", "settings-modes-synthesis-style", disabled=True),
            self._field("Local policy", "settings-modes-local-policy", disabled=True),
            self._field("Remote policy", "settings-modes-remote-policy", disabled=True),
            self._field("Model knowledge", "settings-modes-model-policy", disabled=True),
            self._save_reset_row("modes"),
            classes="settings-section",
        )

    def refresh_fields(self, config: Any) -> None:
        modes = getattr(config, "modes", None) or {}
        self._set_select_options("settings-modes-mode", list(modes.keys()))

        query = getattr(config, "query", None)
        if query is None:
            return

        mode_name = str(getattr(query, "mode", ""))
        self._set_select_value("settings-modes-mode", mode_name)
        self._update_mode_display(config, mode_name)

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        if field_id != "settings-modes-mode":
            return False

        query = getattr(config, "query", None)
        if query is None:
            return False

        mode_name = str(value or "").strip()
        query.mode = mode_name
        self._update_mode_display(config, mode_name)
        return True

    def _update_mode_display(self, config: Any, mode_name: str) -> None:
        if not mode_name:
            return

        try:
            mode = config.get_mode_config(mode_name)
        except Exception:
            return

        local = mode.source_policy.local
        remote = mode.source_policy.remote
        model = mode.source_policy.model_knowledge

        remote_providers = []
        for item in remote.remote_providers or []:
            if isinstance(item, RemoteProviderRef):
                if item.weight is not None:
                    remote_providers.append(f"{item.source} ({item.weight})")
                else:
                    remote_providers.append(item.source)
            else:
                remote_providers.append(str(item))

        self._set_input("settings-modes-synthesis-style", mode.synthesis_style)
        self._set_input(
            "settings-modes-local-policy",
            f"enabled={local.enabled}, min={local.min_relevance}, k={local.k}, k_rerank={local.k_rerank}",
        )
        self._set_input(
            "settings-modes-remote-policy",
            "enabled={enabled}, rank={rank}, max={max_results}, providers={providers}".format(
                enabled=remote.enabled,
                rank=remote.rank_strategy,
                max_results=remote.max_results,
                providers=remote_providers,
            ),
        )
        self._set_input(
            "settings-modes-model-policy",
            f"enabled={model.enabled}, require_label={model.require_label}",
        )

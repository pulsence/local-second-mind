"""Global TUI application state container."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

ContextType = Literal["ingest", "query", "settings", "remote", "agents"]
DensityMode = Literal["auto", "compact", "comfortable"]
Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class Notification:
    """Single UI notification event."""

    message: str
    severity: Severity = "info"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class AppStateSnapshot:
    """Immutable snapshot of global UI state."""

    active_context: ContextType
    density_mode: DensityMode
    selected_agent_id: Optional[str]
    notifications: tuple[Notification, ...]


class AppState:
    """Typed mutable container for cross-screen UI state."""

    def __init__(
        self,
        *,
        active_context: ContextType = "query",
        density_mode: DensityMode = "auto",
        selected_agent_id: Optional[str] = None,
    ) -> None:
        self._active_context: ContextType = active_context
        self._density_mode: DensityMode = density_mode
        self._selected_agent_id: Optional[str] = selected_agent_id
        self._notifications: list[Notification] = []

    @property
    def active_context(self) -> ContextType:
        return self._active_context

    @property
    def density_mode(self) -> DensityMode:
        return self._density_mode

    @property
    def selected_agent_id(self) -> Optional[str]:
        return self._selected_agent_id

    @property
    def notifications(self) -> tuple[Notification, ...]:
        return tuple(self._notifications)

    def snapshot(self) -> AppStateSnapshot:
        """Return an immutable snapshot of current app UI state."""
        return AppStateSnapshot(
            active_context=self._active_context,
            density_mode=self._density_mode,
            selected_agent_id=self._selected_agent_id,
            notifications=tuple(self._notifications),
        )

    def set_active_context(self, context: ContextType) -> None:
        self._active_context = context

    def set_density_mode(self, mode: DensityMode) -> None:
        self._density_mode = mode

    def set_selected_agent_id(self, agent_id: Optional[str]) -> None:
        self._selected_agent_id = agent_id

    def push_notification(self, message: str, *, severity: Severity = "info") -> Notification:
        notification = Notification(message=message, severity=severity)
        self._notifications.append(notification)
        return notification

    def clear_notifications(self) -> None:
        self._notifications.clear()

    def drain_notifications(self) -> list[Notification]:
        items = list(self._notifications)
        self._notifications.clear()
        return items

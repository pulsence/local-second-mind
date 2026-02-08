"""
Global chat transcript configuration model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatsConfig:
    """
    Global configuration for query chat transcript saving.
    """

    enabled: bool = True
    """Whether chat transcript saving is enabled."""

    dir: str = "Chats"
    """Directory name for chat transcripts, relative to global folder if not absolute."""

    auto_save: bool = True
    """Whether chat transcripts should auto-save after each chat turn."""

    format: str = "markdown"
    """Transcript output format. Currently supports 'markdown'."""

    def __post_init__(self) -> None:
        self.dir = str(self.dir).strip() or "Chats"
        self.format = str(self.format).strip().lower() or "markdown"

    def validate(self) -> None:
        """Validate chat configuration."""
        if not self.dir or not self.dir.strip():
            raise ValueError("chats.dir cannot be empty")
        if self.format not in {"markdown"}:
            raise ValueError("chats.format must be 'markdown'")

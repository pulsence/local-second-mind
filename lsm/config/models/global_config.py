"""
Global configuration model for Local Second Mind.

Contains settings shared across multiple modules (ingest, query, agents, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Dict, List, Any

from lsm.paths import get_global_folder

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EMBED_MODEL,
    WELL_KNOWN_EMBED_MODELS,
)


@dataclass
class MCPServerConfig:
    """
    Configuration for a single MCP server connection.
    """

    name: str
    """Server identifier used for tool names."""

    command: str
    """Command to launch the MCP server."""

    args: List[str] = field(default_factory=list)
    """Command arguments."""

    env: Dict[str, str] = field(default_factory=dict)
    """Environment variables applied to the MCP server process."""

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        self.command = str(self.command or "").strip()
        self.args = [str(arg).strip() for arg in (self.args or []) if str(arg).strip()]
        self.env = {
            str(key).strip(): str(value)
            for key, value in (self.env or {}).items()
            if str(key).strip()
        }

    def validate(self) -> None:
        if not self.name:
            raise ValueError("mcp_servers[].name is required")
        if not self.command:
            raise ValueError(f"mcp_servers[{self.name or '?'}].command is required")


@dataclass
class GlobalConfig:
    """
    Global configuration settings used by multiple modules.

    These fields are shared across ingest, query, agents, and other subsystems.
    """

    global_folder: Optional[Path] = None
    """Global Local Second Mind folder for notes, chats, agents, etc."""

    embed_model: str = DEFAULT_EMBED_MODEL
    """Sentence-transformers model for embeddings."""

    device: str = DEFAULT_DEVICE
    """Device for embedding: 'cpu', 'cuda', 'cuda:0', etc."""

    batch_size: int = DEFAULT_BATCH_SIZE
    """Batch size for embedding operations."""

    embedding_dimension: Optional[int] = None
    """Embedding vector dimension. Auto-detected from well-known models if None."""

    tui_density_mode: Literal["auto", "compact", "comfortable"] = "auto"
    """TUI layout density mode: auto, compact, or comfortable."""

    mcp_servers: List[MCPServerConfig] = field(default_factory=list)
    """Configured MCP servers for tool discovery and execution."""

    def __post_init__(self):
        """Normalize paths and load environment variable overrides."""
        if isinstance(self.global_folder, str):
            self.global_folder = Path(self.global_folder)

        if self.global_folder is None:
            env_folder = os.environ.get("LSM_GLOBAL_FOLDER")
            if env_folder:
                self.global_folder = Path(env_folder).expanduser().resolve()
            else:
                self.global_folder = get_global_folder()
        else:
            self.global_folder = self.global_folder.expanduser().resolve()

        env_model = os.environ.get("LSM_EMBED_MODEL")
        if env_model:
            self.embed_model = env_model

        env_device = os.environ.get("LSM_DEVICE")
        if env_device:
            self.device = env_device

        # Auto-detect embedding dimension from well-known models
        if self.embedding_dimension is None:
            self.embedding_dimension = WELL_KNOWN_EMBED_MODELS.get(self.embed_model)

        servers: List[MCPServerConfig] = []
        for entry in self.mcp_servers or []:
            if isinstance(entry, MCPServerConfig):
                servers.append(entry)
            elif isinstance(entry, dict):
                servers.append(MCPServerConfig(**entry))
            else:
                servers.append(MCPServerConfig(name=str(entry), command=""))
        self.mcp_servers = servers

    def validate(self) -> None:
        """Validate global configuration.

        Raises:
            ValueError: If any field has an invalid value.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.embedding_dimension is not None and self.embedding_dimension < 1:
            raise ValueError(
                f"embedding_dimension must be positive, got {self.embedding_dimension}"
            )

        valid_devices = {"cpu", "cuda", "mps"}
        device_base = self.device.split(":")[0]
        if device_base not in valid_devices:
            raise ValueError(
                f"device must start with one of {valid_devices}, got '{self.device}'"
            )

        if self.tui_density_mode not in {"auto", "compact", "comfortable"}:
            raise ValueError(
                "tui_density_mode must be one of {'auto', 'compact', 'comfortable'}"
            )

        seen_names = set()
        for server in self.mcp_servers:
            server.validate()
            if server.name in seen_names:
                raise ValueError(f"Duplicate MCP server name: {server.name}")
            seen_names.add(server.name)

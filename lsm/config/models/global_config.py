"""
Global configuration model for Local Second Mind.

Contains settings shared across multiple modules (ingest, query, agents, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lsm.paths import get_global_folder

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EMBED_MODEL,
)


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

    def validate(self) -> None:
        """Validate global configuration.

        Raises:
            ValueError: If any field has an invalid value.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        valid_devices = {"cpu", "cuda", "mps"}
        device_base = self.device.split(":")[0]
        if device_base not in valid_devices:
            raise ValueError(
                f"device must start with one of {valid_devices}, got '{self.device}'"
            )

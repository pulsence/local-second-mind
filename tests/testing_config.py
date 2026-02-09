"""Test runtime configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


VALID_TEST_TIERS = {"smoke", "integration", "live"}
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEST_TIER = "smoke"


def _env(name: str) -> Optional[str]:
    """Return a stripped env value or None when unset/empty."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class TestConfig:
    """Resolved test runtime configuration values."""

    config_path: Optional[Path]
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    google_api_key: Optional[str]
    ollama_base_url: Optional[str]
    brave_api_key: Optional[str]
    semantic_scholar_api_key: Optional[str]
    core_api_key: Optional[str]
    embed_model: str
    tier: str

    def validate(self) -> None:
        """Validate resolved configuration."""
        if self.tier not in VALID_TEST_TIERS:
            raise ValueError(
                f"Invalid LSM_TEST_TIER '{self.tier}'. "
                f"Expected one of: {sorted(VALID_TEST_TIERS)}"
            )
        if not self.embed_model.strip():
            raise ValueError("LSM_TEST_EMBED_MODEL must not be empty")


def load_test_config(env_path: Optional[Path] = None) -> TestConfig:
    """
    Load test configuration from `.env.test` and `LSM_TEST_*` variables.

    Args:
        env_path: Optional explicit dotenv path for tests.
    """
    dotenv_path = env_path or (Path(__file__).parent / ".env.test")
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)

    raw_config_path = _env("LSM_TEST_CONFIG")
    config_path = Path(raw_config_path).expanduser() if raw_config_path else None
    embed_model = _env("LSM_TEST_EMBED_MODEL") or DEFAULT_EMBED_MODEL
    tier = (_env("LSM_TEST_TIER") or DEFAULT_TEST_TIER).lower()

    config = TestConfig(
        config_path=config_path,
        openai_api_key=_env("LSM_TEST_OPENAI_API_KEY"),
        anthropic_api_key=_env("LSM_TEST_ANTHROPIC_API_KEY"),
        google_api_key=_env("LSM_TEST_GOOGLE_API_KEY"),
        ollama_base_url=_env("LSM_TEST_OLLAMA_BASE_URL"),
        brave_api_key=_env("LSM_TEST_BRAVE_API_KEY"),
        semantic_scholar_api_key=_env("LSM_TEST_SEMANTIC_SCHOLAR_API_KEY"),
        core_api_key=_env("LSM_TEST_CORE_API_KEY"),
        embed_model=embed_model,
        tier=tier,
    )
    config.validate()
    return config

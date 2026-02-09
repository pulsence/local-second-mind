from __future__ import annotations
from pathlib import Path

import pytest

from tests.testing_config import DEFAULT_EMBED_MODEL, load_test_config


LSM_TEST_KEYS = (
    "LSM_TEST_CONFIG",
    "LSM_TEST_OPENAI_API_KEY",
    "LSM_TEST_ANTHROPIC_API_KEY",
    "LSM_TEST_GOOGLE_API_KEY",
    "LSM_TEST_OLLAMA_BASE_URL",
    "LSM_TEST_BRAVE_API_KEY",
    "LSM_TEST_SEMANTIC_SCHOLAR_API_KEY",
    "LSM_TEST_CORE_API_KEY",
    "LSM_TEST_EMBED_MODEL",
    "LSM_TEST_TIER",
)


def _clear_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in LSM_TEST_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_test_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_env(monkeypatch)

    cfg = load_test_config(env_path=Path("does-not-exist.env"))

    assert cfg.config_path is None
    assert cfg.openai_api_key is None
    assert cfg.embed_model == DEFAULT_EMBED_MODEL
    assert cfg.tier == "smoke"


def test_load_test_config_reads_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_env(monkeypatch)
    monkeypatch.setenv("LSM_TEST_CONFIG", "~/test-config.json")
    monkeypatch.setenv("LSM_TEST_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LSM_TEST_EMBED_MODEL", "custom/embed-model")
    monkeypatch.setenv("LSM_TEST_TIER", "integration")

    cfg = load_test_config(env_path=Path("does-not-exist.env"))

    assert cfg.config_path == Path("~/test-config.json").expanduser()
    assert cfg.openai_api_key == "sk-test"
    assert cfg.embed_model == "custom/embed-model"
    assert cfg.tier == "integration"


def test_load_test_config_reads_dotenv_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_test_env(monkeypatch)
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "\n".join(
            (
                "LSM_TEST_GOOGLE_API_KEY=google-test",
                "LSM_TEST_TIER=live",
            )
        ),
        encoding="utf-8",
    )

    cfg = load_test_config(env_path=env_file)

    assert cfg.google_api_key == "google-test"
    assert cfg.tier == "live"


def test_load_test_config_rejects_invalid_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_test_env(monkeypatch)
    monkeypatch.setenv("LSM_TEST_TIER", "invalid")

    with pytest.raises(ValueError, match="Invalid LSM_TEST_TIER"):
        load_test_config(env_path=Path("does-not-exist.env"))

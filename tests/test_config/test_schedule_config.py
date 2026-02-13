from __future__ import annotations

from pathlib import Path

import pytest

from lsm.config.loader import build_config_from_raw, config_to_raw
from lsm.config.models.agents import AgentConfig, ScheduleConfig


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {
            "global_folder": str(tmp_path / "lsm-global"),
        },
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "test_collection",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {
                "query": {
                    "provider": "openai",
                    "model": "gpt-5.2",
                }
            },
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "test_collection",
        },
        "query": {"mode": "grounded"},
    }


def test_schedule_config_validate_happy_path() -> None:
    cfg = ScheduleConfig(
        agent_name=" research ",
        params={"topic": "Aquinas"},
        interval=" DAILY ",
        enabled=True,
        concurrency_policy="Queue",
        confirmation_mode="Confirm",
    )
    cfg.validate()

    assert cfg.agent_name == "research"
    assert cfg.interval == "daily"
    assert cfg.concurrency_policy == "queue"
    assert cfg.confirmation_mode == "confirm"


def test_schedule_config_validate_accepts_seconds_and_cron() -> None:
    ScheduleConfig(agent_name="research", interval="3600s").validate()
    ScheduleConfig(agent_name="research", interval="*/15 * * * *").validate()


def test_schedule_config_validate_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="agent_name"):
        ScheduleConfig(agent_name="").validate()

    with pytest.raises(ValueError, match="interval"):
        ScheduleConfig(agent_name="research", interval="sometimes").validate()

    with pytest.raises(ValueError, match="concurrency_policy"):
        ScheduleConfig(
            agent_name="research",
            concurrency_policy="parallel",
        ).validate()

    with pytest.raises(ValueError, match="confirmation_mode"):
        ScheduleConfig(
            agent_name="research",
            confirmation_mode="ask",
        ).validate()

    with pytest.raises(ValueError, match="params"):
        ScheduleConfig(
            agent_name="research",
            params="not-a-dict",  # type: ignore[arg-type]
        ).validate()


def test_agent_config_normalizes_schedule_entries() -> None:
    cfg = AgentConfig(
        enabled=True,
        schedules=[
            {
                "agent_name": "research",
                "interval": "hourly",
                "params": {"topic": "memory"},
            }
        ],
    )
    cfg.validate()
    assert len(cfg.schedules) == 1
    assert isinstance(cfg.schedules[0], ScheduleConfig)
    assert cfg.schedules[0].interval == "hourly"


def test_build_and_serialize_schedules(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["agents"] = {
        "enabled": True,
        "agents_folder": "Agents",
        "schedules": [
            {
                "agent_name": "research",
                "params": {"topic": "weekly roundup"},
                "interval": "weekly",
                "enabled": True,
                "concurrency_policy": "skip",
                "confirmation_mode": "auto",
            },
            {
                "agent_name": "curator",
                "params": {"mode": "memory"},
                "interval": "0 2 * * 1",
                "enabled": False,
                "concurrency_policy": "queue",
                "confirmation_mode": "confirm",
            },
        ],
    }

    config = build_config_from_raw(raw, tmp_path / "config.json")
    assert config.agents is not None
    assert len(config.agents.schedules) == 2
    assert config.agents.schedules[0].agent_name == "research"
    assert config.agents.schedules[0].interval == "weekly"
    assert config.agents.schedules[1].agent_name == "curator"
    assert config.agents.schedules[1].confirmation_mode == "confirm"

    serialized = config_to_raw(config)
    assert "schedules" in serialized["agents"]
    assert serialized["agents"]["schedules"][0]["agent_name"] == "research"
    assert serialized["agents"]["schedules"][0]["params"]["topic"] == "weekly roundup"
    assert serialized["agents"]["schedules"][1]["interval"] == "0 2 * * 1"

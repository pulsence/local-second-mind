from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from lsm.config.models import ScheduleConfig
from lsm.ui.shell.commands import agents as agent_commands


@dataclass
class _FakeScheduler:
    config: object
    collection: object = None
    embedder: object = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return None

    def stop(self, **kwargs) -> None:
        _ = kwargs
        self.stopped = True

    def list_schedules(self):
        schedules = getattr(self.config.agents, "schedules", [])
        rows = []
        for index, schedule in enumerate(schedules):
            rows.append(
                {
                    "id": f"{index}:{schedule.agent_name}",
                    "index": index,
                    "agent_name": schedule.agent_name,
                    "interval": schedule.interval,
                    "enabled": schedule.enabled,
                    "concurrency_policy": schedule.concurrency_policy,
                    "confirmation_mode": schedule.confirmation_mode,
                    "params": dict(schedule.params),
                    "last_run_at": None,
                    "next_run_at": "2026-02-13T12:00:00+00:00",
                    "last_status": "idle",
                    "last_error": None,
                    "queued_runs": 0,
                    "running": False,
                }
            )
        return rows


def _app(enabled: bool = True):
    return SimpleNamespace(
        config=SimpleNamespace(
            agents=SimpleNamespace(
                enabled=enabled,
                schedules=[],
            ),
            batch_size=32,
        ),
        query_provider=None,
        query_embedder=None,
    )


def test_schedule_commands_add_list_toggle_remove_and_status(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    monkeypatch.setattr(agent_commands, "AgentScheduler", _FakeScheduler)
    app = _app(enabled=True)

    add_out = agent_commands.handle_agent_command(
        '/agent schedule add research daily --params \'{"topic":"daily summary","allow_network":true}\'',
        app,
    )
    assert "Added schedule '0:research'" in add_out
    assert len(app.config.agents.schedules) == 1
    assert app.config.agents.schedules[0].params["topic"] == "daily summary"
    assert app.config.agents.schedules[0].params["allow_network"] is True

    list_out = agent_commands.handle_agent_command("/agent schedule list", app)
    assert "Schedules: 1" in list_out
    assert "0:research" in list_out

    disable_out = agent_commands.handle_agent_command("/agent schedule disable 0:research", app)
    assert "Disabled schedule '0:research'" in disable_out
    assert app.config.agents.schedules[0].enabled is False

    enable_out = agent_commands.handle_agent_command("/agent schedule enable 0:research", app)
    assert "Enabled schedule '0:research'" in enable_out
    assert app.config.agents.schedules[0].enabled is True

    status_out = agent_commands.handle_agent_command("/agent schedule status", app)
    assert "Scheduler status (1 schedule(s))" in status_out
    assert "0:research" in status_out

    remove_out = agent_commands.handle_agent_command("/agent schedule remove 0:research", app)
    assert "Removed schedule '0:research'" in remove_out
    assert app.config.agents.schedules == []


def test_schedule_commands_validate_usage_and_json(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    monkeypatch.setattr(agent_commands, "AgentScheduler", _FakeScheduler)
    app = _app(enabled=True)

    help_out = agent_commands.handle_agent_command("/agent schedule", app)
    assert "Agent schedule commands:" in help_out

    bad_json = agent_commands.handle_agent_command(
        "/agent schedule add research daily --params not-json",
        app,
    )
    assert "Invalid --params JSON" in bad_json

    bad_object = agent_commands.handle_agent_command(
        '/agent schedule add research daily --params \'["not","object"]\'',
        app,
    )
    assert "--params must decode to a JSON object" in bad_object


def test_schedule_commands_require_agents_enabled(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    monkeypatch.setattr(agent_commands, "AgentScheduler", _FakeScheduler)
    app = _app(enabled=False)

    out = agent_commands.handle_agent_command("/agent schedule add research daily", app)
    assert "Failed to add schedule" in out
    assert "Agents are disabled" in out


def test_schedule_config_objects_roundtrip_in_fake_scheduler(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    monkeypatch.setattr(agent_commands, "AgentScheduler", _FakeScheduler)
    app = _app(enabled=True)
    app.config.agents.schedules = [
        ScheduleConfig(
            agent_name="curator",
            params={"topic": "--mode memory"},
            interval="daily",
            enabled=True,
            concurrency_policy="skip",
            confirmation_mode="auto",
        )
    ]

    out = agent_commands.handle_agent_command("/agent schedule list", app)
    assert "curator" in out
    assert "daily" in out

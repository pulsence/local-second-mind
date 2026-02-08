from __future__ import annotations

from datetime import datetime, timedelta

from lsm.agents.base import AgentState, AgentStatus, BaseAgent
from lsm.agents.models import AgentContext, AgentLogEntry


class DummyAgent(BaseAgent):
    name = "dummy"
    description = "Dummy test agent"

    def run(self, initial_context: AgentContext) -> AgentState:
        self.state.set_status(AgentStatus.RUNNING)
        self.state.current_task = "process"
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor="agent",
                content=f"messages={len(initial_context.messages)}",
            )
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state


def test_agent_status_values() -> None:
    assert AgentStatus.IDLE.value == "idle"
    assert AgentStatus.RUNNING.value == "running"
    assert AgentStatus.PAUSED.value == "paused"
    assert AgentStatus.WAITING_USER.value == "waiting_user"
    assert AgentStatus.COMPLETED.value == "completed"
    assert AgentStatus.FAILED.value == "failed"


def test_agent_state_touch_and_log() -> None:
    state = AgentState()
    old_updated_at = state.updated_at - timedelta(seconds=1)
    state.updated_at = old_updated_at
    state.set_status(AgentStatus.RUNNING)
    assert state.status == AgentStatus.RUNNING
    assert state.updated_at > old_updated_at

    entry = AgentLogEntry(timestamp=datetime.utcnow(), actor="agent", content="step")
    state.add_log(entry)
    assert len(state.log_entries) == 1
    assert state.log_entries[0].content == "step"


def test_base_agent_pause_resume_stop() -> None:
    agent = DummyAgent()
    assert agent.state.status == AgentStatus.IDLE

    agent.pause()
    assert agent.state.status == AgentStatus.PAUSED

    agent.resume()
    assert agent.state.status == AgentStatus.RUNNING

    agent.stop()
    assert agent.state.status == AgentStatus.COMPLETED


def test_base_agent_run_updates_state_and_logs() -> None:
    agent = DummyAgent()
    context = AgentContext(messages=[{"role": "user", "content": "hello"}])
    state = agent.run(context)
    assert state.status == AgentStatus.COMPLETED
    assert state.current_task == "process"
    assert len(state.log_entries) == 1
    assert state.log_entries[0].content == "messages=1"


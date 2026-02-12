from __future__ import annotations

from types import SimpleNamespace

from lsm.agents.memory import Memory, MemoryCandidate
from lsm.ui.shell.commands import agents as agent_commands


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.memory = Memory(
            type="task_state",
            key="writing_style",
            value={"tone": "concise"},
            scope="agent",
            tags=["style"],
            source_run_id="run-1",
        )
        self.candidate_id = "cand-1"
        self.status = "pending"
        self.closed = False

    def list_candidates(self, status=None, limit=1000):
        _ = limit
        candidate = MemoryCandidate(
            id=self.candidate_id,
            memory=self.memory,
            provenance="curator",
            rationale="Repeated preference",
            status=self.status,
        )
        if status and status != candidate.status:
            return []
        return [candidate]

    def promote(self, candidate_id: str):
        if candidate_id != self.candidate_id:
            raise KeyError(candidate_id)
        self.status = "promoted"
        return self.memory

    def reject(self, candidate_id: str) -> None:
        if candidate_id != self.candidate_id:
            raise KeyError(candidate_id)
        self.status = "rejected"

    def delete(self, memory_id: str) -> None:
        if memory_id != self.memory.id:
            raise KeyError(memory_id)

    def put_candidate(self, memory: Memory, provenance: str, rationale: str) -> str:
        _ = provenance, rationale
        self.memory = memory
        self.candidate_id = "cand-2"
        self.status = "pending"
        return self.candidate_id

    def close(self) -> None:
        self.closed = True


def _app(memory_enabled: bool = True):
    return SimpleNamespace(
        config=SimpleNamespace(
            agents=SimpleNamespace(
                enabled=True,
                memory=SimpleNamespace(enabled=memory_enabled),
            ),
            vectordb=SimpleNamespace(provider="chromadb"),
        )
    )


def test_handle_memory_command_help_and_disabled(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    app = _app(memory_enabled=False)

    out = agent_commands.handle_memory_command("/memory", app)
    assert "Memory commands:" in out

    out2 = agent_commands.handle_memory_command("/memory candidates", app)
    assert "Agent memory is disabled" in out2


def test_handle_memory_command_candidates_promote_reject(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    store = _FakeMemoryStore()
    monkeypatch.setattr(agent_commands, "create_memory_store", lambda *args, **kwargs: store)
    app = _app(memory_enabled=True)

    listed = agent_commands.handle_memory_command("/memory candidates", app)
    assert "Memory candidates (pending): 1" in listed
    assert "key=writing_style" in listed

    promoted = agent_commands.handle_memory_command("/memory promote cand-1", app)
    assert "Promoted memory candidate 'cand-1'" in promoted

    rejected = agent_commands.handle_memory_command("/memory reject cand-1", app)
    assert "Rejected memory candidate 'cand-1'" in rejected


def test_handle_memory_command_ttl_update(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    store = _FakeMemoryStore()
    monkeypatch.setattr(agent_commands, "create_memory_store", lambda *args, **kwargs: store)
    app = _app(memory_enabled=True)

    bad_ttl = agent_commands.handle_memory_command("/memory ttl cand-1 nope", app)
    assert "TTL days must be an integer." in bad_ttl

    ttl_out = agent_commands.handle_memory_command("/memory ttl cand-1 30", app)
    assert "Updated TTL for memory candidate 'cand-1'" in ttl_out
    assert "Replacement candidate: cand-2" in ttl_out


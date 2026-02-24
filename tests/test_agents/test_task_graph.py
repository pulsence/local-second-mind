from __future__ import annotations

import pytest

from lsm.agents.meta import AgentTask, TaskGraph


def test_task_graph_topological_sort_orders_dependencies() -> None:
    graph = TaskGraph(
        goal="Assemble briefing package",
        tasks=[
            AgentTask(id="research_1", agent_name="research"),
            AgentTask(id="writing_1", agent_name="writing", depends_on=["research_1"]),
            AgentTask(
                id="synthesis_1",
                agent_name="synthesis",
                depends_on=["research_1", "writing_1"],
            ),
        ],
    )

    ordered_ids = [task.id for task in graph.topological_sort()]
    assert ordered_ids == ["research_1", "writing_1", "synthesis_1"]


def test_task_graph_next_ready_mark_complete_and_is_done() -> None:
    graph = TaskGraph(
        goal="Assemble briefing package",
        tasks=[
            AgentTask(id="research_1", agent_name="research"),
            AgentTask(id="writing_1", agent_name="writing", depends_on=["research_1"]),
        ],
    )

    first_ready = graph.next_ready()
    assert first_ready is not None
    assert first_ready.id == "research_1"

    graph.mark_running("research_1")
    graph.mark_complete("research_1")
    second_ready = graph.next_ready()
    assert second_ready is not None
    assert second_ready.id == "writing_1"

    graph.mark_complete("writing_1")
    assert graph.next_ready() is None
    assert graph.is_done() is True


def test_task_graph_rejects_unknown_dependency_id() -> None:
    with pytest.raises(ValueError, match="depends on unknown task id"):
        TaskGraph(
            goal="Invalid graph",
            tasks=[
                AgentTask(
                    id="writing_1",
                    agent_name="writing",
                    depends_on=["research_missing"],
                )
            ],
        )


def test_task_graph_rejects_dependency_cycles() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        TaskGraph(
            goal="Cyclic graph",
            tasks=[
                AgentTask(id="a", agent_name="research", depends_on=["b"]),
                AgentTask(id="b", agent_name="writing", depends_on=["a"]),
            ],
        )


def test_task_graph_mark_running_rejects_terminal_state() -> None:
    graph = TaskGraph(
        goal="Terminal transition",
        tasks=[AgentTask(id="t1", agent_name="research", status="completed")],
    )
    with pytest.raises(ValueError, match="terminal state"):
        graph.mark_running("t1")

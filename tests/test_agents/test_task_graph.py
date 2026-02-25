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


def test_task_graph_builds_parallel_groups_with_dependency_gates() -> None:
    graph = TaskGraph(
        goal="Parallel plan",
        tasks=[
            AgentTask(id="a", agent_name="research", parallel_group="group_1"),
            AgentTask(id="b", agent_name="writing", parallel_group="group_1"),
            AgentTask(
                id="c",
                agent_name="synthesis",
                depends_on=["a"],
                parallel_group="group_2",
            ),
        ],
    )

    groups = graph.parallel_groups()
    assert [group.id for group in groups] == ["group_1", "group_2"]
    assert groups[0].depends_on == []
    assert groups[1].depends_on == ["group_1"]


def test_task_graph_parallel_groups_use_deterministic_task_order() -> None:
    graph = TaskGraph(
        goal="Deterministic parallel order",
        tasks=[
            AgentTask(id="t1", agent_name="research", parallel_group="alpha"),
            AgentTask(id="t2", agent_name="writing", parallel_group="alpha"),
            AgentTask(id="t3", agent_name="synthesis", parallel_group="beta"),
        ],
    )

    groups = graph.parallel_groups()
    alpha = next(group for group in groups if group.id == "alpha")
    assert [task.id for task in alpha.tasks] == ["t1", "t2"]


def test_task_graph_serialization_round_trip_includes_parallel_groups() -> None:
    graph = TaskGraph(
        goal="Serialization",
        tasks=[
            AgentTask(id="t1", agent_name="research", parallel_group="alpha"),
            AgentTask(id="t2", agent_name="writing", depends_on=["t1"]),
        ],
    )

    payload = graph.to_dict()
    rebuilt = TaskGraph.from_dict(payload)
    assert [task.id for task in rebuilt.tasks] == ["t1", "t2"]
    assert rebuilt.tasks[0].parallel_group == "alpha"
    assert "parallel_groups" in payload

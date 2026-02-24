"""
Task graph models for meta-agent orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_VALID_TASK_STATUSES = {"pending", "running", "completed", "failed"}


@dataclass
class AgentTask:
    """
    One task node in a meta-agent execution graph.
    """

    id: str
    """Unique task identifier."""

    agent_name: str
    """Agent to run for this task."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Task-specific parameters passed to the target agent."""

    expected_artifacts: List[str] = field(default_factory=list)
    """Expected output artifact names or paths."""

    depends_on: List[str] = field(default_factory=list)
    """Task IDs that must complete before this task is ready."""

    status: str = "pending"
    """Execution status: pending, running, completed, failed."""

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.agent_name = str(self.agent_name or "").strip().lower()
        self.params = dict(self.params or {})
        self.expected_artifacts = [
            str(item).strip()
            for item in self.expected_artifacts
            if str(item).strip()
        ]
        self.depends_on = [
            str(item).strip()
            for item in self.depends_on
            if str(item).strip()
        ]
        self.status = str(self.status or "pending").strip().lower()

    def validate(self) -> None:
        """
        Validate task shape.
        """
        if not self.id:
            raise ValueError("task.id must be non-empty")
        if not self.agent_name:
            raise ValueError(f"task '{self.id}' agent_name must be non-empty")
        if self.status not in _VALID_TASK_STATUSES:
            raise ValueError(
                f"task '{self.id}' status must be one of: pending, running, completed, failed"
            )


@dataclass
class TaskGraph:
    """
    Dependency graph of agent tasks for a single goal.
    """

    goal: str
    """User goal represented by this task graph."""

    tasks: List[AgentTask]
    """Tasks participating in this graph."""

    def __post_init__(self) -> None:
        self.goal = str(self.goal or "").strip()
        normalized_tasks: list[AgentTask] = []
        for task in self.tasks:
            if isinstance(task, AgentTask):
                normalized_tasks.append(task)
                continue
            if isinstance(task, dict):
                normalized_tasks.append(AgentTask(**task))
                continue
            raise ValueError("tasks entries must be AgentTask objects or task dictionaries")
        self.tasks = normalized_tasks
        self.validate()

    def validate(self) -> None:
        """
        Validate task IDs, dependency references, and graph acyclicity.
        """
        if not self.goal:
            raise ValueError("goal must be non-empty")

        seen_ids: set[str] = set()
        for task in self.tasks:
            task.validate()
            if task.id in seen_ids:
                raise ValueError(f"duplicate task id: '{task.id}'")
            seen_ids.add(task.id)

        for task in self.tasks:
            for dependency in task.depends_on:
                if dependency not in seen_ids:
                    raise ValueError(
                        f"task '{task.id}' depends on unknown task id '{dependency}'"
                    )

        # Force cycle validation.
        self.topological_sort()

    def topological_sort(self) -> List[AgentTask]:
        """
        Return tasks in dependency-safe order.

        Raises:
            ValueError: If the graph contains a dependency cycle.
        """
        task_by_id = {task.id: task for task in self.tasks}
        dependents: Dict[str, List[str]] = {task.id: [] for task in self.tasks}
        indegree: Dict[str, int] = {task.id: 0 for task in self.tasks}
        position: Dict[str, int] = {task.id: idx for idx, task in enumerate(self.tasks)}

        for task in self.tasks:
            indegree[task.id] = len(task.depends_on)
            for dependency in task.depends_on:
                dependents[dependency].append(task.id)

        ready = [task.id for task in self.tasks if indegree[task.id] == 0]
        ordered_ids: list[str] = []
        while ready:
            # Stable order by original declaration index.
            ready.sort(key=lambda task_id: position[task_id])
            current = ready.pop(0)
            ordered_ids.append(current)
            for dependent in dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if len(ordered_ids) != len(self.tasks):
            raise ValueError("task graph contains a dependency cycle")
        return [task_by_id[task_id] for task_id in ordered_ids]

    def next_ready(self) -> Optional[AgentTask]:
        """
        Return the next pending task whose dependencies are completed.
        """
        task_by_id = {task.id: task for task in self.tasks}
        for task in self.topological_sort():
            if task.status != "pending":
                continue
            if all(task_by_id[dependency].status == "completed" for dependency in task.depends_on):
                return task
        return None

    def mark_running(self, task_id: str) -> None:
        """
        Mark a task as running.
        """
        task = self._lookup(task_id)
        if task.status in {"completed", "failed"}:
            raise ValueError(
                f"cannot mark task '{task.id}' running from terminal state '{task.status}'"
            )
        task.status = "running"

    def mark_complete(self, task_id: str) -> None:
        """
        Mark a task as completed.
        """
        task = self._lookup(task_id)
        task.status = "completed"

    def mark_failed(self, task_id: str) -> None:
        """
        Mark a task as failed.
        """
        task = self._lookup(task_id)
        task.status = "failed"

    def is_done(self) -> bool:
        """
        Return True when all tasks are completed.
        """
        return all(task.status == "completed" for task in self.tasks)

    def _lookup(self, task_id: str) -> AgentTask:
        normalized = str(task_id or "").strip()
        for task in self.tasks:
            if task.id == normalized:
                return task
        raise KeyError(f"unknown task id: '{task_id}'")


"""
Meta-agent orchestrator core implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig

from .base import AgentStatus, BaseAgent
from .models import AgentContext
from .task_graph import AgentTask, TaskGraph
from .tools.base import ToolRegistry
from .tools.sandbox import ToolSandbox

_SUPPORTED_SUB_AGENTS = {"curator", "research", "synthesis", "writing"}

_DEFAULT_ARTIFACTS = {
    "curator": ["curation_report.md"],
    "research": ["research_outline.md"],
    "writing": ["draft.md"],
    "synthesis": ["synthesis.md"],
}


@dataclass
class MetaPlanResult:
    """
    Structured result from a meta-agent planning run.
    """

    goal: str
    task_graph: TaskGraph
    execution_order: List[str]


class MetaAgent(BaseAgent):
    """
    Orchestrate multiple agents toward a single goal.

    This core version builds and validates an execution graph, computes a
    dependency-safe sequence, and records the plan. It does not execute
    sub-agent domain work directly.
    """

    name = "meta"
    description = "Orchestrate multiple agents toward a single goal."

    def __init__(
        self,
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
        agent_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=self.name, description=self.description)
        self.llm_registry = llm_registry
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.agent_config = agent_config
        self.agent_overrides = dict(agent_overrides or {})
        self.max_iterations = int(
            self.agent_overrides.get("max_iterations", self.agent_config.max_iterations)
        )
        self.max_tokens_budget = int(
            self.agent_overrides.get(
                "max_tokens_budget",
                self.agent_config.max_tokens_budget,
            )
        )
        self.last_result: Optional[MetaPlanResult] = None
        self.last_task_graph: Optional[TaskGraph] = None
        self.last_execution_order: List[str] = []

    def run(self, initial_context: AgentContext) -> Any:
        """
        Build and validate an orchestration plan from the goal prompt.
        """
        self._tokens_used = 0
        self.state.set_status(AgentStatus.RUNNING)
        goal = self._extract_goal(initial_context)
        self.state.current_task = f"Planning orchestration for: {goal}"

        try:
            graph = self._build_task_graph(goal)
            ordered_tasks = graph.topological_sort()
            execution_order: list[str] = []
            for task in ordered_tasks:
                graph.mark_running(task.id)
                self._log(
                    (
                        f"Planned task '{task.id}' agent={task.agent_name} "
                        f"depends_on={task.depends_on or []} "
                        f"expected_artifacts={task.expected_artifacts or []}"
                    )
                )
                # 6.1 core mode: planning only; sub-agent execution is implemented later.
                graph.mark_complete(task.id)
                execution_order.append(task.id)

            self.last_task_graph = graph
            self.last_execution_order = execution_order
            self.last_result = MetaPlanResult(
                goal=graph.goal,
                task_graph=graph,
                execution_order=execution_order,
            )
            self.state.current_task = (
                f"Planned {len(execution_order)} task(s) in dependency order."
            )
            self._log(
                f"Planned task graph with {len(graph.tasks)} task(s): {execution_order}"
            )
            self.state.set_status(AgentStatus.COMPLETED)
            return self.state
        except Exception as exc:
            self._log(f"Meta-agent planning failed: {exc}")
            self.state.set_status(AgentStatus.FAILED)
            return self.state

    def _extract_goal(self, context: AgentContext) -> str:
        for message in reversed(context.messages):
            if str(message.get("role", "")).lower() != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return "Untitled meta-agent goal"

    def _build_task_graph(self, goal: str) -> TaskGraph:
        structured_tasks = self._try_parse_structured_tasks(goal)
        if structured_tasks:
            return TaskGraph(goal=self._structured_goal_title(goal) or goal, tasks=structured_tasks)
        return TaskGraph(goal=goal, tasks=self._default_tasks_for_goal(goal))

    def _try_parse_structured_tasks(self, goal: str) -> Optional[List[AgentTask]]:
        parsed = self._parse_json(goal)
        if not isinstance(parsed, dict):
            return None
        raw_tasks = parsed.get("tasks")
        if not isinstance(raw_tasks, list):
            return None

        tasks: list[AgentTask] = []
        for index, raw_task in enumerate(raw_tasks):
            if not isinstance(raw_task, dict):
                raise ValueError("structured meta goal 'tasks' entries must be objects")
            task_id = str(raw_task.get("id", "")).strip() or f"task_{index + 1}"
            agent_name = str(raw_task.get("agent_name", "")).strip().lower()
            if agent_name not in _SUPPORTED_SUB_AGENTS:
                raise ValueError(
                    f"unsupported sub-agent '{agent_name}' in structured meta plan"
                )
            params = raw_task.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ValueError(f"task '{task_id}' params must be an object")
            depends_on = raw_task.get("depends_on", [])
            if depends_on is None:
                depends_on = []
            if not isinstance(depends_on, list):
                raise ValueError(f"task '{task_id}' depends_on must be a list")
            expected_artifacts = raw_task.get("expected_artifacts", [])
            if expected_artifacts is None:
                expected_artifacts = []
            if not isinstance(expected_artifacts, list):
                raise ValueError(f"task '{task_id}' expected_artifacts must be a list")

            tasks.append(
                AgentTask(
                    id=task_id,
                    agent_name=agent_name,
                    params=dict(params),
                    expected_artifacts=list(expected_artifacts),
                    depends_on=list(depends_on),
                    status="pending",
                )
            )
        return tasks

    def _structured_goal_title(self, goal: str) -> str:
        parsed = self._parse_json(goal)
        if isinstance(parsed, dict):
            title = str(parsed.get("goal", "")).strip()
            if title:
                return title
        return ""

    def _default_tasks_for_goal(self, goal: str) -> List[AgentTask]:
        lower_goal = goal.lower()
        topic = goal.strip()
        if not topic:
            topic = "Untitled meta-agent goal"

        if any(token in lower_goal for token in ("curate", "cleanup", "clean up", "dedup")):
            return [
                AgentTask(
                    id="curator_1",
                    agent_name="curator",
                    params={"topic": topic},
                    expected_artifacts=list(_DEFAULT_ARTIFACTS["curator"]),
                    depends_on=[],
                )
            ]

        # Default pipeline: research -> writing -> synthesis.
        return [
            AgentTask(
                id="research_1",
                agent_name="research",
                params={"topic": topic},
                expected_artifacts=list(_DEFAULT_ARTIFACTS["research"]),
                depends_on=[],
            ),
            AgentTask(
                id="writing_1",
                agent_name="writing",
                params={"topic": topic},
                expected_artifacts=list(_DEFAULT_ARTIFACTS["writing"]),
                depends_on=["research_1"],
            ),
            AgentTask(
                id="synthesis_1",
                agent_name="synthesis",
                params={"topic": topic},
                expected_artifacts=list(_DEFAULT_ARTIFACTS["synthesis"]),
                depends_on=["research_1", "writing_1"],
            ),
        ]

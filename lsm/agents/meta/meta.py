"""
Meta-agent orchestrator implementation with shared-workspace execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig
from lsm.config.models.agents import SandboxConfig
from lsm.providers.factory import create_provider

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from .task_graph import AgentTask, TaskGraph
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace

_SUPPORTED_SUB_AGENTS = {"curator", "research", "synthesis", "writing"}
_META_SYSTEM_TOOL_NAMES = {"spawn_agent", "await_agent", "collect_artifacts"}

_DEFAULT_ARTIFACTS = {
    "curator": ["curation_report.md"],
    "research": ["research_outline.md"],
    "writing": ["deliverable.md"],
    "synthesis": ["synthesis.md"],
}


@dataclass
class MetaTaskRun:
    """
    Runtime execution record for a single sub-agent task.
    """

    task_id: str
    agent_name: str
    status: str
    sub_agent_dir: Path
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class MetaPlanResult:
    """
    Structured result from a meta-agent run.
    """

    goal: str
    task_graph: TaskGraph
    execution_order: List[str]
    run_root: Optional[Path] = None
    workspace: Optional[Path] = None
    final_result_path: Optional[Path] = None
    meta_log_path: Optional[Path] = None
    task_runs: List[MetaTaskRun] = field(default_factory=list)


class MetaAgent(BaseAgent):
    """
    Orchestrate multiple agents toward a single goal.

    The meta agent plans a dependency-safe task graph, executes supported
    sub-agents with monotone sandbox restrictions, and writes final run
    artifacts (`final_result.md`, `meta_log.md`) in a shared workspace layout.
    """

    name = "meta"
    tier = "complex"
    description = "Orchestrate multiple agents toward a single goal."
    risk_posture = "writes_workspace"

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
        self.execution_mode = str(
            self.agent_overrides.get("execution_mode", "execute")
        ).strip().lower()
        self.enable_final_synthesis = bool(
            self.agent_overrides.get("enable_final_synthesis", True)
        )
        self.last_result: Optional[MetaPlanResult] = None
        self.last_task_graph: Optional[TaskGraph] = None
        self.last_execution_order: List[str] = []

    def run(self, initial_context: AgentContext) -> Any:
        """
        Execute orchestration for the requested goal.
        """
        self._tokens_used = 0
        self.state.set_status(AgentStatus.RUNNING)
        ensure_agent_workspace(
            self.name,
            self.agent_config.agents_folder,
            sandbox=self.sandbox,
        )
        goal = self._extract_goal(initial_context)
        self.state.current_task = f"Orchestrating goal: {goal}"

        try:
            graph = self._build_task_graph(goal)
            ordered_tasks = graph.topological_sort()
            run_root, shared_workspace, sub_agents_root = self._initialize_run_workspace(initial_context)

            execution_order: list[str] = []
            task_runs: list[MetaTaskRun] = []
            sub_agent_counters: Dict[str, int] = {}

            for task in ordered_tasks:
                sub_dir_name = self._next_sub_agent_dir_name(task.agent_name, sub_agent_counters)
                sub_dir = sub_agents_root / sub_dir_name

                if not self._dependencies_completed(graph, task):
                    graph.mark_failed(task.id)
                    blocked = MetaTaskRun(
                        task_id=task.id,
                        agent_name=task.agent_name,
                        status="failed",
                        sub_agent_dir=sub_dir,
                        artifacts=[],
                        error="Blocked by incomplete dependency.",
                    )
                    task_runs.append(blocked)
                    self._log(
                        f"Task '{task.id}' blocked due to failed or incomplete dependency."
                    )
                    continue

                graph.mark_running(task.id)
                self._log(
                    (
                        f"Planned task '{task.id}' agent={task.agent_name} "
                        f"depends_on={task.depends_on or []} "
                        f"expected_artifacts={task.expected_artifacts or []}"
                    )
                )

                if self.execution_mode in {"plan", "planning", "plan_only"}:
                    graph.mark_complete(task.id)
                    execution_order.append(task.id)
                    task_runs.append(
                        MetaTaskRun(
                            task_id=task.id,
                            agent_name=task.agent_name,
                            status="completed",
                            sub_agent_dir=sub_dir,
                            artifacts=[],
                            error=None,
                        )
                    )
                    continue

                run_result = self._execute_sub_agent_task(
                    task=task,
                    goal=goal,
                    sub_dir=sub_dir,
                    shared_workspace=shared_workspace,
                )
                task_runs.append(run_result)

                if run_result.status == "completed":
                    graph.mark_complete(task.id)
                    execution_order.append(task.id)
                else:
                    graph.mark_failed(task.id)

            self.last_task_graph = graph
            self.last_execution_order = execution_order

            final_result_text = self._synthesize_final_result(
                goal=goal,
                graph=graph,
                execution_order=execution_order,
                task_runs=task_runs,
            )
            final_result_path = run_root / "final_result.md"
            final_result_path.write_text(final_result_text, encoding="utf-8")
            self.state.add_artifact(str(final_result_path))

            meta_log_text = self._build_meta_log(
                goal=goal,
                graph=graph,
                execution_order=execution_order,
                task_runs=task_runs,
                final_result_path=final_result_path,
            )
            meta_log_path = run_root / "meta_log.md"
            meta_log_path.write_text(meta_log_text, encoding="utf-8")
            self.state.add_artifact(str(meta_log_path))

            self.last_result = MetaPlanResult(
                goal=graph.goal,
                task_graph=graph,
                execution_order=execution_order,
                run_root=run_root,
                workspace=shared_workspace,
                final_result_path=final_result_path,
                meta_log_path=meta_log_path,
                task_runs=task_runs,
            )

            failed_count = sum(1 for item in task_runs if item.status == "failed")
            completed_count = sum(1 for item in task_runs if item.status == "completed")
            self.state.current_task = (
                f"Completed orchestration: {completed_count} completed, {failed_count} failed task(s)."
            )
            self._log(
                f"Planned task graph with {len(graph.tasks)} task(s): {execution_order}"
            )

            if completed_count == 0 and failed_count > 0:
                self.state.set_status(AgentStatus.FAILED)
            else:
                self.state.set_status(AgentStatus.COMPLETED)
            return self.state
        except Exception as exc:
            self._log(f"Meta-agent orchestration failed: {exc}")
            self.state.set_status(AgentStatus.FAILED)
            return self.state

    def _initialize_run_workspace(self, initial_context: AgentContext) -> tuple[Path, Path, Path]:
        workspace_hint = str(initial_context.run_workspace or "").strip()
        if workspace_hint:
            shared_workspace = Path(workspace_hint).expanduser().resolve(strict=False)
            run_root = shared_workspace.parent
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            agent_root = ensure_agent_workspace(
                self.name,
                self.agent_config.agents_folder,
                sandbox=self.sandbox,
            )
            run_root = (agent_root / "logs" / f"{self.name}_{timestamp}").resolve(strict=False)
            shared_workspace = run_root / "workspace"

        self.sandbox.check_write_path(run_root)
        self.sandbox.check_read_path(run_root)

        sub_agents_root = run_root / "sub_agents"
        shared_workspace.mkdir(parents=True, exist_ok=True)
        sub_agents_root.mkdir(parents=True, exist_ok=True)

        initial_context.run_workspace = str(shared_workspace)
        self.state.add_artifact(str(shared_workspace))
        self.state.add_artifact(str(sub_agents_root))
        return run_root, shared_workspace, sub_agents_root

    @staticmethod
    def _next_sub_agent_dir_name(agent_name: str, counters: Dict[str, int]) -> str:
        normalized = str(agent_name or "agent").strip().lower() or "agent"
        current = int(counters.get(normalized, 0)) + 1
        counters[normalized] = current
        return f"{normalized}_{current:03d}"

    @staticmethod
    def _dependencies_completed(graph: TaskGraph, task: AgentTask) -> bool:
        by_id = {item.id: item for item in graph.tasks}
        return all(by_id[dep_id].status == "completed" for dep_id in task.depends_on)

    def _execute_sub_agent_task(
        self,
        *,
        task: AgentTask,
        goal: str,
        sub_dir: Path,
        shared_workspace: Path,
    ) -> MetaTaskRun:
        sub_dir.mkdir(parents=True, exist_ok=True)

        try:
            child_registry = self._build_child_tool_registry()
            child_sandbox = self._build_child_sandbox(
                sub_dir=sub_dir,
                shared_workspace=shared_workspace,
            )
            child_agent_config = replace(self.agent_config, agents_folder=sub_dir)

            from ..factory import create_agent

            child_agent = create_agent(
                name=task.agent_name,
                llm_registry=self.llm_registry,
                tool_registry=child_registry,
                sandbox=child_sandbox,
                agent_config=child_agent_config,
            )
            effective_agent_config = getattr(child_agent, "agent_config", child_agent_config)
            topic = self._build_sub_agent_topic(task, goal, shared_workspace)
            child_context = AgentContext(
                messages=[{"role": "user", "content": topic}],
                tool_definitions=child_registry.list_definitions(),
                budget_tracking={
                    "tokens_used": 0,
                    "max_tokens_budget": int(effective_agent_config.max_tokens_budget),
                    "iterations": 0,
                    "spawned_by": "meta",
                },
                run_workspace=str(sub_dir),
            )
            state = child_agent.run(child_context)
            status_value = (
                str(state.status.value)
                if hasattr(state.status, "value")
                else str(state.status)
            )
            artifacts = self._collect_sub_agent_artifacts(child_agent, sub_dir)
            for artifact in artifacts:
                self.state.add_artifact(str(artifact))

            error = None
            if status_value != "completed":
                error = self._extract_task_error(state)

            self._log(
                f"Executed task '{task.id}' with sub-agent '{task.agent_name}' status={status_value}"
            )
            return MetaTaskRun(
                task_id=task.id,
                agent_name=task.agent_name,
                status=status_value,
                sub_agent_dir=sub_dir,
                artifacts=[str(path) for path in artifacts],
                error=error,
            )
        except Exception as exc:
            self._log(
                f"Task '{task.id}' failed for sub-agent '{task.agent_name}': {exc}"
            )
            return MetaTaskRun(
                task_id=task.id,
                agent_name=task.agent_name,
                status="failed",
                sub_agent_dir=sub_dir,
                artifacts=[],
                error=str(exc),
            )

    def _build_child_tool_registry(self) -> ToolRegistry:
        child_registry = ToolRegistry()
        for tool in self.tool_registry.list_tools():
            name = str(tool.name).strip()
            if name in _META_SYSTEM_TOOL_NAMES:
                continue
            child_registry.register(tool)
        return child_registry

    def _build_child_sandbox(
        self,
        *,
        sub_dir: Path,
        shared_workspace: Path,
    ) -> ToolSandbox:
        parent = self.sandbox.config

        self.sandbox.check_write_path(sub_dir)
        self.sandbox.check_read_path(sub_dir)
        self.sandbox.check_read_path(shared_workspace)

        read_paths = list(self.sandbox.effective_read_paths())
        read_paths.append(shared_workspace)
        read_paths.append(sub_dir)

        deduped_read_paths: list[Path] = []
        seen: set[str] = set()
        for path in read_paths:
            marker = str(path)
            if marker in seen:
                continue
            seen.add(marker)
            deduped_read_paths.append(path)

        child_config = SandboxConfig(
            allowed_read_paths=deduped_read_paths,
            allowed_write_paths=[sub_dir],
            allow_url_access=bool(parent.allow_url_access),
            require_user_permission=dict(parent.require_user_permission),
            require_permission_by_risk=dict(parent.require_permission_by_risk),
            execution_mode=str(parent.execution_mode),
            force_docker=bool(parent.force_docker),
            limits=dict(parent.limits),
            docker=dict(parent.docker),
            tool_llm_assignments=dict(parent.tool_llm_assignments),
        )
        return ToolSandbox(
            child_config,
            global_sandbox=parent,
            local_runner=self.sandbox.local_runner,
            docker_runner=self.sandbox.docker_runner,
        )

    @staticmethod
    def _build_sub_agent_topic(task: AgentTask, goal: str, shared_workspace: Path) -> str:
        topic = str(task.params.get("topic", task.params.get("goal", ""))).strip()
        if not topic:
            topic = goal

        params = {key: value for key, value in task.params.items() if key not in {"topic", "goal"}}
        if params:
            params_block = json.dumps(params, indent=2, ensure_ascii=True)
            topic = f"{topic}\n\nTask parameters:\n{params_block}"

        topic += (
            "\n\nShared workspace read path: "
            f"{shared_workspace}"
        )
        return topic

    @staticmethod
    def _collect_sub_agent_artifacts(agent: Any, sub_dir: Path) -> list[Path]:
        artifacts: list[Path] = []

        state = getattr(agent, "state", None)
        if state is not None:
            for value in getattr(state, "artifacts", []):
                candidate = Path(str(value).strip())
                if candidate.exists() and candidate.is_file():
                    artifacts.append(candidate.resolve(strict=False))

        if sub_dir.exists():
            for candidate in sorted(sub_dir.rglob("*")):
                if candidate.is_file():
                    artifacts.append(candidate.resolve(strict=False))

        deduped: list[Path] = []
        seen: set[str] = set()
        for artifact in artifacts:
            marker = str(artifact)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(artifact)
        return deduped

    @staticmethod
    def _extract_task_error(state: Any) -> Optional[str]:
        entries = getattr(state, "log_entries", [])
        for entry in reversed(entries):
            text = str(getattr(entry, "content", "")).strip()
            if not text:
                continue
            if "error" in text.lower() or "failed" in text.lower():
                return text
        return None

    def _synthesize_final_result(
        self,
        *,
        goal: str,
        graph: TaskGraph,
        execution_order: List[str],
        task_runs: List[MetaTaskRun],
    ) -> str:
        fallback = self._fallback_final_result(goal, graph, execution_order, task_runs)
        if not self.enable_final_synthesis:
            return fallback

        excerpts = self._build_artifact_excerpts(task_runs)
        payload = {
            "goal": goal,
            "execution_order": execution_order,
            "task_results": [
                {
                    "task_id": run.task_id,
                    "agent_name": run.agent_name,
                    "status": run.status,
                    "error": run.error,
                    "artifact_count": len(run.artifacts),
                }
                for run in task_runs
            ],
            "artifact_excerpts": excerpts,
        }
        context = json.dumps(payload, indent=2, ensure_ascii=True)
        self._consume_tokens(context)

        try:
            provider = create_provider(self._resolve_llm_config(self.llm_registry))
            question = (
                "Synthesize a final markdown result from the executed task graph. "
                "Include: objective, completed outputs, failed tasks, and next steps."
            )
            response = provider.synthesize(question=question, context=context, mode="grounded")
            self._consume_tokens(response)
            text = str(response).strip()
            if text:
                return text + ("\n" if not text.endswith("\n") else "")
        except Exception as exc:
            self._log(f"Final synthesis fallback triggered: {exc}")

        return fallback

    @staticmethod
    def _build_artifact_excerpts(task_runs: List[MetaTaskRun], max_items: int = 12) -> List[Dict[str, str]]:
        excerpts: list[Dict[str, str]] = []
        allowed_suffixes = {".md", ".txt", ".json"}

        for run in task_runs:
            for artifact in run.artifacts:
                path = Path(artifact)
                if not path.exists() or not path.is_file():
                    continue
                if path.suffix.lower() not in allowed_suffixes:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                excerpts.append(
                    {
                        "task_id": run.task_id,
                        "artifact": str(path),
                        "excerpt": text[:800],
                    }
                )
                if len(excerpts) >= max_items:
                    return excerpts
        return excerpts

    @staticmethod
    def _fallback_final_result(
        goal: str,
        graph: TaskGraph,
        execution_order: List[str],
        task_runs: List[MetaTaskRun],
    ) -> str:
        lines = [
            f"# Final Result: {goal}",
            "",
            "## Summary",
            "",
            f"- Planned tasks: {len(graph.tasks)}",
            f"- Completed tasks: {sum(1 for run in task_runs if run.status == 'completed')}",
            f"- Failed tasks: {sum(1 for run in task_runs if run.status == 'failed')}",
            "",
            "## Execution Order",
            "",
        ]
        if execution_order:
            lines.extend([f"- {task_id}" for task_id in execution_order])
        else:
            lines.append("- None")

        lines.extend(["", "## Task Outputs", ""])
        by_task = {run.task_id: run for run in task_runs}
        for task in graph.tasks:
            run = by_task.get(task.id)
            status = run.status if run is not None else task.status
            lines.append(f"### {task.id} ({task.agent_name})")
            lines.append("")
            lines.append(f"- Status: {status}")
            if run is not None and run.error:
                lines.append(f"- Error: {run.error}")
            artifact_count = len(run.artifacts) if run is not None else 0
            lines.append(f"- Artifacts: {artifact_count}")
            if run is not None and run.artifacts:
                lines.append("- Artifact paths:")
                for artifact in run.artifacts[:10]:
                    lines.append(f"  - `{artifact}`")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _build_meta_log(
        self,
        *,
        goal: str,
        graph: TaskGraph,
        execution_order: List[str],
        task_runs: List[MetaTaskRun],
        final_result_path: Path,
    ) -> str:
        lines = [
            "# Meta Log",
            "",
            "## Run",
            "",
            f"- Goal: {goal}",
            f"- Execution mode: {self.execution_mode}",
            f"- Planned task count: {len(graph.tasks)}",
            f"- Completed task count: {sum(1 for run in task_runs if run.status == 'completed')}",
            f"- Failed task count: {sum(1 for run in task_runs if run.status == 'failed')}",
            f"- Final result: `{final_result_path}`",
            "",
            "## Execution Order",
            "",
        ]
        if execution_order:
            lines.extend([f"- {task_id}" for task_id in execution_order])
        else:
            lines.append("- None")

        lines.extend(["", "## Task Trace", ""])
        by_task = {run.task_id: run for run in task_runs}
        for task in graph.tasks:
            run = by_task.get(task.id)
            lines.append(f"### Task `{task.id}`")
            lines.append("")
            lines.append(f"- Agent: {task.agent_name}")
            lines.append(f"- Depends on: {task.depends_on or []}")
            lines.append(f"- Expected artifacts: {task.expected_artifacts or []}")
            lines.append(f"- Status: {run.status if run is not None else task.status}")
            if run is not None:
                lines.append(f"- Sub-agent directory: `{run.sub_agent_dir}`")
                lines.append(f"- Artifacts produced: {len(run.artifacts)}")
                if run.error:
                    lines.append(f"- Error: {run.error}")
            lines.append("")

        lines.extend(["## Agent Log Entries", ""])
        for entry in self.state.log_entries:
            stamp = entry.timestamp.isoformat()
            actor = str(entry.actor or "agent")
            content = str(entry.content or "").strip()
            lines.append(f"- [{stamp}] {actor}: {content}")
        lines.append("")

        return "\n".join(lines).strip() + "\n"

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

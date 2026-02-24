"""
Benchmark harness for agent/tool regression tracking.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from lsm.agents.tools.base import ToolRegistry
from lsm.agents.tools.read_file import ReadFileTool
from lsm.agents.tools.read_folder import ReadFolderTool
from lsm.agents.tools.write_file import WriteFileTool
from lsm.agents.tools.find_file import FindFileTool
from lsm.agents.tools.find_section import FindSectionTool
from lsm.agents.tools.edit_file import EditFileTool
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


DEFAULT_BASELINE_PATH = Path(__file__).parent / "baselines" / "file_ops.json"


@dataclass
class RegressionThresholds:
    """Allowed variance before flagging a regression."""

    max_wall_time_pct: float = 0.25
    max_tokens_pct: float = 0.25
    max_tool_calls_increase: int = 1
    max_quality_drop: float = 0.1


@dataclass
class BenchmarkMetrics:
    """Captured metrics for a benchmark run."""

    wall_time_s: Optional[float]
    tool_calls: Optional[int]
    tokens_used: Optional[int]
    quality_score: Optional[float]


@dataclass
class BenchmarkTask:
    """Executable benchmark task definition."""

    name: str
    description: str
    scenario: str
    expected: str
    scoring: Iterable[str] = field(default_factory=lambda: ("time", "tools", "tokens", "quality"))
    thresholds: RegressionThresholds = field(default_factory=RegressionThresholds)
    setup: Optional[Callable[["BenchmarkContext"], None]] = None
    run: Optional[Callable[["BenchmarkContext"], Any]] = None
    evaluate: Optional[Callable[["BenchmarkContext", Any], float]] = None


@dataclass
class BenchmarkResult:
    """Result payload for a benchmark run."""

    task: BenchmarkTask
    metrics: BenchmarkMetrics
    regressions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.regressions


class BenchmarkContext:
    """Runtime context for executing benchmark tasks."""

    def __init__(self, workspace: Path, tool_registry: ToolRegistry, sandbox: ToolSandbox) -> None:
        self.workspace = workspace
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.tool_calls = 0
        self.tokens_used = 0

    def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        tool = self.tool_registry.lookup(name)
        output = self.sandbox.execute(tool, args)
        self.tool_calls += 1
        return output

    def consume_tokens(self, text: str) -> int:
        estimated = max(1, len(str(text)) // 4)
        self.tokens_used += estimated
        return estimated


class BenchmarkRunner:
    """Execute benchmark tasks with a controlled tool sandbox."""

    def run_task(self, task: BenchmarkTask, baseline: Optional[BenchmarkMetrics]) -> BenchmarkResult:
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            registry = self._build_tool_registry()
            sandbox = self._build_sandbox(workspace_path)
            context = BenchmarkContext(workspace_path, registry, sandbox)

            if task.setup:
                task.setup(context)
            if task.run is None:
                raise ValueError(f"Benchmark task '{task.name}' has no run() callback")

            start = time.perf_counter()
            output = task.run(context)
            wall_time_s = time.perf_counter() - start

            quality_score = None
            if task.evaluate is not None:
                quality_score = float(task.evaluate(context, output))

            metrics = BenchmarkMetrics(
                wall_time_s=round(wall_time_s, 6),
                tool_calls=context.tool_calls,
                tokens_used=context.tokens_used,
                quality_score=quality_score,
            )

            regressions = []
            if baseline is not None:
                regressions = compare_metrics(baseline, metrics, task.thresholds)

            return BenchmarkResult(task=task, metrics=metrics, regressions=regressions)

    @staticmethod
    def _build_tool_registry() -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(ReadFolderTool())
        registry.register(ReadFileTool())
        registry.register(FindFileTool())
        registry.register(FindSectionTool())
        registry.register(EditFileTool())
        registry.register(WriteFileTool())
        return registry

    @staticmethod
    def _build_sandbox(workspace: Path) -> ToolSandbox:
        sandbox_config = SandboxConfig(
            allowed_read_paths=[workspace],
            allowed_write_paths=[workspace],
            allow_url_access=False,
            require_user_permission={"write_file": False},
            require_permission_by_risk={"writes_workspace": False},
            execution_mode="local_only",
        )
        return ToolSandbox(sandbox_config)


def compare_metrics(
    baseline: BenchmarkMetrics,
    current: BenchmarkMetrics,
    thresholds: RegressionThresholds,
) -> list[str]:
    regressions: list[str] = []

    if baseline.wall_time_s is not None and current.wall_time_s is not None:
        limit = baseline.wall_time_s * (1.0 + thresholds.max_wall_time_pct)
        if current.wall_time_s > limit:
            regressions.append(
                f"Wall time regression: {current.wall_time_s:.4f}s > {limit:.4f}s"
            )

    if baseline.tool_calls is not None and current.tool_calls is not None:
        limit = baseline.tool_calls + thresholds.max_tool_calls_increase
        if current.tool_calls > limit:
            regressions.append(
                f"Tool call regression: {current.tool_calls} > {limit}"
            )

    if baseline.tokens_used is not None and current.tokens_used is not None:
        limit = baseline.tokens_used * (1.0 + thresholds.max_tokens_pct)
        if current.tokens_used > limit:
            regressions.append(
                f"Token regression: {current.tokens_used} > {int(limit)}"
            )

    if baseline.quality_score is not None and current.quality_score is not None:
        limit = baseline.quality_score - thresholds.max_quality_drop
        if current.quality_score < limit:
            regressions.append(
                f"Quality regression: {current.quality_score:.3f} < {limit:.3f}"
            )

    return regressions


def load_tasks(task_names: Optional[Iterable[str]] = None) -> list[BenchmarkTask]:
    from tests.benchmarks import tasks as tasks_pkg

    name_filter = {name.strip() for name in task_names or [] if name.strip()}

    loaded: list[BenchmarkTask] = []
    for info in pkgutil.iter_modules(tasks_pkg.__path__, tasks_pkg.__name__ + "."):
        module = importlib.import_module(info.name)
        builder = getattr(module, "build_task", None)
        if not callable(builder):
            continue
        task = builder()
        if name_filter and task.name not in name_filter:
            continue
        loaded.append(task)

    return sorted(loaded, key=lambda item: item.name)


def load_baseline(path: Path) -> Dict[str, BenchmarkMetrics]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = raw.get("tasks", {}) if isinstance(raw, dict) else {}
    baseline: Dict[str, BenchmarkMetrics] = {}
    for name, metrics in tasks.items():
        if not isinstance(metrics, dict):
            continue
        baseline[name] = BenchmarkMetrics(
            wall_time_s=_coerce_optional_float(metrics.get("wall_time_s")),
            tool_calls=_coerce_optional_int(metrics.get("tool_calls")),
            tokens_used=_coerce_optional_int(metrics.get("tokens_used")),
            quality_score=_coerce_optional_float(metrics.get("quality_score")),
        )
    return baseline


def write_baseline(path: Path, results: Iterable[BenchmarkResult]) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": {},
    }
    for result in results:
        payload["tasks"][result.task.name] = serialize_metrics(result.metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def serialize_metrics(metrics: BenchmarkMetrics) -> Dict[str, Any]:
    return {
        "wall_time_s": metrics.wall_time_s,
        "tool_calls": metrics.tool_calls,
        "tokens_used": metrics.tokens_used,
        "quality_score": metrics.quality_score,
    }


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _render_result(result: BenchmarkResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    metrics = result.metrics
    lines = [
        f"[{status}] {result.task.name}",
        f"  wall_time_s: {metrics.wall_time_s}",
        f"  tool_calls: {metrics.tool_calls}",
        f"  tokens_used: {metrics.tokens_used}",
        f"  quality_score: {metrics.quality_score}",
    ]
    for regression in result.regressions:
        lines.append(f"  regression: {regression}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Local Second Mind benchmark tasks")
    parser.add_argument(
        "--tasks",
        help="Comma-separated task names (default: all)",
        default="",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to baseline JSON file",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite baseline file with current results",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmark tasks",
    )

    args = parser.parse_args(argv)
    task_filter = [name.strip() for name in args.tasks.split(",") if name.strip()]

    tasks = load_tasks(task_filter)
    if args.list:
        for task in tasks:
            print(task.name)
        return 0

    if not tasks:
        print("No benchmark tasks found.")
        return 1

    baseline = load_baseline(args.baseline)
    runner = BenchmarkRunner()
    results = [runner.run_task(task, baseline.get(task.name)) for task in tasks]

    for result in results:
        print(_render_result(result))

    if args.write_baseline:
        write_baseline(args.baseline, results)
        print(f"Baseline updated: {args.baseline}")
        return 0

    failed = [result for result in results if not result.passed]
    if failed:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

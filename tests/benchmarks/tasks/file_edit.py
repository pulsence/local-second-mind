"""Benchmark task: file edit via write tool."""

from __future__ import annotations

import json
import os

from tests.benchmarks.harness import BenchmarkContext, BenchmarkTask, RegressionThresholds

_FILLER = "\n".join([f"# filler {idx}" for idx in range(120)])
_INITIAL = (
    f"{_FILLER}\n\n"
    "def helper():\n    return 1\n\n\n"
    "def target_function():\n    return 2\n"
)
_UPDATED = "def target_function():\n    return 3\n"


def _setup(context: BenchmarkContext) -> None:
    path = context.workspace / "edit.py"
    path.write_text(_INITIAL, encoding="utf-8")


def _run(context: BenchmarkContext) -> str:
    mode = os.getenv("LSM_BENCHMARK_MODE", "advanced").lower()
    target_path = context.workspace / "edit.py"

    if mode == "naive":
        content = context.execute_tool("read_file", {"path": str(target_path)})
        context.consume_tokens(content)
        updated = content.replace("return 2", "return 3")
        output = context.execute_tool(
            "write_file",
            {"path": str(target_path), "content": updated},
        )
        context.consume_tokens(output)
        confirmation = context.execute_tool("read_file", {"path": str(target_path)})
        context.consume_tokens(confirmation)
        return confirmation

    section_output = context.execute_tool(
        "find_section",
        {
            "path": str(target_path),
            "section": "target_function",
            "node_type": "function",
            "max_results": 1,
        },
    )
    context.consume_tokens(section_output)
    payload = json.loads(section_output)
    entry = payload[0] if payload else {}
    output = context.execute_tool(
        "edit_file",
        {
            "path": str(target_path),
            "start_hash": entry.get("start_hash", ""),
            "end_hash": entry.get("end_hash", ""),
            "new_content": _UPDATED,
        },
    )
    context.consume_tokens(output)
    return output


def _evaluate(context: BenchmarkContext, _: str) -> float:
    path = context.workspace / "edit.py"
    content = path.read_text(encoding="utf-8")
    return 1.0 if "return 3" in content else 0.0


def build_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="file_edit",
        description="Edit a function via naive vs line-hash tools.",
        scenario="Replace a line inside a target function.",
        expected="Target function reflects the updated return value.",
        thresholds=RegressionThresholds(max_wall_time_pct=5.0),
        setup=_setup,
        run=_run,
        evaluate=_evaluate,
    )

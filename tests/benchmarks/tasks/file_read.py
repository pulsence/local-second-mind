"""Benchmark task: file read."""

from __future__ import annotations

import json
import os

from tests.benchmarks.harness import BenchmarkContext, BenchmarkTask, RegressionThresholds

_SAMPLE_TEXT = "Local Second Mind\nBenchmark Read\n"
_TARGET_SECTION = "## Target Section\n\nLine A\nLine B\n"


def _setup(context: BenchmarkContext) -> None:
    path = context.workspace / "readme.md"
    filler = "\n".join([f"Line {idx}" for idx in range(400)])
    path.write_text(
        f"{_SAMPLE_TEXT}\n{filler}\n\n{_TARGET_SECTION}\n",
        encoding="utf-8",
    )


def _run(context: BenchmarkContext) -> str:
    mode = os.getenv("LSM_BENCHMARK_MODE", "advanced").lower()
    if mode == "naive":
        output = context.execute_tool(
            "read_file",
            {"path": str(context.workspace / "readme.md")},
        )
        context.consume_tokens(output)
        return output

    output = context.execute_tool(
        "read_file",
        {
            "path": str(context.workspace / "readme.md"),
            "section": "Target Section",
            "max_depth": 1,
        },
    )
    context.consume_tokens(output)
    payload = json.loads(output)
    return str(payload.get("content", ""))


def _evaluate(_: BenchmarkContext, content: str) -> float:
    return 1.0 if "Target Section" in content else 0.0


def build_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="file_read",
        description="Read a UTF-8 text file from disk.",
        scenario="Open a small markdown-like note file and return its contents.",
        expected="Exact file contents are returned.",
        thresholds=RegressionThresholds(max_wall_time_pct=5.0),
        setup=_setup,
        run=_run,
        evaluate=_evaluate,
    )

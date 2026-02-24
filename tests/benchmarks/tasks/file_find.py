"""Benchmark task: recursive file discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tests.benchmarks.harness import BenchmarkContext, BenchmarkTask, RegressionThresholds


def _setup(context: BenchmarkContext) -> None:
    root = context.workspace / "docs"
    nested = root / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (root / "note.txt").write_text("Just a note.", encoding="utf-8")
    (root / "alpha.txt").write_text("alpha content", encoding="utf-8")
    (nested / "beta.txt").write_text("beta content", encoding="utf-8")
    target = nested / "target.txt"
    target.write_text("Find the needle in this file.", encoding="utf-8")


def _run(context: BenchmarkContext) -> str:
    mode = os.getenv("LSM_BENCHMARK_MODE", "advanced").lower()
    if mode == "naive":
        output = context.execute_tool(
            "read_folder",
            {"path": str(context.workspace), "recursive": True},
        )
        context.consume_tokens(output)
        entries = json.loads(output)
        for entry in entries:
            if entry.get("is_dir"):
                continue
            path = str(entry.get("path"))
            content = context.execute_tool("read_file", {"path": path})
            context.consume_tokens(content)
            if "needle" in content:
                return path
        return ""

    output = context.execute_tool(
        "find_file",
        {
            "path": str(context.workspace),
            "content_pattern": "needle",
            "max_results": 1,
        },
    )
    context.consume_tokens(output)
    entries = json.loads(output)
    if entries:
        return str(entries[0].get("path", ""))
    return ""


def _evaluate(context: BenchmarkContext, found_path: str) -> float:
    expected = context.workspace / "docs" / "nested" / "target.txt"
    return 1.0 if Path(found_path).resolve() == expected.resolve() else 0.0


def build_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="file_find",
        description="Locate a nested file via a recursive folder listing.",
        scenario="Search for a known filename within a nested directory tree.",
        expected="Returns the absolute path to docs/nested/target.txt.",
        thresholds=RegressionThresholds(max_wall_time_pct=5.0),
        setup=_setup,
        run=_run,
        evaluate=_evaluate,
    )

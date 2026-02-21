"""Benchmark task: recursive file discovery."""

from __future__ import annotations

import json
from pathlib import Path

from tests.benchmarks.harness import BenchmarkContext, BenchmarkTask


def _setup(context: BenchmarkContext) -> None:
    root = context.workspace / "docs" / "nested"
    root.mkdir(parents=True, exist_ok=True)
    (context.workspace / "docs" / "note.txt").write_text("note", encoding="utf-8")
    target = root / "target.txt"
    target.write_text("target", encoding="utf-8")


def _run(context: BenchmarkContext) -> str:
    output = context.execute_tool(
        "read_folder",
        {"path": str(context.workspace), "recursive": True},
    )
    entries = json.loads(output)
    for entry in entries:
        if entry.get("name") == "target.txt":
            return str(entry.get("path"))
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
        setup=_setup,
        run=_run,
        evaluate=_evaluate,
    )

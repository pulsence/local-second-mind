"""Benchmark task: file edit via write tool."""

from __future__ import annotations

from tests.benchmarks.harness import BenchmarkContext, BenchmarkTask

_INITIAL = "alpha\n"
_APPEND = "beta\n"


def _setup(context: BenchmarkContext) -> None:
    path = context.workspace / "edit.txt"
    path.write_text(_INITIAL, encoding="utf-8")


def _run(context: BenchmarkContext) -> str:
    return context.execute_tool(
        "write_file",
        {
            "path": str(context.workspace / "edit.txt"),
            "content": _APPEND,
            "append": True,
        },
    )


def _evaluate(context: BenchmarkContext, _: str) -> float:
    path = context.workspace / "edit.txt"
    content = path.read_text(encoding="utf-8")
    return 1.0 if content == _INITIAL + _APPEND else 0.0


def build_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="file_edit",
        description="Append content to a file via the write tool.",
        scenario="Append a single line to an existing file.",
        expected="File contains the appended line in order.",
        setup=_setup,
        run=_run,
        evaluate=_evaluate,
    )

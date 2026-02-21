"""Benchmark task: file read."""

from __future__ import annotations

from tests.benchmarks.harness import BenchmarkContext, BenchmarkTask

_SAMPLE_TEXT = "Local Second Mind\nBenchmark Read\n"


def _setup(context: BenchmarkContext) -> None:
    path = context.workspace / "readme.txt"
    path.write_text(_SAMPLE_TEXT, encoding="utf-8")


def _run(context: BenchmarkContext) -> str:
    return context.execute_tool("read_file", {"path": str(context.workspace / "readme.txt")})


def _evaluate(_: BenchmarkContext, content: str) -> float:
    return 1.0 if content == _SAMPLE_TEXT else 0.0


def build_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="file_read",
        description="Read a UTF-8 text file from disk.",
        scenario="Open a small markdown-like note file and return its contents.",
        expected="Exact file contents are returned.",
        setup=_setup,
        run=_run,
        evaluate=_evaluate,
    )

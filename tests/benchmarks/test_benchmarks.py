from __future__ import annotations

import os

from tests.benchmarks.harness import BenchmarkRunner, DEFAULT_BASELINE_PATH, load_baseline, load_tasks


def test_benchmark_tasks_against_baseline() -> None:
    os.environ.pop("LSM_BENCHMARK_MODE", None)
    baseline = load_baseline(DEFAULT_BASELINE_PATH)
    runner = BenchmarkRunner()
    results = [runner.run_task(task, baseline.get(task.name)) for task in load_tasks()]
    assert results
    assert all(result.passed for result in results)

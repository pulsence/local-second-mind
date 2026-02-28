"""Tests for evaluation harness, dataset loading, and baselines."""

from __future__ import annotations

from pathlib import Path

import pytest

from lsm.eval.baselines import (
    delete_baseline,
    list_baselines,
    load_baseline,
    save_baseline,
)
from lsm.eval.dataset import EvalDataset, load_bundled_dataset, load_dataset
from lsm.eval.harness import EvalHarness, EvalResult


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestDataset:
    def test_load_bundled_dataset(self):
        ds = load_bundled_dataset()
        assert ds.num_queries >= 50
        assert ds.num_documents >= 50
        assert len(ds.qrels) >= 50

    def test_relevant_for(self):
        ds = load_bundled_dataset()
        relevant = ds.relevant_for("q01")
        assert isinstance(relevant, set)
        assert len(relevant) >= 1

    def test_relevant_for_missing_query(self):
        ds = load_bundled_dataset()
        relevant = ds.relevant_for("nonexistent")
        assert relevant == set()

    def test_load_dataset_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path")

    def test_load_dataset_roundtrip(self, tmp_path: Path):
        import json

        queries = {"q1": "test query"}
        corpus = {"d1": "test doc"}
        qrels = {"q1": {"d1": 1}}

        (tmp_path / "queries.json").write_text(json.dumps(queries))
        (tmp_path / "corpus.json").write_text(json.dumps(corpus))
        (tmp_path / "qrels.json").write_text(json.dumps(qrels))

        ds = load_dataset(tmp_path)
        assert ds.num_queries == 1
        assert ds.num_documents == 1
        assert ds.relevant_for("q1") == {"d1"}


# ---------------------------------------------------------------------------
# Harness tests
# ---------------------------------------------------------------------------


def _make_simple_dataset():
    return EvalDataset(
        name="test",
        queries={"q1": "find alpha", "q2": "find beta"},
        corpus={
            "d1": "alpha content",
            "d2": "beta content",
            "d3": "gamma content",
        },
        qrels={
            "q1": {"d1": 1},
            "q2": {"d2": 1},
        },
    )


class TestHarness:
    def test_run_perfect_retrieval(self):
        ds = _make_simple_dataset()
        harness = EvalHarness(ds)

        def perfect_fn(query_text):
            if "alpha" in query_text:
                return [("d1", "alpha.md"), ("d3", "gamma.md")]
            return [("d2", "beta.md"), ("d3", "gamma.md")]

        result = harness.run(perfect_fn, profile="test")
        assert result.recall_at_5 == 1.0
        assert result.mrr_value == 1.0
        assert result.num_queries == 2
        assert len(result.per_query) == 2

    def test_run_empty_retrieval(self):
        ds = _make_simple_dataset()
        harness = EvalHarness(ds)

        result = harness.run(lambda q: [], profile="empty")
        assert result.recall_at_5 == 0.0
        assert result.mrr_value == 0.0

    def test_run_partial_retrieval(self):
        ds = _make_simple_dataset()
        harness = EvalHarness(ds)

        # Only q1 gets correct result; q2 gets wrong result
        def partial_fn(query_text):
            if "alpha" in query_text:
                return [("d1", "alpha.md")]
            return [("d3", "gamma.md")]

        result = harness.run(partial_fn, profile="partial")
        assert result.recall_at_5 == 0.5  # 1 of 2 queries hit
        assert result.mrr_value == 0.5  # avg of 1.0 and 0.0

    def test_result_serialization_roundtrip(self):
        ds = _make_simple_dataset()
        harness = EvalHarness(ds)

        result = harness.run(lambda q: [("d1", "a.md")], profile="rt")
        d = result.to_dict()
        restored = EvalResult.from_dict(d)

        assert restored.profile == result.profile
        assert restored.recall_at_5 == result.recall_at_5
        assert restored.mrr_value == result.mrr_value
        assert len(restored.per_query) == len(result.per_query)

    def test_comparison_report(self):
        ds = _make_simple_dataset()
        harness = EvalHarness(ds)

        current = harness.run(
            lambda q: [("d1", "a.md"), ("d2", "b.md")],
            profile="current",
        )
        baseline = harness.run(lambda q: [], profile="baseline")

        report = EvalHarness.compare(current, baseline)
        assert report.deltas["recall@5"] > 0
        assert report.deltas["mrr"] > 0

        summary = report.summary()
        assert "current" in summary
        assert "baseline" in summary

    def test_latency_tracked(self):
        ds = _make_simple_dataset()
        harness = EvalHarness(ds)

        result = harness.run(lambda q: [("d1", "a.md")], profile="latency")
        assert result.latency["mean"] >= 0
        assert all(q.latency_ms >= 0 for q in result.per_query)


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        result_dict = {"recall@5": 0.8, "profile": "test"}
        save_baseline(result_dict, "my-baseline", tmp_path)

        loaded = load_baseline("my-baseline", tmp_path)
        assert loaded is not None
        assert loaded["recall@5"] == 0.8

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        assert load_baseline("nonexistent", tmp_path) is None

    def test_list_baselines(self, tmp_path: Path):
        save_baseline({"a": 1}, "first", tmp_path)
        save_baseline({"b": 2}, "second", tmp_path)

        names = list_baselines(tmp_path)
        assert "first" in names
        assert "second" in names

    def test_list_baselines_empty(self, tmp_path: Path):
        assert list_baselines(tmp_path) == []

    def test_delete_baseline(self, tmp_path: Path):
        save_baseline({"a": 1}, "to-delete", tmp_path)
        assert delete_baseline("to-delete", tmp_path)
        assert load_baseline("to-delete", tmp_path) is None

    def test_delete_nonexistent(self, tmp_path: Path):
        assert not delete_baseline("nonexistent", tmp_path)

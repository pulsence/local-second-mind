"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import math

import pytest

from lsm.eval.metrics import (
    diversity_at_k,
    latency_stats,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "e"}
        assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)

    def test_zero_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_cutoff_matters(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, 5) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a", "b"], {"a"}) == 0.5

    def test_third_position(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_returns_first(self):
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_empty_lists(self):
        assert mrr([], {"a"}) == 0.0


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_ndcg(self):
        # Only first of 2 relevant is in top-2
        retrieved = ["a", "x"]
        relevant = {"a", "b"}
        result = ndcg_at_k(retrieved, relevant, 2)
        assert 0.0 < result < 1.0

    def test_order_matters(self):
        relevant = {"a", "b"}
        # a at rank 1 is better than a at rank 2
        ndcg_first = ndcg_at_k(["a", "x"], relevant, 2)
        ndcg_second = ndcg_at_k(["x", "a"], relevant, 2)
        assert ndcg_first > ndcg_second

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 2) == 0.0


class TestDiversityAtK:
    def test_all_unique(self):
        sources = ["file1.md", "file2.md", "file3.md"]
        assert diversity_at_k(sources, 3) == 1.0

    def test_all_same(self):
        sources = ["file1.md", "file1.md", "file1.md"]
        assert diversity_at_k(sources, 3) == pytest.approx(1 / 3)

    def test_partial_diversity(self):
        sources = ["a.md", "a.md", "b.md", "c.md"]
        assert diversity_at_k(sources, 4) == pytest.approx(3 / 4)

    def test_cutoff(self):
        sources = ["a.md", "a.md", "b.md"]
        assert diversity_at_k(sources, 2) == pytest.approx(1 / 2)

    def test_empty(self):
        assert diversity_at_k([], 5) == 0.0


class TestLatencyStats:
    def test_single_value(self):
        stats = latency_stats([100.0])
        assert stats["mean"] == 100.0
        assert stats["p50"] == 100.0

    def test_multiple_values(self):
        timings = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = latency_stats(timings)
        assert stats["mean"] == 30.0
        assert stats["p50"] == 30.0

    def test_percentiles_ordered(self):
        timings = list(range(1, 101))
        stats = latency_stats([float(t) for t in timings])
        assert stats["p50"] <= stats["p95"] <= stats["p99"]

    def test_empty(self):
        stats = latency_stats([])
        assert stats["mean"] == 0.0
        assert stats["p50"] == 0.0
        assert stats["p95"] == 0.0
        assert stats["p99"] == 0.0

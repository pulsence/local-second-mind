"""
Retrieval evaluation harness.

Runs evaluation queries against a pipeline or retrieval function,
collects metrics, and supports baseline comparisons.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from lsm.eval.dataset import EvalDataset
from lsm.eval.metrics import (
    diversity_at_k,
    latency_stats,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from lsm.logging import get_logger

logger = get_logger(__name__)

# Type for a retrieval function: query_text -> list of (doc_id, source_path) tuples
RetrievalFn = Callable[[str], List[tuple]]


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query_id: str
    recall_at_5: float
    recall_at_10: float
    mrr_value: float
    ndcg_at_5: float
    ndcg_at_10: float
    diversity_at_5: float
    latency_ms: float
    num_retrieved: int


@dataclass
class EvalResult:
    """Aggregate evaluation result across all queries."""

    dataset_name: str
    profile: str
    num_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr_value: float
    ndcg_at_5: float
    ndcg_at_10: float
    diversity_at_5: float
    latency: dict
    per_query: List[QueryMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "profile": self.profile,
            "num_queries": self.num_queries,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr_value,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "diversity@5": self.diversity_at_5,
            "latency": self.latency,
            "per_query": [
                {
                    "query_id": q.query_id,
                    "recall@5": q.recall_at_5,
                    "recall@10": q.recall_at_10,
                    "mrr": q.mrr_value,
                    "ndcg@5": q.ndcg_at_5,
                    "ndcg@10": q.ndcg_at_10,
                    "diversity@5": q.diversity_at_5,
                    "latency_ms": q.latency_ms,
                    "num_retrieved": q.num_retrieved,
                }
                for q in self.per_query
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalResult:
        per_query = [
            QueryMetrics(
                query_id=q["query_id"],
                recall_at_5=q["recall@5"],
                recall_at_10=q["recall@10"],
                mrr_value=q["mrr"],
                ndcg_at_5=q["ndcg@5"],
                ndcg_at_10=q["ndcg@10"],
                diversity_at_5=q["diversity@5"],
                latency_ms=q["latency_ms"],
                num_retrieved=q["num_retrieved"],
            )
            for q in data.get("per_query", [])
        ]
        return cls(
            dataset_name=data["dataset_name"],
            profile=data["profile"],
            num_queries=data["num_queries"],
            recall_at_5=data["recall@5"],
            recall_at_10=data["recall@10"],
            mrr_value=data["mrr"],
            ndcg_at_5=data["ndcg@5"],
            ndcg_at_10=data["ndcg@10"],
            diversity_at_5=data["diversity@5"],
            latency=data["latency"],
            per_query=per_query,
        )


@dataclass
class ComparisonReport:
    """Comparison between two evaluation results."""

    current: EvalResult
    baseline: EvalResult
    deltas: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "current_profile": self.current.profile,
            "baseline_profile": self.baseline.profile,
            "deltas": self.deltas,
        }

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [
            f"Comparison: {self.current.profile} vs {self.baseline.profile} (baseline)",
            "",
        ]
        metric_names = {
            "recall@5": "Recall@5",
            "recall@10": "Recall@10",
            "mrr": "MRR",
            "ndcg@5": "nDCG@5",
            "ndcg@10": "nDCG@10",
            "diversity@5": "Diversity@5",
        }
        for key, label in metric_names.items():
            delta = self.deltas.get(key, 0.0)
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {label:15s}: {sign}{delta:.4f}")
        return "\n".join(lines)


class EvalHarness:
    """
    Retrieval evaluation harness.

    Runs queries from a dataset through a retrieval function and computes
    standard IR metrics.
    """

    def __init__(self, dataset: EvalDataset) -> None:
        self.dataset = dataset

    def run(
        self,
        retrieve_fn: RetrievalFn,
        profile: str = "default",
        k_values: tuple = (5, 10),
    ) -> EvalResult:
        """
        Run evaluation across all dataset queries.

        Args:
            retrieve_fn: Function that takes a query string and returns
                a list of (doc_id, source_path) tuples in ranked order.
            profile: Name of the retrieval profile being evaluated.
            k_values: Tuple of k values for recall/nDCG computation.

        Returns:
            EvalResult with aggregate and per-query metrics.
        """
        per_query: List[QueryMetrics] = []
        timings: List[float] = []

        for query_id, query_text in self.dataset.queries.items():
            relevant = self.dataset.relevant_for(query_id)

            start = time.perf_counter()
            results = retrieve_fn(query_text)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            retrieved_ids = [r[0] for r in results]
            retrieved_sources = [r[1] for r in results]

            qm = QueryMetrics(
                query_id=query_id,
                recall_at_5=recall_at_k(retrieved_ids, relevant, 5),
                recall_at_10=recall_at_k(retrieved_ids, relevant, 10),
                mrr_value=mrr(retrieved_ids, relevant),
                ndcg_at_5=ndcg_at_k(retrieved_ids, relevant, 5),
                ndcg_at_10=ndcg_at_k(retrieved_ids, relevant, 10),
                diversity_at_5=diversity_at_k(retrieved_sources, 5),
                latency_ms=elapsed_ms,
                num_retrieved=len(results),
            )
            per_query.append(qm)
            timings.append(elapsed_ms)

        n = len(per_query)

        def _avg(attr: str) -> float:
            if n == 0:
                return 0.0
            return sum(getattr(q, attr) for q in per_query) / n

        return EvalResult(
            dataset_name=self.dataset.name,
            profile=profile,
            num_queries=n,
            recall_at_5=_avg("recall_at_5"),
            recall_at_10=_avg("recall_at_10"),
            mrr_value=_avg("mrr_value"),
            ndcg_at_5=_avg("ndcg_at_5"),
            ndcg_at_10=_avg("ndcg_at_10"),
            diversity_at_5=_avg("diversity_at_5"),
            latency=latency_stats(timings),
            per_query=per_query,
        )

    @staticmethod
    def compare(current: EvalResult, baseline: EvalResult) -> ComparisonReport:
        """
        Compare two evaluation results.

        Args:
            current: Current evaluation result.
            baseline: Baseline evaluation result.

        Returns:
            ComparisonReport with per-metric deltas.
        """
        deltas = {
            "recall@5": current.recall_at_5 - baseline.recall_at_5,
            "recall@10": current.recall_at_10 - baseline.recall_at_10,
            "mrr": current.mrr_value - baseline.mrr_value,
            "ndcg@5": current.ndcg_at_5 - baseline.ndcg_at_5,
            "ndcg@10": current.ndcg_at_10 - baseline.ndcg_at_10,
            "diversity@5": current.diversity_at_5 - baseline.diversity_at_5,
        }
        return ComparisonReport(current=current, baseline=baseline, deltas=deltas)

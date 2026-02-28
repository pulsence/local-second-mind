"""
Standard information retrieval metrics.

All functions accept lists of retrieved document IDs and sets of relevant
document IDs, following BEIR / TREC conventions.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Set


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Recall@k: fraction of relevant documents found in the top-k retrieved.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs.
        k: Cutoff rank.

    Returns:
        Recall value in [0.0, 1.0].
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant)
    return found / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of first relevant result.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs.

    Returns:
        MRR value in [0.0, 1.0], or 0.0 if no relevant result found.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(gains: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    total = 0.0
    for i, g in enumerate(gains[:k]):
        total += g / math.log2(i + 2)  # i+2 because rank starts at 1
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs.
        k: Cutoff rank.

    Returns:
        nDCG value in [0.0, 1.0].
    """
    if not relevant:
        return 0.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
    dcg = _dcg_at_k(gains, k)
    ideal_gains = sorted(gains, reverse=True)
    # For ideal DCG, we also consider relevant docs not in retrieved
    n_relevant_in_k = min(len(relevant), k)
    ideal_gains_full = [1.0] * n_relevant_in_k + [0.0] * (k - n_relevant_in_k)
    idcg = _dcg_at_k(ideal_gains_full, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def diversity_at_k(retrieved_sources: Sequence[str], k: int) -> float:
    """
    Source diversity at k: fraction of unique sources in top-k results.

    Args:
        retrieved_sources: Ordered list of source identifiers (e.g. file paths)
            for each retrieved chunk.
        k: Cutoff rank.

    Returns:
        Diversity value in [0.0, 1.0].
    """
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    return len(set(top_k)) / len(top_k)


def latency_stats(timings: Sequence[float]) -> dict:
    """
    Compute latency statistics from a list of query durations (in ms).

    Args:
        timings: List of timing values in milliseconds.

    Returns:
        Dict with mean, p50, p95, p99 keys.
    """
    if not timings:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_t = sorted(timings)
    n = len(sorted_t)

    def percentile(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_t[lo] * (1 - frac) + sorted_t[hi] * frac

    return {
        "mean": sum(sorted_t) / n,
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
    }

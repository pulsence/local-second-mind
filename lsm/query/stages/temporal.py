"""
Temporal-aware ranking stage.

Applies recency boost and time-range filtering using ``mtime_ns``
metadata from indexed chunks.
"""

from __future__ import annotations

import time
from typing import List, Optional

from lsm.logging import get_logger
from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate

logger = get_logger(__name__)

# Nanoseconds per day
_NS_PER_DAY = 86_400_000_000_000


def apply_temporal_boost(
    candidates: List[Candidate],
    boost_days: int = 30,
    boost_factor: float = 1.5,
) -> List[Candidate]:
    """Apply recency boost to candidates with recent modification times.

    Candidates whose ``mtime_ns`` metadata falls within ``boost_days``
    of the current time receive their distance multiplied by ``1/boost_factor``
    (lower distance = higher relevance). The ``temporal_boost`` field is
    set on their ``ScoreBreakdown``.

    Args:
        candidates: Input candidates (must have ``mtime_ns`` in metadata).
        boost_days: Window in days for recency boost.
        boost_factor: Multiplier for boosted candidates (>1.0 = stronger boost).

    Returns:
        Candidates with temporal boost applied and re-sorted by distance.
    """
    if not candidates or boost_factor <= 1.0:
        return candidates

    now_ns = int(time.time() * 1_000_000_000)
    cutoff_ns = now_ns - (boost_days * _NS_PER_DAY)

    result: List[Candidate] = []
    for c in candidates:
        mtime = c.meta.get("mtime_ns") if c.meta else None
        if mtime is not None and int(mtime) >= cutoff_ns:
            # Apply boost — reduce distance
            boosted_distance = (c.distance or 0.0) / boost_factor
            existing = c.score_breakdown or ScoreBreakdown()
            result.append(
                Candidate(
                    cid=c.cid,
                    text=c.text,
                    meta=c.meta,
                    distance=boosted_distance,
                    score_breakdown=ScoreBreakdown(
                        dense_score=existing.dense_score,
                        dense_rank=existing.dense_rank,
                        sparse_score=existing.sparse_score,
                        sparse_rank=existing.sparse_rank,
                        fused_score=existing.fused_score,
                        rerank_score=existing.rerank_score,
                        temporal_boost=boost_factor,
                    ),
                    embedding=c.embedding,
                )
            )
        else:
            result.append(c)

    # Re-sort by distance (ascending = most relevant first)
    result.sort(key=lambda c: c.distance if c.distance is not None else float("inf"))
    return result


def filter_time_range(
    candidates: List[Candidate],
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> List[Candidate]:
    """Filter candidates to a specific time range.

    Args:
        candidates: Input candidates.
        start_ns: Minimum mtime_ns (inclusive). None = no lower bound.
        end_ns: Maximum mtime_ns (inclusive). None = no upper bound.

    Returns:
        Candidates within the specified time range.
    """
    if start_ns is None and end_ns is None:
        return candidates

    result: List[Candidate] = []
    for c in candidates:
        mtime = c.meta.get("mtime_ns") if c.meta else None
        if mtime is None:
            # Include candidates without time metadata
            result.append(c)
            continue
        mtime_val = int(mtime)
        if start_ns is not None and mtime_val < start_ns:
            continue
        if end_ns is not None and mtime_val > end_ns:
            continue
        result.append(c)

    return result

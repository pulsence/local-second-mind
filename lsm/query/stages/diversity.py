"""
Diversity selection stages: MMR and per-section caps.

Maximal Marginal Relevance (MMR) selects candidates that are both
relevant and diverse, reducing redundancy in the final context.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from lsm.logging import get_logger
from lsm.query.session import Candidate

logger = get_logger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def mmr_select(
    candidates: List[Candidate],
    query_embedding: Optional[List[float]] = None,
    lambda_param: float = 0.7,
    k: int = 10,
) -> List[Candidate]:
    """Select diverse candidates using Maximal Marginal Relevance.

    MMR formula: λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))

    Candidates must have ``embedding`` populated or a ``distance`` to
    compute a relevance proxy. If embeddings are not available, returns
    candidates in original order.

    Args:
        candidates: Input candidates with embeddings.
        query_embedding: Query embedding for relevance scoring.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
        k: Number of candidates to select.

    Returns:
        Selected candidates ordered by MMR score.
    """
    if not candidates or k <= 0:
        return []

    # Check if embeddings are available
    has_embeddings = all(c.embedding is not None for c in candidates)
    if not has_embeddings or query_embedding is None:
        logger.debug("MMR: embeddings not available; returning top-k by order")
        return candidates[:k]

    n = len(candidates)
    k = min(k, n)

    # Pre-compute relevance scores (query-doc similarity)
    relevance = np.array(
        [_cosine_similarity(query_embedding, c.embedding) for c in candidates],
        dtype=np.float32,
    )

    # Pre-compute pairwise similarities
    embs = np.array([c.embedding for c in candidates], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embs / norms
    sim_matrix = normed @ normed.T

    selected: List[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_score = float("-inf")

        for idx in remaining:
            rel = float(relevance[idx])

            if selected:
                max_sim = max(float(sim_matrix[idx][s]) for s in selected)
            else:
                max_sim = 0.0

            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.discard(best_idx)

    return [candidates[i] for i in selected]


def _get_heading_prefix(candidate: Candidate, depth: int) -> str:
    """Extract heading prefix at the given depth for grouping."""
    heading_path = candidate.heading_path
    if not heading_path:
        return ""
    prefix = heading_path[: min(depth, len(heading_path))]
    return " > ".join(prefix)


def per_section_cap(
    candidates: List[Candidate],
    max_per_section: int = 3,
    heading_depth: int = 2,
) -> List[Candidate]:
    """Cap the number of candidates per heading section.

    Groups candidates by their ``heading_path`` prefix at the specified
    depth, then limits each group to ``max_per_section`` candidates.

    Args:
        candidates: Input candidates.
        max_per_section: Maximum candidates per section group.
        heading_depth: Heading depth for grouping.

    Returns:
        Capped candidates preserving input order.
    """
    if max_per_section is None or max_per_section < 1:
        return candidates

    counts: Dict[str, int] = defaultdict(int)
    result: List[Candidate] = []

    for c in candidates:
        section_key = _get_heading_prefix(c, heading_depth)
        if counts[section_key] < max_per_section:
            result.append(c)
            counts[section_key] += 1

    removed = len(candidates) - len(result)
    if removed > 0:
        logger.debug("Per-section cap removed %d candidates", removed)

    return result

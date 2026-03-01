"""
Multi-vector retrieval stage.

Retrieves candidates at multiple granularities (chunk, section_summary,
file_summary) and fuses them using RRF.  Section/file matches without a
corresponding chunk match are expanded to top-k chunks from the same
file or section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate

if TYPE_CHECKING:
    from lsm.vectordb.base import BaseVectorDBProvider


def multi_vector_recall(
    query_embedding: List[float],
    db: "BaseVectorDBProvider",
    top_k: int = 12,
    k_section: int = 5,
    k_file: int = 3,
    expansion_k: int = 3,
    filters: Optional[dict] = None,
) -> List[Candidate]:
    """Retrieve at chunk, section, and file granularities with RRF fusion.

    Args:
        query_embedding: Query vector.
        db: Vector database provider.
        top_k: Number of chunk-level candidates.
        k_section: Number of section-level summary candidates.
        k_file: Number of file-level summary candidates.
        expansion_k: Chunks to expand from file/section matches.
        filters: Optional metadata filters.

    Returns:
        Fused candidates sorted by multi-vector RRF score.
    """
    # 1. Chunk-level recall
    chunk_filters = dict(filters or {})
    chunk_filters["node_type"] = "chunk"
    chunk_result = db.query(query_embedding, top_k, filters=chunk_filters)
    chunk_candidates = _results_to_candidates(chunk_result, "chunk")

    # 2. Section-level recall (if section summaries exist)
    section_candidates: List[Candidate] = []
    if k_section > 0:
        section_filters = dict(filters or {})
        section_filters["node_type"] = "section_summary"
        section_result = db.query(query_embedding, k_section, filters=section_filters)
        section_candidates = _results_to_candidates(section_result, "section")

    # 3. File-level recall (if file summaries exist)
    file_candidates: List[Candidate] = []
    if k_file > 0:
        file_filters = dict(filters or {})
        file_filters["node_type"] = "file_summary"
        file_result = db.query(query_embedding, k_file, filters=file_filters)
        file_candidates = _results_to_candidates(file_result, "file")

    # 4. Expand file/section matches to chunks
    chunk_source_paths: Set[str] = {
        c.meta.get("source_path", "") for c in chunk_candidates
    }
    expanded: List[Candidate] = []

    for c in section_candidates + file_candidates:
        source_path = c.meta.get("source_path", "")
        if source_path and source_path not in chunk_source_paths:
            expansion = _expand_to_chunks(
                query_embedding, db, source_path, expansion_k, filters,
            )
            expanded.extend(expansion)
            chunk_source_paths.add(source_path)

    # 5. RRF fusion across granularities
    all_chunk_level = chunk_candidates + expanded
    fused = _multi_rrf_fuse(all_chunk_level, section_candidates, file_candidates)

    return fused


def _results_to_candidates(
    result,
    granularity: str,
) -> List[Candidate]:
    """Convert VectorDBQueryResult to Candidates with granularity tag."""
    candidates = []
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(result.ids, result.documents, result.metadatas, result.distances),
        start=1,
    ):
        score = 1.0 - (dist or 0.0)
        candidates.append(
            Candidate(
                cid=cid,
                text=doc,
                meta=meta,
                distance=dist,
                score_breakdown=ScoreBreakdown(
                    dense_score=score,
                    dense_rank=rank,
                ),
            )
        )
    return candidates


def _expand_to_chunks(
    query_embedding: List[float],
    db: "BaseVectorDBProvider",
    source_path: str,
    k: int,
    base_filters: Optional[dict],
) -> List[Candidate]:
    """Retrieve top-k chunks from a specific file."""
    expand_filters = dict(base_filters or {})
    expand_filters["source_path"] = source_path
    expand_filters["node_type"] = "chunk"
    result = db.query(query_embedding, k, filters=expand_filters)
    return _results_to_candidates(result, "chunk_expanded")


def _multi_rrf_fuse(
    chunk_candidates: List[Candidate],
    section_candidates: List[Candidate],
    file_candidates: List[Candidate],
    chunk_weight: float = 0.6,
    section_weight: float = 0.25,
    file_weight: float = 0.15,
    k: int = 60,
) -> List[Candidate]:
    """RRF fusion across three granularity levels.

    Chunk candidates are the primary results. Section and file candidates
    contribute indirect signal: they boost chunks from the same source.
    """
    # Build rank maps for each granularity
    chunk_ranks: Dict[str, int] = {}
    chunk_by_id: Dict[str, Candidate] = {}
    for rank, c in enumerate(chunk_candidates, start=1):
        if c.cid not in chunk_ranks:
            chunk_ranks[c.cid] = rank
            chunk_by_id[c.cid] = c

    # Section/file matches contribute to source-based boosting
    section_source_ranks: Dict[str, int] = {}
    for rank, c in enumerate(section_candidates, start=1):
        sp = c.meta.get("source_path", "")
        if sp and sp not in section_source_ranks:
            section_source_ranks[sp] = rank

    file_source_ranks: Dict[str, int] = {}
    for rank, c in enumerate(file_candidates, start=1):
        sp = c.meta.get("source_path", "")
        if sp and sp not in file_source_ranks:
            file_source_ranks[sp] = rank

    # Default ranks
    default_chunk_rank = len(chunk_candidates) + 1
    default_section_rank = len(section_candidates) + 1
    default_file_rank = len(file_candidates) + 1

    # Score all chunk-level candidates
    scored: List[tuple] = []
    for cid, candidate in chunk_by_id.items():
        c_rank = chunk_ranks.get(cid, default_chunk_rank)
        source_path = candidate.meta.get("source_path", "")

        s_rank = section_source_ranks.get(source_path, default_section_rank)
        f_rank = file_source_ranks.get(source_path, default_file_rank)

        fused_score = (
            chunk_weight / (k + c_rank)
            + section_weight / (k + s_rank)
            + file_weight / (k + f_rank)
        )

        scored.append((fused_score, cid, candidate))

    scored.sort(key=lambda x: x[0], reverse=True)

    result: List[Candidate] = []
    for fused_score, cid, base in scored:
        breakdown = ScoreBreakdown(
            dense_score=base.score_breakdown.dense_score if base.score_breakdown else None,
            dense_rank=chunk_ranks.get(cid),
            fused_score=fused_score,
        )
        result.append(
            Candidate(
                cid=base.cid,
                text=base.text,
                meta=base.meta,
                distance=base.distance,
                score_breakdown=breakdown,
            )
        )

    return result

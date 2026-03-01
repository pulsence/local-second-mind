"""
Cross-encoder reranking stage.

Uses a local cross-encoder model from sentence-transformers to rerank
dense recall candidates based on query-document relevance.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from lsm.logging import get_logger
from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate

logger = get_logger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker.

    The model is downloaded and loaded on first ``rerank()`` call.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the cross-encoder.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: str = "cpu",
        cache_conn: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_conn = cache_conn
        self._model: Any = None
        self._cache_ready: bool = False

    def _load_model(self) -> Any:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading cross-encoder model %s on %s",
                self.model_name,
                self.device,
            )
            self._model = CrossEncoder(self.model_name, device=self.device)
            return self._model
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder model %s: %s", self.model_name, exc
            )
            return None

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: int,
    ) -> List[Candidate]:
        """Rerank candidates using cross-encoder scoring.

        Each candidate receives a ``ScoreBreakdown`` with ``rerank_score``
        populated by the cross-encoder output.

        Args:
            query: The user's query text.
            candidates: Candidates from dense recall.
            top_k: Maximum number of candidates to return.

        Returns:
            Reranked candidates sorted by cross-encoder score (descending),
            truncated to ``top_k``.
        """
        if not candidates:
            return []

        cached_scores = self._load_cached_scores(query, candidates)
        score_map: Dict[str, float] = dict(cached_scores)

        uncached_candidates = [
            c for c in candidates
            if self._build_cache_key(query, c.cid) not in cached_scores
        ]

        model = self._load_model()
        if model is None and uncached_candidates:
            logger.warning(
                "Cross-encoder unavailable; returning dense recall order"
            )
            return candidates[:top_k]

        if uncached_candidates:
            # Prepare query-document pairs only for uncached entries.
            pairs = [(query, c.text or "") for c in uncached_candidates]

            try:
                scores = model.predict(pairs)
            except Exception as exc:
                logger.warning(
                    "Cross-encoder inference failed: %s; returning dense order", exc
                )
                return candidates[:top_k]

            new_cache_rows: Dict[str, float] = {}
            for cand, score in zip(uncached_candidates, scores):
                key = self._build_cache_key(query, cand.cid)
                score_val = float(score)
                score_map[key] = score_val
                new_cache_rows[key] = score_val
            self._store_scores(new_cache_rows)

        # Pair candidates with scores and sort descending.
        scored = []
        for cand in candidates:
            key = self._build_cache_key(query, cand.cid)
            score = score_map.get(key)
            if score is None:
                # Cache miss without inference should be rare; preserve dense order.
                score = float("-inf")
            scored.append((cand, score))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        result: List[Candidate] = []
        for rank, (cand, score) in enumerate(scored[:top_k], start=1):
            existing = cand.score_breakdown or ScoreBreakdown()
            result.append(
                Candidate(
                    cid=cand.cid,
                    text=cand.text,
                    meta=cand.meta,
                    distance=cand.distance,
                    score_breakdown=ScoreBreakdown(
                        dense_score=existing.dense_score,
                        dense_rank=existing.dense_rank,
                        rerank_score=float(score),
                    ),
                )
            )
        return result

    def _build_cache_key(self, query: str, chunk_id: str) -> str:
        raw = f"{query}\n{chunk_id}\n{self.model_name}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _ensure_cache_table(self) -> None:
        if self.cache_conn is None or self._cache_ready:
            return
        try:
            self.cache_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lsm_reranker_cache (
                    cache_key TEXT PRIMARY KEY,
                    score REAL,
                    created_at TEXT
                )
                """
            )
            self.cache_conn.commit()
            self._cache_ready = True
        except Exception:
            # Cache is best-effort and must never break retrieval.
            self._cache_ready = False

    def _load_cached_scores(
        self,
        query: str,
        candidates: List[Candidate],
    ) -> Dict[str, float]:
        if self.cache_conn is None or not candidates:
            return {}
        self._ensure_cache_table()
        if not self._cache_ready:
            return {}
        keys = [self._build_cache_key(query, c.cid) for c in candidates]
        placeholders = ", ".join("?" for _ in keys)
        try:
            rows = self.cache_conn.execute(
                f"""
                SELECT cache_key, score
                FROM lsm_reranker_cache
                WHERE cache_key IN ({placeholders})
                """,
                keys,
            ).fetchall()
        except Exception:
            return {}
        return {str(row[0]): float(row[1]) for row in rows}

    def _store_scores(self, scores: Dict[str, float]) -> None:
        if self.cache_conn is None or not scores:
            return
        self._ensure_cache_table()
        if not self._cache_ready:
            return
        try:
            self.cache_conn.executemany(
                """
                INSERT OR REPLACE INTO lsm_reranker_cache (cache_key, score, created_at)
                VALUES (?, ?, datetime('now'))
                """,
                [(k, float(v)) for k, v in scores.items()],
            )
            self.cache_conn.commit()
        except Exception:
            return

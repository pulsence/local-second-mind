"""
Cross-encoder reranking stage.

Uses a local cross-encoder model from sentence-transformers to rerank
dense recall candidates based on query-document relevance.
"""

from __future__ import annotations

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
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

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

        model = self._load_model()
        if model is None:
            logger.warning(
                "Cross-encoder unavailable; returning dense recall order"
            )
            return candidates[:top_k]

        # Prepare query-document pairs
        pairs = [(query, c.text or "") for c in candidates]

        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.warning(
                "Cross-encoder inference failed: %s; returning dense order", exc
            )
            return candidates[:top_k]

        # Pair candidates with scores and sort descending
        scored = list(zip(candidates, scores))
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

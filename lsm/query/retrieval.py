"""
Retrieval module for semantic search and filtering.

Handles embedding queries, retrieving candidates from vector DB, and applying filters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lsm.logging import get_logger
from lsm.vectordb.base import BaseVectorDBProvider
from .session import Candidate

logger = get_logger(__name__)


def _import_sentence_transformer():
    """Lazy-import SentenceTransformer to avoid heavy ML imports at module load time."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer, None
    except Exception as exc:
        return None, exc


# -----------------------------
# Constants
# -----------------------------
DEFAULT_COLLECTION = "local_kb"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Embeddings
# -----------------------------
def init_embedder(model_name: str, device: str = "cpu"):
    """
    Initialize the sentence transformer model for embeddings.

    Args:
        model_name: Hugging Face model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        device: Device to run on ("cpu", "cuda", etc.)

    Returns:
        Initialized SentenceTransformer model

    Example:
        >>> embedder = init_embedder("sentence-transformers/all-MiniLM-L6-v2", "cpu")
    """
    logger.debug(f"Initializing embedder: {model_name} on {device}")

    SentenceTransformer, import_error = _import_sentence_transformer()
    if SentenceTransformer is None:
        raise RuntimeError(
            "Failed to import sentence-transformers. "
            "Install compatible dependencies (e.g. `pip install sentence-transformers`) "
            "and ensure torch is available."
        ) from import_error

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:
        # Common local setup issue: config asks for CUDA but installed torch is CPU-only.
        if str(device).lower().startswith("cuda"):
            message = str(exc).lower()
            if "not compiled with cuda" in message or "cuda" in message:
                logger.warning(
                    "Embedder init failed on device '%s' (%s). Falling back to CPU.",
                    device,
                    exc,
                )
                model = SentenceTransformer(model_name, device="cpu")
            else:
                raise
        else:
            raise
    logger.info(f"Embedder ready: {model_name}")
    return model


def embed_text(
    model: "SentenceTransformer",
    text: str,
    batch_size: int = 32,
) -> List[float]:
    """
    Embed a single text query into a vector.

    Args:
        model: SentenceTransformer model
        text: Text to embed
        batch_size: Batch size for encoding

    Returns:
        Embedding vector as list of floats

    Note:
        Uses normalize_embeddings=True to match ingest pipeline.

    Example:
        >>> embedder = init_embedder("sentence-transformers/all-MiniLM-L6-v2")
        >>> vector = embed_text(embedder, "What is Python?")
        >>> len(vector)
        384
    """
    logger.debug(f"Embedding query text ({len(text)} chars)")

    # Match ingest.py: normalize_embeddings=True
    vec = model.encode(
        [text],
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    first = vec[0]
    if hasattr(first, "tolist"):
        return first.tolist()
    return list(first)


# -----------------------------
# Retrieval
# -----------------------------
def retrieve_candidates(
    collection: BaseVectorDBProvider,
    query_embedding: List[float],
    k: int,
    where_filter: Optional[Dict[str, Any]] = None,
) -> List[Candidate]:
    """
    Retrieve candidates from a vector DB using similarity search.

    Args:
        collection: Vector DB provider instance.
        query_embedding: Query vector
        k: Number of candidates to retrieve
        where_filter: Optional metadata filter (e.g. ``{"is_current": True}``)

    Returns:
        List of Candidate objects sorted by distance (most similar first)
    """
    logger.debug(f"Retrieving {k} candidates from vector DB")

    res = collection.query(query_embedding, top_k=k, filters=where_filter)
    ids = res.ids
    docs = res.documents
    metas = res.metadatas
    dists = res.distances

    candidates: List[Candidate] = []
    for idx, cid in enumerate(ids):
        doc = docs[idx] if idx < len(docs) else ""
        meta = metas[idx] if idx < len(metas) else {}
        dist = dists[idx] if idx < len(dists) else None

        candidates.append(
            Candidate(
                cid=str(cid),
                text=(doc or ""),
                meta=(meta or {}),
                distance=dist,
            )
        )

    logger.info(f"Retrieved {len(candidates)} candidates")
    return candidates


# -----------------------------
# Filtering
# -----------------------------
def _normalize_ext(ext: str) -> str:
    """
    Normalize file extension.

    Args:
        ext: File extension (e.g., "md", ".md", ".MD")

    Returns:
        Normalized extension (lowercase with leading dot)

    Example:
        >>> _normalize_ext("md")
        '.md'
        >>> _normalize_ext(".PDF")
        '.pdf'
    """
    ext = (ext or "").strip().lower()
    if not ext:
        return ""
    return ext if ext.startswith(".") else f".{ext}"


def filter_candidates(
    candidates: List[Candidate],
    path_contains: Optional[Any] = None,
    ext_allow: Optional[List[str]] = None,
    ext_deny: Optional[List[str]] = None,
) -> List[Candidate]:
    """
    Apply post-retrieval filters to candidates.

    Filters candidates based on metadata fields:
    - source_path: Must contain specified substring(s)
    - ext: File extension allow/deny lists

    Args:
        candidates: List of candidates to filter
        path_contains: String or list of strings that must appear in source_path
        ext_allow: List of allowed extensions (e.g., [".md", ".pdf"])
        ext_deny: List of denied extensions (e.g., [".tmp"])

    Returns:
        Filtered list of candidates

    Example:
        >>> # Filter for markdown files in "notes" directory
        >>> filtered = filter_candidates(
        ...     candidates,
        ...     path_contains="notes",
        ...     ext_allow=[".md"]
        ... )
    """
    if not candidates:
        return []

    # Normalize path_contains to list of strings
    path_filters: List[str] = []
    if isinstance(path_contains, str) and path_contains.strip():
        path_filters = [path_contains.strip().lower()]
    elif isinstance(path_contains, list):
        path_filters = [str(p).strip().lower() for p in path_contains if str(p).strip()]

    # Normalize extension lists
    allow = {_normalize_ext(e) for e in (ext_allow or []) if _normalize_ext(e)}
    deny = {_normalize_ext(e) for e in (ext_deny or []) if _normalize_ext(e)}

    filtered: List[Candidate] = []
    for candidate in candidates:
        meta = candidate.meta or {}
        source_path = str(meta.get("source_path", "") or "")
        source_path_lower = source_path.lower()

        # Apply path filter
        if path_filters:
            if not any(p in source_path_lower for p in path_filters):
                continue

        # Apply extension filters
        ext = _normalize_ext(str(meta.get("ext", "") or ""))
        if allow and ext and ext not in allow:
            continue
        if deny and ext and ext in deny:
            continue

        filtered.append(candidate)

    logger.info(
        f"Filtered {len(candidates)} → {len(filtered)} candidates "
        f"(path_contains={bool(path_filters)}, "
        f"ext_allow={bool(allow)}, ext_deny={bool(deny)})"
    )

    return filtered


# -----------------------------
# Relevance Metrics
# -----------------------------
def compute_relevance(candidates: List[Candidate]) -> float:
    """
    Compute best relevance score from candidates.

    Converts ChromaDB distance (lower is better) to relevance (higher is better).
    For cosine distance: relevance ≈ 1 - distance

    Args:
        candidates: List of candidates with distance scores

    Returns:
        Best relevance score (clamped to [-1, 1])

    Example:
        >>> relevance = compute_relevance(candidates)
        >>> relevance <= 1.0
        True
    """
    if not candidates:
        return -1.0

    dists = [c.distance for c in candidates if c.distance is not None]
    if not dists:
        return -1.0

    best_dist = min(dists)
    rel = 1.0 - float(best_dist)

    # Clamp to reasonable range
    if rel > 1.0:
        rel = 1.0
    if rel < -1.0:
        rel = -1.0

    return rel

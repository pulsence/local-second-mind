"""
HyDE (Hypothetical Document Embeddings) stage.

Generates hypothetical documents via LLM, embeds them, and uses the
pooled embedding for retrieval instead of the raw query embedding.

Reference: Gao et al. 2023, "Precise Zero-Shot Dense Retrieval without
Relevance Labels."
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from lsm.logging import get_logger
from lsm.providers.base import BaseLLMProvider

logger = get_logger(__name__)

HYDE_INSTRUCTIONS = """You are a helpful assistant. Given the user's question,
write a short paragraph that could be found in a document that directly answers
the question. Do NOT cite sources or say "I don't know." Just write the answer
paragraph as if it were from a reference document.

Write only the paragraph, no preamble or meta-commentary."""


def generate_hypothetical_docs(
    query: str,
    provider: BaseLLMProvider,
    num_samples: int = 2,
    temperature: float = 0.2,
) -> List[str]:
    """Generate hypothetical answer documents for a query.

    Args:
        query: The user's query.
        provider: LLM provider for generation.
        num_samples: Number of hypothetical docs to generate.
        temperature: Sampling temperature.

    Returns:
        List of hypothetical document strings.
    """
    docs: List[str] = []
    for i in range(num_samples):
        try:
            doc = provider.send_message(
                input=query,
                instruction=HYDE_INSTRUCTIONS,
                temperature=temperature,
                max_tokens=300,
            )
            if doc and doc.strip():
                docs.append(doc.strip())
        except Exception as exc:
            logger.warning("HyDE generation %d/%d failed: %s", i + 1, num_samples, exc)
    return docs


def pool_embeddings(
    embeddings: List[Any],
    strategy: str = "mean",
) -> List[float]:
    """Pool multiple embeddings into a single vector.

    Args:
        embeddings: List of embedding arrays/lists.
        strategy: Pooling strategy — ``"mean"`` or ``"max"``.

    Returns:
        Single pooled embedding vector.
    """
    if not embeddings:
        return []

    arr = np.array(embeddings, dtype=np.float32)
    if strategy == "max":
        pooled = np.max(arr, axis=0)
    else:
        pooled = np.mean(arr, axis=0)

    # Normalize to unit vector
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm

    return pooled.tolist()


def hyde_recall(
    query: str,
    provider: BaseLLMProvider,
    embedder: Any,
    db: Any,
    *,
    num_samples: int = 2,
    temperature: float = 0.2,
    pooling: str = "mean",
    top_k: int = 100,
    batch_size: int = 32,
    filters: Optional[dict] = None,
) -> tuple:
    """HyDE retrieval: generate hypothetical docs, embed, and retrieve.

    Args:
        query: User's query.
        provider: LLM provider for hypothetical doc generation.
        embedder: SentenceTransformer model.
        db: Vector database provider.
        num_samples: Number of hypothetical documents.
        temperature: Generation temperature.
        pooling: Embedding pooling strategy.
        top_k: Number of candidates to retrieve.
        batch_size: Embedding batch size.
        filters: Optional metadata filters.

    Returns:
        Tuple of (query_embedding, hypothetical_docs) where query_embedding
        is the pooled HyDE embedding to use for retrieval.
    """
    # Generate hypothetical documents
    hyp_docs = generate_hypothetical_docs(query, provider, num_samples, temperature)

    if not hyp_docs:
        logger.warning("HyDE generated no documents; using raw query embedding")
        # Fall back to raw query embedding
        vec = embedder.encode(
            [query],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vec[0].tolist(), []

    # Embed all hypothetical docs + the original query
    texts_to_embed = hyp_docs + [query]
    embeddings = embedder.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Pool all embeddings (hypothetical docs + original query)
    pooled = pool_embeddings(embeddings.tolist(), strategy=pooling)

    return pooled, hyp_docs

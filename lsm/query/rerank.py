"""
Reranking module for improving candidate quality.

Provides lexical scoring, deduplication, and diversity enforcement.
"""

from __future__ import annotations

import re
from typing import Any, List, Dict

from lsm.logging import get_logger
from lsm.providers.base import BaseLLMProvider
from lsm.query.stages.llm_rerank import llm_rerank
from .session import Candidate

logger = get_logger(__name__)


# -----------------------------
# Stopwords & Tokenization
# -----------------------------
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "into",
    "about", "over", "under", "after", "before", "between", "through", "during",
    "without", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "done", "doing", "have", "has", "had", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their",
    "our", "this", "that", "these", "those", "there", "here", "not", "no", "yes",
    "so", "than", "too", "very", "can", "could", "should", "would", "may", "might",
    "must", "will", "just", "what", "how"
}


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for lexical scoring.

    Extracts words and numbers, converts to lowercase, removes stopwords.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens (lowercased, stopwords removed)

    Example:
        >>> tokenize("What is Python programming?")
        ['python', 'programming']
    """
    raw = (text or "").lower()
    # Normalize numeric separators so "3.9" and "3-9" become "39".
    raw = re.sub(r"(?<=\d)[\.\-](?=\d)", "", raw)
    # Extract words and numbers
    tokens = re.findall(r"[a-zA-Z0-9']+", raw)

    # Filter out short tokens and stopwords
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


# -----------------------------
# Lexical Scoring
# -----------------------------
def compute_lexical_score(question: str, passage: str) -> float:
    """
    Compute lexical relevance score between question and passage.

    Uses:
    - Token overlap (weighted toward question tokens)
    - Phrase bonus for contiguous query substrings in passage

    Args:
        question: User's question
        passage: Candidate passage text

    Returns:
        Lexical score (0.0 to ~1.2, higher is better)

    Example:
        >>> score = compute_lexical_score(
        ...     "What is Python programming?",
        ...     "Python is a high-level programming language."
        ... )
        >>> score > 0.5  # Good lexical match
        True
    """
    q_tokens = tokenize(question)
    p_tokens = tokenize(passage)

    if not q_tokens or not p_tokens:
        return 0.0

    q_set = set(q_tokens)
    p_set = set(p_tokens)

    # Base score: token overlap
    overlap = len(q_set & p_set)
    base = overlap / max(1, len(q_set))  # 0..1

    # Phrase bonus: reward if meaningful phrases appear verbatim
    q_norm = " ".join(q_tokens)
    p_norm = " ".join(p_tokens)

    bonus = 0.0
    for n in (4, 3, 2):  # Try longer phrases first, then bigrams
        if len(q_tokens) >= n:
            # Sliding window, but cap to avoid expensive scans
            max_windows = min(10, len(q_tokens) - n + 1)
            for i in range(max_windows):
                phrase = " ".join(q_tokens[i : i + n])
                if phrase and phrase in p_norm:
                    bonus += 0.10
                    break

    return base + bonus


def rerank_lexical(question: str, candidates: List[Candidate]) -> List[Candidate]:
    """
    Rerank candidates by lexical relevance to question.

    Args:
        question: User's question
        candidates: List of candidates to rerank

    Returns:
        Candidates sorted by lexical score (highest first)

    Example:
        >>> reranked = rerank_lexical("Python programming", candidates)
    """
    if not candidates:
        return []

    logger.debug(f"Lexical reranking {len(candidates)} candidates")

    scored = []
    for c in candidates:
        score = compute_lexical_score(question, c.text or "")
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    logger.info(
        f"Lexical reranked {len(candidates)} candidates "
        f"(top score: {scored[0][0]:.3f})"
    )

    return [c for _, c in scored]


# -----------------------------
# Deduplication
# -----------------------------
def deduplicate(
    candidates: List[Candidate],
    max_chars: int = 2000,
) -> List[Candidate]:
    """
    Remove duplicate candidates by normalized text hash.

    Keeps first occurrence (which will be higher-ranked if called after reranking).

    Args:
        candidates: List of candidates
        max_chars: Maximum characters to consider for hashing

    Returns:
        Deduplicated list of candidates

    Example:
        >>> unique_candidates = deduplicate(candidates)
    """
    if not candidates:
        return []

    logger.debug(f"Deduplicating {len(candidates)} candidates")

    seen = set()
    unique: List[Candidate] = []

    for c in candidates:
        text = (c.text or "").strip()
        if not text:
            continue

        # Normalize whitespace and case, cap length for efficient hashing
        norm = re.sub(r"\s+", " ", text).strip().lower()
        norm = norm[:max_chars]

        h = hash(norm)
        if h in seen:
            continue

        seen.add(h)
        unique.append(c)

    logger.info(f"Deduplicated {len(candidates)} → {len(unique)} candidates")
    return unique


# -----------------------------
# Diversity Enforcement
# -----------------------------
def enforce_diversity(
    candidates: List[Candidate],
    max_per_file: int = 2,
) -> List[Candidate]:
    """
    Enforce diversity by limiting chunks per source file.

    Keeps order and fills as many candidates as possible while respecting the limit.

    Args:
        candidates: List of candidates (in rank order)
        max_per_file: Maximum chunks allowed from same source file

    Returns:
        Filtered candidates with diversity constraint

    Example:
        >>> diverse = enforce_diversity(candidates, max_per_file=2)
    """
    if max_per_file <= 0 or not candidates:
        return candidates

    logger.debug(
        f"Enforcing diversity: max {max_per_file} chunks per file "
        f"across {len(candidates)} candidates"
    )

    counts: Dict[str, int] = {}
    diverse: List[Candidate] = []

    for c in candidates:
        source_path = str((c.meta or {}).get("source_path", "unknown"))
        current_count = counts.get(source_path, 0)

        if current_count >= max_per_file:
            continue

        counts[source_path] = current_count + 1
        diverse.append(c)

    logger.info(
        f"Enforced diversity: {len(candidates)} → {len(diverse)} candidates "
        f"({len(counts)} unique files)"
    )

    return diverse


# -----------------------------
# Combined Reranking Pipeline
# -----------------------------
def apply_local_reranking(
    question: str,
    candidates: List[Candidate],
    max_per_file: int = 2,
    local_pool: int = 36,
) -> List[Candidate]:
    """
    Apply full local reranking pipeline.

    Steps:
    1. Deduplicate to remove overlaps
    2. Lexical rerank to surface keyword matches
    3. Limit to local_pool size
    4. Enforce per-file diversity

    Args:
        question: User's question
        candidates: List of candidates (from vector search)
        max_per_file: Maximum chunks per source file
        local_pool: Maximum candidates to keep before diversity

    Returns:
        Reranked and filtered candidates

    Example:
        >>> reranked = apply_local_reranking(
        ...     "Python programming",
        ...     candidates,
        ...     max_per_file=2,
        ...     local_pool=36
        ... )
    """
    logger.info(f"Starting local reranking pipeline with {len(candidates)} candidates")

    # Step 1: Deduplicate early to remove overlap noise
    local = deduplicate(candidates)

    # Step 2: Lexical rerank to surface proper-noun / keyword matches
    local = rerank_lexical(question, local)

    # Step 3: Keep manageable pool before diversity
    local = local[: min(local_pool, len(local))]

    # Step 4: Enforce per-file diversity
    local = enforce_diversity(local, max_per_file=max_per_file)

    logger.info(f"Local reranking complete: {len(local)} candidates")

    return local


def llm_rerank_candidates(
    question: str,
    candidates: List[Dict[str, Any]],
    k: int,
    provider: BaseLLMProvider,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate dictionaries through a provider transport call.

    Falls back to local candidate order when the provider response is invalid.
    """
    if not candidates:
        return []

    return llm_rerank(candidates, question, provider, k=k)

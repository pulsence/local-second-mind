"""
MinHash near-duplicate detection stage.

Uses token-level MinHash signatures to detect and suppress near-duplicate
chunks in retrieval results.
"""

from __future__ import annotations

import hashlib
import struct
from typing import List, Set, Tuple

from lsm.logging import get_logger
from lsm.query.session import Candidate

logger = get_logger(__name__)

# Large prime for hash mixing
_HASH_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _tokenize(text: str) -> List[str]:
    """Split text into word-level tokens (lowercased)."""
    return text.lower().split()


def _shingle(tokens: List[str], k: int = 3) -> Set[str]:
    """Generate character-level k-shingles from tokens."""
    text = " ".join(tokens)
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def compute_minhash(text: str, num_perm: int = 128) -> Tuple[int, ...]:
    """Compute a MinHash signature for a text.

    Args:
        text: Input text.
        num_perm: Number of permutations (signature length).

    Returns:
        Tuple of ``num_perm`` minimum hash values.
    """
    shingles = _shingle(_tokenize(text))
    if not shingles:
        return tuple([_MAX_HASH] * num_perm)

    # Initialize with max values
    sig = [_MAX_HASH] * num_perm

    for shingle in shingles:
        h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16) & _MAX_HASH
        for i in range(num_perm):
            # Simple hash family: h_i(x) = (a_i * h + b_i) mod prime
            a = (i + 1) * 0x5BD1E995
            b = (i + 1) * 0x1B873593
            val = ((a * h + b) % _HASH_PRIME) & _MAX_HASH
            if val < sig[i]:
                sig[i] = val

    return tuple(sig)


def jaccard_estimate(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures.

    Args:
        sig_a: First MinHash signature.
        sig_b: Second MinHash signature.

    Returns:
        Estimated Jaccard similarity in [0, 1].
    """
    if len(sig_a) != len(sig_b):
        return 0.0
    if not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def are_near_duplicates(
    sig_a: Tuple[int, ...],
    sig_b: Tuple[int, ...],
    threshold: float = 0.8,
) -> bool:
    """Check if two texts are near-duplicates based on MinHash signatures.

    Args:
        sig_a: MinHash signature of first text.
        sig_b: MinHash signature of second text.
        threshold: Jaccard similarity threshold for duplicate detection.

    Returns:
        True if estimated Jaccard similarity >= threshold.
    """
    return jaccard_estimate(sig_a, sig_b) >= threshold


def deduplicate_candidates(
    candidates: List[Candidate],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> List[Candidate]:
    """Remove near-duplicate candidates based on MinHash similarity.

    Preserves the first occurrence of each near-duplicate group (i.e.,
    the highest-ranked candidate is kept).

    Args:
        candidates: Input candidates in ranked order.
        threshold: Jaccard similarity threshold.
        num_perm: Number of MinHash permutations.

    Returns:
        Deduplicated candidate list preserving original order.
    """
    if not candidates:
        return []

    # Compute signatures
    sigs = [compute_minhash(c.text or "", num_perm=num_perm) for c in candidates]

    kept: List[Candidate] = []
    kept_sigs: List[Tuple[int, ...]] = []

    for cand, sig in zip(candidates, sigs):
        is_dup = False
        for kept_sig in kept_sigs:
            if are_near_duplicates(sig, kept_sig, threshold):
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
            kept_sigs.append(sig)

    removed = len(candidates) - len(kept)
    if removed > 0:
        logger.debug("MinHash dedup removed %d near-duplicates", removed)

    return kept

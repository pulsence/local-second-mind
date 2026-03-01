"""
Simhash computation for near-duplicate detection at ingest time.

Produces a 64-bit integer hash for each chunk that can be stored in the
``simhash`` column and used for fast duplicate lookups.
"""

from __future__ import annotations

import hashlib


def compute_simhash(text: str) -> int:
    """Compute a 64-bit SimHash for *text*.

    The algorithm:
    1. Tokenize into whitespace-separated words.
    2. Hash each token to a 64-bit integer.
    3. Build a weighted bit-vector (set bit → +1, unset → −1).
    4. Collapse to a 64-bit fingerprint (positive → 1, else → 0).

    Returns:
        Non-negative 64-bit integer.
    """
    bits = 64
    vector = [0] * bits
    tokens = text.lower().split()
    if not tokens:
        return 0

    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) & ((1 << bits) - 1)
        for i in range(bits):
            if h & (1 << i):
                vector[i] += 1
            else:
                vector[i] -= 1

    fingerprint = 0
    for i in range(bits):
        if vector[i] > 0:
            fingerprint |= 1 << i

    # Convert to signed 64-bit so it fits in SQLite INTEGER
    if fingerprint >= (1 << 63):
        fingerprint -= 1 << 64
    return fingerprint

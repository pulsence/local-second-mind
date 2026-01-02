from __future__ import annotations

from typing import List

from lsm.ingest.utils import normalize_whitespace
from lsm.ingest.config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

# -----------------------------
# Chunk (simple, minimal)
# -----------------------------
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """
    Minimal chunker:
    - Normalize whitespace
    - Split into chunks by char count with overlap
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)

    return chunks
from __future__ import annotations

from typing import List, Dict, Any, Tuple

from lsm.config.models.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from lsm.ingest.utils import normalize_whitespace


# -----------------------------
# Chunk (with position tracking)
# -----------------------------
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    track_positions: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Chunk text into overlapping segments with position tracking.

    Args:
        text: Input text to chunk
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between consecutive chunks in characters
        track_positions: If True, return position information for each chunk

    Returns:
        Tuple of (chunks, positions) where positions is a list of dicts with start/end offsets
    """
    # Normalize whitespace
    normalized_text = normalize_whitespace(text)

    if not normalized_text:
        return [], []

    chunks: List[str] = []
    positions: List[Dict[str, Any]] = []

    i = 0
    n = len(normalized_text)
    chunk_index = 0

    while i < n:
        j = min(i + chunk_size, n)
        chunk = normalized_text[i:j].strip()

        if chunk:
            chunks.append(chunk)

            if track_positions:
                # Calculate actual start/end after stripping
                # Find where stripped chunk starts in the original slice
                orig_slice = normalized_text[i:j]
                start_offset = i + (len(orig_slice) - len(orig_slice.lstrip()))
                end_offset = start_offset + len(chunk)

                positions.append({
                    "chunk_index": chunk_index,
                    "start_char": start_offset,
                    "end_char": end_offset,
                    "length": len(chunk),
                })

            chunk_index += 1

        if j == n:
            break

        step = chunk_size - overlap
        if step <= 0:
            # Avoid infinite loop when overlap >= chunk_size
            step = max(1, chunk_size)
        i += step

    return chunks, positions

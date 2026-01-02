from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# -----------------------------
# Threading Classes
# -----------------------------
@dataclass
class ParseResult:
    source_path: str
    fp: Path
    mtime_ns: int
    size: int
    file_hash: str
    chunks: List[str]
    ext: str
    had_prev: bool  # whether manifest had an entry (controls delete(where))
    ok: bool
    err: Optional[str] = None

@dataclass
class WriteJob:
    # One job corresponds to one file (so writer can delete/manifest-update per file)
    source_path: str
    fp: Path
    mtime_ns: int
    size: int
    file_hash: str
    ext: str
    chunks: List[str]
    embeddings: List[List[float]]
    had_prev: bool


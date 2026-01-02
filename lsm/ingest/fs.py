from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# -----------------------------
# Crawl
# -----------------------------
def iter_files(
    roots: List[Path],
    exts: set[str],
    exclude_dirs: set[str],
) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Skip excluded dirs by checking any parent folder name
            if any(parent.name in exclude_dirs for parent in p.parents):
                continue
            if p.suffix.lower() in exts:
                yield p
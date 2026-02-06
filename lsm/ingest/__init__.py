"""
Ingest subsystem for Local Second Mind.

Responsibilities:
- File discovery
- Parsing and chunking
- Embedding
- Persistence to Chroma
- Manifest tracking

Public API:
- ingest(config): run the ingest pipeline
"""

from __future__ import annotations

from typing import Any


def ingest(*args: Any, **kwargs: Any):
    """Run the ingest pipeline with lazy import to avoid heavy import-time dependencies."""
    from lsm.ingest.pipeline import ingest as _ingest

    return _ingest(*args, **kwargs)


__all__ = ["ingest"]

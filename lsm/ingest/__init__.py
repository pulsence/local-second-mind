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

# Public entrypoint (stable)
from lsm.ingest.pipeline import ingest

__all__ = [
    "ingest",
]

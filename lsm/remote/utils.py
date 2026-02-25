"""
Utility helpers for remote provider normalization.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)


def normalize_doi(value: str | None) -> str | None:
    """Normalize DOI values to bare DOI strings."""
    if not value:
        return None
    doi = str(value).strip()
    if not doi:
        return None
    lower = doi.lower()
    if lower.startswith("doi:"):
        doi = doi[4:].strip()
        lower = doi.lower()
    for prefix in _DOI_PREFIXES:
        if lower.startswith(prefix):
            doi = doi[len(prefix):].strip()
            break
    return doi or None


def sanitize_filename(value: str, *, fallback: str = "item") -> str:
    """Create a filesystem-safe filename component."""
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text.strip("._") or fallback


def filename_from_url(url: str, *, fallback: str = "item") -> str:
    """Derive a filename from a URL path."""
    try:
        parsed = urlparse(url)
        name = parsed.path.rsplit("/", 1)[-1]
    except Exception:
        name = ""
    if not name:
        return sanitize_filename(fallback, fallback=fallback)
    return sanitize_filename(name, fallback=fallback)

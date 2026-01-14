"""
Citation export utilities (BibTeX and Zotero JSON).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Citation:
    key: str
    title: str
    author: Optional[str]
    year: Optional[str]
    date: Optional[str]
    source_path: str
    entry_type: str
    item_type: str


def _slugify(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "source"


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _extract_date(meta: Dict[str, Optional[str]], source_path: str) -> Optional[datetime]:
    ingested_at = _parse_datetime(meta.get("ingested_at"))
    if ingested_at:
        return ingested_at

    mtime_ns = meta.get("mtime_ns")
    if isinstance(mtime_ns, (int, float)) and mtime_ns:
        try:
            return datetime.fromtimestamp(mtime_ns / 1_000_000_000)
        except Exception:
            pass

    try:
        if source_path and os.path.exists(source_path):
            return datetime.fromtimestamp(os.path.getmtime(source_path))
    except Exception:
        pass

    return None


def _infer_types(source_path: str) -> Dict[str, str]:
    ext = Path(source_path).suffix.lower()
    if source_path.startswith("http"):
        return {"entry_type": "misc", "item_type": "webpage"}
    if ext in {".pdf"}:
        return {"entry_type": "article", "item_type": "journalArticle"}
    if ext in {".html", ".htm"}:
        return {"entry_type": "misc", "item_type": "webpage"}
    if ext in {".md", ".txt"}:
        return {"entry_type": "misc", "item_type": "document"}
    return {"entry_type": "misc", "item_type": "document"}


def sources_to_citations(sources: List[Dict[str, Optional[str]]]) -> List[Citation]:
    citations: List[Citation] = []
    for source in sources:
        source_path = source.get("source_path") or "unknown"
        title = source.get("title") or source.get("source_name") or Path(source_path).name
        author = source.get("author")
        date = _extract_date(source, source_path)
        year = str(date.year) if date else None
        types = _infer_types(source_path)
        key = _slugify(f"{author or title}-{year or 'n.d.'}")
        citations.append(
            Citation(
                key=key,
                title=title,
                author=author,
                year=year,
                date=date.isoformat() if date else None,
                source_path=source_path,
                entry_type=types["entry_type"],
                item_type=types["item_type"],
            )
        )
    return citations


def citations_to_bibtex(citations: List[Citation]) -> str:
    entries: List[str] = []
    for c in citations:
        lines = [f"@{c.entry_type}{{{c.key},"]
        if c.author:
            lines.append(f"  author = {{{c.author}}},")
        if c.title:
            lines.append(f"  title = {{{c.title}}},")
        if c.year:
            lines.append(f"  year = {{{c.year}}},")
        if c.source_path:
            lines.append(f"  howpublished = {{{c.source_path}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries)


def citations_to_zotero(citations: List[Citation]) -> str:
    items = []
    for c in citations:
        item = {
            "itemType": c.item_type,
            "title": c.title,
            "creators": [],
            "date": c.date or c.year or "",
            "url": c.source_path if c.source_path.startswith("http") else "",
            "filePath": c.source_path if not c.source_path.startswith("http") else "",
        }
        if c.author:
            item["creators"].append({"creatorType": "author", "name": c.author})
        items.append(item)
    return json.dumps(items, indent=2)


def export_citations_from_sources(
    sources: List[Dict[str, Optional[str]]],
    fmt: str = "bibtex",
    output_path: Optional[Path] = None,
) -> Path:
    citations = sources_to_citations(sources)
    if fmt == "bibtex":
        content = citations_to_bibtex(citations)
        suffix = ".bib"
    else:
        content = citations_to_zotero(citations)
        suffix = ".json"

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"citations-{timestamp}{suffix}")

    output_path.write_text(content, encoding="utf-8")
    return output_path


def parse_note_sources(note_text: str) -> List[Dict[str, Optional[str]]]:
    sources: List[Dict[str, Optional[str]]] = []
    current: Dict[str, Optional[str]] = {}

    def _flush() -> None:
        nonlocal current
        if current.get("source_path"):
            sources.append(current)
        current = {}

    for line in note_text.splitlines():
        line = line.strip()
        if line.startswith("### Source"):
            _flush()
            continue
        if line.startswith("**Title:**"):
            current["title"] = line.replace("**Title:**", "").strip()
        elif line.startswith("**Author:**"):
            current["author"] = line.replace("**Author:**", "").strip()
        elif line.startswith("**Path:**"):
            match = re.search(r"`([^`]+)`", line)
            if match:
                current["source_path"] = match.group(1)
                current["source_name"] = Path(match.group(1)).name
    _flush()
    return sources


def export_citations_from_note(note_path: Path, fmt: str = "bibtex") -> Path:
    content = note_path.read_text(encoding="utf-8")
    sources = parse_note_sources(content)
    if not sources:
        raise ValueError("No sources found in note.")
    output_path = note_path.with_suffix(".bib" if fmt == "bibtex" else ".json")
    return export_citations_from_sources(sources, fmt=fmt, output_path=output_path)

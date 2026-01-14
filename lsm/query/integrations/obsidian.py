"""
Obsidian integration helpers.
"""

from __future__ import annotations

from pathlib import Path


def format_wikilink(name: str) -> str:
    return f"[[{name}]]"


def normalize_tag(tag: str) -> str:
    tag = tag.strip().lower().replace(" ", "-")
    return f"#{tag}"


def export_note(content: str, vault_path: Path, filename: str) -> Path:
    vault_path.mkdir(parents=True, exist_ok=True)
    note_path = vault_path / filename
    note_path.write_text(content, encoding="utf-8")
    return note_path

"""
Notes writing system for query sessions.

Provides automatic Markdown note generation for query sessions,
saving the query, sources, and answer for future reference.
"""

from __future__ import annotations

import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from lsm.gui.shell.logging import get_logger
import tempfile
import subprocess
import os

logger = get_logger(__name__)


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to a filename-safe slug.

    Args:
        text: Text to slugify
        max_length: Maximum slug length

    Returns:
        Slugified string
    """
    # Convert to lowercase
    text = text.lower()

    # Replace non-alphanumeric characters with hyphens
    text = re.sub(r'[^a-z0-9]+', '-', text)

    # Remove leading/trailing hyphens
    text = text.strip('-')

    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length].rstrip('-')

    return text


def generate_timestamp() -> str:
    """
    Generate a timestamp for filename.

    Returns:
        Timestamp string in format: YYYYMMDD-HHMMSS
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_note_filename(
    query: str,
    format: str = "timestamp",
) -> str:
    """
    Generate a filename for a note.

    Args:
        query: Query string
        format: Filename format ('timestamp', 'query_slug', 'incremental')

    Returns:
        Filename (without directory path)
    """
    if format == "timestamp":
        # Format: YYYYMMDD-HHMMSS.md
        return f"{generate_timestamp()}.md"

    elif format == "query_slug":
        # Format: query-slug-YYYYMMDD.md
        slug = slugify(query)
        date = datetime.now().strftime("%Y%m%d")
        return f"{slug}-{date}.md"

    else:
        # Default to timestamp
        logger.warning(f"Unknown filename format '{format}', using timestamp")
        return f"{generate_timestamp()}.md"


def _deserialize_tags(tags_value: Any) -> List[str]:
    if not tags_value:
        return []
    if isinstance(tags_value, list):
        return [t for t in tags_value if isinstance(t, str)]
    if isinstance(tags_value, str):
        try:
            data = json.loads(tags_value)
            if isinstance(data, list):
                return [t for t in data if isinstance(t, str)]
        except json.JSONDecodeError:
            pass
    return []


def format_local_sources(
    sources: List[Dict[str, Any]],
    use_wikilinks: bool = False,
    include_tags: bool = False,
) -> str:
    """
    Format local knowledge base sources as Markdown.

    Args:
        sources: List of source dicts with 'text', 'meta', 'distance' fields

    Returns:
        Markdown formatted sources section
    """
    if not sources:
        return "No local sources found.\n"

    lines = []

    for i, source in enumerate(sources, 1):
        meta = source.get("meta", {})
        text = source.get("text", "")
        distance = source.get("distance", 0.0)
        relevance = 1.0 - distance

        # Extract metadata
        source_path = meta.get("source_path", "unknown")
        chunk_idx = meta.get("chunk_index", 0)
        title = meta.get("title", "")
        author = meta.get("author", "")

        # Format source header
        display_name = Path(source_path).stem or Path(source_path).name
        header_name = f"[[{display_name}]]" if use_wikilinks else Path(source_path).name
        lines.append(f"### Source {i}: {header_name}")
        lines.append(f"**Relevance:** {relevance:.2f} | **Chunk:** {chunk_idx}")

        if title:
            lines.append(f"**Title:** {title}")
        if author:
            lines.append(f"**Author:** {author}")

        lines.append(f"**Path:** `{source_path}`")
        lines.append("")  # Blank line

        if include_tags:
            ai_tags = _deserialize_tags(meta.get("ai_tags"))
            user_tags = _deserialize_tags(meta.get("user_tags"))
            tags = sorted(set(ai_tags + user_tags))
            if tags:
                lines.append(f"**Tags:** {' '.join('#' + t for t in tags)}")
                lines.append("")

        # Add text snippet
        lines.append("**Content:**")
        lines.append("```")
        # Limit snippet length
        snippet = text[:500] + "..." if len(text) > 500 else text
        lines.append(snippet)
        lines.append("```")
        lines.append("")  # Blank line

    return "\n".join(lines)


def format_remote_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format remote sources as Markdown.

    Args:
        sources: List of RemoteResult dicts

    Returns:
        Markdown formatted remote sources section
    """
    if not sources:
        return "No remote sources used.\n"

    lines = []

    for i, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        snippet = source.get("snippet", "")
        score = source.get("score", 0.0)

        lines.append(f"### Remote Source {i}: {title}")
        lines.append(f"**Score:** {score:.2f}")
        lines.append(f"**URL:** {url}")
        lines.append("")

        if snippet:
            lines.append("**Summary:**")
            lines.append(f"> {snippet}")
            lines.append("")

    return "\n".join(lines)


def generate_note_content(
    query: str,
    answer: str,
    local_sources: Optional[List[Dict[str, Any]]] = None,
    remote_sources: Optional[List[Dict[str, Any]]] = None,
    mode: str = "grounded",
    use_wikilinks: bool = False,
    include_backlinks: bool = False,
    include_tags: bool = False,
) -> str:
    """
    Generate note content as Markdown string.

    Args:
        query: Query string
        answer: Answer text from LLM
        local_sources: Optional list of local sources used
        remote_sources: Optional list of remote sources used
        mode: Query mode used

    Returns:
        Markdown content string
    """
    lines = []

    # Header
    lines.append("# Query Session")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Mode:** {mode}")
    lines.append("")

    if include_tags and local_sources:
        all_tags: List[str] = []
        for source in local_sources:
            meta = source.get("meta", {})
            all_tags.extend(_deserialize_tags(meta.get("ai_tags")))
            all_tags.extend(_deserialize_tags(meta.get("user_tags")))
        all_tags = sorted(set(t for t in all_tags if t))
        if all_tags:
            lines.append(f"**Tags:** {' '.join('#' + t for t in all_tags)}")
            lines.append("")

    # Query
    lines.append("## Query")
    lines.append("")
    lines.append(query)
    lines.append("")

    # Local Sources
    lines.append("## Local Sources")
    lines.append("")
    if local_sources:
        lines.append(
            format_local_sources(
                local_sources,
                use_wikilinks=use_wikilinks,
                include_tags=include_tags,
            )
        )
    else:
        lines.append("No local sources used.\n")
    lines.append("")

    # Remote Sources (if any)
    if remote_sources:
        lines.append("## Remote Sources")
        lines.append("")
        lines.append(format_remote_sources(remote_sources))
        lines.append("")

    # Answer
    lines.append("## Answer")
    lines.append("")
    lines.append(answer)
    lines.append("")

    if include_backlinks and local_sources:
        backlinks = []
        for source in local_sources:
            meta = source.get("meta", {})
            path = meta.get("source_path", "")
            name = Path(path).stem or Path(path).name
            if name:
                backlinks.append(name)
        backlinks = sorted(set(backlinks))
        if backlinks:
            lines.append("## Backlinks")
            lines.append("")
            for name in backlinks:
                if use_wikilinks:
                    lines.append(f"- [[{name}]]")
                else:
                    lines.append(f"- {name}")
            lines.append("")

    return "\n".join(lines)


def edit_note_in_editor(content: str) -> Optional[str]:
    """
    Open note content in user's default editor for editing.

    Args:
        content: Initial note content

    Returns:
        Edited content, or None if user cancelled

    Raises:
        OSError: If editor cannot be launched
    """
    # Determine editor
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")

    if not editor:
        # Platform-specific defaults
        import platform
        system = platform.system()
        if system == "Windows":
            editor = "notepad"
        elif system == "Darwin":  # macOS
            editor = "open -e"  # TextEdit
        else:  # Linux/Unix
            editor = "nano"

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
        tmp.write(content)

    try:
        # Open editor
        subprocess.run(f"{editor} {tmp_path}", shell=True, check=True)

        # Read edited content
        edited_content = Path(tmp_path).read_text(encoding='utf-8')

        return edited_content

    except Exception as e:
        logger.error(f"Failed to open editor: {e}")
        raise OSError(f"Could not open editor: {e}")

    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


def write_note(
    notes_dir: Path,
    query: str,
    answer: str,
    local_sources: Optional[List[Dict[str, Any]]] = None,
    remote_sources: Optional[List[Dict[str, Any]]] = None,
    mode: str = "grounded",
    filename_format: str = "timestamp",
) -> Path:
    """
    Write a query session note to Markdown.

    Args:
        notes_dir: Directory to write notes to
        query: Query string
        answer: Answer text from LLM
        local_sources: Optional list of local sources used
        remote_sources: Optional list of remote sources used
        mode: Query mode used
        filename_format: Filename format ('timestamp', 'query_slug')

    Returns:
        Path to written note file

    Raises:
        OSError: If note cannot be written
    """
    # Ensure notes directory exists
    notes_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = get_note_filename(query, format=filename_format)
    note_path = notes_dir / filename

    # Generate content
    content = generate_note_content(query, answer, local_sources, remote_sources, mode)

    try:
        note_path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote note to: {note_path}")
        return note_path

    except Exception as e:
        logger.error(f"Failed to write note: {e}")
        raise OSError(f"Could not write note to {note_path}: {e}")

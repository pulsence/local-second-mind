"""
Session state management for query REPL.

Manages user session state, filters, and query artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from lsm.paths import get_chats_folder

if TYPE_CHECKING:
    from .cost_tracking import CostTracker


@dataclass
class Candidate:
    """
    A retrieved document chunk candidate.

    Represents a single chunk from the vector database with metadata.
    """

    cid: str
    """Chunk ID from ChromaDB."""

    text: str
    """Chunk text content."""

    meta: Dict[str, Any]
    """Metadata from ChromaDB (source_path, chunk_index, etc.)."""

    distance: Optional[float] = None
    """Distance score from vector search (lower is better)."""

    @property
    def source_path(self) -> str:
        """Get source file path from metadata."""
        return self.meta.get("source_path", "unknown")

    @property
    def source_name(self) -> str:
        """Get source file name from metadata."""
        return self.meta.get("source_name", "unknown")

    @property
    def chunk_index(self) -> int:
        """Get chunk index from metadata."""
        return self.meta.get("chunk_index", 0)

    @property
    def ext(self) -> str:
        """Get file extension from metadata."""
        return self.meta.get("ext", "")

    @property
    def heading_path(self) -> List[str]:
        """Get normalized heading path from metadata."""
        raw_value = self.meta.get("heading_path")
        if raw_value is None:
            return []

        if isinstance(raw_value, list):
            return [str(item) for item in raw_value if str(item).strip()]

        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except Exception:
                return [text]
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
            return [text]

        return []

    @property
    def relevance(self) -> float:
        """
        Convert distance to relevance score.

        Returns:
            Relevance score (1.0 - distance), clamped to [-1, 1]
        """
        if self.distance is None:
            return 0.0

        rel = 1.0 - self.distance

        # Clamp to reasonable range
        if rel > 1.0:
            return 1.0
        if rel < -1.0:
            return -1.0

        return rel

    def format(self, label: str = "", expanded: bool = False) -> str:
        """
        Format this candidate for display.

        Args:
            label: Citation label (e.g., "S1")
            expanded: If True, show full text without truncation

        Returns:
            Formatted chunk string
        """
        chunk_index = self.chunk_index
        distance = self.distance

        lines = []
        if expanded:
            lines.append(f"\n{label} — {self.source_path}")
            lines.append(f"chunk_index={chunk_index}, distance={distance}")
            lines.append("=" * 80)
            lines.append((self.text or "").strip())
            lines.append("=" * 80)
            lines.append("")
        else:
            lines.append(f"\n{label} — {self.source_path} (chunk_index={chunk_index}, distance={distance})")
            lines.append("-" * 80)
            lines.append((self.text or "").strip())
            lines.append("-" * 80)
            lines.append("")

        return "\n".join(lines)


@dataclass
class SessionState:
    """
    REPL session state.

    Tracks user filters, model selection, and last query artifacts.
    """

    # Session-level filter overrides
    path_contains: Optional[Any] = None
    """Filter for files containing this string in path."""

    ext_allow: Optional[List[str]] = None
    """Only include files with these extensions."""

    ext_deny: Optional[List[str]] = None
    """Exclude files with these extensions."""

    # Model override (session)
    model: str = "gpt-5.2"
    """LLM model to use for this session."""

    available_models: Optional[List[str]] = None
    """List of available models (populated on demand)."""

    # Last-turn artifacts (for debugging and inspection)
    last_question: Optional[str] = None
    """Last question asked by user."""

    last_all_candidates: List[Candidate] = None
    """All candidates retrieved from vector DB."""

    last_filtered_candidates: List[Candidate] = None
    """Candidates after applying filters."""

    last_chosen: List[Candidate] = None
    """Final candidates used for answer generation."""

    last_label_to_candidate: Dict[str, Candidate] = None
    """Mapping from source labels ([S1], etc.) to candidates."""

    last_debug: Dict[str, Any] = None
    """Debug information from last query."""

    last_answer: Optional[str] = None
    """Last answer generated (for note saving)."""

    last_remote_sources: Optional[List[Dict[str, Any]]] = None
    """Last remote sources fetched (for note saving)."""

    last_local_sources_for_notes: Optional[List[Dict[str, Any]]] = None
    """Last local sources in dict format (for note saving)."""

    pinned_chunks: List[str] = None
    """Chunk IDs pinned for forced inclusion in context."""

    context_documents: Optional[List[str]] = None
    """Source document paths selected as context anchors."""

    context_chunks: Optional[List[str]] = None
    """Chunk IDs selected as context anchors."""

    conversation_history: List[Dict[str, str]] = None
    """Chat-mode conversation turns (role/content)."""

    llm_server_cache_ids: Dict[str, str] = None
    """Provider/model cache chain IDs for server-side conversation continuation."""

    cost_tracker: Optional["CostTracker"] = None
    """Session-level cost tracker for API usage."""

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.available_models is None:
            self.available_models = []
        if self.last_all_candidates is None:
            self.last_all_candidates = []
        if self.last_filtered_candidates is None:
            self.last_filtered_candidates = []
        if self.last_chosen is None:
            self.last_chosen = []
        if self.last_label_to_candidate is None:
            self.last_label_to_candidate = {}
        if self.last_debug is None:
            self.last_debug = {}
        if self.pinned_chunks is None:
            self.pinned_chunks = []
        if self.context_documents is None:
            self.context_documents = []
        if self.context_chunks is None:
            self.context_chunks = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.llm_server_cache_ids is None:
            self.llm_server_cache_ids = {}

    def clear_artifacts(self) -> None:
        """Clear last query artifacts."""
        self.last_question = None
        self.last_all_candidates = []
        self.last_filtered_candidates = []
        self.last_chosen = []
        self.last_label_to_candidate = {}
        self.last_debug = {}

    def has_filters(self) -> bool:
        """Check if any filters are active."""
        return bool(
            self.path_contains or
            self.ext_allow or
            self.ext_deny
        )

    def get_filter_summary(self) -> str:
        """
        Get a summary of active filters.

        Returns:
            Human-readable filter summary
        """
        if not self.has_filters():
            return "No filters active"

        parts = []

        if self.path_contains:
            if isinstance(self.path_contains, list):
                parts.append(f"Path contains: {', '.join(self.path_contains)}")
            else:
                parts.append(f"Path contains: {self.path_contains}")

        if self.ext_allow:
            parts.append(f"Extensions: {', '.join(self.ext_allow)}")

        if self.ext_deny:
            parts.append(f"Excluding: {', '.join(self.ext_deny)}")

        return "; ".join(parts)

    def format_debug(self) -> str:
        """
        Format debug information from last query.

        Returns:
            Formatted debug string
        """
        if not self.last_debug:
            return "No debug info yet. Ask a question first.\n"

        lines = ["\nDebug (last query):"]
        for key, value in self.last_debug.items():
            lines.append(f"- {key}: {value}")

        lines.append("\nTop candidates (post-filter):")
        max_display = min(10, len(self.last_filtered_candidates))
        for i, c in enumerate(self.last_filtered_candidates[:max_display], start=1):
            source_name = c.source_name or Path(c.source_path).name
            chunk_index = c.chunk_index
            lines.append(f"  {i:02d}. {source_name} (chunk_index={chunk_index}, distance={c.distance})")
        lines.append("")

        return "\n".join(lines)

    def format_costs(self) -> str:
        """
        Format current session cost summary.

        Returns:
            Formatted cost summary string
        """
        tracker = self.cost_tracker
        if not tracker:
            return "Cost tracking is not initialized.\n"
        if not tracker.entries:
            return "No costs recorded for this session.\n"
        return f"\n{tracker.format_summary()}\n"


def get_default_chats_dir(global_folder: Optional[str | Path] = None) -> Path:
    """Get default chat-save directory."""
    return get_chats_folder(global_folder)


def append_chat_turn(
    state: SessionState,
    role: str,
    content: str,
) -> None:
    """
    Append a conversation turn to session history.
    """
    if not content:
        return
    state.conversation_history.append(
        {"role": str(role).strip().lower(), "content": str(content)}
    )


def format_conversation_markdown(
    state: SessionState,
    mode_name: str,
) -> str:
    """
    Format current conversation history as Markdown.
    """
    lines = [
        "# Query Chat Transcript",
        "",
        f"- Mode: {mode_name}",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for turn in state.conversation_history:
        role = (turn.get("role") or "user").capitalize()
        content = turn.get("content") or ""
        lines.append(f"## {role}")
        lines.append("")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def save_conversation_markdown(
    state: SessionState,
    chats_dir: Path,
    mode_name: str,
) -> Path:
    """
    Persist conversation history to a markdown file.
    """
    chats_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"chat-{mode_name}-{timestamp}.md"
    output_path = chats_dir / filename
    output_path.write_text(format_conversation_markdown(state, mode_name), encoding="utf-8")
    return output_path


def serialize_conversation(state: SessionState) -> str:
    """
    Serialize conversation history for caching keys.
    """
    return json.dumps(state.conversation_history, sort_keys=True, separators=(",", ":"))

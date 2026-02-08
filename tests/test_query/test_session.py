"""
Tests for query session module.
"""

import pytest

from pathlib import Path

from lsm.query.session import (
    Candidate,
    SessionState,
    append_chat_turn,
    get_default_chats_dir,
    save_conversation_markdown,
)


class TestCandidate:
    """Tests for Candidate dataclass."""

    def test_candidate_creation(self):
        """Test creating a candidate."""
        candidate = Candidate(
            cid="chunk123",
            text="Sample text",
            meta={"source_path": "/docs/test.md", "chunk_index": 0},
            distance=0.25,
        )

        assert candidate.cid == "chunk123"
        assert candidate.text == "Sample text"
        assert candidate.meta["source_path"] == "/docs/test.md"
        assert candidate.distance == 0.25

    def test_candidate_without_distance(self):
        """Test candidate with no distance."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
        )

        assert candidate.distance is None

    def test_source_path_property(self):
        """Test source_path property."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={"source_path": "/docs/file.md"},
        )

        assert candidate.source_path == "/docs/file.md"

    def test_source_path_missing(self):
        """Test source_path when not in metadata."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
        )

        assert candidate.source_path == "unknown"

    def test_source_name_property(self):
        """Test source_name property."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={"source_name": "file.md"},
        )

        assert candidate.source_name == "file.md"

    def test_source_name_missing(self):
        """Test source_name when not in metadata."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
        )

        assert candidate.source_name == "unknown"

    def test_chunk_index_property(self):
        """Test chunk_index property."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={"chunk_index": 5},
        )

        assert candidate.chunk_index == 5

    def test_chunk_index_missing(self):
        """Test chunk_index when not in metadata."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
        )

        assert candidate.chunk_index == 0

    def test_ext_property(self):
        """Test ext property."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={"ext": ".md"},
        )

        assert candidate.ext == ".md"

    def test_ext_missing(self):
        """Test ext when not in metadata."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
        )

        assert candidate.ext == ""

    def test_relevance_calculation(self):
        """Test relevance calculation from distance."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
            distance=0.3,
        )

        assert candidate.relevance == 0.7  # 1.0 - 0.3

    def test_relevance_no_distance(self):
        """Test relevance when distance is None."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
            distance=None,
        )

        assert candidate.relevance == 0.0

    def test_relevance_clamped_high(self):
        """Test relevance is clamped to 1.0."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
            distance=-0.5,  # Would give 1.5
        )

        assert candidate.relevance == 1.0

    def test_relevance_clamped_low(self):
        """Test relevance is clamped to -1.0."""
        candidate = Candidate(
            cid="1",
            text="Text",
            meta={},
            distance=2.5,  # Would give -1.5
        )

        assert candidate.relevance == -1.0


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_session_state_default_initialization(self):
        """Test session state with defaults."""
        state = SessionState()

        assert state.path_contains is None
        assert state.ext_allow is None
        assert state.ext_deny is None
        assert state.model == "gpt-5.2"
        assert state.available_models == []
        assert state.last_question is None
        assert state.last_all_candidates == []
        assert state.context_documents == []
        assert state.context_chunks == []
        assert state.conversation_history == []
        assert state.llm_server_cache_ids == {}

    def test_session_state_with_filters(self):
        """Test session state with filters."""
        state = SessionState(
            path_contains="docs",
            ext_allow=[".md", ".txt"],
            ext_deny=[".pdf"],
            model="gpt-4",
        )

        assert state.path_contains == "docs"
        assert state.ext_allow == [".md", ".txt"]
        assert state.ext_deny == [".pdf"]
        assert state.model == "gpt-4"

    def test_has_filters_true(self):
        """Test has_filters returns True when filters active."""
        state = SessionState(path_contains="docs")
        assert state.has_filters() is True

        state = SessionState(ext_allow=[".md"])
        assert state.has_filters() is True

        state = SessionState(ext_deny=[".pdf"])
        assert state.has_filters() is True

    def test_has_filters_false(self):
        """Test has_filters returns False when no filters."""
        state = SessionState()
        assert state.has_filters() is False

    def test_get_filter_summary_no_filters(self):
        """Test filter summary with no filters."""
        state = SessionState()
        summary = state.get_filter_summary()

        assert summary == "No filters active"

    def test_get_filter_summary_path_contains(self):
        """Test filter summary with path_contains."""
        state = SessionState(path_contains="docs")
        summary = state.get_filter_summary()

        assert "Path contains: docs" in summary

    def test_get_filter_summary_path_contains_list(self):
        """Test filter summary with path_contains as list."""
        state = SessionState(path_contains=["docs", "guides"])
        summary = state.get_filter_summary()

        assert "Path contains:" in summary
        assert "docs" in summary
        assert "guides" in summary

    def test_get_filter_summary_ext_allow(self):
        """Test filter summary with ext_allow."""
        state = SessionState(ext_allow=[".md", ".txt"])
        summary = state.get_filter_summary()

        assert "Extensions:" in summary
        assert ".md" in summary
        assert ".txt" in summary

    def test_get_filter_summary_ext_deny(self):
        """Test filter summary with ext_deny."""
        state = SessionState(ext_deny=[".pdf", ".docx"])
        summary = state.get_filter_summary()

        assert "Excluding:" in summary
        assert ".pdf" in summary
        assert ".docx" in summary

    def test_get_filter_summary_combined(self):
        """Test filter summary with multiple filters."""
        state = SessionState(
            path_contains="docs",
            ext_allow=[".md"],
            ext_deny=[".pdf"],
        )
        summary = state.get_filter_summary()

        assert "Path contains: docs" in summary
        assert "Extensions: .md" in summary
        assert "Excluding: .pdf" in summary

    def test_clear_artifacts(self):
        """Test clearing query artifacts."""
        state = SessionState()

        # Set some artifacts
        state.last_question = "What is Python?"
        state.last_all_candidates = [
            Candidate(cid="1", text="Text", meta={}, distance=0.1)
        ]
        state.last_filtered_candidates = [
            Candidate(cid="2", text="Text2", meta={}, distance=0.2)
        ]
        state.last_chosen = [
            Candidate(cid="3", text="Text3", meta={}, distance=0.15)
        ]
        state.last_label_to_candidate = {"S1": Candidate(cid="1", text="", meta={})}
        state.last_debug = {"key": "value"}

        # Clear artifacts
        state.clear_artifacts()

        # Check all cleared
        assert state.last_question is None
        assert state.last_all_candidates == []
        assert state.last_filtered_candidates == []
        assert state.last_chosen == []
        assert state.last_label_to_candidate == {}
        assert state.last_debug == {}

    def test_clear_artifacts_preserves_filters(self):
        """Test that clearing artifacts preserves filters."""
        state = SessionState(
            path_contains="docs",
            ext_allow=[".md"],
            model="gpt-4",
        )

        state.last_question = "Question?"
        state.clear_artifacts()

        # Filters should be preserved
        assert state.path_contains == "docs"
        assert state.ext_allow == [".md"]
        assert state.model == "gpt-4"

        # Artifacts should be cleared
        assert state.last_question is None

    def test_post_init_initializes_mutable_defaults(self):
        """Test that __post_init__ properly initializes mutable fields."""
        state = SessionState()

        # These should be lists/dicts, not None
        assert isinstance(state.available_models, list)
        assert isinstance(state.last_all_candidates, list)
        assert isinstance(state.last_filtered_candidates, list)
        assert isinstance(state.last_chosen, list)
        assert isinstance(state.last_label_to_candidate, dict)
        assert isinstance(state.last_debug, dict)

    def test_session_state_independent_instances(self):
        """Test that session states don't share mutable defaults."""
        state1 = SessionState()
        state2 = SessionState()

        state1.last_all_candidates.append(
            Candidate(cid="1", text="Text", meta={})
        )

        # state2 should not be affected
        assert len(state1.last_all_candidates) == 1
        assert len(state2.last_all_candidates) == 0

    def test_last_label_to_candidate_mapping(self):
        """Test label to candidate mapping."""
        state = SessionState()

        candidate = Candidate(cid="1", text="Text", meta={})
        state.last_label_to_candidate["S1"] = candidate

        assert state.last_label_to_candidate["S1"] == candidate
        assert state.last_label_to_candidate["S1"].cid == "1"

    def test_last_debug_storage(self):
        """Test debug information storage."""
        state = SessionState()

        state.last_debug = {
            "question": "What is Python?",
            "k": 12,
            "best_relevance": 0.85,
            "filters_active": True,
        }

        assert state.last_debug["question"] == "What is Python?"
        assert state.last_debug["k"] == 12
        assert state.last_debug["best_relevance"] == 0.85
        assert state.last_debug["filters_active"] is True


def test_get_default_chats_dir(tmp_path: Path):
    resolved = get_default_chats_dir(tmp_path / "lsm-home")
    assert resolved == (tmp_path / "lsm-home" / "Chats").resolve()


def test_chat_history_append_and_save(tmp_path: Path):
    state = SessionState()
    append_chat_turn(state, "user", "hello")
    append_chat_turn(state, "assistant", "hi")
    out = save_conversation_markdown(state, tmp_path, mode_name="grounded")
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "## User" in content
    assert "## Assistant" in content

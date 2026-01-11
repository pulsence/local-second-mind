"""
Tests for query rerank module.
"""

import pytest

from lsm.query.rerank import (
    tokenize,
    compute_lexical_score,
    rerank_lexical,
    deduplicate,
    enforce_diversity,
    apply_local_reranking,
)
from lsm.query.session import Candidate


class TestTokenize:
    """Tests for tokenization."""

    def test_tokenize_simple_text(self):
        """Test tokenizing simple text."""
        result = tokenize("What is Python programming?")
        assert "python" in result
        assert "programming" in result
        # Stopwords removed
        assert "what" not in result
        assert "is" not in result

    def test_tokenize_with_punctuation(self):
        """Test tokenizing text with punctuation."""
        result = tokenize("Hello, world! How are you?")
        assert "hello" in result
        assert "world" in result
        # Stopwords and short words removed
        assert "how" not in result
        assert "are" not in result
        assert "you" not in result

    def test_tokenize_with_numbers(self):
        """Test tokenizing text with numbers."""
        result = tokenize("Python 3.9 released in 2020")
        assert "python" in result
        assert "39" in result  # Combined
        assert "released" in result
        assert "2020" in result

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        result = tokenize("")
        assert result == []

    def test_tokenize_only_stopwords(self):
        """Test tokenizing text with only stopwords."""
        result = tokenize("a the and or is are")
        assert result == []

    def test_tokenize_case_insensitive(self):
        """Test tokenization is case-insensitive."""
        result = tokenize("PYTHON Python python")
        assert all(t == "python" for t in result)


class TestComputeLexicalScore:
    """Tests for lexical scoring."""

    def test_lexical_score_exact_match(self):
        """Test lexical score with exact match."""
        score = compute_lexical_score(
            "Python programming",
            "Python programming language"
        )
        assert score > 0.5
        assert score <= 1.2

    def test_lexical_score_partial_match(self):
        """Test lexical score with partial match."""
        score = compute_lexical_score(
            "Python programming",
            "Python is great"
        )
        assert score > 0
        assert score < 1.0

    def test_lexical_score_no_match(self):
        """Test lexical score with no match."""
        score = compute_lexical_score(
            "Python programming",
            "Java development"
        )
        assert score == 0.0

    def test_lexical_score_phrase_bonus(self):
        """Test lexical score with phrase bonus."""
        # Should get phrase bonus for "python programming"
        score_with_phrase = compute_lexical_score(
            "learn python programming language",
            "Python programming is a skill"
        )

        # Without phrase match
        score_without_phrase = compute_lexical_score(
            "python language",
            "Python is a programming skill"
        )

        assert score_with_phrase > score_without_phrase

    def test_lexical_score_empty_question(self):
        """Test lexical score with empty question."""
        score = compute_lexical_score("", "Some text")
        assert score == 0.0

    def test_lexical_score_empty_passage(self):
        """Test lexical score with empty passage."""
        score = compute_lexical_score("Question", "")
        assert score == 0.0


class TestRerankLexical:
    """Tests for lexical reranking."""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates for reranking."""
        return [
            Candidate(
                cid="1",
                text="Python is a high-level programming language",
                meta={"source_path": "/docs/python.md"},
                distance=0.2,
            ),
            Candidate(
                cid="2",
                text="Java is also used for programming",
                meta={"source_path": "/docs/java.md"},
                distance=0.1,
            ),
            Candidate(
                cid="3",
                text="Python programming tutorials and guides",
                meta={"source_path": "/docs/tutorial.md"},
                distance=0.15,
            ),
        ]

    def test_rerank_lexical_basic(self, sample_candidates):
        """Test basic lexical reranking."""
        result = rerank_lexical("Python programming", sample_candidates)

        assert len(result) == 3
        # Candidates should be reordered by lexical relevance
        assert isinstance(result[0], Candidate)

    def test_rerank_lexical_empty_list(self):
        """Test reranking empty candidate list."""
        result = rerank_lexical("Question", [])
        assert result == []

    def test_rerank_lexical_preserves_all_candidates(self, sample_candidates):
        """Test that reranking preserves all candidates."""
        result = rerank_lexical("Test", sample_candidates)
        assert len(result) == len(sample_candidates)
        assert set(c.cid for c in result) == set(c.cid for c in sample_candidates)


class TestDeduplicate:
    """Tests for deduplication."""

    def test_deduplicate_exact_duplicates(self):
        """Test removing exact duplicate texts."""
        candidates = [
            Candidate(cid="1", text="Same text", meta={}, distance=0.1),
            Candidate(cid="2", text="Same text", meta={}, distance=0.2),
            Candidate(cid="3", text="Different text", meta={}, distance=0.15),
        ]

        result = deduplicate(candidates)

        assert len(result) == 2
        assert result[0].text == "Same text"
        assert result[1].text == "Different text"

    def test_deduplicate_whitespace_normalization(self):
        """Test deduplication normalizes whitespace."""
        candidates = [
            Candidate(cid="1", text="Text  with   spaces", meta={}, distance=0.1),
            Candidate(cid="2", text="Text with spaces", meta={}, distance=0.2),
            Candidate(cid="3", text="Unique text", meta={}, distance=0.15),
        ]

        result = deduplicate(candidates)

        # Should treat both as duplicates (normalized whitespace)
        assert len(result) == 2

    def test_deduplicate_case_insensitive(self):
        """Test deduplication is case-insensitive."""
        candidates = [
            Candidate(cid="1", text="Python Programming", meta={}, distance=0.1),
            Candidate(cid="2", text="python programming", meta={}, distance=0.2),
            Candidate(cid="3", text="Java Development", meta={}, distance=0.15),
        ]

        result = deduplicate(candidates)

        assert len(result) == 2

    def test_deduplicate_empty_list(self):
        """Test deduplicating empty list."""
        result = deduplicate([])
        assert result == []

    def test_deduplicate_keeps_first_occurrence(self):
        """Test deduplication keeps first occurrence."""
        candidates = [
            Candidate(cid="first", text="Duplicate", meta={"rank": 1}, distance=0.1),
            Candidate(cid="second", text="Duplicate", meta={"rank": 2}, distance=0.2),
        ]

        result = deduplicate(candidates)

        assert len(result) == 1
        assert result[0].cid == "first"
        assert result[0].meta["rank"] == 1

    def test_deduplicate_ignores_empty_text(self):
        """Test deduplication ignores empty text candidates."""
        candidates = [
            Candidate(cid="1", text="", meta={}, distance=0.1),
            Candidate(cid="2", text="Valid text", meta={}, distance=0.2),
            Candidate(cid="3", text="  ", meta={}, distance=0.15),
        ]

        result = deduplicate(candidates)

        assert len(result) == 1
        assert result[0].text == "Valid text"


class TestEnforceDiversity:
    """Tests for diversity enforcement."""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates from different files."""
        return [
            Candidate(
                cid="1",
                text="Python chunk 1",
                meta={"source_path": "/docs/python.md"},
                distance=0.1,
            ),
            Candidate(
                cid="2",
                text="Python chunk 2",
                meta={"source_path": "/docs/python.md"},
                distance=0.2,
            ),
            Candidate(
                cid="3",
                text="Python chunk 3",
                meta={"source_path": "/docs/python.md"},
                distance=0.25,
            ),
            Candidate(
                cid="4",
                text="Java chunk 1",
                meta={"source_path": "/docs/java.md"},
                distance=0.15,
            ),
            Candidate(
                cid="5",
                text="Java chunk 2",
                meta={"source_path": "/docs/java.md"},
                distance=0.3,
            ),
        ]

    def test_enforce_diversity_basic(self, sample_candidates):
        """Test basic diversity enforcement."""
        result = enforce_diversity(sample_candidates, max_per_file=2)

        assert len(result) == 4
        # Should keep first 2 from python.md and first 2 from java.md
        python_count = sum(1 for c in result if "python.md" in c.source_path)
        java_count = sum(1 for c in result if "java.md" in c.source_path)
        assert python_count == 2
        assert java_count == 2

    def test_enforce_diversity_max_one(self, sample_candidates):
        """Test diversity with max_per_file=1."""
        result = enforce_diversity(sample_candidates, max_per_file=1)

        assert len(result) == 2
        # Should keep first from each file
        assert result[0].cid == "1"
        assert result[1].cid == "4"

    def test_enforce_diversity_max_zero(self, sample_candidates):
        """Test diversity with max_per_file=0."""
        result = enforce_diversity(sample_candidates, max_per_file=0)
        # Should return all candidates when max_per_file <= 0
        assert len(result) == len(sample_candidates)

    def test_enforce_diversity_preserves_order(self, sample_candidates):
        """Test that diversity enforcement preserves order."""
        result = enforce_diversity(sample_candidates, max_per_file=2)

        # Should keep candidates in original order
        prev_idx = -1
        for c in result:
            curr_idx = next(
                i for i, sc in enumerate(sample_candidates) if sc.cid == c.cid
            )
            assert curr_idx > prev_idx
            prev_idx = curr_idx

    def test_enforce_diversity_empty_list(self):
        """Test diversity enforcement on empty list."""
        result = enforce_diversity([], max_per_file=2)
        assert result == []


class TestApplyLocalReranking:
    """Tests for complete local reranking pipeline."""

    @pytest.fixture
    def diverse_candidates(self):
        """Create diverse candidates for pipeline testing."""
        return [
            Candidate(
                cid="1",
                text="Python programming language overview",
                meta={"source_path": "/docs/python.md"},
                distance=0.1,
            ),
            Candidate(
                cid="2",
                text="Python programming language overview",  # Duplicate
                meta={"source_path": "/docs/python_copy.md"},
                distance=0.15,
            ),
            Candidate(
                cid="3",
                text="Python tutorials and examples",
                meta={"source_path": "/docs/python.md"},
                distance=0.2,
            ),
            Candidate(
                cid="4",
                text="Java programming basics",
                meta={"source_path": "/docs/java.md"},
                distance=0.25,
            ),
            Candidate(
                cid="5",
                text="Python advanced topics",
                meta={"source_path": "/docs/python.md"},
                distance=0.3,
            ),
        ]

    def test_apply_local_reranking_full_pipeline(self, diverse_candidates):
        """Test full local reranking pipeline."""
        result = apply_local_reranking(
            "Python programming",
            diverse_candidates,
            max_per_file=2,
            local_pool=10,
        )

        # Should deduplicate, rerank, and enforce diversity
        assert len(result) <= 4  # After dedup and diversity
        assert isinstance(result[0], Candidate)

    def test_apply_local_reranking_respects_local_pool(self, diverse_candidates):
        """Test that local pool limit is respected."""
        result = apply_local_reranking(
            "Python programming",
            diverse_candidates,
            max_per_file=2,
            local_pool=2,  # Very small pool
        )

        # Should be limited by local_pool after lexical rerank
        assert len(result) <= 2

    def test_apply_local_reranking_empty_list(self):
        """Test pipeline with empty candidate list."""
        result = apply_local_reranking(
            "Question",
            [],
            max_per_file=2,
            local_pool=10,
        )

        assert result == []

    def test_apply_local_reranking_removes_duplicates(self, diverse_candidates):
        """Test that pipeline removes duplicates."""
        result = apply_local_reranking(
            "Python programming",
            diverse_candidates,
            max_per_file=5,
            local_pool=50,
        )

        # Should have removed the duplicate
        texts = [c.text for c in result]
        assert len(texts) == len(set(texts))  # No duplicates

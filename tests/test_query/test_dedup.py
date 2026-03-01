"""Tests for MinHash near-duplicate detection stage."""

from __future__ import annotations

import pytest

from lsm.query.session import Candidate
from lsm.query.stages.dedup import (
    are_near_duplicates,
    compute_minhash,
    deduplicate_candidates,
    jaccard_estimate,
)


def _candidate(cid, text=""):
    return Candidate(cid=cid, text=text, meta={}, distance=0.2)


class TestComputeMinHash:
    def test_returns_tuple_of_correct_length(self):
        sig = compute_minhash("hello world", num_perm=64)
        assert isinstance(sig, tuple)
        assert len(sig) == 64

    def test_identical_texts_produce_same_signature(self):
        sig_a = compute_minhash("the quick brown fox jumps")
        sig_b = compute_minhash("the quick brown fox jumps")
        assert sig_a == sig_b

    def test_different_texts_produce_different_signatures(self):
        sig_a = compute_minhash("the quick brown fox jumps over the lazy dog")
        sig_b = compute_minhash("completely unrelated topic about quantum physics")
        assert sig_a != sig_b

    def test_empty_text(self):
        sig = compute_minhash("")
        assert len(sig) == 128


class TestJaccardEstimate:
    def test_identical_signatures(self):
        sig = compute_minhash("same text here")
        assert jaccard_estimate(sig, sig) == 1.0

    def test_different_lengths_returns_zero(self):
        assert jaccard_estimate((1, 2, 3), (1, 2)) == 0.0

    def test_empty_signatures(self):
        assert jaccard_estimate((), ()) == 0.0


class TestAreNearDuplicates:
    def test_identical_texts_are_duplicates(self):
        sig_a = compute_minhash("this is a test document about Python programming")
        sig_b = compute_minhash("this is a test document about Python programming")
        assert are_near_duplicates(sig_a, sig_b, threshold=0.8)

    def test_similar_texts_detected(self):
        sig_a = compute_minhash("the quick brown fox jumps over the lazy dog and runs away")
        sig_b = compute_minhash("the quick brown fox jumps over the lazy dog and runs fast")
        # Very similar texts should have high Jaccard estimate
        similarity = jaccard_estimate(sig_a, sig_b)
        assert similarity > 0.5

    def test_different_texts_not_duplicates(self):
        sig_a = compute_minhash("python is a programming language for data science")
        sig_b = compute_minhash("the weather forecast shows rain tomorrow afternoon")
        assert not are_near_duplicates(sig_a, sig_b, threshold=0.8)

    def test_threshold_parameter(self):
        sig_a = compute_minhash("hello world foo bar baz")
        sig_b = compute_minhash("hello world foo bar qux")
        sim = jaccard_estimate(sig_a, sig_b)
        # At a very low threshold, they should be considered duplicates
        assert are_near_duplicates(sig_a, sig_b, threshold=0.0)


class TestDeduplicateCandidates:
    def test_empty_input(self):
        assert deduplicate_candidates([]) == []

    def test_no_duplicates_preserved(self):
        candidates = [
            _candidate("a", "python programming language"),
            _candidate("b", "quantum physics research paper"),
            _candidate("c", "cooking recipe for pasta carbonara"),
        ]
        result = deduplicate_candidates(candidates, threshold=0.8)
        assert len(result) == 3

    def test_exact_duplicates_removed(self):
        text = "this is a test document about Python programming features and patterns"
        candidates = [
            _candidate("a", text),
            _candidate("b", text),
            _candidate("c", text),
        ]
        result = deduplicate_candidates(candidates, threshold=0.8)
        assert len(result) == 1
        assert result[0].cid == "a"  # First occurrence preserved

    def test_preserves_order(self):
        candidates = [
            _candidate("a", "unique text about databases and SQL queries in production"),
            _candidate("b", "unique text about databases and SQL queries in production"),
            _candidate("c", "another unique document about machine learning algorithms"),
        ]
        result = deduplicate_candidates(candidates, threshold=0.8)
        cids = [c.cid for c in result]
        assert cids[0] == "a"
        assert "c" in cids

    def test_threshold_controls_sensitivity(self):
        candidates = [
            _candidate("a", "the quick brown fox jumps over the lazy dog"),
            _candidate("b", "the quick brown fox leaps over the lazy dog"),
        ]
        # With threshold=1.0, nothing should be deduped (exact match only)
        strict = deduplicate_candidates(candidates, threshold=1.0)
        assert len(strict) == 2

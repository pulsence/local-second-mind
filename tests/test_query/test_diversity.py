"""Tests for MMR diversity selection and per-section cap stages."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lsm.query.session import Candidate
from lsm.query.stages.diversity import mmr_select, per_section_cap


def _candidate(cid, text="", embedding=None, heading_path=None, distance=0.2):
    meta = {}
    if heading_path is not None:
        meta["heading_path"] = json.dumps(heading_path)
    return Candidate(
        cid=cid, text=text, meta=meta, distance=distance, embedding=embedding
    )


# ------------------------------------------------------------------
# MMR Selection
# ------------------------------------------------------------------


class TestMMRSelect:
    def test_empty_input(self):
        assert mmr_select([], query_embedding=[0.1] * 10) == []

    def test_k_zero(self):
        candidates = [_candidate("a")]
        assert mmr_select(candidates, query_embedding=[0.1] * 10, k=0) == []

    def test_returns_top_k(self):
        emb = [0.1] * 10
        candidates = [_candidate(f"c{i}", embedding=emb) for i in range(5)]
        result = mmr_select(candidates, query_embedding=emb, k=3)
        assert len(result) == 3

    def test_no_embeddings_returns_order(self):
        """Without embeddings, returns candidates in original order."""
        candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
        result = mmr_select(candidates, query_embedding=[0.1] * 10, k=2)
        assert [c.cid for c in result] == ["a", "b"]

    def test_no_query_embedding_returns_order(self):
        emb = [0.1] * 10
        candidates = [_candidate("a", embedding=emb), _candidate("b", embedding=emb)]
        result = mmr_select(candidates, query_embedding=None, k=2)
        assert [c.cid for c in result] == ["a", "b"]

    def test_diversity_selection(self):
        """Diverse candidates should be preferred when lambda is low."""
        # Create candidates: two very similar, one different
        similar_emb_1 = [1.0, 0.0, 0.0]
        similar_emb_2 = [0.99, 0.01, 0.0]
        diverse_emb = [0.0, 1.0, 0.0]
        query_emb = [0.5, 0.5, 0.0]

        candidates = [
            _candidate("sim1", embedding=similar_emb_1),
            _candidate("sim2", embedding=similar_emb_2),
            _candidate("div", embedding=diverse_emb),
        ]
        # Low lambda = prioritize diversity
        result = mmr_select(candidates, query_embedding=query_emb, lambda_param=0.3, k=2)
        cids = {c.cid for c in result}
        # Should pick one similar and one diverse
        assert "div" in cids

    def test_high_lambda_prioritizes_relevance(self):
        """High lambda should pick most relevant candidates."""
        query_emb = [1.0, 0.0, 0.0]
        candidates = [
            _candidate("relevant", embedding=[0.9, 0.1, 0.0]),
            _candidate("diverse", embedding=[0.0, 0.0, 1.0]),
        ]
        result = mmr_select(
            candidates, query_embedding=query_emb, lambda_param=0.99, k=1
        )
        assert result[0].cid == "relevant"


# ------------------------------------------------------------------
# Per-Section Cap
# ------------------------------------------------------------------


class TestPerSectionCap:
    def test_empty_input(self):
        assert per_section_cap([]) == []

    def test_no_cap(self):
        candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
        result = per_section_cap(candidates, max_per_section=None)
        assert len(result) == 3

    def test_caps_within_section(self):
        candidates = [
            _candidate("a", heading_path=["Chapter 1", "Section A"]),
            _candidate("b", heading_path=["Chapter 1", "Section A"]),
            _candidate("c", heading_path=["Chapter 1", "Section A"]),
            _candidate("d", heading_path=["Chapter 2", "Section B"]),
        ]
        result = per_section_cap(candidates, max_per_section=2, heading_depth=2)
        assert len(result) == 3  # 2 from Ch1/SecA + 1 from Ch2/SecB
        cids = [c.cid for c in result]
        assert "a" in cids
        assert "b" in cids
        assert "d" in cids
        assert "c" not in cids

    def test_preserves_order(self):
        candidates = [
            _candidate("a", heading_path=["H1"]),
            _candidate("b", heading_path=["H1"]),
            _candidate("c", heading_path=["H2"]),
        ]
        result = per_section_cap(candidates, max_per_section=1)
        assert [c.cid for c in result] == ["a", "c"]

    def test_heading_depth_grouping(self):
        """depth=1 groups by top-level heading only."""
        candidates = [
            _candidate("a", heading_path=["Chapter 1", "Section A"]),
            _candidate("b", heading_path=["Chapter 1", "Section B"]),
            _candidate("c", heading_path=["Chapter 2", "Section A"]),
        ]
        result = per_section_cap(candidates, max_per_section=1, heading_depth=1)
        assert len(result) == 2  # 1 from Ch1, 1 from Ch2

    def test_no_heading_treated_as_empty_group(self):
        candidates = [
            _candidate("a"),
            _candidate("b"),
            _candidate("c"),
        ]
        result = per_section_cap(candidates, max_per_section=2)
        assert len(result) == 2

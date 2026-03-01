"""Tests for temporal-aware ranking stage."""

from __future__ import annotations

import time

import pytest

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate
from lsm.query.stages.temporal import (
    apply_temporal_boost,
    filter_time_range,
)

_NS_PER_DAY = 86_400_000_000_000


def _now_ns():
    return int(time.time() * 1_000_000_000)


def _candidate(cid, distance=0.3, days_ago=None):
    meta = {}
    if days_ago is not None:
        meta["mtime_ns"] = _now_ns() - (days_ago * _NS_PER_DAY)
    return Candidate(cid=cid, text="", meta=meta, distance=distance)


# ------------------------------------------------------------------
# apply_temporal_boost
# ------------------------------------------------------------------


class TestApplyTemporalBoost:
    def test_empty_input(self):
        assert apply_temporal_boost([]) == []

    def test_recent_files_get_boost(self):
        candidates = [
            _candidate("recent", distance=0.4, days_ago=5),
            _candidate("old", distance=0.4, days_ago=60),
        ]
        result = apply_temporal_boost(candidates, boost_days=30, boost_factor=2.0)
        # Recent candidate should have lower distance (boosted)
        by_id = {c.cid: c for c in result}
        assert by_id["recent"].distance < by_id["old"].distance
        assert by_id["recent"].distance == pytest.approx(0.2)  # 0.4 / 2.0
        assert by_id["old"].distance == 0.4  # unchanged

    def test_temporal_boost_in_score_breakdown(self):
        candidates = [_candidate("recent", distance=0.3, days_ago=5)]
        result = apply_temporal_boost(candidates, boost_days=30, boost_factor=1.5)
        assert result[0].score_breakdown is not None
        assert result[0].score_breakdown.temporal_boost == 1.5

    def test_no_boost_for_old_files(self):
        candidates = [_candidate("old", distance=0.3, days_ago=60)]
        result = apply_temporal_boost(candidates, boost_days=30, boost_factor=2.0)
        assert result[0].distance == 0.3
        # No temporal_boost should be set on old files
        if result[0].score_breakdown:
            assert result[0].score_breakdown.temporal_boost is None

    def test_no_mtime_metadata(self):
        candidates = [Candidate(cid="no_time", text="", meta={}, distance=0.3)]
        result = apply_temporal_boost(candidates, boost_days=30, boost_factor=2.0)
        assert result[0].distance == 0.3

    def test_boost_factor_one_no_change(self):
        candidates = [_candidate("a", distance=0.3, days_ago=5)]
        result = apply_temporal_boost(candidates, boost_factor=1.0)
        assert result[0].distance == 0.3

    def test_resorting_by_distance(self):
        """After boosting, candidates should be re-sorted by distance."""
        candidates = [
            _candidate("old_close", distance=0.2, days_ago=60),
            _candidate("recent_far", distance=0.5, days_ago=5),
        ]
        result = apply_temporal_boost(candidates, boost_days=30, boost_factor=3.0)
        # recent_far: 0.5/3.0 = 0.167; old_close: 0.2
        assert result[0].cid == "recent_far"
        assert result[1].cid == "old_close"

    def test_configurable_boost_days(self):
        candidates = [_candidate("a", distance=0.3, days_ago=10)]
        # 10 days ago is within 15-day window
        result15 = apply_temporal_boost(candidates, boost_days=15, boost_factor=2.0)
        assert result15[0].distance == pytest.approx(0.15)
        # 10 days ago is NOT within 5-day window
        result5 = apply_temporal_boost(candidates, boost_days=5, boost_factor=2.0)
        assert result5[0].distance == 0.3


# ------------------------------------------------------------------
# filter_time_range
# ------------------------------------------------------------------


class TestFilterTimeRange:
    def test_empty_input(self):
        assert filter_time_range([]) == []

    def test_no_bounds_returns_all(self):
        candidates = [_candidate("a"), _candidate("b")]
        result = filter_time_range(candidates)
        assert len(result) == 2

    def test_start_filter(self):
        now = _now_ns()
        candidates = [
            _candidate("recent", days_ago=5),
            _candidate("old", days_ago=60),
        ]
        # Only include items from last 30 days
        start = now - (30 * _NS_PER_DAY)
        result = filter_time_range(candidates, start_ns=start)
        assert len(result) == 1
        assert result[0].cid == "recent"

    def test_end_filter(self):
        now = _now_ns()
        candidates = [
            _candidate("recent", days_ago=5),
            _candidate("old", days_ago=60),
        ]
        # Only include items older than 30 days
        end = now - (30 * _NS_PER_DAY)
        result = filter_time_range(candidates, end_ns=end)
        assert len(result) == 1
        assert result[0].cid == "old"

    def test_both_bounds(self):
        now = _now_ns()
        candidates = [
            _candidate("recent", days_ago=5),
            _candidate("middle", days_ago=20),
            _candidate("old", days_ago=60),
        ]
        start = now - (30 * _NS_PER_DAY)
        end = now - (10 * _NS_PER_DAY)
        result = filter_time_range(candidates, start_ns=start, end_ns=end)
        assert len(result) == 1
        assert result[0].cid == "middle"

    def test_no_mtime_included(self):
        """Candidates without mtime_ns are always included."""
        candidates = [
            Candidate(cid="no_time", text="", meta={}, distance=0.3),
        ]
        now = _now_ns()
        result = filter_time_range(candidates, start_ns=now - _NS_PER_DAY)
        assert len(result) == 1

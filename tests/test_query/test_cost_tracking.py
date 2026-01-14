"""
Tests for cost tracking utilities.
"""

from datetime import datetime

from lsm.query.cost_tracking import CostTracker, estimate_tokens, estimate_output_tokens


def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2


def test_estimate_output_tokens():
    assert estimate_output_tokens("abcd", None) == 1
    assert estimate_output_tokens(None, 120) == 120
    assert estimate_output_tokens(None, None) == 0


def test_cost_tracker_summary_and_budget():
    tracker = CostTracker(budget_limit=1.0, warn_threshold=0.8)
    tracker.add_entry("openai", "gpt-5.2", 100, 50, 0.4, "synthesize")
    tracker.add_entry("openai", "gpt-5.2", 50, 20, 0.5, "rerank")

    assert tracker.total_calls() == 2
    assert abs(tracker.total_cost() - 0.9) < 1e-6
    status = tracker.budget_status()
    assert status is not None
    assert "warning" in status.lower()

    summary = tracker.format_summary()
    assert "SESSION COST SUMMARY" in summary
    assert "openai" in summary


def test_cost_tracker_period_summary():
    tracker = CostTracker()
    tracker.add_entry("openai", "gpt-5.2", 100, 50, 0.4, "synthesize")
    tracker.add_entry("openai", "gpt-5.2", 100, 50, 0.6, "synthesize")

    # Force timestamps into two different months
    tracker.entries[0].timestamp = datetime(2025, 1, 15)
    tracker.entries[1].timestamp = datetime(2025, 2, 3)

    monthly = tracker.period_summary("monthly")
    assert monthly["2025-01"] == 0.4
    assert monthly["2025-02"] == 0.6


def test_cost_tracker_export_csv(tmp_path):
    tracker = CostTracker()
    tracker.add_entry("openai", "gpt-5.2", 100, 50, 0.4, "synthesize")
    out_path = tmp_path / "costs.csv"
    tracker.export_csv(out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "timestamp,provider,model" in content
    assert "openai" in content

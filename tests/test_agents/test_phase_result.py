from __future__ import annotations

import pytest

from lsm.agents.phase import PhaseResult


def test_phase_result_construction() -> None:
    result = PhaseResult(
        final_text="All done",
        tool_calls=[{"name": "echo", "result": "hello"}],
        stop_reason="done",
    )
    assert result.final_text == "All done"
    assert result.tool_calls == [{"name": "echo", "result": "hello"}]
    assert result.stop_reason == "done"


def test_phase_result_has_no_tokens_used() -> None:
    result = PhaseResult(final_text="", tool_calls=[], stop_reason="done")
    with pytest.raises(AttributeError):
        _ = result.tokens_used  # type: ignore[attr-defined]


def test_phase_result_has_no_cost_usd() -> None:
    result = PhaseResult(final_text="", tool_calls=[], stop_reason="done")
    with pytest.raises(AttributeError):
        _ = result.cost_usd  # type: ignore[attr-defined]


def test_phase_result_all_stop_reasons_accepted() -> None:
    for reason in ("done", "max_iterations", "budget_exhausted", "stop_requested"):
        result = PhaseResult(final_text="", tool_calls=[], stop_reason=reason)
        assert result.stop_reason == reason

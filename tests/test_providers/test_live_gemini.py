"""Live Gemini provider tests."""

from __future__ import annotations

import pytest

from tests.test_providers.live_llm_checks import exercise_live_provider_contract


pytestmark = [pytest.mark.live, pytest.mark.live_llm]


def test_live_gemini_provider_contract(real_gemini_provider) -> None:
    exercise_live_provider_contract(real_gemini_provider, expect_response_id=True)

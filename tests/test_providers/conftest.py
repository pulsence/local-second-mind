"""
Shared fixtures for provider tests.
"""

import pytest

from lsm.providers.base import BaseLLMProvider


@pytest.fixture(autouse=True)
def reset_health_stats():
    BaseLLMProvider._GLOBAL_HEALTH_STATS = {}

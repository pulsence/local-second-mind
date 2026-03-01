"""
Tests for agent mode validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import pytest

from lsm.agents.tools.mode_validation import validate_agent_mode, VALID_PROFILES


# ---------------------------------------------------------------------------
# Fake config structures
# ---------------------------------------------------------------------------


@dataclass
class FakeLocalPolicy:
    enabled: bool = True
    k: int = 5
    min_relevance: float = 0.3


@dataclass
class FakeRemotePolicy:
    enabled: bool = False


@dataclass
class FakeModeConfig:
    retrieval_profile: Optional[str] = None
    local_policy: FakeLocalPolicy = field(default_factory=FakeLocalPolicy)
    remote_policy: FakeRemotePolicy = field(default_factory=FakeRemotePolicy)


@dataclass
class FakeSandboxConfig:
    allow_url_access: bool = False
    allowed_read_paths: List[Path] = field(default_factory=list)
    allowed_write_paths: List[Path] = field(default_factory=list)


class FakeLSMConfig:
    def __init__(self, modes=None):
        self._modes = modes or {}

    def get_mode_config(self, name):
        if name in self._modes:
            return self._modes[name]
        raise KeyError(f"Unknown mode: {name}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_none_mode_is_valid():
    config = FakeLSMConfig()
    assert validate_agent_mode(None, config) is None


def test_valid_mode_accepted():
    config = FakeLSMConfig(modes={
        "grounded": FakeModeConfig(retrieval_profile="dense_only")
    })
    assert validate_agent_mode("grounded", config) is None


def test_unknown_mode_rejected():
    config = FakeLSMConfig(modes={})
    error = validate_agent_mode("nonexistent", config)
    assert error is not None
    assert "Unknown mode" in error


def test_invalid_retrieval_profile_rejected():
    config = FakeLSMConfig(modes={
        "bad": FakeModeConfig(retrieval_profile="invalid_profile")
    })
    error = validate_agent_mode("bad", config)
    assert error is not None
    assert "Invalid retrieval profile" in error


def test_valid_retrieval_profiles_accepted():
    for profile in VALID_PROFILES:
        config = FakeLSMConfig(modes={
            "test": FakeModeConfig(retrieval_profile=profile)
        })
        assert validate_agent_mode("test", config) is None


def test_remote_enabled_without_url_access_rejected():
    config = FakeLSMConfig(modes={
        "remote_mode": FakeModeConfig(
            remote_policy=FakeRemotePolicy(enabled=True)
        )
    })
    sandbox = FakeSandboxConfig(allow_url_access=False)
    error = validate_agent_mode("remote_mode", config, sandbox)
    assert error is not None
    assert "allow_url_access" in error


def test_remote_enabled_with_url_access_accepted():
    config = FakeLSMConfig(modes={
        "remote_mode": FakeModeConfig(
            remote_policy=FakeRemotePolicy(enabled=True)
        )
    })
    sandbox = FakeSandboxConfig(allow_url_access=True)
    assert validate_agent_mode("remote_mode", config, sandbox) is None


def test_remote_disabled_without_url_access_accepted():
    config = FakeLSMConfig(modes={
        "local_mode": FakeModeConfig(
            remote_policy=FakeRemotePolicy(enabled=False)
        )
    })
    sandbox = FakeSandboxConfig(allow_url_access=False)
    assert validate_agent_mode("local_mode", config, sandbox) is None


def test_no_sandbox_config_skips_url_check():
    config = FakeLSMConfig(modes={
        "remote_mode": FakeModeConfig(
            remote_policy=FakeRemotePolicy(enabled=True)
        )
    })
    # No sandbox_config — validation should pass
    assert validate_agent_mode("remote_mode", config, None) is None


def test_no_profile_in_mode_accepted():
    config = FakeLSMConfig(modes={
        "simple": FakeModeConfig(retrieval_profile=None)
    })
    assert validate_agent_mode("simple", config) is None

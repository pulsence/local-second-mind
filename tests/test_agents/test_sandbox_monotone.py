from __future__ import annotations

from pathlib import Path

import pytest

from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


def _base_parent(tmp_path: Path) -> SandboxConfig:
    return SandboxConfig(
        allowed_read_paths=[tmp_path],
        allowed_write_paths=[tmp_path],
        allow_url_access=False,
        require_user_permission={"spawn_agent": True},
        require_permission_by_risk={"exec": True},
        execution_mode="prefer_docker",
        force_docker=True,
        limits={
            "timeout_s_default": 30,
            "max_stdout_kb": 256,
            "max_file_write_mb": 10,
        },
    )


def test_child_sandbox_cannot_enable_network_when_parent_disables(tmp_path: Path) -> None:
    parent = _base_parent(tmp_path)
    with pytest.raises(ValueError, match="allow_url_access"):
        ToolSandbox(
            SandboxConfig(
                allowed_read_paths=[tmp_path],
                allowed_write_paths=[tmp_path],
                allow_url_access=True,
                execution_mode="prefer_docker",
                force_docker=True,
            ),
            global_sandbox=parent,
        )


def test_child_sandbox_cannot_relax_permission_gates(tmp_path: Path) -> None:
    parent = _base_parent(tmp_path)
    with pytest.raises(ValueError, match="require_permission_by_risk"):
        ToolSandbox(
            SandboxConfig(
                allowed_read_paths=[tmp_path],
                allowed_write_paths=[tmp_path],
                allow_url_access=False,
                require_user_permission={"spawn_agent": True},
                require_permission_by_risk={},
                execution_mode="prefer_docker",
                force_docker=True,
            ),
            global_sandbox=parent,
        )


def test_child_sandbox_cannot_relax_runner_policy_or_limits(tmp_path: Path) -> None:
    parent = _base_parent(tmp_path)
    with pytest.raises(ValueError, match="execution_mode"):
        ToolSandbox(
            SandboxConfig(
                allowed_read_paths=[tmp_path],
                allowed_write_paths=[tmp_path],
                allow_url_access=False,
                require_user_permission={"spawn_agent": True},
                require_permission_by_risk={"exec": True},
                execution_mode="local_only",
                force_docker=True,
            ),
            global_sandbox=parent,
        )

    with pytest.raises(ValueError, match="max_stdout_kb"):
        ToolSandbox(
            SandboxConfig(
                allowed_read_paths=[tmp_path],
                allowed_write_paths=[tmp_path],
                allow_url_access=False,
                require_user_permission={"spawn_agent": True},
                require_permission_by_risk={"exec": True},
                execution_mode="prefer_docker",
                force_docker=True,
                limits={
                    "timeout_s_default": 30,
                    "max_stdout_kb": 1024,
                    "max_file_write_mb": 10,
                },
            ),
            global_sandbox=parent,
        )


def test_child_sandbox_can_be_more_restrictive_than_parent(tmp_path: Path) -> None:
    parent = SandboxConfig(
        allowed_read_paths=[tmp_path],
        allowed_write_paths=[tmp_path],
        allow_url_access=True,
        require_user_permission={"spawn_agent": True},
        require_permission_by_risk={"exec": True, "network": True},
        execution_mode="local_only",
        force_docker=False,
        limits={
            "timeout_s_default": 30,
            "max_stdout_kb": 256,
            "max_file_write_mb": 10,
        },
    )
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path / "child"],
            allowed_write_paths=[tmp_path / "child"],
            allow_url_access=False,
            require_user_permission={"spawn_agent": True},
            require_permission_by_risk={"exec": True, "network": True},
            execution_mode="prefer_docker",
            force_docker=True,
            limits={
                "timeout_s_default": 15,
                "max_stdout_kb": 128,
                "max_file_write_mb": 5,
            },
        ),
        global_sandbox=parent,
    )
    assert sandbox.config.force_docker is True
    assert sandbox.config.execution_mode == "prefer_docker"
    assert sandbox.config.allow_url_access is False

from __future__ import annotations

from pathlib import Path

import pytest

from lsm.agents.tools.base import BaseTool
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


class LoadURLToolStub(BaseTool):
    name = "load_url"
    description = "network load"
    risk_level = "network"
    needs_network = True
    input_schema = {"type": "object", "properties": {"url": {"type": "string"}}}

    def execute(self, args: dict) -> str:
        _ = args
        return "ok"


class QueryRemoteToolStub(BaseTool):
    name = "query_remote"
    description = "network remote"
    risk_level = "network"
    needs_network = True
    input_schema = {"type": "object", "properties": {"input": {"type": "object"}}}

    def execute(self, args: dict) -> str:
        _ = args
        return "ok"


class QueryRemoteChainToolStub(BaseTool):
    name = "query_remote_chain"
    description = "network chain"
    risk_level = "network"
    needs_network = True
    input_schema = {"type": "object", "properties": {"input": {"type": "object"}}}

    def execute(self, args: dict) -> str:
        _ = args
        return "ok"


class ReadOnlyToolStub(BaseTool):
    name = "read_file"
    description = "local read"
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        _ = args
        return "ok"


def test_security_network_load_url_blocked_when_disabled() -> None:
    sandbox = ToolSandbox(SandboxConfig(allow_url_access=False))
    with pytest.raises(PermissionError, match="Network access is disabled"):
        sandbox.execute(LoadURLToolStub(), {"url": "https://example.com"})


def test_security_network_query_remote_blocked_when_disabled() -> None:
    sandbox = ToolSandbox(SandboxConfig(allow_url_access=False))
    with pytest.raises(PermissionError, match="Network access is disabled"):
        sandbox.execute(QueryRemoteToolStub(), {"input": {}})


def test_security_network_query_remote_chain_blocked_when_disabled() -> None:
    sandbox = ToolSandbox(SandboxConfig(allow_url_access=False))
    with pytest.raises(PermissionError, match="Network access is disabled"):
        sandbox.execute(QueryRemoteChainToolStub(), {"input": {}})


def test_security_network_tools_allowed_when_enabled() -> None:
    sandbox = ToolSandbox(SandboxConfig(allow_url_access=True))
    assert sandbox.execute(LoadURLToolStub(), {"url": "https://example.com"}) == "ok"
    assert sandbox.execute(QueryRemoteToolStub(), {"input": {}}) == "ok"
    assert sandbox.execute(QueryRemoteChainToolStub(), {"input": {}}) == "ok"


def test_security_non_network_tools_unaffected_by_url_access_setting(tmp_path: Path) -> None:
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=False,
            allowed_read_paths=[tmp_path],
        )
    )
    assert sandbox.execute(ReadOnlyToolStub(), {"path": str(tmp_path / "doc.txt")}) == "ok"

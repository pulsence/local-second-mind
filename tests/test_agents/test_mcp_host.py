from __future__ import annotations

from typing import Any, Dict, List

import pytest

from lsm.agents.tools import ToolRegistry
from lsm.agents.tools.mcp_host import MCPError, MCPHost
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig
from lsm.config.models.global_config import MCPServerConfig


class FakeMCPServer:
    def __init__(
        self,
        name: str,
        tools: List[Dict[str, Any]],
        *,
        fail_list_once: bool = False,
    ) -> None:
        self.name = name
        self.tools = tools
        self.fail_list_once = fail_list_once
        self.list_calls = 0
        self.call_calls = 0
        self.start_calls = 0
        self.shutdown_calls = 0
        self._running = False
        self._failed_once = False

    def start(self) -> None:
        self.start_calls += 1
        self._running = True

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def list_tools(self) -> List[Dict[str, Any]]:
        self.list_calls += 1
        if not self._running:
            self.start()
        if self.fail_list_once and not self._failed_once:
            self._failed_once = True
            self._running = False
            raise MCPError("list failed")
        return self.tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        self.call_calls += 1
        if not self._running:
            self.start()
        return f"{tool_name}:{arguments.get('value', '')}"


def _tool_defs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "echo",
            "description": "Echo MCP tool.",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        }
    ]


def test_mcp_host_registers_tools() -> None:
    config = MCPServerConfig(name="demo", command="demo-server")
    server = FakeMCPServer("demo", _tool_defs())
    host = MCPHost([config], server_factory=lambda _: server)

    registry = ToolRegistry()
    host.register_tools(registry)

    tool = registry.lookup("mcp_demo_echo")
    assert tool.description == "Echo MCP tool."
    assert tool.execute({"value": "hi"}) == "echo:hi"
    assert server.list_calls == 1
    assert server.call_calls == 1


def test_mcp_host_restarts_server_after_failure() -> None:
    config = MCPServerConfig(name="demo", command="demo-server")
    server = FakeMCPServer("demo", _tool_defs(), fail_list_once=True)
    host = MCPHost([config], server_factory=lambda _: server)

    registry = ToolRegistry()
    host.register_tools(registry)

    assert registry.lookup("mcp_demo_echo").name == "mcp_demo_echo"
    assert server.start_calls >= 2


def test_mcp_tool_blocked_by_sandbox() -> None:
    config = MCPServerConfig(name="demo", command="demo-server")
    server = FakeMCPServer("demo", _tool_defs())
    host = MCPHost([config], server_factory=lambda _: server)
    registry = ToolRegistry()
    host.register_tools(registry)

    tool = registry.lookup("mcp_demo_echo")
    sandbox = ToolSandbox(SandboxConfig(allow_url_access=False))
    with pytest.raises(PermissionError, match="Network access is disabled"):
        sandbox.execute(tool, {"value": "hi"})


def test_mcp_host_shutdown_calls_servers() -> None:
    config = MCPServerConfig(name="demo", command="demo-server")
    server = FakeMCPServer("demo", _tool_defs())
    host = MCPHost([config], server_factory=lambda _: server)
    registry = ToolRegistry()
    host.register_tools(registry)

    host.shutdown_all()
    assert server.shutdown_calls == 1

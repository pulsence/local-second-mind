"""
MCP host support for external tool servers.

Implements a JSON-RPC 2.0 client over stdio with LSP-style message framing.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from lsm.config.models import MCPServerConfig
from lsm.logging import get_logger

from .base import BaseTool, ToolRegistry

logger = get_logger(__name__)


class MCPError(RuntimeError):
    """Base MCP client error."""


class MCPServer(Protocol):
    """Protocol for MCP server implementations."""

    name: str

    def start(self) -> None:
        """Start the server if needed."""

    def shutdown(self) -> None:
        """Shut down the server."""

    def is_running(self) -> bool:
        """Return True if the server process is running."""

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions."""

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return text output."""


@dataclass
class MCPToolDefinition:
    """Normalized MCP tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPTool(BaseTool):
    """Tool wrapper for an MCP tool exposed by a server."""

    risk_level = "network"
    needs_network = True

    def __init__(
        self,
        host: "MCPHost",
        *,
        server_name: str,
        tool_name: str,
        display_name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> None:
        self._host = host
        self._server_name = server_name
        self._tool_name = tool_name
        self.name = display_name
        self.description = description
        self.input_schema = input_schema

    def execute(self, args: Dict[str, Any]) -> str:
        return self._host.call_tool(self._server_name, self._tool_name, args)


class MCPStdioServer:
    """JSON-RPC client for MCP servers over stdio with LSP framing."""

    _PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, config: MCPServerConfig, *, timeout_s: float = 30.0) -> None:
        self.config = config
        self.name = config.name
        self.timeout_s = float(timeout_s)
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._responses: Dict[int, Dict[str, Any]] = {}
        self._response_cond = threading.Condition()
        self._closed = False
        self._next_id = 0

    def start(self) -> None:
        if self.is_running():
            return
        self._spawn_process()
        self._start_reader()
        self._initialize_session()

    def shutdown(self) -> None:
        if not self._process:
            return
        self._closed = True
        try:
            if self.is_running():
                try:
                    self._request("shutdown", {})
                except MCPError:
                    pass
                self._notify("exit", {})
        finally:
            self._terminate_process()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def list_tools(self) -> List[Dict[str, Any]]:
        self.start()
        result = self._request("tools/list", {})
        if isinstance(result, dict):
            tools = result.get("tools")
            if isinstance(tools, list):
                return [tool for tool in tools if isinstance(tool, dict)]
        if isinstance(result, list):
            return [tool for tool in result if isinstance(tool, dict)]
        return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        self.start()
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": dict(arguments or {})},
        )
        return self._format_tool_result(result)

    def _spawn_process(self) -> None:
        args = [self.config.command] + list(self.config.args or [])
        env = os.environ.copy()
        env.update(self.config.env or {})
        try:
            self._process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise MCPError(f"MCP server command not found: {self.config.command}") from exc
        self._closed = False
        self._responses.clear()
        self._next_id = 0

    def _terminate_process(self) -> None:
        proc = self._process
        self._process = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            logger.exception("Failed to terminate MCP server '%s'", self.name)

    def _start_reader(self) -> None:
        if self._process is None or self._process.stdout is None:
            raise MCPError("MCP server stdout not available")
        thread = threading.Thread(target=self._reader_loop, daemon=True)
        thread.start()
        self._reader_thread = thread

    def _initialize_session(self) -> None:
        params = {
            "protocolVersion": self._PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "Local Second Mind",
                "version": "0.0",
            },
        }
        self._request("initialize", params)
        self._notify("initialized", {})

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_running():
            raise MCPError(f"MCP server '{self.name}' is not running")
        request_id = self._next_request_id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        self._send(payload)
        response = self._wait_for_response(request_id)
        if not isinstance(response, dict):
            raise MCPError(f"Malformed MCP response for {method}")
        if response.get("error"):
            raise MCPError(
                f"MCP server '{self.name}' error for {method}: {response['error']}"
            )
        return response.get("result")

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        if not self.is_running():
            return
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._send(payload)

    def _send(self, payload: Dict[str, Any]) -> None:
        proc = self._process
        if proc is None or proc.stdin is None:
            raise MCPError("MCP server stdin not available")
        data = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        with self._lock:
            try:
                proc.stdin.write(header + data)
                proc.stdin.flush()
            except BrokenPipeError as exc:
                raise MCPError("MCP server pipe closed") from exc

    def _reader_loop(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        while not self._closed:
            message = self._read_message(proc.stdout)
            if message is None:
                break
            if "id" not in message:
                continue
            request_id = message.get("id")
            if not isinstance(request_id, int):
                continue
            with self._response_cond:
                self._responses[request_id] = message
                self._response_cond.notify_all()

    def _read_message(self, stream) -> Optional[Dict[str, Any]]:
        headers: Dict[str, str] = {}
        while True:
            line = stream.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            decoded = line.decode("utf-8", errors="ignore")
            if ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        length_value = headers.get("content-length")
        if not length_value:
            return None
        try:
            length = int(length_value)
        except ValueError:
            return None
        body = stream.read(length)
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid MCP JSON message from '%s'", self.name)
            return None

    def _wait_for_response(self, request_id: int) -> Dict[str, Any]:
        deadline = time.time() + self.timeout_s
        with self._response_cond:
            while request_id not in self._responses:
                if self._closed or not self.is_running():
                    raise MCPError(f"MCP server '{self.name}' stopped responding")
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise MCPError(
                        f"Timed out waiting for MCP response ({self.name}, id={request_id})"
                    )
                self._response_cond.wait(timeout=remaining)
            return self._responses.pop(request_id)

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        if isinstance(result, dict):
            if result.get("isError"):
                raise MCPError(str(result.get("content") or "MCP tool error"))
            content = result.get("content")
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("text") is not None:
                            parts.append(str(part.get("text")))
                        elif part.get("data") is not None:
                            parts.append(str(part.get("data")))
                        else:
                            parts.append(json.dumps(part, ensure_ascii=True))
                    else:
                        parts.append(str(part))
                return "\n".join([p for p in parts if p]).strip()
            return json.dumps(result, ensure_ascii=True)
        if isinstance(result, list):
            return json.dumps(result, ensure_ascii=True)
        return str(result)


class MCPHost:
    """Host wrapper for MCP server tool discovery and execution."""

    def __init__(
        self,
        servers: Iterable[MCPServerConfig],
        *,
        server_factory: Optional[Callable[[MCPServerConfig], MCPServer]] = None,
    ) -> None:
        self._server_configs = {server.name: server for server in servers}
        self._server_factory = server_factory or (lambda cfg: MCPStdioServer(cfg))
        self._servers: Dict[str, MCPServer] = {}
        atexit.register(self.shutdown_all)

    def register_tools(self, registry: ToolRegistry) -> None:
        for server_name, server_cfg in self._server_configs.items():
            server = self._get_server(server_name, server_cfg)
            try:
                tools = self._list_tools_with_restart(server)
            except MCPError as exc:
                logger.warning("Failed to load MCP tools from '%s': %s", server_name, exc)
                continue
            for tool_def in self._normalize_tool_defs(tools):
                tool_name = self._format_tool_name(server_name, tool_def.name)
                mcp_tool = MCPTool(
                    self,
                    server_name=server_name,
                    tool_name=tool_def.name,
                    display_name=tool_name,
                    description=tool_def.description,
                    input_schema=tool_def.input_schema,
                )
                registry.register(mcp_tool)

    def call_tool(self, server_name: str, tool_name: str, args: Dict[str, Any]) -> str:
        server_cfg = self._server_configs.get(server_name)
        if server_cfg is None:
            raise MCPError(f"Unknown MCP server: {server_name}")
        server = self._get_server(server_name, server_cfg)
        return self._call_tool_with_restart(server, tool_name, args)

    def shutdown_all(self) -> None:
        for server in self._servers.values():
            try:
                server.shutdown()
            except Exception:
                logger.exception("Failed to shutdown MCP server '%s'", server.name)

    def _get_server(self, name: str, config: MCPServerConfig) -> MCPServer:
        if name not in self._servers:
            self._servers[name] = self._server_factory(config)
        return self._servers[name]

    def _call_tool_with_restart(
        self,
        server: MCPServer,
        tool_name: str,
        args: Dict[str, Any],
    ) -> str:
        for attempt in range(2):
            try:
                return server.call_tool(tool_name, args)
            except Exception as exc:
                if attempt == 0 and not server.is_running():
                    logger.warning(
                        "Restarting MCP server '%s' after failure: %s",
                        server.name,
                        exc,
                    )
                    server.start()
                    continue
                raise
        raise MCPError(f"Failed to call MCP tool '{tool_name}' on '{server.name}'")

    def _list_tools_with_restart(self, server: MCPServer) -> List[Dict[str, Any]]:
        for attempt in range(2):
            try:
                return server.list_tools()
            except Exception as exc:
                if attempt == 0 and not server.is_running():
                    logger.warning(
                        "Restarting MCP server '%s' after list failure: %s",
                        server.name,
                        exc,
                    )
                    server.start()
                    continue
                raise
        return []

    def _normalize_tool_defs(self, tools: Iterable[Dict[str, Any]]) -> List[MCPToolDefinition]:
        normalized: List[MCPToolDefinition] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            description = str(tool.get("description", "")).strip()
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            if not isinstance(input_schema, dict):
                input_schema = {}
            normalized.append(
                MCPToolDefinition(
                    name=name,
                    description=description or f"MCP tool '{name}'",
                    input_schema=input_schema,
                )
            )
        return normalized

    @staticmethod
    def _format_tool_name(server_name: str, tool_name: str) -> str:
        safe_server = _sanitize_tool_component(server_name)
        safe_tool = _sanitize_tool_component(tool_name)
        return f"mcp_{safe_server}_{safe_tool}"


def _sanitize_tool_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "tool"

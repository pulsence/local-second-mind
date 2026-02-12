"""
Shell-level agent command helpers.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Optional

from lsm.agents import create_agent
from lsm.agents.log_formatter import format_agent_log
from lsm.agents.memory import create_memory_store
from lsm.agents.models import AgentContext
from lsm.agents.tools import ToolSandbox, create_default_tool_registry


class AgentRuntimeManager:
    """
    Manage active agent runtime instances for UI command surfaces.
    """

    def __init__(self) -> None:
        self._active_agent = None
        self._active_thread: Optional[threading.Thread] = None
        self._active_name: Optional[str] = None
        self._lock = threading.Lock()

    def start(self, app: Any, agent_name: str, topic: str) -> str:
        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                return (
                    f"Agent '{self._active_name}' is already running. "
                    "Use /agent status or /agent stop first.\n"
                )

            agent_cfg = getattr(app.config, "agents", None)
            if agent_cfg is None or not getattr(agent_cfg, "enabled", False):
                return "Agents are disabled. Enable `agents.enabled` in config.\n"

            memory_store = None
            if getattr(agent_cfg, "memory", None) is not None and agent_cfg.memory.enabled:
                try:
                    memory_store = create_memory_store(agent_cfg, app.config.vectordb)
                except Exception as exc:
                    return f"Failed to initialize agent memory store: {exc}\n"

            tool_registry = create_default_tool_registry(
                app.config,
                collection=getattr(app, "query_provider", None),
                embedder=getattr(app, "query_embedder", None),
                batch_size=getattr(app.config, "batch_size", 32),
                memory_store=memory_store,
            )
            sandbox = ToolSandbox(agent_cfg.sandbox)
            agent = create_agent(
                name=agent_name,
                llm_registry=app.config.llm,
                tool_registry=tool_registry,
                sandbox=sandbox,
                agent_config=agent_cfg,
            )
            context = AgentContext(
                messages=[{"role": "user", "content": topic}],
                tool_definitions=tool_registry.list_definitions(),
                budget_tracking={
                    "tokens_used": 0,
                    "max_tokens_budget": agent_cfg.max_tokens_budget,
                    "started_at": datetime.utcnow().isoformat(),
                },
            )

            def _run() -> None:
                try:
                    agent.run(context)
                finally:
                    if memory_store is not None:
                        memory_store.close()

            thread = threading.Thread(
                target=_run,
                daemon=True,
                name=f"Agent-{agent_name}",
            )
            self._active_agent = agent
            self._active_thread = thread
            self._active_name = agent_name
            thread.start()
            return f"Started agent '{agent_name}' with topic: {topic}\n"

    def status(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        alive = bool(self._active_thread and self._active_thread.is_alive())
        status = self._active_agent.state.status.value
        return (
            f"Agent: {self._active_name}\n"
            f"Status: {status}\n"
            f"Thread alive: {alive}\n"
            f"Logs: {len(self._active_agent.state.log_entries)} entries\n"
        )

    def stop(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        self._active_agent.stop()
        return f"Stop requested for agent '{self._active_name}'.\n"

    def pause(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        self._active_agent.pause()
        return f"Paused agent '{self._active_name}'.\n"

    def resume(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        self._active_agent.resume()
        return f"Resumed agent '{self._active_name}'.\n"

    def log(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        entries = self._active_agent.state.log_entries
        if not entries:
            return "No log entries yet.\n"
        return format_agent_log(entries)

    def get_active_agent(self):
        """Return the active agent instance for UI screens."""
        return self._active_agent


_MANAGER = AgentRuntimeManager()


def get_agent_runtime_manager() -> AgentRuntimeManager:
    """Return the singleton agent runtime manager."""
    return _MANAGER


def handle_agent_command(command: str, app: Any) -> str:
    """
    Parse and execute `/agent ...` commands.

    Supported:
    - `/agent start <name> <topic...>`
    - `/agent status`
    - `/agent stop`
    - `/agent pause`
    - `/agent resume`
    - `/agent log`
    """
    text = command.strip()
    parts = text.split()
    if len(parts) < 2:
        return _agent_help()

    manager = get_agent_runtime_manager()
    action = parts[1].lower()
    if action == "start":
        if len(parts) < 4:
            return "Usage: /agent start <name> <topic>\n"
        name = parts[2].strip()
        topic = text.split(maxsplit=3)[3].strip()
        if not topic:
            return "Usage: /agent start <name> <topic>\n"
        return manager.start(app, name, topic)
    if action == "status":
        return manager.status()
    if action == "stop":
        return manager.stop()
    if action == "pause":
        return manager.pause()
    if action == "resume":
        return manager.resume()
    if action == "log":
        return manager.log()

    return _agent_help()


def _agent_help() -> str:
    return (
        "Agent commands:\n"
        "  /agent start <name> <topic>\n"
        "  /agent status\n"
        "  /agent stop\n"
        "  /agent pause\n"
        "  /agent resume\n"
        "  /agent log\n"
    )

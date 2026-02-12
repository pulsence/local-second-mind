"""
Agent system configuration models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_VALID_RISK_LEVELS = {"read_only", "writes_workspace", "network", "exec"}
_VALID_EXECUTION_MODES = {"local_only", "prefer_docker"}
_DEFAULT_SANDBOX_LIMITS = {
    "timeout_s_default": 30.0,
    "max_stdout_kb": 256,
    "max_file_write_mb": 10.0,
}
_DEFAULT_SANDBOX_DOCKER = {
    "enabled": False,
    "image": "lsm-agent-sandbox:latest",
    "network_default": "none",
    "cpu_limit": 1.0,
    "mem_limit_mb": 512,
    "read_only_root": True,
}
_VALID_MEMORY_BACKENDS = {"auto", "sqlite", "postgresql"}


@dataclass
class SandboxConfig:
    """
    Agent tool sandbox configuration.
    """

    allowed_read_paths: List[Path] = field(default_factory=list)
    """Paths that tools may read from."""

    allowed_write_paths: List[Path] = field(default_factory=list)
    """Paths that tools may write to."""

    allow_url_access: bool = False
    """Whether URL fetch tools are allowed."""

    require_user_permission: Dict[str, bool] = field(default_factory=dict)
    """Per-tool permission gate (`tool_name -> bool`)."""

    require_permission_by_risk: Dict[str, bool] = field(default_factory=dict)
    """Per-risk permission gate (`risk_level -> bool`)."""

    execution_mode: str = "local_only"
    """Runner selection policy (`local_only` or `prefer_docker`)."""

    limits: Dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_SANDBOX_LIMITS)
    )
    """Sandbox runtime limits (`timeout_s_default`, `max_stdout_kb`, `max_file_write_mb`)."""

    docker: Dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_SANDBOX_DOCKER)
    )
    """Docker runner configuration (reserved for sandbox container execution)."""

    tool_llm_assignments: Dict[str, str] = field(default_factory=dict)
    """Per-tool LLM service assignment (`tool_name -> service_name`)."""

    def __post_init__(self) -> None:
        self.allowed_read_paths = [Path(path).expanduser() for path in self.allowed_read_paths]
        self.allowed_write_paths = [Path(path).expanduser() for path in self.allowed_write_paths]
        self.require_user_permission = {
            str(key).strip(): bool(value)
            for key, value in self.require_user_permission.items()
            if str(key).strip()
        }
        self.require_permission_by_risk = {
            str(key).strip().lower(): bool(value)
            for key, value in self.require_permission_by_risk.items()
            if str(key).strip()
        }
        self.execution_mode = str(self.execution_mode or "local_only").strip().lower()
        limits = self.limits if isinstance(self.limits, dict) else {}
        docker = self.docker if isinstance(self.docker, dict) else {}
        self.limits = dict(_DEFAULT_SANDBOX_LIMITS)
        self.limits.update(limits)
        self.docker = dict(_DEFAULT_SANDBOX_DOCKER)
        self.docker.update(docker)
        self.tool_llm_assignments = {
            str(key).strip(): str(value).strip()
            for key, value in self.tool_llm_assignments.items()
            if str(key).strip()
        }

    def validate(self) -> None:
        """Validate sandbox configuration."""
        for idx, path in enumerate(self.allowed_read_paths):
            if not str(path).strip():
                raise ValueError(f"sandbox.allowed_read_paths[{idx}] cannot be empty")
        for idx, path in enumerate(self.allowed_write_paths):
            if not str(path).strip():
                raise ValueError(f"sandbox.allowed_write_paths[{idx}] cannot be empty")
        for risk_level in self.require_permission_by_risk.keys():
            if risk_level not in _VALID_RISK_LEVELS:
                raise ValueError(
                    "sandbox.require_permission_by_risk keys must be one of: "
                    "read_only, writes_workspace, network, exec"
                )
        if self.execution_mode not in _VALID_EXECUTION_MODES:
            raise ValueError("sandbox.execution_mode must be 'local_only' or 'prefer_docker'")
        try:
            timeout_s = float(self.limits.get("timeout_s_default", 30.0))
            max_stdout_kb = int(self.limits.get("max_stdout_kb", 256))
            max_file_write_mb = float(self.limits.get("max_file_write_mb", 10.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("sandbox.limits values must be numeric") from exc
        if timeout_s <= 0:
            raise ValueError("sandbox.limits.timeout_s_default must be > 0")
        if max_stdout_kb < 1:
            raise ValueError("sandbox.limits.max_stdout_kb must be >= 1")
        if max_file_write_mb <= 0:
            raise ValueError("sandbox.limits.max_file_write_mb must be > 0")


@dataclass
class MemoryConfig:
    """
    Agent memory storage configuration.
    """

    enabled: bool = True
    """Whether persistent memory storage is enabled."""

    storage_backend: str = "auto"
    """Backend selection: 'auto', 'sqlite', or 'postgresql'."""

    sqlite_path: Path = Path("memory.sqlite3")
    """SQLite memory database path."""

    postgres_connection_string: Optional[str] = None
    """Optional PostgreSQL connection string override."""

    postgres_table_prefix: str = "agent_memory"
    """Table name prefix for PostgreSQL memory tables."""

    ttl_project_fact_days: int = 90
    """TTL cap for project_fact memories."""

    ttl_task_state_days: int = 7
    """TTL cap for task_state memories."""

    ttl_cache_hours: int = 24
    """TTL cap for cache memories."""

    def __post_init__(self) -> None:
        self.storage_backend = str(self.storage_backend or "auto").strip().lower()
        self.sqlite_path = Path(self.sqlite_path).expanduser()
        self.postgres_connection_string = (
            str(self.postgres_connection_string).strip()
            if self.postgres_connection_string is not None
            else None
        )
        self.postgres_table_prefix = (
            str(self.postgres_table_prefix or "agent_memory")
            .strip()
            .lower()
        )
        if not self.postgres_table_prefix:
            self.postgres_table_prefix = "agent_memory"

    def validate(self) -> None:
        """Validate memory configuration."""
        if self.storage_backend not in _VALID_MEMORY_BACKENDS:
            raise ValueError(
                "agents.memory.storage_backend must be one of: "
                "auto, sqlite, postgresql"
            )
        if self.ttl_project_fact_days < 1:
            raise ValueError("agents.memory.ttl_project_fact_days must be >= 1")
        if self.ttl_task_state_days < 1:
            raise ValueError("agents.memory.ttl_task_state_days must be >= 1")
        if self.ttl_cache_hours < 1:
            raise ValueError("agents.memory.ttl_cache_hours must be >= 1")

    def ttl_cap_for_type(self, memory_type: str) -> Optional[timedelta]:
        """
        Return TTL cap for a memory type.
        """
        normalized = str(memory_type or "").strip().lower()
        if normalized == "pinned":
            return None
        if normalized == "project_fact":
            return timedelta(days=int(self.ttl_project_fact_days))
        if normalized == "task_state":
            return timedelta(days=int(self.ttl_task_state_days))
        if normalized == "cache":
            return timedelta(hours=int(self.ttl_cache_hours))
        raise ValueError(
            "memory_type must be one of: pinned, project_fact, task_state, cache"
        )


@dataclass
class AgentConfig:
    """
    Top-level configuration for the agent system.
    """

    enabled: bool = False
    """Whether the agent runtime is enabled."""

    agents_folder: Path = Path("Agents")
    """Folder where agent logs/state are written."""

    max_tokens_budget: int = 200_000
    """Maximum total token budget per run."""

    max_iterations: int = 25
    """Maximum tool/model loop iterations per run."""

    context_window_strategy: str = "compact"
    """Context strategy: 'compact' or 'fresh'."""

    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    """Sandbox policy for tool execution."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    """Persistent memory storage configuration."""

    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-agent overrides keyed by agent name."""

    def __post_init__(self) -> None:
        self.agents_folder = Path(self.agents_folder).expanduser()
        self.context_window_strategy = (self.context_window_strategy or "compact").strip().lower()

    def validate(self) -> None:
        """Validate agent configuration."""
        if self.max_tokens_budget < 1:
            raise ValueError("agents.max_tokens_budget must be positive")
        if self.max_iterations < 1:
            raise ValueError("agents.max_iterations must be positive")
        if self.context_window_strategy not in {"compact", "fresh"}:
            raise ValueError("agents.context_window_strategy must be 'compact' or 'fresh'")
        if not isinstance(self.agent_configs, dict):
            raise ValueError("agents.agent_configs must be a dict")
        self.sandbox.validate()
        self.memory.validate()

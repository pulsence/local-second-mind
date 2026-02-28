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
_DEFAULT_SANDBOX_WSL2 = {
    "enabled": False,
    "distro": "",
    "wsl_bin": "wsl",
    "shell": "bash",
}
_VALID_MEMORY_BACKENDS = {"auto", "sqlite", "postgresql"}
_VALID_SCHEDULE_CONCURRENCY_POLICIES = {"skip", "queue", "cancel"}
_VALID_SCHEDULE_CONFIRMATION_MODES = {"auto", "confirm", "deny"}
_VALID_INTERACTION_TIMEOUT_ACTIONS = {"deny", "approve"}
_SCHEDULE_INTERVAL_ALIASES = {"hourly", "daily", "weekly"}


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

    force_docker: bool = False
    """Require Docker runner for all tool risks; blocks execution when unavailable."""

    limits: Dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_SANDBOX_LIMITS)
    )
    """Sandbox runtime limits (`timeout_s_default`, `max_stdout_kb`, `max_file_write_mb`)."""

    docker: Dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_SANDBOX_DOCKER)
    )
    """Docker runner configuration (reserved for sandbox container execution)."""

    wsl2: Dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_SANDBOX_WSL2)
    )
    """WSL2 runner configuration for Windows-hosted execution."""

    command_allowlist: List[str] = field(default_factory=list)
    """Allowed command prefixes for shell tools."""

    command_denylist: List[str] = field(default_factory=list)
    """Denied command prefixes for shell tools."""

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
        self.force_docker = bool(self.force_docker)
        limits = self.limits if isinstance(self.limits, dict) else {}
        docker = self.docker if isinstance(self.docker, dict) else {}
        wsl2 = self.wsl2 if isinstance(self.wsl2, dict) else {}
        self.limits = dict(_DEFAULT_SANDBOX_LIMITS)
        self.limits.update(limits)
        self.docker = dict(_DEFAULT_SANDBOX_DOCKER)
        self.docker.update(docker)
        self.wsl2 = dict(_DEFAULT_SANDBOX_WSL2)
        self.wsl2.update(wsl2)
        self.command_allowlist = [
            str(cmd).strip() for cmd in (self.command_allowlist or [])
        ]
        self.command_denylist = [
            str(cmd).strip() for cmd in (self.command_denylist or [])
        ]
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
        for idx, cmd in enumerate(self.command_allowlist):
            if not str(cmd).strip():
                raise ValueError(f"sandbox.command_allowlist[{idx}] cannot be empty")
        for idx, cmd in enumerate(self.command_denylist):
            if not str(cmd).strip():
                raise ValueError(f"sandbox.command_denylist[{idx}] cannot be empty")


@dataclass
class MemoryConfig:
    """
    Agent memory storage configuration.
    """

    enabled: bool = True
    """Whether persistent memory storage is enabled."""

    storage_backend: str = "auto"
    """Backend selection: 'auto', 'sqlite', or 'postgresql'."""

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
class ScheduleConfig:
    """
    Agent scheduler entry configuration.
    """

    agent_name: str
    """Registered agent name to execute on schedule."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary schedule parameters passed to the scheduled run."""

    interval: str = "daily"
    """Run interval: alias (`hourly|daily|weekly`), seconds (`3600s`), or cron syntax."""

    enabled: bool = True
    """Whether this schedule is active."""

    concurrency_policy: str = "skip"
    """Overlap policy: `skip`, `queue`, or `cancel`."""

    confirmation_mode: str = "auto"
    """Approval mode for scheduled actions: `auto`, `confirm`, or `deny`."""

    def __post_init__(self) -> None:
        self.agent_name = str(self.agent_name or "").strip()
        self.interval = str(self.interval or "").strip().lower()
        self.concurrency_policy = str(self.concurrency_policy or "skip").strip().lower()
        self.confirmation_mode = str(self.confirmation_mode or "auto").strip().lower()
        if self.params is None:
            self.params = {}

    def validate(self) -> None:
        """Validate schedule configuration."""
        if not self.agent_name:
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(self.params, dict):
            raise ValueError("params must be an object")
        if not self.interval:
            raise ValueError("interval must be non-empty")

        if self.interval in _SCHEDULE_INTERVAL_ALIASES:
            pass
        elif self.interval.endswith("s") and self.interval[:-1].isdigit():
            if int(self.interval[:-1]) < 1:
                raise ValueError("interval seconds value must be >= 1")
        else:
            # Cron support is syntax-agnostic at config layer; parsing is scheduler-engine responsibility.
            parts = self.interval.split()
            if len(parts) not in {5, 6}:
                raise ValueError(
                    "interval must be one of: hourly, daily, weekly, '<seconds>s', or cron syntax"
                )

        if self.concurrency_policy not in _VALID_SCHEDULE_CONCURRENCY_POLICIES:
            raise ValueError("concurrency_policy must be one of: skip, queue, cancel")
        if self.confirmation_mode not in _VALID_SCHEDULE_CONFIRMATION_MODES:
            raise ValueError("confirmation_mode must be one of: auto, confirm, deny")


@dataclass
class InteractionConfig:
    """
    Agent interaction channel configuration.
    """

    timeout_seconds: int = 300
    """Maximum seconds to wait for a user interaction response."""

    timeout_action: str = "deny"
    """Timeout behavior (`deny` or `approve`)."""

    auto_continue: bool = False
    """Auto-respond to ask_user prompts with a continue message."""

    acknowledged_timeout_seconds: int = 0
    """Timeout after acknowledgment; 0 means infinite (no timeout once acknowledged)."""

    def __post_init__(self) -> None:
        self.timeout_seconds = int(self.timeout_seconds)
        self.timeout_action = str(self.timeout_action or "deny").strip().lower()
        self.auto_continue = bool(self.auto_continue)
        self.acknowledged_timeout_seconds = int(self.acknowledged_timeout_seconds)

    def validate(self) -> None:
        """Validate interaction configuration."""
        if self.timeout_seconds < 1:
            raise ValueError("agents.interaction.timeout_seconds must be >= 1")
        if self.timeout_action not in _VALID_INTERACTION_TIMEOUT_ACTIONS:
            raise ValueError(
                "agents.interaction.timeout_action must be one of: deny, approve"
            )
        if self.acknowledged_timeout_seconds < 0:
            raise ValueError("agents.interaction.acknowledged_timeout_seconds must be >= 0")


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

    max_concurrent: int = 5
    """Maximum number of concurrent agent runs."""

    log_stream_queue_limit: int = 500
    """Maximum buffered log entries per agent stream before oldest entries are dropped."""

    context_window_strategy: str = "compact"
    """Context strategy: 'compact' or 'fresh'."""

    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    """Sandbox policy for tool execution."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    """Persistent memory storage configuration."""

    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    """Interaction request timeout and fallback policy."""

    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Per-agent overrides keyed by agent name."""

    schedules: List[ScheduleConfig] = field(default_factory=list)
    """Configured scheduled agent runs."""

    def __post_init__(self) -> None:
        self.agents_folder = Path(self.agents_folder).expanduser()
        self.context_window_strategy = (self.context_window_strategy or "compact").strip().lower()
        self.max_concurrent = int(self.max_concurrent)
        self.log_stream_queue_limit = int(self.log_stream_queue_limit)
        if isinstance(self.interaction, dict):
            self.interaction = InteractionConfig(**self.interaction)
        elif not isinstance(self.interaction, InteractionConfig):
            raise ValueError("agents.interaction must be an object")
        normalized_schedules: List[ScheduleConfig] = []
        for entry in self.schedules or []:
            if isinstance(entry, ScheduleConfig):
                normalized_schedules.append(entry)
            elif isinstance(entry, dict):
                normalized_schedules.append(ScheduleConfig(**entry))
            else:
                raise ValueError("agents.schedules entries must be objects")
        self.schedules = normalized_schedules

    def validate(self) -> None:
        """Validate agent configuration."""
        if self.max_tokens_budget < 1:
            raise ValueError("agents.max_tokens_budget must be positive")
        if self.max_iterations < 1:
            raise ValueError("agents.max_iterations must be positive")
        if self.max_concurrent < 1:
            raise ValueError("agents.max_concurrent must be >= 1")
        if self.log_stream_queue_limit < 1:
            raise ValueError("agents.log_stream_queue_limit must be >= 1")
        if self.context_window_strategy not in {"compact", "fresh"}:
            raise ValueError("agents.context_window_strategy must be 'compact' or 'fresh'")
        if not isinstance(self.agent_configs, dict):
            raise ValueError("agents.agent_configs must be a dict")
        if not isinstance(self.schedules, list):
            raise ValueError("agents.schedules must be a list")
        for idx, schedule in enumerate(self.schedules):
            if not isinstance(schedule, ScheduleConfig):
                raise ValueError(f"agents.schedules[{idx}] must be an object")
            try:
                schedule.validate()
            except ValueError as exc:
                raise ValueError(f"agents.schedules[{idx}] {exc}") from exc
        self.sandbox.validate()
        self.memory.validate()
        self.interaction.validate()

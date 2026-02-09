"""
Agent system configuration models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

_VALID_RISK_LEVELS = {"read_only", "writes_workspace", "network", "exec"}


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

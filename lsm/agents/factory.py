"""
Agent factory and registry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig

from .base import BaseAgent
from .academic import AGENT_SPECS as ACADEMIC_SPECS
from .assistants import AGENT_SPECS as ASSISTANTS_SPECS
from .meta import AGENT_SPECS as META_SPECS
from .productivity import AGENT_SPECS as PRODUCTIVITY_SPECS
from .tools.base import ToolRegistry
from .tools.sandbox import ToolSandbox


AgentBuilder = Callable[
    [LLMRegistryConfig, ToolRegistry, ToolSandbox, AgentConfig, Optional[dict]],
    BaseAgent,
]


@dataclass(frozen=True)
class AgentRegistryEntry:
    """
    Metadata describing a registered agent builder.
    """

    name: str
    builder: AgentBuilder
    theme: str
    category: str


@dataclass(frozen=True)
class AgentRegistryGroup:
    """
    Grouped registry entries for UI presentation.
    """

    theme: str
    entries: tuple[AgentRegistryEntry, ...]


def _load_agent_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for group in (
        ACADEMIC_SPECS,
        PRODUCTIVITY_SPECS,
        META_SPECS,
        ASSISTANTS_SPECS,
    ):
        specs.extend(group)
    return specs


class AgentRegistry:
    """
    Registry for built-in and custom agent constructors.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, AgentRegistryEntry] = {}
        for spec in _load_agent_specs():
            self.register_spec(spec)

    def register(
        self,
        name: str,
        builder: AgentBuilder,
        *,
        theme: str,
        category: str,
    ) -> None:
        normalized = str(name).strip().lower()
        if not normalized:
            raise ValueError("Agent name cannot be empty")
        entry = AgentRegistryEntry(
            name=normalized,
            builder=builder,
            theme=str(theme or "Other").strip() or "Other",
            category=str(category or normalized).strip() or normalized,
        )
        self._entries[normalized] = entry

    def register_spec(self, spec: dict[str, Any]) -> None:
        name = str(spec.get("name", "")).strip().lower()
        agent_cls = spec.get("agent_cls")
        theme = spec.get("theme", "Other")
        category = spec.get("category", name)
        if not name:
            raise ValueError("Agent spec missing name")
        if agent_cls is None:
            raise ValueError(f"Agent spec '{name}' missing agent_cls")

        def _builder(
            llm_registry: LLMRegistryConfig,
            tool_registry: ToolRegistry,
            sandbox: ToolSandbox,
            agent_config: AgentConfig,
            overrides: Optional[dict],
            _agent_cls=agent_cls,
            _name=name,
        ) -> BaseAgent:
            if overrides and "enabled" in overrides and not bool(overrides["enabled"]):
                raise ValueError(f"Agent '{_name}' is disabled by configuration override")
            if overrides and "max_iterations" in overrides:
                agent_config = replace(
                    agent_config,
                    max_iterations=int(overrides["max_iterations"]),
                )
            return _agent_cls(
                llm_registry=llm_registry,
                tool_registry=tool_registry,
                sandbox=sandbox,
                agent_config=agent_config,
                agent_overrides=overrides,
            )

        self.register(
            name,
            _builder,
            theme=theme,
            category=category,
        )

    def list_agents(self) -> list[str]:
        return sorted(self._entries.keys())

    def list_entries(self) -> list[AgentRegistryEntry]:
        return sorted(
            self._entries.values(),
            key=lambda entry: (entry.theme.lower(), entry.category.lower(), entry.name),
        )

    def list_groups(self) -> list[AgentRegistryGroup]:
        grouped: dict[str, list[AgentRegistryEntry]] = {}
        for entry in self.list_entries():
            grouped.setdefault(entry.theme, []).append(entry)
        return [
            AgentRegistryGroup(theme=theme, entries=tuple(grouped[theme]))
            for theme in sorted(grouped.keys(), key=str.lower)
        ]

    def get_entry(self, name: str) -> Optional[AgentRegistryEntry]:
        normalized = str(name).strip().lower()
        if not normalized:
            return None
        return self._entries.get(normalized)

    def create(
        self,
        name: str,
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
    ) -> BaseAgent:
        normalized = str(name).strip().lower()
        if normalized not in self._entries:
            raise ValueError(
                f"Unknown agent '{name}'. Available agents: {self.list_agents()}"
            )
        overrides = None
        if isinstance(agent_config.agent_configs, dict):
            overrides = agent_config.agent_configs.get(normalized)
        return self._entries[normalized].builder(
            llm_registry,
            tool_registry,
            sandbox,
            agent_config,
            overrides,
        )


_DEFAULT_REGISTRY = AgentRegistry()


def create_agent(
    name: str,
    llm_registry: LLMRegistryConfig,
    tool_registry: ToolRegistry,
    sandbox: ToolSandbox,
    agent_config: AgentConfig,
) -> BaseAgent:
    """
    Create an agent instance by name.
    """
    return _DEFAULT_REGISTRY.create(
        name=name,
        llm_registry=llm_registry,
        tool_registry=tool_registry,
        sandbox=sandbox,
        agent_config=agent_config,
    )

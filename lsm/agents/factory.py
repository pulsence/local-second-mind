"""
Agent factory and registry.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Optional, Type

from lsm.config.models import AgentConfig, LLMRegistryConfig

from .base import BaseAgent
from .research import ResearchAgent
from .tools.base import ToolRegistry
from .tools.sandbox import ToolSandbox


AgentBuilder = Callable[
    [LLMRegistryConfig, ToolRegistry, ToolSandbox, AgentConfig, Optional[dict]],
    BaseAgent,
]


class AgentRegistry:
    """
    Registry for built-in and custom agent constructors.
    """

    def __init__(self) -> None:
        self._builders: Dict[str, AgentBuilder] = {}
        self.register("research", self._build_research_agent)

    def register(self, name: str, builder: AgentBuilder) -> None:
        normalized = str(name).strip().lower()
        if not normalized:
            raise ValueError("Agent name cannot be empty")
        self._builders[normalized] = builder

    def list_agents(self) -> list[str]:
        return sorted(self._builders.keys())

    def create(
        self,
        name: str,
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
    ) -> BaseAgent:
        normalized = str(name).strip().lower()
        if normalized not in self._builders:
            raise ValueError(
                f"Unknown agent '{name}'. Available agents: {self.list_agents()}"
            )
        overrides = None
        if isinstance(agent_config.agent_configs, dict):
            overrides = agent_config.agent_configs.get(normalized)
        return self._builders[normalized](
            llm_registry,
            tool_registry,
            sandbox,
            agent_config,
            overrides,
        )

    @staticmethod
    def _build_research_agent(
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
        overrides: Optional[dict],
    ) -> BaseAgent:
        if overrides and "enabled" in overrides and not bool(overrides["enabled"]):
            raise ValueError("Agent 'research' is disabled by configuration override")
        if overrides and "max_iterations" in overrides:
            agent_config = replace(agent_config, max_iterations=int(overrides["max_iterations"]))
        return ResearchAgent(
            llm_registry=llm_registry,
            tool_registry=tool_registry,
            sandbox=sandbox,
            agent_config=agent_config,
            agent_overrides=overrides,
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


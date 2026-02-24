"""
General agent implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace


class GeneralAgent(BaseAgent):
    """
    General-purpose task agent for tool-driven workflows.
    """

    name = "general"
    tier = "normal"
    description = "General-purpose agent for multi-step tasks using local tools."
    tool_allowlist = {
        "read_file",
        "read_folder",
        "find_file",
        "find_section",
        "file_metadata",
        "hash_file",
        "write_file",
        "append_file",
        "create_folder",
        "edit_file",
        "query_embeddings",
        "extract_snippets",
        "source_map",
    }
    risk_posture = "writes_workspace"

    def __init__(
        self,
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
        agent_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=self.name, description=self.description)
        self.llm_registry = llm_registry
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.agent_config = agent_config
        self.agent_overrides = agent_overrides or {}
        self.max_iterations = int(
            self.agent_overrides.get("max_iterations", self.agent_config.max_iterations)
        )
        self.max_tokens_budget = int(
            self.agent_overrides.get(
                "max_tokens_budget",
                self.agent_config.max_tokens_budget,
            )
        )

    def run(self, initial_context: AgentContext) -> Any:
        """
        Placeholder run loop for the General agent.
        """
        self.state.set_status(AgentStatus.RUNNING)
        ensure_agent_workspace(
            self.name,
            self.agent_config.agents_folder,
            sandbox=self.sandbox,
        )
        self.state.current_task = "General task"
        self._log("General agent placeholder run.")
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

"""
General agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Any, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig, LSMConfig

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox


@dataclass
class GeneralResult:
    """
    Structured result payload for a general agent run.
    """

    topic: str
    summary_path: Path
    artifacts: list[str]


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
        "query_knowledge_base",
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
        agent_overrides: Optional[dict[str, Any]] = None,
        lsm_config: Optional[LSMConfig] = None,
    ) -> None:
        super().__init__(name=self.name, description=self.description)
        self.llm_registry = llm_registry
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.agent_config = agent_config
        self.agent_overrides = agent_overrides or {}
        self.lsm_config = lsm_config
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
        Run the general-purpose tool loop and emit a summary artifact.
        """
        self._reset_harness()
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        self._workspace_root()

        topic = self._extract_topic(initial_context)
        self.state.current_task = f"General: {topic}"

        try:
            self._run_phase(
                system_prompt=self._system_prompt(),
                user_message=topic,
                max_iterations=self.max_iterations,
            )
        except Exception as exc:
            self._log(f"Phase execution failed: {exc}")
            self.state.set_status(AgentStatus.FAILED)
        else:
            if self._harness is not None:
                self.state = self._harness.state
            self.state.set_status(AgentStatus.COMPLETED)

        summary_path = self._write_summary(topic)
        self.last_result = GeneralResult(
            topic=topic,
            summary_path=summary_path,
            artifacts=list(self.state.artifacts),
        )
        return self.state

    def _extract_topic(self, context: AgentContext) -> str:
        for message in reversed(context.messages):
            if str(message.get("role", "")).lower() == "user":
                topic = str(message.get("content", "")).strip()
                if topic:
                    return topic
        return "Untitled General Task"

    def _system_prompt(self) -> str:
        return (
            "You are the General agent. Use tools sparingly and only when needed. "
            "Respect sandbox boundaries, summarize progress clearly, and stop when "
            "the task is complete."
        )

    def _write_summary(self, topic: str) -> Path:
        safe_topic = re.sub(r"[^\w\-]", "_", topic.strip())[:80] or "general"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        summary_path = self._artifacts_dir() / f"general_{safe_topic}_{timestamp}_summary.md"

        status = self.state.status.value
        tools_used: dict[str, int] = {}
        if self._harness is not None:
            tools_used = dict(getattr(self._harness, "_tool_usage_counts", {}))
        artifacts = list(self.state.artifacts)

        lines = [
            "# General Agent Summary",
            "",
            f"- Topic: {topic}",
            f"- Status: {status}",
            "",
            "## Tools Used",
        ]
        if tools_used:
            for tool_name, count in tools_used.items():
                lines.append(f"- {tool_name}: {count}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Artifacts")
        if artifacts:
            for artifact in artifacts:
                lines.append(f"- `{artifact}`")
        else:
            lines.append("- None")

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        self.state.add_artifact(str(summary_path))
        self._log(f"Saved general agent summary to {summary_path}")
        return summary_path

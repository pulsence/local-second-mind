"""
General agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace


@dataclass
class GeneralResult:
    """
    Structured result payload for a general agent run.
    """

    topic: str
    summary_path: Path
    run_summary_path: Optional[Path]
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
        Run the general-purpose tool loop and emit a summary artifact.
        """
        self._tokens_used = 0
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        ensure_agent_workspace(
            self.name,
            self.agent_config.agents_folder,
            sandbox=self.sandbox,
        )

        topic = self._extract_topic(initial_context)
        self.state.current_task = f"General: {topic}"

        effective_config = replace(
            self.agent_config,
            max_iterations=self.max_iterations,
            max_tokens_budget=self.max_tokens_budget,
        )
        llm_selection = self._get_llm_selection()
        from ..harness import AgentHarness

        harness = AgentHarness(
            effective_config,
            self.tool_registry,
            self.llm_registry,
            self.sandbox,
            agent_name=self.name,
            tool_allowlist=self.tool_allowlist,
            llm_service=llm_selection.get("service"),
            llm_tier=llm_selection.get("tier"),
            llm_provider=llm_selection.get("provider"),
            llm_model=llm_selection.get("model"),
            llm_temperature=llm_selection.get("temperature"),
            llm_max_tokens=llm_selection.get("max_tokens"),
            interaction_channel=self.sandbox.interaction_channel,
            system_prompt=self._system_prompt(),
        )
        state = harness.run(initial_context)
        self.state = state

        run_summary_path = self._resolve_run_summary_path(harness)
        summary_path = self._write_summary(topic, run_summary_path)
        artifacts = list(self.state.artifacts)
        self.last_result = GeneralResult(
            topic=topic,
            summary_path=summary_path,
            run_summary_path=run_summary_path,
            artifacts=artifacts,
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

    def _resolve_run_summary_path(self, harness: Any) -> Optional[Path]:
        state_path = harness.get_state_path()
        if state_path is None:
            return None
        summary_path = state_path.parent / "run_summary.json"
        if summary_path.exists():
            return summary_path
        return None

    def _write_summary(self, topic: str, run_summary_path: Optional[Path]) -> Path:
        safe_topic = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip()
        )
        safe_topic = safe_topic[:80] or "general"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        summary_path = (
            self.agent_config.agents_folder
            / f"{self.name}_{safe_topic}_{timestamp}_summary.md"
        )

        summary_payload = self._load_run_summary(run_summary_path)
        status = summary_payload.get("status", self.state.status.value)
        tools_used = summary_payload.get("tools_used") or {}
        artifacts = summary_payload.get("artifacts_created") or list(self.state.artifacts)
        if run_summary_path is not None:
            run_summary_str = str(run_summary_path)
            if run_summary_str not in artifacts:
                artifacts.append(run_summary_str)
        token_usage = summary_payload.get("token_usage") or {}

        lines = [
            "# General Agent Summary",
            "",
            f"- Topic: {topic}",
            f"- Status: {status}",
            f"- Iterations: {token_usage.get('iterations', 'unknown')}",
            f"- Tokens Used: {token_usage.get('tokens_used', 'unknown')}",
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

    def _load_run_summary(self, run_summary_path: Optional[Path]) -> dict[str, Any]:
        if run_summary_path is None:
            return {}
        try:
            return json.loads(run_summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

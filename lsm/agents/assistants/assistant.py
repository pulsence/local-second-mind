"""
Assistant agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class AssistantResult:
    """
    Structured result payload for assistant summaries.
    """

    summary_path: Path
    summary_json_path: Path
    log_path: Path
    memory_payloads: list[dict[str, Any]]


class AssistantAgent(BaseAgent):
    """
    Summarize cross-agent activity and suggest memory updates.
    """

    name = "assistant"
    tier = "normal"
    description = "Summarize recent agent activity and propose memory updates."
    tool_allowlist = {
        "memory_put",
        "memory_search",
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
        Summarize run summaries and propose memory updates.
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
        self.state.current_task = f"Assistant: {topic}"

        run_summaries = self._load_run_summaries()
        summary_payload = self._build_summary_payload(run_summaries)
        run_dir = self._resolve_output_dir(topic, initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_json_path = run_dir / "assistant_summary.json"
        summary_json_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        summary_path = run_dir / "assistant_summary.md"
        summary_path.write_text(
            self._format_summary_markdown(summary_payload),
            encoding="utf-8",
        )
        self.state.add_artifact(str(summary_json_path))
        self.state.add_artifact(str(summary_path))

        memory_payloads = self._propose_memory(summary_payload)
        log_path = self._save_log()
        self.last_result = AssistantResult(
            summary_path=summary_path,
            summary_json_path=summary_json_path,
            log_path=log_path,
            memory_payloads=memory_payloads,
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def _extract_topic(self, context: AgentContext) -> str:
        for message in reversed(context.messages):
            if str(message.get("role", "")).lower() == "user":
                topic = str(message.get("content", "")).strip()
                if topic:
                    return topic
        return "Agent Activity Summary"

    def _resolve_output_dir(self, topic: str, initial_context: AgentContext) -> Path:
        workspace = str(initial_context.run_workspace or "").strip()
        if workspace:
            return Path(workspace)

        safe_topic = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip()
        )
        safe_topic = safe_topic[:80] or "assistant"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}"

    def _load_run_summaries(self) -> list[dict[str, Any]]:
        run_summaries: list[dict[str, Any]] = []
        root = Path(self.agent_config.agents_folder)
        if not root.exists():
            return run_summaries
        for path in root.rglob("run_summary.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._log(f"Failed to parse run summary {path}: {exc}")
                continue
            if isinstance(payload, dict):
                payload["summary_path"] = str(path)
                run_summaries.append(payload)
        return run_summaries

    def _build_summary_payload(self, run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        by_agent: dict[str, int] = {}
        by_status: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        topics: list[str] = []
        artifacts: list[str] = []
        denials = 0
        exec_tools = {"bash", "powershell"}
        network_tools = {"query_remote", "query_remote_chain", "query_llm", "load_url"}
        flagged_tools: set[str] = set()

        for summary in run_summaries:
            agent_name = str(summary.get("agent_name", "agent")).strip() or "agent"
            by_agent[agent_name] = by_agent.get(agent_name, 0) + 1
            status = str(summary.get("status", "unknown")).strip() or "unknown"
            by_status[status] = by_status.get(status, 0) + 1
            topic = str(summary.get("topic", "")).strip()
            if topic:
                topics.append(topic)
            for artifact in summary.get("artifacts_created", []) or []:
                artifacts.append(str(artifact))
            tools_used = summary.get("tools_used") or {}
            if isinstance(tools_used, dict):
                for tool_name, count in tools_used.items():
                    count_value = int(count) if isinstance(count, (int, float)) else 0
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + count_value
                    if tool_name in exec_tools or tool_name in network_tools:
                        flagged_tools.add(tool_name)
            approvals = summary.get("approvals_denials") or {}
            denials += int(approvals.get("denials", 0) or 0)

        frequent_topics = self._top_counts(topics, limit=5)
        top_tools = sorted(tool_counts.items(), key=lambda item: item[1], reverse=True)[:6]
        concerns: list[str] = []
        if denials:
            concerns.append(f"Permission denials recorded: {denials}.")
        if flagged_tools:
            concerns.append(
                f"High-risk tools used: {', '.join(sorted(flagged_tools))}."
            )
        if by_status.get("failed", 0) > 0:
            concerns.append("Some runs failed; review failure logs.")

        recommendations: list[str] = []
        if not run_summaries:
            recommendations.append("No run summaries found. Run agents to generate activity.")
        if by_status.get("failed", 0) > 0:
            recommendations.append("Follow up on failed runs and re-run as needed.")
        if denials:
            recommendations.append("Review denied tool requests for policy adjustments.")

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "run_count": len(run_summaries),
            "by_agent": by_agent,
            "by_status": by_status,
            "top_tools": top_tools,
            "frequent_topics": frequent_topics,
            "artifacts": artifacts,
            "concerns": concerns,
            "recommendations": recommendations,
            "run_summaries": run_summaries,
        }

    def _format_summary_markdown(self, payload: dict[str, Any]) -> str:
        lines = ["# Assistant Summary", ""]
        lines.append(f"- Generated at: {payload.get('generated_at', '-')}")
        lines.append(f"- Total runs: {payload.get('run_count', 0)}")
        lines.append("")
        lines.append("## Runs by Agent")
        by_agent = payload.get("by_agent") or {}
        if by_agent:
            for name, count in by_agent.items():
                lines.append(f"- {name}: {count}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Top Tools")
        top_tools = payload.get("top_tools") or []
        if top_tools:
            for name, count in top_tools:
                lines.append(f"- {name}: {count}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Frequent Topics")
        frequent_topics = payload.get("frequent_topics") or []
        if frequent_topics:
            for topic, count in frequent_topics:
                lines.append(f"- {topic}: {count}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Concerns")
        concerns = payload.get("concerns") or []
        if concerns:
            for concern in concerns:
                lines.append(f"- {concern}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Recommendations")
        recommendations = payload.get("recommendations") or []
        if recommendations:
            for recommendation in recommendations:
                lines.append(f"- {recommendation}")
        else:
            lines.append("- None")

        return "\n".join(lines).strip() + "\n"

    def _propose_memory(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        if "memory_put" not in self._available_tools():
            return payloads

        candidates = [
            {
                "key": "assistant:recent_topics",
                "value": {"topics": payload.get("frequent_topics", [])},
                "rationale": "Capture frequent topics across recent agent runs.",
            },
            {
                "key": "assistant:top_tools",
                "value": {"tools": payload.get("top_tools", [])},
                "rationale": "Capture top tool usage across recent agent runs.",
            },
        ]

        for candidate in candidates:
            args = {
                "key": candidate["key"],
                "value": candidate["value"],
                "type": "project_fact",
                "scope": "project",
                "tags": ["assistant", "summary"],
                "rationale": candidate["rationale"],
            }
            output = self._run_tool("memory_put", args)
            parsed = self._parse_json(output)
            if isinstance(parsed, dict):
                payloads.append(parsed)
        return payloads

    def _available_tools(self) -> set[str]:
        return {
            str(item.get("name", "")).strip()
            for item in self._get_tool_definitions(
                self.tool_registry,
                tool_allowlist=self.tool_allowlist,
            )
            if str(item.get("name", "")).strip()
        }

    def _run_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if self._handle_stop_request():
            return ""
        try:
            tool = self.tool_registry.lookup(tool_name)
        except KeyError:
            self._log(f"Skipping unknown tool '{tool_name}'.")
            return ""
        try:
            output = self.sandbox.execute(tool, args)
            self._consume_tokens(output)
            self._log(
                output,
                actor="tool",
                action=tool_name,
                action_arguments=args,
            )
            return output
        except Exception as exc:
            self._log(f"Tool '{tool_name}' failed: {exc}")
            return ""

    @staticmethod
    def _top_counts(items: list[str], limit: int = 5) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for item in items:
            normalized = str(item).strip()
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
        return sorted(counts.items(), key=lambda row: row[1], reverse=True)[:limit]


_ASSISTANT_VALIDATION_RULES = {
    "permission denied": "Permission denial detected.",
    "denied": "Permission denial detected.",
    "todo": "Outstanding TODO item found.",
    "tbd": "Unresolved TBD item found.",
    "error": "Potential error reported.",
    "failed": "Failure reported.",
}


def scan_assistant_findings(text: str) -> list[str]:
    """
    Scan text for compliance or review findings.
    """
    findings: list[str] = []
    haystack = str(text or "").lower()
    for token, message in _ASSISTANT_VALIDATION_RULES.items():
        if token in haystack:
            findings.append(message)
    return findings


def build_action_recommendations(findings: list[str]) -> list[str]:
    """
    Convert findings into actionable recommendations.
    """
    recommendations: list[str] = []
    if any("permission" in item.lower() for item in findings):
        recommendations.append("Review permission denials and update approvals.")
    if any("todo" in item.lower() or "tbd" in item.lower() for item in findings):
        recommendations.append("Resolve outstanding TODO/TBD items.")
    if any("error" in item.lower() or "failed" in item.lower() for item in findings):
        recommendations.append("Investigate errors and re-run failed steps.")
    if not recommendations and findings:
        recommendations.append("Review flagged findings and take follow-up actions.")
    return recommendations

"""
Synthesis agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig, LSMConfig

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..phase import PhaseResult
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a synthesis agent. Your workflow has three phases: "
    "PLAN (determine scope: query, format bullets/outline/narrative/qa, and target length), "
    "EVIDENCE (use tools to gather relevant information: read_folder, query_knowledge_base, "
    "extract_snippets, source_map, read_file), and "
    "SYNTHESIZE (produce a concise grounded markdown synthesis, tighten it for concision, "
    "and verify that it covers the core evidence). "
    "Follow phase instructions carefully and output in the format requested."
)


@dataclass
class SynthesisResult:
    """
    Structured result payload for a synthesis run.
    """

    topic: str
    synthesis_markdown: str
    source_map_markdown: str
    output_path: Path
    source_map_path: Path
    log_path: Path


class SynthesisAgent(BaseAgent):
    """
    Distill multiple documents into compact summaries.
    """

    name = "synthesis"
    tier = "complex"
    description = "Distill multiple documents into compact summaries."
    risk_posture = "writes_workspace"
    tool_allowlist = {
        "read_folder",
        "query_knowledge_base",
        "read_file",
        "write_file",
        "extract_snippets",
        "source_map",
    }

    def __init__(
        self,
        llm_registry: LLMRegistryConfig,
        tool_registry: ToolRegistry,
        sandbox: ToolSandbox,
        agent_config: AgentConfig,
        agent_overrides: Optional[Dict[str, Any]] = None,
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
        self.last_result: Optional[SynthesisResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the synthesis workflow.

        Args:
            initial_context: Agent context containing the user request/topic.

        Returns:
            Agent state after execution.
        """
        self._reset_harness()
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        self._workspace_root()
        topic = self._extract_topic(initial_context)
        self.state.current_task = f"Synthesizing: {topic}"

        # Phase 1 — PLAN: LLM determines scope and evidence strategy
        self._run_phase(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_message=(
                f"Phase: PLAN. Topic: '{topic}'. "
                "Determine the synthesis scope, query, target format "
                "(bullets/outline/narrative/qa), and target length in words. "
                "Think through your evidence-gathering approach."
            ),
            tool_names=[],
            max_iterations=1,
        )

        # Phase 2 — EVIDENCE: LLM gathers evidence using available tools
        evidence_result = self._run_phase(
            user_message=(
                f"Phase: EVIDENCE. Topic: '{topic}'. "
                "Use the available tools (read_folder, query_knowledge_base, "
                "extract_snippets, source_map, read_file) to gather relevant "
                "evidence for synthesis."
            ),
            tool_names=None,  # use all allowed tools from harness allowlist
            max_iterations=self.max_iterations,
        )

        # Phase 3 — SYNTHESIZE + TIGHTEN + COVERAGE CHECK
        synthesize_result = self._run_phase(
            user_message=(
                f"Phase: SYNTHESIZE. Topic: '{topic}'. "
                "Based on the evidence gathered above, produce a concise grounded "
                "synthesis in markdown. "
                "Tighten it for concision while preserving factual grounding. "
                "Verify that it covers the core evidence and note any gaps. "
                "Output the final synthesis markdown only."
            ),
            tool_names=[],
            max_iterations=3,
        )

        synthesis_markdown = synthesize_result.final_text or (
            f"# Synthesis: {topic}\n\nNo content generated.\n"
        )
        source_map_markdown = self._extract_source_map_markdown(evidence_result)

        output_path, source_map_path = self._save_outputs(
            topic=topic,
            synthesis_markdown=synthesis_markdown,
            source_map_markdown=source_map_markdown,
            initial_context=initial_context,
        )
        log_path = self._save_log()
        self.last_result = SynthesisResult(
            topic=topic,
            synthesis_markdown=synthesis_markdown,
            source_map_markdown=source_map_markdown,
            output_path=output_path,
            source_map_path=source_map_path,
            log_path=log_path,
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def _extract_topic(self, context: AgentContext) -> str:
        for message in reversed(context.messages):
            if str(message.get("role", "")).lower() == "user":
                topic = str(message.get("content", "")).strip()
                if topic:
                    return topic
        return "Untitled Synthesis Topic"

    def _extract_source_map_markdown(self, evidence_result: PhaseResult) -> str:
        """Build source_map markdown from the source_map tool call in the evidence phase."""
        for tool_call in evidence_result.tool_calls:
            if tool_call.get("name") == "source_map":
                parsed = self._parse_json(tool_call.get("result", ""))
                if isinstance(parsed, dict):
                    return self._build_source_map_markdown(parsed)
        return self._build_source_map_markdown({})

    def _build_source_map_markdown(self, source_map: Any) -> str:
        lines = ["# Source Map", ""]
        if not isinstance(source_map, dict) or not source_map:
            lines.append("No source evidence was mapped.")
            return "\n".join(lines).strip() + "\n"

        for source_path in sorted(source_map.keys()):
            details = source_map.get(source_path, {})
            if not isinstance(details, dict):
                details = {}
            count = int(details.get("count", 0))
            outline = details.get("outline", [])
            if not isinstance(outline, list):
                outline = []

            lines.append(f"## {source_path}")
            lines.append("")
            lines.append(f"- Evidence items: {count}")
            if outline:
                lines.append("- Outline:")
                for node in outline:
                    if not isinstance(node, dict):
                        continue
                    node_type = str(node.get("node_type", "")).strip() or "node"
                    name = str(node.get("name", "")).strip()
                    span = node.get("span") or {}
                    start_line = span.get("start_line")
                    end_line = span.get("end_line")
                    label = f"{node_type}: {name}" if name else node_type
                    if start_line and end_line:
                        label = f"{label} (lines {start_line}-{end_line})"
                    lines.append(f"- {label}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _save_outputs(
        self,
        topic: str,
        synthesis_markdown: str,
        source_map_markdown: str,
        initial_context: AgentContext,
    ) -> tuple[Path, Path]:
        run_dir = self._resolve_output_dir(initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)

        synthesis_path = run_dir / "synthesis.md"
        source_map_path = run_dir / "source_map.md"
        synthesis_path.write_text(synthesis_markdown, encoding="utf-8")
        source_map_path.write_text(source_map_markdown, encoding="utf-8")

        self.state.add_artifact(str(synthesis_path))
        self.state.add_artifact(str(source_map_path))
        self._log(f"Saved synthesis outputs to {run_dir}")
        return synthesis_path, source_map_path

    def _resolve_output_dir(self, initial_context: AgentContext) -> Path:
        workspace = str(initial_context.run_workspace or "").strip()
        if workspace:
            return Path(workspace)
        return self._artifacts_dir()

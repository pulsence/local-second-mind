"""
Research agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig, LSMConfig

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox

RESEARCH_SYSTEM_PROMPT = (
    "You are a research agent. Your workflow has four phases: "
    "DECOMPOSE (return a JSON array of focused research subtopics), "
    "RESEARCH (use tools to gather findings for a subtopic, then summarise in markdown), "
    "SYNTHESIZE (produce a structured markdown research outline with heading "
    "'# Research Outline: <topic>'), and "
    "REVIEW (assess the outline, return JSON: {\"sufficient\": true/false, \"suggestions\": [\"...\"]}). "
    "Follow the phase instructions carefully and respond in the format requested."
)


@dataclass
class ResearchResult:
    """
    Structured result payload for a research run.
    """

    topic: str
    outline_markdown: str
    output_path: Path
    log_path: Path


class ResearchAgent(BaseAgent):
    """
    LLM-guided agent for iterative topic research and outline generation.
    """

    name = "research"
    tier = "complex"
    description = (
        "Decomposes a topic, gathers evidence via tools, and writes a structured outline."
    )
    tool_allowlist = {
        "query_knowledge_base",
        "query_remote_chain",
    }
    risk_posture = "network"

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
        self.last_result: Optional[ResearchResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the research workflow.

        Args:
            initial_context: Agent context containing initial user topic.

        Returns:
            Agent state after execution.
        """
        self._reset_harness()
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        self._workspace_root()
        topic = self._extract_topic(initial_context)
        self.state.current_task = f"Researching: {topic}"

        # Phase 1 — DECOMPOSE
        decompose_result = self._run_phase(
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            user_message=(
                f"Phase: DECOMPOSE. Topic: '{topic}'. "
                "Return a JSON array of focused research subtopics."
            ),
            tool_names=[],
            max_iterations=1,
        )
        subtopics = self._parse_subtopics(decompose_result.final_text)
        if not subtopics:
            subtopics = [topic]

        iteration = 1
        subtopic_lines = "\n".join(f"  [{i}] {st}" for i, st in enumerate(subtopics, 1))
        self._log(
            f"Research iteration {iteration} — {len(subtopics)} subtopics:\n{subtopic_lines}"
        )

        # Phase 2 — RESEARCH per subtopic
        research_findings: List[Dict[str, str]] = []
        for subtopic in subtopics:
            self._log(f"Collecting findings for subtopic: {subtopic}")
            result = self._run_phase(
                user_message=(
                    f"Phase: RESEARCH. Subtopic: '{subtopic}'. "
                    "Use query_knowledge_base to gather relevant information. "
                    "Summarise your findings in markdown bullet points."
                ),
                tool_names=["query_knowledge_base"],
                max_iterations=3,
                context_label=f"subtopic:{subtopic}",
            )
            research_findings.append({"subtopic": subtopic, "findings": result.final_text or ""})
            if result.stop_reason in ("budget_exhausted", "stop_requested"):
                break

        # Phase 3 — SYNTHESIZE
        findings_block = "\n\n".join(
            f"### {item['subtopic']}\n{item['findings']}" for item in research_findings
        )
        synthesize_result = self._run_phase(
            user_message=(
                f"Phase: SYNTHESIZE. Topic: '{topic}'.\n\n"
                f"Research findings by subtopic:\n\n{findings_block}\n\n"
                f"Write a structured markdown research outline based on these findings. "
                f"Use '# Research Outline: {topic}' as the top-level heading."
            ),
            tool_names=[],
            max_iterations=1,
            context_label=None,
        )
        outline_markdown = synthesize_result.final_text or self._build_outline(
            topic,
            [{"subtopic": item["subtopic"], "summary": item["findings"]} for item in research_findings],
        )

        # Phase 4 — REVIEW
        review_result = self._run_phase(
            user_message=(
                "Phase: REVIEW. Review the research outline above. "
                'Return JSON: {"sufficient": true/false, "suggestions": ["..."]}.'
            ),
            tool_names=[],
            max_iterations=1,
        )
        parsed_review = self._parse_json(review_result.final_text or "")
        if isinstance(parsed_review, dict):
            suggestions = parsed_review.get("suggestions") or []
            normalized = [str(s).strip() for s in suggestions if str(s).strip()]
            if normalized:
                suggestion_lines = "\n".join(
                    f"  [{i}] {s}" for i, s in enumerate(normalized, 1)
                )
                self._log(
                    f"Refining with {len(normalized)} review suggestions:\n{suggestion_lines}"
                )

        output_path = self._save_outline(topic, outline_markdown)
        log_path = self._save_log()
        self.last_result = ResearchResult(
            topic=topic,
            outline_markdown=outline_markdown,
            output_path=output_path,
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
        return "Untitled Research Topic"

    def _parse_subtopics(self, text: str) -> List[str]:
        parsed = self._parse_json(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return self._fallback_line_list(text)

    def _build_outline(self, topic: str, sections: List[Dict[str, str]]) -> str:
        lines = [f"# Research Outline: {topic}", ""]
        for section in sections:
            lines.append(f"## {section['subtopic']}")
            lines.append("")
            lines.append(section["summary"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _save_outline(self, topic: str, outline: str) -> Path:
        safe_topic = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip()
        )
        safe_topic = safe_topic[:80] or "research"
        output_path = self._artifacts_dir() / self._artifact_filename(safe_topic)
        output_path.write_text(outline, encoding="utf-8")
        self.state.add_artifact(str(output_path))
        self._log(f"Saved research outline to {output_path}")
        return output_path

    def _fallback_line_list(self, text: str) -> List[str]:
        lines = []
        for line in str(text).splitlines():
            stripped = line.strip("-* \t")
            if stripped:
                lines.append(stripped)
        return lines

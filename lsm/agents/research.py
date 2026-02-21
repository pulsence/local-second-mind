"""
Research agent implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig
from lsm.logging import get_logger
from lsm.providers.factory import create_provider

from .base import AgentStatus, BaseAgent
from .models import AgentContext
from .tools.base import ToolRegistry
from .tools.sandbox import ToolSandbox

logger = get_logger(__name__)


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
    description = (
        "Decomposes a topic, gathers evidence via tools, and writes a structured outline."
    )

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
        self.last_result: Optional[ResearchResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the research workflow.

        Args:
            initial_context: Agent context containing initial user topic.

        Returns:
            Agent state after execution.
        """
        self._tokens_used = 0
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        topic = self._extract_topic(initial_context)
        self.state.current_task = f"Researching: {topic}"

        provider = create_provider(self.llm_registry.resolve_service("default"))
        subtopics = self._decompose_topic(provider, topic)
        if not subtopics:
            subtopics = [topic]

        iteration = 0
        outline_sections: List[Dict[str, str]] = []

        while iteration < self.max_iterations and not self._budget_exhausted():
            if self._handle_stop_request():
                break
            iteration += 1
            self._log(f"Research iteration {iteration} with {len(subtopics)} subtopics.")

            outline_sections = []
            for subtopic in subtopics:
                if self._handle_stop_request():
                    break
                if self._budget_exhausted():
                    self._log("Budget exhausted; stopping subtopic processing.")
                    break

                findings = self._collect_findings(provider, subtopic)
                if self._handle_stop_request():
                    break
                summary = self._summarize_findings(provider, subtopic, findings)
                outline_sections.append({"subtopic": subtopic, "summary": summary})

            if self._handle_stop_request():
                break
            outline = self._build_outline(topic, outline_sections)
            review = self._review_outline(provider, topic, outline)
            if review.get("sufficient", False):
                self._log("Review marked outline as sufficient.")
                break

            suggestions = review.get("suggestions") or []
            normalized = [str(item).strip() for item in suggestions if str(item).strip()]
            if not normalized:
                self._log("No suggestions returned; stopping iterations.")
                break
            subtopics = normalized
            self._log(f"Refining with {len(subtopics)} review suggestions.")

        outline_markdown = self._build_outline(topic, outline_sections)
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

    def _decompose_topic(self, provider: Any, topic: str) -> List[str]:
        if self._is_stop_requested():
            return []
        prompt = (
            "Decompose the topic into focused research subtopics. "
            "Respond as JSON array of strings.\n"
            f"Topic: {topic}"
        )
        response = provider.synthesize(prompt, "", mode="insight")
        self._consume_tokens(response)
        self._log(
            "Generated subtopics from topic decomposition.",
            actor="llm",
            provider_name=getattr(provider, "name", None),
            model_name=getattr(provider, "model", None),
        )
        parsed = self._parse_json(response)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return self._fallback_line_list(response)

    def _collect_findings(self, provider: Any, subtopic: str) -> List[Dict[str, Any]]:
        tool_names = self._select_tools(provider, subtopic)
        findings: List[Dict[str, Any]] = []
        for tool_name in tool_names:
            if self._handle_stop_request():
                break
            try:
                tool = self.tool_registry.lookup(tool_name)
            except KeyError:
                self._log(f"Skipping unknown tool '{tool_name}'.")
                continue

            args = self._build_tool_args(tool_name, subtopic)
            try:
                output = self.sandbox.execute(tool, args)
            except Exception as exc:
                self._log(f"Tool '{tool_name}' failed: {exc}")
                continue

            self._consume_tokens(output)
            findings.append({"tool": tool_name, "output": output})
            self._log(
                output,
                actor="tool",
                action=tool_name,
                action_arguments=args,
            )
        return findings

    def _select_tools(self, provider: Any, subtopic: str) -> List[str]:
        if self._is_stop_requested():
            return []
        tool_definitions = self._get_tool_definitions(self.tool_registry)
        available = [str(item.get("name", "")).strip() for item in tool_definitions]
        available = [name for name in available if name]
        prompt = (
            "Select the best tools for this subtopic. "
            "Return JSON object: {\"tools\":[\"tool_name\", ...]}.\n"
            f"Subtopic: {subtopic}\n"
            "Available tools (name, description, args schema):\n"
            f"{self._format_tool_definitions_for_prompt(self.tool_registry)}"
        )
        response = provider.synthesize(prompt, "", mode="insight")
        self._consume_tokens(response)
        selected = self._parse_tool_selection(response, available)
        if selected:
            return selected

        defaults = [name for name in ("query_embeddings", "query_remote") if name in available]
        if defaults:
            return defaults
        return available[:1]

    def _build_tool_args(self, tool_name: str, subtopic: str) -> Dict[str, Any]:
        if tool_name == "query_embeddings":
            return {"query": subtopic, "top_k": 5}
        if tool_name == "query_remote":
            provider_name = self._first_remote_provider_name()
            return {"provider": provider_name, "input": {"query": subtopic}, "max_results": 5}
        if tool_name == "query_remote_chain":
            return {"chain": "Research Digest", "input": {"query": subtopic}, "max_results": 5}
        if tool_name == "query_llm":
            return {"prompt": f"Summarize key ideas for: {subtopic}"}
        return {"text": subtopic}

    def _first_remote_provider_name(self) -> str:
        # Best effort default used by query_remote args.
        configured = self.agent_overrides.get("default_remote_provider")
        if configured:
            return str(configured)
        return "arxiv"

    def _summarize_findings(
        self,
        provider: Any,
        subtopic: str,
        findings: List[Dict[str, Any]],
    ) -> str:
        if self._is_stop_requested():
            return ""
        findings_block = json.dumps(findings, indent=2)
        prompt = (
            f"Summarize findings for subtopic '{subtopic}'. "
            "Produce concise markdown bullet points."
        )
        response = provider.synthesize(prompt, findings_block, mode="grounded")
        self._consume_tokens(response)
        return str(response).strip()

    def _review_outline(self, provider: Any, topic: str, outline: str) -> Dict[str, Any]:
        if self._is_stop_requested():
            return {"sufficient": True, "suggestions": []}
        prompt = (
            "Review this research outline and decide if it is sufficient. "
            "Return JSON object: {\"sufficient\": bool, \"suggestions\": [\"...\"]}.\n"
            f"Topic: {topic}"
        )
        response = provider.synthesize(prompt, outline, mode="insight")
        self._consume_tokens(response)
        parsed = self._parse_json(response)
        if isinstance(parsed, dict):
            sufficient = bool(parsed.get("sufficient", False))
            suggestions = parsed.get("suggestions") or []
            if not isinstance(suggestions, list):
                suggestions = []
            return {"sufficient": sufficient, "suggestions": suggestions}
        return {"sufficient": True, "suggestions": []}

    def _build_outline(self, topic: str, sections: List[Dict[str, str]]) -> str:
        lines = [f"# Research Outline: {topic}", ""]
        for section in sections:
            lines.append(f"## {section['subtopic']}")
            lines.append("")
            lines.append(section["summary"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _save_outline(self, topic: str, outline: str) -> Path:
        safe_topic = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip())
        safe_topic = safe_topic[:80] or "research"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        output_path = self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(outline, encoding="utf-8")
        self._log(f"Saved research outline to {output_path}")
        return output_path

    def _fallback_line_list(self, text: str) -> List[str]:
        lines = []
        for line in str(text).splitlines():
            stripped = line.strip("-* \t")
            if stripped:
                lines.append(stripped)
        return lines

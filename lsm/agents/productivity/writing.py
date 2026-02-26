"""
Writing agent implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig
from lsm.providers.factory import create_provider

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace


@dataclass
class WritingResult:
    """
    Structured result payload for a writing run.
    """

    topic: str
    deliverable_markdown: str
    output_path: Path
    log_path: Path


class WritingAgent(BaseAgent):
    """
    Generate grounded written deliverables from local evidence.
    """

    name = "writing"
    tier = "complex"
    description = "Generate grounded written deliverables from the knowledge base."
    risk_posture = "writes_workspace"
    tool_allowlist = {
        "query_knowledge_base",
        "read_file",
        "read_folder",
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
        self.last_result: Optional[WritingResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the writing workflow.

        Args:
            initial_context: Agent context containing the user request/topic.

        Returns:
            Agent state after execution.
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
        self.state.current_task = f"Writing: {topic}"

        provider = create_provider(self._resolve_llm_config(self.llm_registry))
        grounding = self._collect_grounding(topic)
        outline = ""
        draft = ""
        revised = ""
        if not self._is_stop_requested():
            outline = self._build_outline(provider, topic, grounding)
        if not self._is_stop_requested():
            draft = self._draft_deliverable(provider, topic, outline, grounding)
        if not self._is_stop_requested():
            revised = self._review_deliverable(provider, topic, draft, grounding)
        if self._handle_stop_request():
            revised = revised or draft or (
                f"# Deliverable: {topic}\n\nRun stopped before full completion.\n"
            )

        output_path = self._save_deliverable(topic, revised)
        log_path = self._save_log()
        self.last_result = WritingResult(
            topic=topic,
            deliverable_markdown=revised,
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
        return "Untitled Writing Task"

    def _collect_grounding(self, topic: str) -> Dict[str, Any]:
        grounding: Dict[str, Any] = {
            "candidates": [],
            "snippets": [],
            "source_map": {},
        }
        if self._is_stop_requested():
            return grounding
        available = {
            str(item.get("name", "")).strip()
            for item in self._get_tool_definitions(self.tool_registry)
            if str(item.get("name", "")).strip()
        }

        if "query_knowledge_base" in available:
            query_args = {"query": topic, "top_k": 8, "max_chars": 700}
            output = self._run_tool("query_knowledge_base", query_args)
            parsed = self._parse_json(output)
            if isinstance(parsed, dict):
                grounding["candidates"] = parsed.get("candidates", [])

        if self._is_stop_requested():
            return grounding
        source_paths = self._extract_source_paths(grounding["candidates"])
        if "extract_snippets" in available and source_paths:
            snippet_args = {
                "query": topic,
                "paths": source_paths[:6],
                "max_snippets": 8,
                "max_chars_per_snippet": 450,
            }
            output = self._run_tool("extract_snippets", snippet_args)
            parsed = self._parse_json(output)
            if isinstance(parsed, list):
                grounding["snippets"] = parsed

        if self._is_stop_requested():
            return grounding
        if "source_map" in available and grounding["snippets"]:
            source_map_args = {
                "evidence": grounding["snippets"],
                "max_depth": 2,
            }
            output = self._run_tool("source_map", source_map_args)
            parsed = self._parse_json(output)
            if isinstance(parsed, dict):
                grounding["source_map"] = parsed

        return grounding

    def _extract_source_paths(self, candidates: List[Dict[str, Any]]) -> List[str]:
        paths: list[str] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            source_path = item.get("source_path")
            if isinstance(metadata, dict):
                source_path = metadata.get("source_path", source_path)
            normalized = str(source_path or "").strip()
            if normalized and normalized not in paths:
                paths.append(normalized)
        return paths

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

    def _build_outline(
        self,
        provider: Any,
        topic: str,
        grounding: Dict[str, Any],
    ) -> str:
        if self._is_stop_requested():
            return f"# Outline: {topic}\n\nRun stopped before outline generation.\n"
        prompt = (
            "Create a concise markdown outline for a grounded deliverable. "
            "Use headings and a logical flow."
        )
        context = json.dumps({"topic": topic, "grounding": grounding}, indent=2)
        response = provider.synthesize(prompt, context, mode="grounded")
        self._consume_tokens(response)
        text = str(response).strip()
        if text:
            return text
        return (
            f"# Outline: {topic}\n\n"
            "## Introduction\n\n"
            "## Core Arguments\n\n"
            "## Synthesis and Conclusion\n"
        )

    def _draft_deliverable(
        self,
        provider: Any,
        topic: str,
        outline: str,
        grounding: Dict[str, Any],
    ) -> str:
        if self._is_stop_requested():
            return f"# Deliverable: {topic}\n\nRun stopped before drafting.\n"
        prompt = (
            "Draft a polished markdown deliverable using the outline and evidence. "
            "Be precise and grounded in provided sources."
        )
        context = json.dumps(
            {
                "topic": topic,
                "outline": outline,
                "grounding": grounding,
            },
            indent=2,
        )
        response = provider.synthesize(prompt, context, mode="grounded")
        self._consume_tokens(response)
        text = str(response).strip()
        if text:
            return text
        return f"# Deliverable: {topic}\n\n_No grounded draft could be generated._\n"

    def _review_deliverable(
        self,
        provider: Any,
        topic: str,
        draft: str,
        grounding: Dict[str, Any],
    ) -> str:
        if self._is_stop_requested():
            return draft or f"# Deliverable: {topic}\n\nRun stopped before review.\n"
        prompt = (
            "Revise the draft for clarity, factual grounding, and concise style. "
            "Return final markdown only."
        )
        context = json.dumps(
            {
                "topic": topic,
                "draft": draft,
                "grounding": grounding,
            },
            indent=2,
        )
        response = provider.synthesize(prompt, context, mode="grounded")
        self._consume_tokens(response)
        revised = str(response).strip()
        return revised or draft

    def _save_deliverable(self, topic: str, content: str) -> Path:
        safe_topic = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip())
        safe_topic = safe_topic[:80] or "writing"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        output_path = self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        self._log(f"Saved writing deliverable to {output_path}")
        return output_path

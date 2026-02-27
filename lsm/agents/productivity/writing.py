"""
Writing agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig, LSMConfig

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox

WRITING_SYSTEM_PROMPT = (
    "You are a writing agent. Your workflow has three phases: "
    "OUTLINE (use query_knowledge_base to gather evidence and produce a grounded outline), "
    "DRAFT (use the outline to write the full grounded deliverable in markdown), and "
    "REVIEW (revise the draft for clarity and factual grounding, return final markdown only). "
    "Follow phase instructions carefully and respond in the format requested."
)


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
        self.last_result: Optional[WritingResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the writing workflow.

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
        self.state.current_task = f"Writing: {topic}"

        # Phase 1 — OUTLINE: gather evidence and produce a grounded outline
        outline_result = self._run_phase(
            system_prompt=WRITING_SYSTEM_PROMPT,
            user_message=(
                f"Phase: OUTLINE. Topic: '{topic}'. "
                "Use query_knowledge_base to gather evidence and produce a grounded outline."
            ),
            tool_names=["query_knowledge_base"],
            max_iterations=3,
            context_label="outline",
        )
        deliverable_markdown = ""
        if outline_result.stop_reason not in ("budget_exhausted", "stop_requested"):
            # Phase 2 — DRAFT: write the full deliverable from the outline
            draft_result = self._run_phase(
                user_message=(
                    f"Phase: DRAFT. Topic: '{topic}'. "
                    "Using the outline produced above, write the full grounded deliverable in markdown."
                ),
                tool_names=[],
                max_iterations=1,
                context_label="draft",
                continue_context=False,
            )
            if draft_result.stop_reason not in ("budget_exhausted", "stop_requested"):
                # Phase 3 — REVIEW: revise for clarity and grounding
                review_result = self._run_phase(
                    user_message=(
                        "Phase: REVIEW. "
                        "Revise the draft for clarity and factual grounding. "
                        "Return final markdown only."
                    ),
                    tool_names=[],
                    max_iterations=1,
                    context_label="draft",
                    continue_context=True,
                )
                deliverable_markdown = review_result.final_text or draft_result.final_text or ""
            else:
                deliverable_markdown = draft_result.final_text or ""
        else:
            deliverable_markdown = outline_result.final_text or ""

        if not deliverable_markdown:
            deliverable_markdown = f"# Deliverable: {topic}\n\nNo content generated.\n"

        output_path = self._save_deliverable(topic, deliverable_markdown)
        log_path = self._save_log()
        self.last_result = WritingResult(
            topic=topic,
            deliverable_markdown=deliverable_markdown,
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

    def _save_deliverable(self, topic: str, content: str) -> Path:
        safe_topic = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip())
        safe_topic = safe_topic[:80] or "writing"
        output_path = self._artifacts_dir() / self._artifact_filename(safe_topic)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        self.state.add_artifact(str(output_path))
        self._log(f"Saved writing deliverable to {output_path}")
        return output_path

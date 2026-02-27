"""
Manuscript editor agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig
from lsm.utils.file_graph import build_graph_outline, get_file_graph

from ..base import AgentStatus, BaseAgent
from ..models import AgentContext
from ..tools.base import ToolRegistry
from ..tools.sandbox import ToolSandbox
from ..workspace import ensure_agent_workspace


@dataclass
class ManuscriptEditResult:
    """
    Structured result payload for manuscript edits.
    """

    source_path: Path
    output_path: Path
    revision_log_path: Path
    log_path: Path


class ManuscriptEditorAgent(BaseAgent):
    """
    Iteratively edit a manuscript with section-level revisions.
    """

    name = "manuscript_editor"
    tier = "normal"
    description = "Iteratively edit manuscripts and produce revision logs."
    tool_allowlist = {
        "read_file",
        "find_section",
        "edit_file",
        "write_file",
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
        self.max_sections = int(self.agent_overrides.get("max_sections", 4))
        self.max_rounds = int(self.agent_overrides.get("max_rounds", 2))
        self.last_result: Optional[ManuscriptEditResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the manuscript editing workflow.
        """
        self._reset_harness()
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        ensure_agent_workspace(
            self.name,
            self.agent_config.agents_folder,
            sandbox=self.sandbox,
        )
        target_path = self._extract_target_path(initial_context)
        if target_path is None:
            self._log("No manuscript path provided; aborting.")
            self.state.set_status(AgentStatus.FAILED)
            return self.state

        self.state.current_task = f"Editing manuscript: {target_path}"
        graph = get_file_graph(target_path)
        outline = build_graph_outline(graph, max_depth=3, node_types=["heading"])
        sections = self._select_sections(outline)
        original_text = target_path.read_text(encoding="utf-8")
        updated_text, revisions = self._edit_sections(
            original_text,
            sections,
            rounds=self.max_rounds,
        )

        run_dir = self._resolve_output_dir(target_path, initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / f"manuscript_final{target_path.suffix}"
        output_path.write_text(updated_text, encoding="utf-8")
        revision_log_path = run_dir / "revision_log.md"
        revision_log_path.write_text(
            self._format_revision_log(target_path, revisions),
            encoding="utf-8",
        )
        self.state.add_artifact(str(output_path))
        self.state.add_artifact(str(revision_log_path))
        log_path = self._save_log()
        self.last_result = ManuscriptEditResult(
            source_path=target_path,
            output_path=output_path,
            revision_log_path=revision_log_path,
            log_path=log_path,
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def _extract_target_path(self, context: AgentContext) -> Optional[Path]:
        for message in reversed(context.messages):
            if str(message.get("role", "")).lower() != "user":
                continue
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if Path(content).exists():
                return Path(content)
            tokens = [token.strip(" ,;\"'") for token in content.split()]
            for token in tokens:
                candidate = Path(token)
                if candidate.exists():
                    return candidate
        return None

    def _select_sections(self, outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not outline:
            return []
        sorted_outline = sorted(outline, key=lambda item: int(item.get("span", {}).get("start_line", 0)))
        return sorted_outline[: max(1, self.max_sections)]

    def _edit_sections(
        self,
        text: str,
        sections: list[dict[str, Any]],
        *,
        rounds: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        lines = text.splitlines()
        revisions: list[dict[str, Any]] = []

        for round_idx in range(max(1, rounds)):
            for section in sorted(
                sections,
                key=lambda item: int(item.get("span", {}).get("start_line", 0)),
                reverse=True,
            ):
                start_line = int(section.get("span", {}).get("start_line", 1)) - 1
                end_line = int(section.get("span", {}).get("end_line", start_line + 1))
                if start_line < 0 or end_line <= start_line:
                    continue
                section_lines = lines[start_line:end_line]
                section_text = "\n".join(section_lines)
                revised_text = self._revise_text(section_text)
                revised_lines = revised_text.splitlines()
                lines[start_line:end_line] = revised_lines
                revisions.append(
                    {
                        "round": round_idx + 1,
                        "heading": section.get("name"),
                        "start_line": start_line + 1,
                        "end_line": start_line + len(revised_lines),
                    }
                )

            if round_idx == 0:
                continue
            lines = [line.rstrip() for line in lines]

        return "\n".join(lines).strip() + "\n", revisions

    def _revise_text(self, text: str) -> str:
        if "query_llm" in self._available_tools():
            prompt = (
                "Revise the following manuscript section for clarity and cohesion. "
                "Return the revised section text only."
            )
            result = self._run_phase(
                direct_tool_calls=[
                    {
                        "name": "query_llm",
                        "arguments": {"prompt": prompt, "context": text, "mode": "grounded"},
                    }
                ]
            )
            for tc in result.tool_calls:
                if tc.get("name") == "query_llm" and "result" in tc:
                    revised = str(tc["result"]).strip()
                    if revised:
                        return revised
        return "\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n"

    def _available_tools(self) -> set[str]:
        return {
            str(item.get("name", "")).strip()
            for item in self._get_tool_definitions(self.tool_registry)
            if str(item.get("name", "")).strip()
        }

    def _format_revision_log(
        self,
        source_path: Path,
        revisions: list[dict[str, Any]],
    ) -> str:
        lines = ["# Revision Log", ""]
        lines.append(f"- Source: `{source_path}`")
        lines.append(f"- Total edits: {len(revisions)}")
        lines.append("")
        lines.append("## Revisions")
        if not revisions:
            lines.append("- None")
        for revision in revisions:
            heading = revision.get("heading") or "(untitled)"
            line_range = f"{revision.get('start_line')}-{revision.get('end_line')}"
            lines.append(
                f"- Round {revision.get('round')}: {heading} (lines {line_range})"
            )
        return "\n".join(lines).strip() + "\n"

    def _resolve_output_dir(self, source_path: Path, initial_context: AgentContext) -> Path:
        workspace = str(initial_context.run_workspace or "").strip()
        if workspace:
            return Path(workspace)
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source_path.stem
        )
        safe_name = safe_name[:80] or "manuscript"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return self._artifacts_dir() / f"{safe_name}_{timestamp}"

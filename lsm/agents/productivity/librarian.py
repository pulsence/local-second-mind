"""
Librarian agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
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
class LibrarianResult:
    """
    Structured result payload for a librarian run.
    """

    topic: str
    idea_graph_path: Path
    graph_json_path: Path
    log_path: Path
    memory_payloads: list[dict[str, Any]]


class LibrarianAgent(BaseAgent):
    """
    Explore the knowledge base and build idea graphs.
    """

    name = "librarian"
    tier = "normal"
    description = "Explore embeddings and build idea graphs with metadata summaries."
    tool_allowlist = {
        "query_embeddings",
        "extract_snippets",
        "source_map",
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
        Run the librarian workflow to generate idea graphs and summaries.
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
        self.state.current_task = f"Librarian: {topic}"

        candidates = self._query_embeddings(topic)
        source_paths = self._extract_source_paths(candidates)
        graph_entries = self._build_graph_entries(source_paths)
        citations = self._build_citations(candidates)

        run_dir = self._resolve_output_dir(topic, initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)
        graph_json_path = run_dir / "idea_graph.json"
        idea_graph_path = run_dir / "idea_graph.md"

        graph_payload = {
            "topic": topic,
            "sources": graph_entries,
            "citations": citations,
        }
        graph_json_path.write_text(
            json.dumps(graph_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        idea_graph_path.write_text(
            self._format_idea_graph_markdown(topic, graph_entries, citations),
            encoding="utf-8",
        )
        self.state.add_artifact(str(graph_json_path))
        self.state.add_artifact(str(idea_graph_path))
        self._log(f"Saved idea graph to {idea_graph_path}")

        memory_payloads = self._propose_memory(topic, graph_entries, citations)
        log_path = self._save_log()
        self.last_result = LibrarianResult(
            topic=topic,
            idea_graph_path=idea_graph_path,
            graph_json_path=graph_json_path,
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
        return "Untitled Librarian Task"

    def _available_tools(self) -> set[str]:
        return {
            str(item.get("name", "")).strip()
            for item in self._get_tool_definitions(
                self.tool_registry,
                tool_allowlist=self.tool_allowlist,
            )
            if str(item.get("name", "")).strip()
        }

    def _query_embeddings(self, topic: str) -> list[dict[str, Any]]:
        if self._handle_stop_request():
            return []
        if "query_embeddings" not in self._available_tools():
            self._log("query_embeddings tool is not available; skipping retrieval.")
            return []
        output = self._run_tool(
            "query_embeddings",
            {"query": topic, "top_k": 8, "max_chars": 500},
        )
        parsed = self._parse_json(output)
        return parsed if isinstance(parsed, list) else []

    def _extract_source_paths(self, candidates: list[dict[str, Any]]) -> list[str]:
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

    def _build_graph_entries(self, source_paths: list[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in source_paths:
            try:
                graph = get_file_graph(path)
                outline = build_graph_outline(graph, max_depth=3)
            except Exception as exc:
                self._log(f"Failed to build graph for {path}: {exc}")
                continue
            entries.append({"path": str(path), "outline": outline})
        return entries

    def _build_citations(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            source_path = item.get("source_path")
            if isinstance(metadata, dict):
                source_path = metadata.get("source_path", source_path)
            normalized = str(source_path or "").strip()
            if not normalized:
                continue
            citations.append(
                {
                    "source_path": normalized,
                    "excerpt": str(item.get("text", ""))[:240],
                    "relevance": item.get("relevance"),
                }
            )
        return citations

    def _resolve_output_dir(self, topic: str, initial_context: AgentContext) -> Path:
        workspace = str(initial_context.run_workspace or "").strip()
        if workspace:
            return Path(workspace)

        safe_topic = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in topic.strip()
        )
        safe_topic = safe_topic[:80] or "librarian"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}"

    def _format_idea_graph_markdown(
        self,
        topic: str,
        entries: list[dict[str, Any]],
        citations: list[dict[str, Any]],
    ) -> str:
        lines = [f"# Idea Graph: {topic}", ""]
        lines.append("## Sources")
        if entries:
            for entry in entries:
                lines.append(f"- `{entry['path']}`")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Citations")
        if citations:
            for citation in citations[:12]:
                excerpt = citation.get("excerpt", "")
                lines.append(f"- `{citation['source_path']}`: {excerpt}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("## Graph Outline")
        for entry in entries:
            lines.append(f"### {entry['path']}")
            outline = entry.get("outline") or []
            if not outline:
                lines.append("- (no outline)")
                continue
            for node in outline[:120]:
                node_name = node.get("name", "-")
                node_type = node.get("node_type", "-")
                span = node.get("span") or {}
                line_range = f"{span.get('start_line', '-')}-{span.get('end_line', '-')}"
                lines.append(f"- [{node_type}] {node_name} (lines {line_range})")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _propose_memory(
        self,
        topic: str,
        entries: list[dict[str, Any]],
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        if "memory_put" not in self._available_tools():
            return payloads
        value = {
            "topic": topic,
            "sources": [entry.get("path") for entry in entries],
            "citations": citations[:5],
            "generated_at": datetime.utcnow().isoformat(),
        }
        args = {
            "key": f"idea_graph:{topic}",
            "value": value,
            "type": "project_fact",
            "scope": "project",
            "tags": ["librarian", "idea_graph"],
            "rationale": f"Idea graph summary for {topic}.",
        }
        try:
            output = self._run_tool("memory_put", args)
            parsed = self._parse_json(output)
            if isinstance(parsed, dict):
                payloads.append(parsed)
        except Exception as exc:
            self._log(f"memory_put failed: {exc}")
        return payloads

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

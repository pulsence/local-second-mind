"""
Synthesis agent implementation.
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
    tool_allowlist = {
        "read_folder",
        "query_embeddings",
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
        self.last_result: Optional[SynthesisResult] = None

    def run(self, initial_context: AgentContext) -> Any:
        """
        Run the synthesis workflow.

        Args:
            initial_context: Agent context containing the user request/topic.

        Returns:
            Agent state after execution.
        """
        self._tokens_used = 0
        self._stop_logged = False
        self.state.set_status(AgentStatus.RUNNING)
        topic = self._extract_topic(initial_context)
        self.state.current_task = f"Synthesizing: {topic}"

        provider = create_provider(self._resolve_llm_config(self.llm_registry))
        scope = self._select_scope(provider, topic)
        candidate_paths = self._collect_candidate_sources(scope)
        evidence = self._retrieve_evidence(scope, candidate_paths)

        draft = ""
        tightened = ""
        coverage: Dict[str, Any] = {
            "sufficient": True,
            "coverage_notes": [],
            "missing_topics": [],
        }
        if not self._is_stop_requested():
            draft = self._synthesize(provider, topic, scope, evidence)
        if not self._is_stop_requested():
            tightened = self._tighten(provider, topic, scope, draft)
        if not self._is_stop_requested():
            coverage = self._coverage_check(provider, topic, scope, evidence, tightened)
        if self._handle_stop_request():
            tightened = tightened or draft or (
                f"# Synthesis: {topic}\n\nRun stopped before full completion.\n"
            )
        final_markdown = self._apply_coverage_notes(tightened, coverage)
        source_map_markdown = self._build_source_map_markdown(
            evidence.get("source_map", {})
        )

        output_path, source_map_path = self._save_outputs(
            topic=topic,
            synthesis_markdown=final_markdown,
            source_map_markdown=source_map_markdown,
            initial_context=initial_context,
        )
        log_path = self._save_log()
        self.last_result = SynthesisResult(
            topic=topic,
            synthesis_markdown=final_markdown,
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

    def _available_tools(self) -> set[str]:
        return {
            str(item.get("name", "")).strip()
            for item in self._get_tool_definitions(
                self.tool_registry,
                tool_allowlist=self.tool_allowlist,
            )
            if str(item.get("name", "")).strip()
        }

    def _select_scope(self, provider: Any, topic: str) -> Dict[str, Any]:
        if self._is_stop_requested():
            return {
                "query": topic,
                "target_format": "bullets",
                "target_length_words": 350,
                "scope_path": self._default_scope_path(),
            }
        default_scope_path = str(
            self.agent_overrides.get(
                "scope_path",
                self._default_scope_path(),
            )
        )
        default_format = str(
            self.agent_overrides.get("target_format", "bullets")
        ).strip()
        default_length = int(self.agent_overrides.get("target_length_words", 350))

        prompt = (
            "Select synthesis scope and output settings. "
            "Return strict JSON object with keys: "
            '{"query":"...","target_format":"bullets|outline|narrative|qa",'
            '"target_length_words": 200, "scope_hint":"optional path"}'
        )
        response = provider.synthesize(prompt, f"Topic:\n{topic}", mode="insight")
        self._consume_tokens(response)
        parsed = self._parse_json(response)

        query = topic
        target_format = default_format
        target_length_words = default_length
        scope_path = default_scope_path
        if isinstance(parsed, dict):
            parsed_query = str(parsed.get("query", "")).strip()
            if parsed_query:
                query = parsed_query
            parsed_format = str(parsed.get("target_format", "")).strip().lower()
            if parsed_format:
                target_format = parsed_format
            try:
                parsed_len = int(parsed.get("target_length_words", target_length_words))
                target_length_words = parsed_len
            except (TypeError, ValueError):
                pass
            scope_hint = str(parsed.get("scope_hint", "")).strip()
            if scope_hint and "scope_path" not in self.agent_overrides:
                scope_path = scope_hint

        if target_format not in {"bullets", "outline", "narrative", "qa"}:
            target_format = "bullets"
        target_length_words = max(120, min(target_length_words, 1500))

        scope = {
            "query": query,
            "target_format": target_format,
            "target_length_words": target_length_words,
            "scope_path": scope_path,
        }
        self._log(f"Selected synthesis scope: {json.dumps(scope)}")
        return scope

    def _default_scope_path(self) -> str:
        if self.sandbox.config.allowed_read_paths:
            return str(self.sandbox.config.allowed_read_paths[0])
        return "."

    def _collect_candidate_sources(self, scope: Dict[str, Any]) -> List[str]:
        if self._is_stop_requested():
            return []
        if "read_folder" not in self._available_tools():
            return []
        args = {"path": str(scope.get("scope_path", ".")), "recursive": True}
        output = self._run_tool("read_folder", args)
        parsed = self._parse_json(output)
        if not isinstance(parsed, list):
            return []

        allowed_exts = {".txt", ".md", ".rst", ".pdf", ".docx", ".html", ".htm"}
        candidates: list[str] = []
        for item in parsed:
            if self._is_stop_requested():
                break
            if not isinstance(item, dict):
                continue
            if bool(item.get("is_dir", False)):
                continue
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            ext = Path(path).suffix.lower()
            if ext and ext not in allowed_exts:
                continue
            if path not in candidates:
                candidates.append(path)
            if len(candidates) >= 40:
                break

        self._log(f"Collected {len(candidates)} candidate source paths.")
        return candidates

    def _retrieve_evidence(
        self,
        scope: Dict[str, Any],
        candidate_paths: List[str],
    ) -> Dict[str, Any]:
        if self._is_stop_requested():
            return {
                "candidates": [],
                "snippets": [],
                "source_map": {},
            }
        available = self._available_tools()
        query = str(scope.get("query", "")).strip()
        evidence: Dict[str, Any] = {
            "candidates": [],
            "snippets": [],
            "source_map": {},
        }

        if query and "query_embeddings" in available:
            query_args = {"query": query, "top_k": 10, "max_chars": 700}
            output = self._run_tool("query_embeddings", query_args)
            parsed = self._parse_json(output)
            if isinstance(parsed, list):
                evidence["candidates"] = parsed

        if self._is_stop_requested():
            return evidence
        snippet_paths = list(candidate_paths[:8])
        if not snippet_paths:
            snippet_paths = self._extract_source_paths(evidence["candidates"])[:8]

        if query and snippet_paths and "extract_snippets" in available:
            snippet_args = {
                "query": query,
                "paths": snippet_paths,
                "max_snippets": 10,
                "max_chars_per_snippet": 500,
            }
            output = self._run_tool("extract_snippets", snippet_args)
            parsed = self._parse_json(output)
            if isinstance(parsed, list):
                evidence["snippets"] = parsed

        if not evidence["snippets"] and "read_file" in available:
            fallback_snippets: list[Dict[str, Any]] = []
            for path in snippet_paths[:3]:
                if self._is_stop_requested():
                    break
                text = self._run_tool("read_file", {"path": path})
                if not text:
                    continue
                fallback_snippets.append(
                    {
                        "source_path": path,
                        "snippet": text[:500],
                        "score": 0.0,
                    }
                )
            evidence["snippets"] = fallback_snippets

        if "source_map" in available:
            source_map_input = evidence["snippets"] or evidence["candidates"]
            if isinstance(source_map_input, list) and source_map_input:
                source_map_args = {
                    "evidence": source_map_input,
                    "max_snippets_per_source": 3,
                }
                output = self._run_tool("source_map", source_map_args)
                parsed = self._parse_json(output)
                if isinstance(parsed, dict):
                    evidence["source_map"] = parsed

        return evidence

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

    def _synthesize(
        self,
        provider: Any,
        topic: str,
        scope: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> str:
        if self._is_stop_requested():
            return f"# Synthesis: {topic}\n\nRun stopped before synthesis.\n"
        target_format = str(scope.get("target_format", "bullets"))
        target_length_words = int(scope.get("target_length_words", 350))
        prompt = (
            "Synthesize grounded output in markdown. "
            f"Target format: {target_format}. "
            f"Target length: about {target_length_words} words. "
            "Use only evidence provided in context."
        )
        context = json.dumps(
            {
                "topic": topic,
                "query": scope.get("query"),
                "evidence": evidence,
            },
            indent=2,
        )
        response = provider.synthesize(prompt, context, mode="grounded")
        self._consume_tokens(response)
        text = str(response).strip()
        if text:
            return text
        return (
            f"# Synthesis: {topic}\n\n"
            "No grounded synthesis could be generated from available evidence.\n"
        )

    def _tighten(
        self,
        provider: Any,
        topic: str,
        scope: Dict[str, Any],
        synthesis: str,
    ) -> str:
        if self._is_stop_requested():
            return synthesis or f"# Synthesis: {topic}\n\nRun stopped before tightening.\n"
        prompt = (
            "Tighten the synthesis for concision while preserving factual grounding. "
            "Return markdown only."
        )
        context = json.dumps(
            {
                "topic": topic,
                "target_format": scope.get("target_format"),
                "target_length_words": scope.get("target_length_words"),
                "draft": synthesis,
            },
            indent=2,
        )
        response = provider.synthesize(prompt, context, mode="grounded")
        self._consume_tokens(response)
        tightened = str(response).strip()
        return tightened or synthesis

    def _coverage_check(
        self,
        provider: Any,
        topic: str,
        scope: Dict[str, Any],
        evidence: Dict[str, Any],
        synthesis: str,
    ) -> Dict[str, Any]:
        if self._is_stop_requested():
            return {"sufficient": True, "coverage_notes": [], "missing_topics": []}
        prompt = (
            "Assess whether the synthesis covers the core evidence. "
            "Return JSON object with keys: "
            '{"sufficient": bool, "coverage_notes": ["..."], "missing_topics": ["..."]}.'
        )
        context = json.dumps(
            {
                "topic": topic,
                "query": scope.get("query"),
                "evidence": evidence,
                "synthesis": synthesis,
            },
            indent=2,
        )
        response = provider.synthesize(prompt, context, mode="insight")
        self._consume_tokens(response)
        parsed = self._parse_json(response)
        if isinstance(parsed, dict):
            sufficient = bool(parsed.get("sufficient", False))
            coverage_notes = parsed.get("coverage_notes")
            missing_topics = parsed.get("missing_topics")
            return {
                "sufficient": sufficient,
                "coverage_notes": coverage_notes if isinstance(coverage_notes, list) else [],
                "missing_topics": missing_topics if isinstance(missing_topics, list) else [],
            }
        return {"sufficient": True, "coverage_notes": [], "missing_topics": []}

    def _apply_coverage_notes(self, synthesis: str, coverage: Dict[str, Any]) -> str:
        if coverage.get("sufficient", False):
            return synthesis
        coverage_notes = [
            str(item).strip()
            for item in coverage.get("coverage_notes", [])
            if str(item).strip()
        ]
        missing_topics = [
            str(item).strip()
            for item in coverage.get("missing_topics", [])
            if str(item).strip()
        ]
        if not coverage_notes and not missing_topics:
            return synthesis

        lines = [synthesis.rstrip(), "", "## Coverage Notes", ""]
        if coverage_notes:
            lines.append("### Notes")
            lines.append("")
            lines.extend([f"- {note}" for note in coverage_notes])
            lines.append("")
        if missing_topics:
            lines.append("### Missing Topics")
            lines.append("")
            lines.extend([f"- {topic}" for topic in missing_topics])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

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
            snippets = details.get("top_snippets", [])
            if not isinstance(snippets, list):
                snippets = []

            lines.append(f"## {source_path}")
            lines.append("")
            lines.append(f"- Evidence items: {count}")
            if snippets:
                lines.append("- Top snippets:")
                for snippet in snippets:
                    cleaned = str(snippet).strip().replace("\n", " ")
                    if cleaned:
                        lines.append(f"  - {cleaned}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _save_outputs(
        self,
        topic: str,
        synthesis_markdown: str,
        source_map_markdown: str,
        initial_context: AgentContext,
    ) -> tuple[Path, Path]:
        run_dir = self._resolve_output_dir(topic, initial_context)
        run_dir.mkdir(parents=True, exist_ok=True)

        synthesis_path = run_dir / "synthesis.md"
        source_map_path = run_dir / "source_map.md"
        synthesis_path.write_text(synthesis_markdown, encoding="utf-8")
        source_map_path.write_text(source_map_markdown, encoding="utf-8")

        self.state.add_artifact(str(synthesis_path))
        self.state.add_artifact(str(source_map_path))
        self._log(f"Saved synthesis outputs to {run_dir}")
        return synthesis_path, source_map_path

    def _resolve_output_dir(self, topic: str, initial_context: AgentContext) -> Path:
        workspace = str(initial_context.run_workspace or "").strip()
        if workspace:
            return Path(workspace)

        safe_topic = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in topic.strip()
        )
        safe_topic = safe_topic[:80] or "synthesis"
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return self.agent_config.agents_folder / f"{self.name}_{safe_topic}_{timestamp}"

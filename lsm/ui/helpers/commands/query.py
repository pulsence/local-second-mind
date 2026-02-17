"""Query command handlers shared by TUI/shell surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

from lsm.logging import get_logger
from lsm.query.citations import export_citations_from_note, export_citations_from_sources
from lsm.query.cost_tracking import estimate_query_cost
from lsm.query.notes import generate_note_content, get_note_filename, resolve_notes_dir
from lsm.ui.helpers.commands.common import (
    ParsedCommand,
    QUERY_EXPORT_FORMATS,
    parse_on_off_value,
    parse_slash_command,
)
from lsm.ui.utils import (
    display_provider_name,
    format_feature_label,
    open_file,
    run_remote_search,
    run_remote_search_all,
)

logger = get_logger(__name__)


@dataclass
class QueryCommandResult:
    """Result from query command handler dispatch."""

    output: str = ""
    handled: bool = True
    should_exit: bool = False


class QueryCommandHost(Protocol):
    """Protocol for command hosts (QueryScreen)."""

    app: Any

    def _get_help_text(self) -> str: ...

    def _format_model_selection(self) -> str: ...

    def _format_models(self, command: str) -> str: ...

    def _format_providers(self) -> str: ...

    def _format_provider_status(self) -> str: ...

    def _format_vectordb_providers(self) -> str: ...

    def _format_vectordb_status(self) -> str: ...

    def _format_remote_providers(self) -> str: ...


def execute_query_command(host: QueryCommandHost, command: str) -> QueryCommandResult:
    """Run grouped query command handlers with shared dispatch logic."""
    parsed = parse_slash_command(command)
    if not parsed.cmd:
        return QueryCommandResult(output="", handled=False)

    if parsed.cmd in {"/exit", "/quit"}:
        return QueryCommandResult(output="", should_exit=True)
    if parsed.cmd in {"/help", "/?"}:
        return QueryCommandResult(output=host._get_help_text())
    if parsed.cmd == "/debug":
        return QueryCommandResult(output=host.app.query_state.format_debug())

    for handler in (
        _handle_ui_commands,
        handle_mode_commands,
        handle_model_commands,
        handle_results_commands,
        handle_agent_commands,
        handle_filter_commands,
        handle_cost_commands,
        handle_remote_commands,
        handle_note_commands,
    ):
        result = handler(host, parsed)
        if result is not None:
            return result

    return QueryCommandResult(output=host._get_help_text())


def handle_mode_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle `/mode` command group."""
    if parsed.cmd != "/mode":
        return None
    parts = parsed.parts

    if len(parts) == 1:
        current_mode = host.app.config.query.mode
        mode_config = host.app.config.get_mode_config(current_mode)
        lines = [
            f"Current mode: {current_mode}",
            f"  Chat mode: {getattr(host.app.config.query, 'chat_mode', 'single')}",
            f"  Synthesis style: {mode_config.synthesis_style}",
            f"  Local sources: enabled (k={mode_config.source_policy.local.k})",
            (
                "  Remote sources: "
                f"{'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}"
            ),
            (
                "  LLM server cache: "
                f"{'enabled' if getattr(host.app.config.query, 'enable_llm_server_cache', True) else 'disabled'}"
            ),
            (
                "  Model knowledge: "
                f"{'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}"
            ),
            f"  Notes: {'enabled' if host.app.config.notes.enabled else 'disabled'}",
            f"\nAvailable modes: {', '.join(host.app.config.modes.keys())}\n",
        ]
        return QueryCommandResult(output="\n".join(lines))

    action = parts[1].lower()
    if action == "set":
        if len(parts) != 4:
            return QueryCommandResult(
                output=(
                    "Usage:\n  /mode set <setting> <on|off>\n"
                    "Settings: model_knowledge, remote, notes (global), llm_cache\n"
                )
            )

        setting = parts[2].strip().lower()
        enabled = parse_on_off_value(parts[3].strip())
        if enabled is None:
            enabled = False

        mode_config = host.app.config.get_mode_config()
        if setting in {"model_knowledge", "model-knowledge"}:
            mode_config.source_policy.model_knowledge.enabled = enabled
        elif setting in {"remote", "remote_sources", "remote-sources"}:
            mode_config.source_policy.remote.enabled = enabled
        elif setting in {"notes"}:
            host.app.config.notes.enabled = enabled
        elif setting in {"llm_cache", "llm-cache", "server_cache", "server-cache"}:
            host.app.config.query.enable_llm_server_cache = enabled
        else:
            return QueryCommandResult(
                output=(
                    f"Unknown setting: {setting}\n"
                    "Settings: model_knowledge, remote, notes (global), llm_cache\n"
                )
            )
        return QueryCommandResult(
            output=f"Mode setting '{setting}' set to: {'on' if enabled else 'off'}\n"
        )

    if len(parts) != 2:
        return QueryCommandResult(
            output=(
                "Usage:\n  /mode           (show current)\n"
                "  /mode <name>    (switch query source mode)\n"
                "  /mode chat|single (switch conversation mode)\n"
            )
        )

    mode_name = parts[1].strip()
    if mode_name in {"chat", "single"}:
        host.app.config.query.chat_mode = mode_name
        return QueryCommandResult(output=f"Chat mode set to: {mode_name}\n")
    if mode_name not in host.app.config.modes:
        return QueryCommandResult(
            output=(
                f"Mode not found: {mode_name}\n"
                f"Available modes: {', '.join(host.app.config.modes.keys())}\n"
            )
        )

    host.app.config.query.mode = mode_name
    mode_config = host.app.config.get_mode_config(mode_name)
    lines = [
        f"Mode switched to: {mode_name}",
        f"  Synthesis style: {mode_config.synthesis_style}",
        (
            "  Remote sources: "
            f"{'enabled' if mode_config.source_policy.remote.enabled else 'disabled'}"
        ),
        (
            "  Model knowledge: "
            f"{'enabled' if mode_config.source_policy.model_knowledge.enabled else 'disabled'}\n"
        ),
    ]
    return QueryCommandResult(output="\n".join(lines))


def handle_model_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle model/provider command group."""
    if parsed.cmd == "/model":
        parts = parsed.parts
        if len(parts) == 1:
            return QueryCommandResult(output=host._format_model_selection())
        if len(parts) != 4:
            return QueryCommandResult(
                output=(
                    "Usage:\n"
                    "  /model                   (show current)\n"
                    "  /model <task> <provider> <model>  (set model for a task)\n"
                    "  /models [provider]       (list available models)\n"
                )
            )

        task = parts[1].strip().lower()
        provider_name = parts[2].strip()
        model_name = parts[3].strip()
        task_map = {
            "query": "query",
            "tag": "tagging",
            "tagging": "tagging",
            "rerank": "ranking",
            "ranking": "ranking",
        }
        feature = task_map.get(task)
        if not feature:
            return QueryCommandResult(output="Unknown task. Use: query, tag, rerank\n")

        try:
            provider_names = host.app.config.llm.get_provider_names()
            normalized_provider = provider_name
            if provider_name == "anthropic" and "claude" in provider_names:
                normalized_provider = "claude"
            elif provider_name == "claude" and "anthropic" in provider_names:
                normalized_provider = "anthropic"

            host.app.config.llm.set_feature_selection(feature, normalized_provider, model_name)
            if feature == "query":
                host.app.query_state.model = model_name
            label = format_feature_label(feature)
            return QueryCommandResult(
                output=(
                    "Model set: "
                    f"{label} = {display_provider_name(normalized_provider)}/{model_name}\n"
                )
            )
        except Exception as exc:
            return QueryCommandResult(output=f"Failed to set model: {exc}\n")

    if parsed.cmd == "/models":
        return QueryCommandResult(output=host._format_models(parsed.text))
    if parsed.cmd == "/providers":
        return QueryCommandResult(output=host._format_providers())
    if parsed.cmd == "/provider-status":
        return QueryCommandResult(output=host._format_provider_status())
    if parsed.cmd == "/vectordb-providers":
        return QueryCommandResult(output=host._format_vectordb_providers())
    if parsed.cmd == "/vectordb-status":
        return QueryCommandResult(output=host._format_vectordb_status())
    return None


def handle_results_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle query result/citation command group."""
    cmd = parsed.cmd
    parts = parsed.parts

    if cmd == "/export-citations":
        fmt = parts[1].strip().lower() if len(parts) > 1 else "bibtex"
        note_path = Path(parts[2]) if len(parts) > 2 else None
        if fmt not in QUERY_EXPORT_FORMATS:
            return QueryCommandResult(output="Format must be 'bibtex' or 'zotero'.\n")

        try:
            if note_path:
                output_path = export_citations_from_note(note_path, fmt=fmt)
            else:
                if not host.app.query_state.last_label_to_candidate:
                    return QueryCommandResult(output="No last query sources available to export.\n")
                sources = [
                    {
                        "source_path": candidate.source_path,
                        "source_name": candidate.source_name,
                        "chunk_index": candidate.chunk_index,
                        "ext": candidate.ext,
                        "label": label,
                        "title": (candidate.meta or {}).get("title"),
                        "author": (candidate.meta or {}).get("author"),
                        "mtime_ns": (candidate.meta or {}).get("mtime_ns"),
                        "ingested_at": (candidate.meta or {}).get("ingested_at"),
                    }
                    for label, candidate in host.app.query_state.last_label_to_candidate.items()
                ]
                output_path = export_citations_from_sources(sources, fmt=fmt)
            return QueryCommandResult(output=f"Citations exported to: {output_path}\n")
        except Exception as exc:
            return QueryCommandResult(output=f"Failed to export citations: {exc}\n")

    if cmd in {"/show", "/expand"}:
        label = parts[1].strip() if len(parts) > 1 else None
        expanded = cmd == "/expand"
        if not label:
            usage = "/show S#   (e.g., /show S2)" if not expanded else "/expand S#   (e.g., /expand S2)"
            return QueryCommandResult(output=f"Usage: {usage}\n")

        normalized_label = label.strip().upper()
        candidate = host.app.query_state.last_label_to_candidate.get(normalized_label)
        if not candidate:
            return QueryCommandResult(output=f"No such label in last results: {normalized_label}\n")
        output = candidate.format(label=normalized_label, expanded=expanded)
        return QueryCommandResult(output=output)

    if cmd == "/open":
        label = parts[1].strip() if len(parts) > 1 else None
        if not label:
            return QueryCommandResult(output="Usage: /open S#   (e.g., /open S2)\n")

        normalized_label = label.strip().upper()
        candidate = host.app.query_state.last_label_to_candidate.get(normalized_label)
        if not candidate:
            return QueryCommandResult(output=f"No such label in last results: {normalized_label}\n")
        path = (candidate.meta or {}).get("source_path")
        if not path:
            return QueryCommandResult(output="No source_path available for this citation.\n")
        if open_file(path):
            return QueryCommandResult(output=f"Opened: {path}\n")
        return QueryCommandResult(output=f"Failed to open file: {path}\n")

    return None


def handle_agent_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle `/agent` + `/memory` command group."""
    if parsed.cmd == "/agent":
        from lsm.ui.shell.commands.agents import handle_agent_command
        return QueryCommandResult(output=handle_agent_command(parsed.text, host.app))
    if parsed.cmd == "/memory":
        from lsm.ui.shell.commands.agents import handle_memory_command
        return QueryCommandResult(output=handle_memory_command(parsed.text, host.app))
    return None


def handle_filter_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle filter/context/load command group."""
    cmd = parsed.cmd
    parts = parsed.parts

    if cmd == "/context":
        if len(parts) == 1:
            docs = host.app.query_state.context_documents or []
            chunks = host.app.query_state.context_chunks or []
            lines = [
                "Context anchors:",
                f"  Documents: {', '.join(docs) if docs else '(none)'}",
                f"  Chunks: {', '.join(chunks) if chunks else '(none)'}",
                "",
                "Usage:",
                "  /context doc <path> [more paths...]",
                "  /context chunk <id> [more ids...]",
                "  /context clear",
            ]
            return QueryCommandResult(output="\n".join(lines))

        action = parts[1].strip().lower()
        if action == "clear":
            host.app.query_state.context_documents = []
            host.app.query_state.context_chunks = []
            return QueryCommandResult(output="Cleared context anchors.\n")
        if action == "doc":
            if len(parts) < 3:
                return QueryCommandResult(output="Usage: /context doc <path> [more paths...]\n")
            host.app.query_state.context_documents = list(parts[2:])
            return QueryCommandResult(
                output=f"Context document anchors set ({len(parts[2:])}).\n"
            )
        if action == "chunk":
            if len(parts) < 3:
                return QueryCommandResult(output="Usage: /context chunk <id> [more ids...]\n")
            host.app.query_state.context_chunks = list(parts[2:])
            return QueryCommandResult(
                output=f"Context chunk anchors set ({len(parts[2:])}).\n"
            )
        return QueryCommandResult(output="Usage: /context [doc|chunk|clear] ...\n")

    if cmd == "/load":
        if len(parts) < 2:
            return QueryCommandResult(
                output=(
                    "Usage: /load <file_path>\n"
                    "Example: /load /docs/important.md\n\n"
                    "This pins a document for forced inclusion in next query context.\n"
                    "Use /load clear to clear pinned chunks.\n"
                )
            )
        arg = parsed.text.split(maxsplit=1)[1].strip()
        if arg.lower() == "clear":
            host.app.query_state.pinned_chunks = []
            return QueryCommandResult(output="Cleared all pinned chunks.\n")

        file_path = arg
        lines = [f"Loading chunks from: {file_path}", "Searching collection..."]
        try:
            result = host.app.query_provider.get(
                filters={"source_path": file_path},
                include=["metadatas"],
            )
            if not result.ids:
                lines.append(f"\nNo chunks found for path: {file_path}")
                lines.append("Tip: Path must match exactly. Use /explore to find exact paths.\n")
                return QueryCommandResult(output="\n".join(lines))

            chunk_ids = result.ids
            for chunk_id in chunk_ids:
                if chunk_id not in host.app.query_state.pinned_chunks:
                    host.app.query_state.pinned_chunks.append(chunk_id)

            lines.append(f"\nPinned {len(chunk_ids)} chunks from {file_path}")
            lines.append(f"Total pinned chunks: {len(host.app.query_state.pinned_chunks)}")
            lines.append("\nThese chunks will be forcibly included in your next query.")
            lines.append("Use /load clear to unpin all chunks.\n")
        except Exception as exc:
            lines.append(f"Error loading chunks: {exc}\n")
            logger.error(f"Load command error: {exc}")
        return QueryCommandResult(output="\n".join(lines))

    if cmd == "/set":
        key = parts[1].strip() if len(parts) > 1 else None
        values = list(parts[2:]) if len(parts) > 2 else []
        if not key or not values:
            return QueryCommandResult(
                output=(
                    "Usage:\n  /set path_contains <substring> [more...]\n"
                    "  /set ext_allow .md .pdf\n"
                    "  /set ext_deny .txt\n"
                )
            )
        if key == "path_contains":
            host.app.query_state.path_contains = values if len(values) > 1 else values[0]
            return QueryCommandResult(
                output=f"path_contains set to: {host.app.query_state.path_contains}\n"
            )
        if key == "ext_allow":
            host.app.query_state.ext_allow = values
            return QueryCommandResult(output=f"ext_allow set to: {host.app.query_state.ext_allow}\n")
        if key == "ext_deny":
            host.app.query_state.ext_deny = values
            return QueryCommandResult(output=f"ext_deny set to: {host.app.query_state.ext_deny}\n")
        return QueryCommandResult(output=f"Unknown filter key: {key}\n")

    if cmd == "/clear":
        key = parts[1].strip() if len(parts) > 1 else None
        if not key:
            return QueryCommandResult(output="Usage: /clear path_contains|ext_allow|ext_deny\n")
        if key == "path_contains":
            host.app.query_state.path_contains = None
            return QueryCommandResult(output="path_contains cleared.\n")
        if key == "ext_allow":
            host.app.query_state.ext_allow = None
            return QueryCommandResult(output="ext_allow cleared.\n")
        if key == "ext_deny":
            host.app.query_state.ext_deny = None
            return QueryCommandResult(output="ext_deny cleared.\n")
        return QueryCommandResult(output=f"Unknown filter key: {key}\n")

    return None


def handle_cost_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle cost tracking command group."""
    cmd = parsed.cmd
    parts = parsed.parts

    if cmd == "/costs":
        tracker = host.app.query_state.cost_tracker
        if not tracker:
            return QueryCommandResult(output="Cost tracking is not initialized.\n")
        if len(parts) == 1:
            return QueryCommandResult(output=host.app.query_state.format_costs())
        if len(parts) >= 2 and parts[1].lower() == "export":
            if len(parts) >= 3:
                export_path = Path(parts[2])
            else:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                export_path = Path(f"costs-{timestamp}.csv")
            try:
                tracker.export_csv(export_path)
                return QueryCommandResult(output=f"Cost data exported to: {export_path}\n")
            except Exception as exc:
                return QueryCommandResult(output=f"Failed to export costs: {exc}\n")
        return QueryCommandResult(output="Usage:\n  /costs\n  /costs export <path>\n")

    if cmd == "/budget":
        tracker = host.app.query_state.cost_tracker
        if not tracker:
            return QueryCommandResult(output="Cost tracking is not initialized.\n")
        if len(parts) == 1:
            if tracker.budget_limit is None:
                return QueryCommandResult(output="No budget set.\n")
            return QueryCommandResult(output=f"Budget limit: ${tracker.budget_limit:.4f}\n")
        if len(parts) == 3 and parts[1].lower() == "set":
            try:
                tracker.budget_limit = float(parts[2])
                return QueryCommandResult(
                    output=f"Budget limit set to: ${tracker.budget_limit:.4f}\n"
                )
            except ValueError:
                return QueryCommandResult(output="Invalid budget amount. Use a numeric value.\n")
        return QueryCommandResult(output="Usage:\n  /budget\n  /budget set <amount>\n")

    if cmd == "/cost-estimate":
        if len(parts) < 2:
            return QueryCommandResult(output="Usage: /cost-estimate <query>\n")
        query = parsed.text.split(maxsplit=1)[1].strip()
        cost = estimate_query_cost(
            query,
            host.app.config,
            host.app.query_state,
            host.app.query_embedder,
            host.app.query_provider,
        )
        return QueryCommandResult(output=f"Estimated cost: ${cost:.4f}\n")

    return None


def handle_remote_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle remote provider command group."""
    parts = parsed.parts

    if parsed.cmd == "/remote-providers":
        return QueryCommandResult(output=host._format_remote_providers())

    if parsed.cmd == "/remote-search":
        if len(parts) < 3:
            return QueryCommandResult(output="Usage: /remote-search <provider> <query>\n")
        provider_name = parts[1].strip()
        query = " ".join(parts[2:]).strip()
        if not query:
            return QueryCommandResult(output="Usage: /remote-search <provider> <query>\n")
        output = run_remote_search(provider_name, query, host.app.config)
        return QueryCommandResult(output=output)

    if parsed.cmd == "/remote-search-all":
        if len(parts) < 2:
            return QueryCommandResult(output="Usage: /remote-search-all <query>\n")
        query = parsed.text.split(maxsplit=1)[1].strip()
        if not query:
            return QueryCommandResult(output="Usage: /remote-search-all <query>\n")
        output = run_remote_search_all(query, host.app.config, host.app.query_state)
        return QueryCommandResult(output=output)

    return None


def handle_note_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    """Handle note persistence command group."""
    if parsed.cmd not in {"/note", "/notes"}:
        return None
    parts = parsed.parts

    if not host.app.query_state.last_question:
        return QueryCommandResult(output="No query to save. Run a query first.\n")
    try:
        notes_config = host.app.config.notes
        notes_dir = resolve_notes_dir(host.app.config, notes_config.dir)
        notes_dir.mkdir(parents=True, exist_ok=True)

        content = generate_note_content(
            query=host.app.query_state.last_question,
            answer=host.app.query_state.last_answer or "No answer generated",
            local_sources=host.app.query_state.last_local_sources_for_notes,
            remote_sources=host.app.query_state.last_remote_sources,
            mode=host.app.config.query.mode,
            use_wikilinks=notes_config.wikilinks,
            include_backlinks=notes_config.backlinks,
            include_tags=notes_config.include_tags,
        )

        filename_override = parsed.text.split(maxsplit=1)[1].strip() if len(parts) > 1 else None
        if filename_override:
            filename = filename_override
            if not filename.lower().endswith(".md"):
                filename += ".md"
            note_path = Path(filename)
            if not note_path.is_absolute():
                note_path = notes_dir / note_path
        else:
            filename = get_note_filename(
                host.app.query_state.last_question,
                format=notes_config.filename_format,
            )
            note_path = notes_dir / filename

        note_path.write_text(content, encoding="utf-8")
        return QueryCommandResult(output=f"Note saved to: {note_path}\n")
    except Exception as exc:
        logger.error(f"Note save error: {exc}")
        return QueryCommandResult(output=f"Failed to save note: {exc}\n")


def _handle_ui_commands(
    host: QueryCommandHost,
    parsed: ParsedCommand,
) -> Optional[QueryCommandResult]:
    if parsed.cmd != "/ui":
        return None
    parts = parsed.parts

    if len(parts) == 1:
        status = getattr(host.app, "density_status_text", None)
        if callable(status):
            return QueryCommandResult(output=f"{status()}\n")
        return QueryCommandResult(output="UI status is unavailable.\n")

    subcommand = parts[1].strip().lower()
    if subcommand != "density":
        return QueryCommandResult(output="Usage: /ui density [auto|compact|comfortable]\n")

    if len(parts) == 2:
        status = getattr(host.app, "density_status_text", None)
        if callable(status):
            return QueryCommandResult(output=f"{status()}\n")
        return QueryCommandResult(output="UI density status is unavailable.\n")

    if len(parts) > 3:
        return QueryCommandResult(output="Usage: /ui density [auto|compact|comfortable]\n")

    setter = getattr(host.app, "set_density_mode", None)
    if not callable(setter):
        return QueryCommandResult(output="UI density control is unavailable.\n")

    _success, message = setter(parts[2])
    return QueryCommandResult(output=f"{message}\n")


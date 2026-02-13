"""
Autocomplete logic for LSM TUI.

Provides command and argument completion for both ingest and query contexts.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Literal

from lsm.logging import get_logger

logger = get_logger(__name__)

# Type alias for context
ContextType = Literal["ingest", "query", "global"]


# Command definitions with descriptions and argument hints
GLOBAL_COMMANDS: Dict[str, str] = {
    "/exit": "Exit the application",
    "/quit": "Exit the application",
    "/help": "Show help",
}

INGEST_COMMANDS: Dict[str, str] = {
    "/info": "Show collection information",
    "/stats": "Show detailed statistics",
    "/explore": "Browse indexed files (optional: path filter)",
    "/show": "Show chunks for a file (requires: path)",
    "/search": "Search metadata (requires: query)",
    "/build": "Run ingest pipeline (optional: --force)",
    "/tag": "Run AI tagging (optional: --max N)",
    "/tags": "Show all tags in collection",
    "/vectordb-providers": "List available vector DB providers",
    "/vectordb-status": "Show vector DB provider status",
    "/wipe": "Clear collection (requires confirmation)",
    "/help": "Show help",
    "/exit": "Exit",
}

QUERY_COMMANDS: Dict[str, str] = {
    "/help": "Show help",
    "/exit": "Exit",
    "/show": "Show cited chunk (requires: S# e.g., S1)",
    "/expand": "Expand citation (requires: S# e.g., S1)",
    "/open": "Open source file (requires: S#)",
    "/model": "Show or set current model",
    "/models": "List available models (optional: provider)",
    "/providers": "List available LLM providers",
    "/provider-status": "Show provider health",
    "/vectordb-providers": "List vector DB providers",
    "/vectordb-status": "Show vector DB status",
    "/remote-providers": "List remote source providers",
    "/remote-search": "Test remote provider (requires: provider query)",
    "/remote-search-all": "Search all providers (requires: query)",
    "/mode": "Show or switch query mode",
    "/context": "Show or set context anchors",
    "/note": "Save last query as note (optional: name)",
    "/notes": "Alias for /note",
    "/load": "Pin document for context (requires: path)",
    "/costs": "Show session cost summary",
    "/budget": "Set session budget (requires: set amount)",
    "/cost-estimate": "Estimate query cost (requires: query)",
    "/export-citations": "Export citations (optional: format path)",
    "/debug": "Show retrieval diagnostics",
    "/set": "Set session filter (requires: filter value)",
    "/clear": "Clear session filter",
    "/agent": "Run or inspect agent workflows",
    "/memory": "Manage agent memory candidates",
    "/ui": "Inspect or change TUI UI settings",
}

# Subcommand completions
MODE_VALUES = ["grounded", "insight", "hybrid", "chat", "single"]
MODE_SETTINGS = ["model_knowledge", "remote", "notes", "llm_cache"]
BUILD_OPTIONS = ["--force"]
TAG_OPTIONS = ["--max"]
EXPORT_FORMATS = ["bibtex", "zotero"]
MEMORY_SUBCOMMANDS = ["candidates", "promote", "reject", "ttl"]
AGENT_SUBCOMMANDS = ["start", "status", "pause", "resume", "stop", "log", "schedule", "meta"]
AGENT_SCHEDULE_SUBCOMMANDS = ["add", "list", "enable", "disable", "remove", "status"]
AGENT_META_SUBCOMMANDS = ["start", "status", "log"]
UI_SUBCOMMANDS = ["density"]
UI_DENSITY_VALUES = ["auto", "compact", "comfortable"]


def get_commands(context: ContextType) -> Dict[str, str]:
    """
    Get available commands for a context.

    Args:
        context: The current context

    Returns:
        Dictionary of command -> description
    """
    if context == "ingest":
        return {**GLOBAL_COMMANDS, **INGEST_COMMANDS}
    elif context == "query":
        return {**GLOBAL_COMMANDS, **QUERY_COMMANDS}
    else:
        return GLOBAL_COMMANDS


def get_completions(
    text: str,
    context: ContextType = "global",
    candidates: Optional[List[str]] = None,
) -> List[str]:
    """
    Get completion suggestions for input text.

    Args:
        text: Current input text
        context: Current context (ingest, query, or global)
        candidates: Optional list of citation candidates (S1, S2, etc.)

    Returns:
        List of completion suggestions
    """
    # Check for trailing space BEFORE stripping
    has_trailing_space = text.endswith(" ")
    text = text.strip()

    # Empty input - suggest all commands
    if not text:
        commands = get_commands(context)
        return sorted(commands.keys())

    # Command completion
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # If we have arguments or trailing space after complete command, suggest arguments
        commands = get_commands(context)
        cmd_is_complete = cmd in commands

        if len(parts) > 1 or (cmd_is_complete and has_trailing_space):
            return _get_argument_completions(
                cmd,
                arg,
                context,
                candidates,
                has_trailing_space=has_trailing_space,
            )

        # Suggest matching commands
        matches = [c for c in commands.keys() if c.startswith(cmd)]
        return sorted(matches)

    # Citation completion (S1, S2, etc.)
    if candidates and text.upper().startswith("S"):
        return [f"S{i}" for i in range(1, len(candidates) + 1)
                if f"S{i}".startswith(text.upper())]

    return []


def _get_argument_completions(
    cmd: str,
    arg: str,
    context: ContextType,
    candidates: Optional[List[str]] = None,
    *,
    has_trailing_space: bool = False,
) -> List[str]:
    """
    Get completions for command arguments.

    Args:
        cmd: The command
        arg: Current argument text
        context: Current context
        candidates: Optional citation candidates

    Returns:
        List of argument completions
    """
    arg = arg.strip().lower()

    # Mode command
    if cmd == "/mode":
        if arg.startswith("set"):
            # /mode set <setting> <value>
            parts = arg.split()
            if len(parts) == 1:
                return [f"set {s}" for s in MODE_SETTINGS if s.startswith("")]
            elif len(parts) == 2:
                setting = parts[1]
                matches = [s for s in MODE_SETTINGS if s.startswith(setting)]
                return [f"set {m}" for m in matches]
            elif len(parts) == 3:
                return ["on", "off"]
        else:
            # /mode <name>
            matches = [m for m in MODE_VALUES if m.startswith(arg)]
            # Also suggest "set" for mode settings
            if "set".startswith(arg):
                matches.append("set")
            return sorted(matches)

    if cmd == "/context":
        options = ["doc", "chunk", "clear"]
        matches = [opt for opt in options if opt.startswith(arg)]
        return matches

    if cmd == "/memory":
        parts = arg.split()
        if not parts:
            return list(MEMORY_SUBCOMMANDS)
        if len(parts) == 1:
            return [item for item in MEMORY_SUBCOMMANDS if item.startswith(parts[0])]
        if len(parts) == 2 and parts[0] == "candidates":
            statuses = ["pending", "promoted", "rejected", "all"]
            return [status for status in statuses if status.startswith(parts[1])]
        return []

    if cmd == "/agent":
        parts = arg.split()
        if has_trailing_space and arg:
            parts.append("")
        if not parts:
            return list(AGENT_SUBCOMMANDS)
        if len(parts) == 1:
            if parts[0] == "schedule":
                return list(AGENT_SCHEDULE_SUBCOMMANDS)
            if parts[0] == "meta":
                return list(AGENT_META_SUBCOMMANDS)
            return [item for item in AGENT_SUBCOMMANDS if item.startswith(parts[0])]
        if parts[0] == "schedule":
            if len(parts) == 2:
                return [item for item in AGENT_SCHEDULE_SUBCOMMANDS if item.startswith(parts[1])]
            if len(parts) == 3 and parts[1] in {"enable", "disable", "remove"}:
                return ["<schedule_id>"]
            if len(parts) == 3 and parts[1] == "add":
                return ["<agent_name>"]
            if len(parts) == 4 and parts[1] == "add":
                return ["hourly", "daily", "weekly", "3600s"]
            if len(parts) >= 5 and parts[1] == "add":
                flags = ["--params", "--concurrency_policy", "--confirmation_mode"]
                return [flag for flag in flags if flag.startswith(parts[-1])] if parts[-1].startswith("--") else flags
        if parts[0] == "meta":
            if len(parts) == 2:
                return [item for item in AGENT_META_SUBCOMMANDS if item.startswith(parts[1])]
            if len(parts) == 3 and parts[1] == "start":
                return ["<goal>"]
        return []

    if cmd == "/ui":
        parts = arg.split()
        if has_trailing_space and arg:
            parts.append("")
        if not parts:
            return list(UI_SUBCOMMANDS)
        if len(parts) == 1:
            return [item for item in UI_SUBCOMMANDS if item.startswith(parts[0])]
        if parts[0] == "density" and len(parts) == 2:
            return [mode for mode in UI_DENSITY_VALUES if mode.startswith(parts[1])]
        return []

    # Build command
    if cmd == "/build":
        matches = [o for o in BUILD_OPTIONS if o.startswith(arg)]
        return matches

    # Tag command
    if cmd == "/tag":
        if arg.startswith("--max"):
            return ["--max "]  # Prompt for number
        matches = [o for o in TAG_OPTIONS if o.startswith(arg)]
        return matches

    # Show/Expand/Open commands - suggest citations
    if cmd in ("/show", "/expand", "/open") and candidates:
        citation_prefix = arg.upper()
        matches = [f"S{i}" for i in range(1, len(candidates) + 1)
                   if f"S{i}".startswith(citation_prefix)]
        return matches

    # Export citations
    if cmd == "/export-citations":
        matches = [f for f in EXPORT_FORMATS if f.startswith(arg)]
        return matches

    # Costs subcommands
    if cmd == "/costs":
        if "export".startswith(arg):
            return ["export"]
        return []

    # Budget subcommands
    if cmd == "/budget":
        if "set".startswith(arg):
            return ["set"]
        return []

    return []


def format_command_help(context: ContextType) -> str:
    """
    Format command help text for display.

    Args:
        context: Current context

    Returns:
        Formatted help text
    """
    commands = get_commands(context)

    lines = [f"Available commands ({context}):"]
    lines.append("")

    # Group commands by category
    categories = {
        "Navigation": ["/exit", "/quit", "/help"],
        "Information": ["/info", "/stats", "/debug", "/costs"],
        "Exploration": ["/explore", "/show", "/search", "/tags"],
        "Operations": ["/build", "/tag", "/wipe", "/load"],
        "Query": ["/mode", "/show", "/expand", "/open", "/note", "/notes"],
        "Agents": ["/agent", "/memory"],
        "UI": ["/ui"],
        "Providers": [
            "/providers", "/provider-status", "/model", "/models",
            "/vectordb-providers", "/vectordb-status",
            "/remote-providers", "/remote-search", "/remote-search-all",
        ],
        "Export": ["/export-citations", "/budget", "/cost-estimate"],
        "Filters": ["/set", "/clear", "/context"],
    }

    for category, cmds in categories.items():
        category_cmds = [(c, commands[c]) for c in cmds if c in commands]
        if category_cmds:
            lines.append(f"  {category}:")
            for cmd, desc in category_cmds:
                lines.append(f"    {cmd:20s} {desc}")
            lines.append("")

    return "\n".join(lines)


def create_completer(
    context: ContextType,
    candidates_getter=None,
):
    """
    Create a completer function for use with CommandInput.

    Args:
        context: Current context
        candidates_getter: Optional callable that returns current candidates

    Returns:
        Completer function
    """
    def completer(text: str) -> List[str]:
        candidates = candidates_getter() if candidates_getter else None
        return get_completions(text, context, candidates)

    return completer

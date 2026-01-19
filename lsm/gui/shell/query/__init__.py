"""LSM Query Shell module - shared query commands and CLI utilities.

Interactive usage is now TUI-only; REPL helpers remain for reuse.
"""

from __future__ import annotations

__all__ = [
    # REPL
    "run_repl",
    "run_query_turn",
    # Display
    "print_banner",
    "print_help",
    "print_source_chunk",
    "print_debug",
    "print_costs",
    "stream_output",
    # Commands
    "handle_command",
    "COMMAND_HINTS",
    "print_providers",
    "print_provider_status",
    "print_vectordb_providers",
    "print_vectordb_status",
    "print_remote_providers",
    "run_remote_search",
    "run_remote_search_all",
    # CLI
    "run_query_cli",
]

from lsm.gui.shell.query.repl import run_repl, run_query_turn
from lsm.gui.shell.query.display import (
    print_banner,
    print_help,
    print_source_chunk,
    print_debug,
    print_costs,
    stream_output,
)
from lsm.gui.shell.query.commands import (
    handle_command,
    COMMAND_HINTS,
    print_providers,
    print_provider_status,
    print_vectordb_providers,
    print_vectordb_status,
    print_remote_providers,
    run_remote_search,
    run_remote_search_all,
)


def __getattr__(name: str):
    """Lazy import for CLI to avoid circular imports."""
    if name == "run_query_cli":
        from lsm.gui.shell.query.cli import run_query_cli
        return run_query_cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

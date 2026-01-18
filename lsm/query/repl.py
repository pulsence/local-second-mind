"""
REPL (Read-Eval-Print Loop) for interactive query sessions.

DEPRECATED: This module is deprecated in favor of lsm.gui.shell.query.
All functionality has been moved to lsm.gui.shell.query.

This module re-exports from lsm.gui.shell.query for backward compatibility.
"""

from __future__ import annotations

import warnings

# Re-export from new location for backward compatibility
from lsm.gui.shell.query.repl import (
    run_repl,
    run_query_turn,
    fetch_remote_sources,
)
from lsm.gui.shell.query.display import (
    print_banner,
    print_help,
    print_source_chunk,
    print_debug,
    print_costs,
    stream_output,
    display_provider_name as _display_provider_name,
    format_feature_label as _format_feature_label,
)
from lsm.gui.shell.query.commands import (
    handle_command,
    COMMAND_HINTS,
    list_models,
    print_models,
    print_providers,
    print_provider_status,
    print_vectordb_providers,
    print_vectordb_status,
    print_remote_providers,
    run_remote_search,
    run_remote_search_all,
    toggle_remote_provider,
    set_remote_provider_weight,
    open_file,
    estimate_synthesis_cost,
    estimate_rerank_cost,
    estimate_query_cost,
    PROVIDER_RECOMMENDATIONS,
)

__all__ = [
    # REPL
    "run_repl",
    "run_query_turn",
    "fetch_remote_sources",
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
    "list_models",
    "print_models",
    "print_providers",
    "print_provider_status",
    "print_vectordb_providers",
    "print_vectordb_status",
    "print_remote_providers",
    "run_remote_search",
    "run_remote_search_all",
    "toggle_remote_provider",
    "set_remote_provider_weight",
    "open_file",
    "estimate_synthesis_cost",
    "estimate_rerank_cost",
    "estimate_query_cost",
    "PROVIDER_RECOMMENDATIONS",
]


def _deprecated_warning():
    warnings.warn(
        "lsm.query.repl is deprecated, use lsm.gui.shell.query instead",
        DeprecationWarning,
        stacklevel=3,
    )

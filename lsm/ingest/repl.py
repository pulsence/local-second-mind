"""
Interactive REPL for ingest pipeline management.

DEPRECATED: This module is deprecated in favor of lsm.gui.shell.ingest.
All functionality has been moved to lsm.gui.shell.ingest.

This module re-exports from lsm.gui.shell.ingest for backward compatibility.
"""

from __future__ import annotations

import warnings

# Re-export from new location for backward compatibility
from lsm.gui.shell.ingest.repl import run_ingest_repl, handle_command
from lsm.gui.shell.ingest.display import (
    print_banner,
    print_help,
    normalize_query_path as _normalize_query_path,
    parse_explore_query as _parse_explore_query,
    new_tree_node as _new_tree_node,
    build_tree as _build_tree,
    print_tree as _print_tree,
    compute_common_parts as _compute_common_parts,
)
from lsm.gui.shell.ingest.commands import (
    handle_info_command,
    handle_stats_command,
    handle_explore_command,
    handle_show_command,
    handle_search_command,
    handle_build_command,
    handle_tag_command,
    handle_tags_command,
    handle_wipe_command,
    handle_vectordb_providers_command,
    handle_vectordb_status_command,
)

__all__ = [
    "run_ingest_repl",
    "handle_command",
    "print_banner",
    "print_help",
    "handle_info_command",
    "handle_stats_command",
    "handle_explore_command",
    "handle_show_command",
    "handle_search_command",
    "handle_build_command",
    "handle_tag_command",
    "handle_tags_command",
    "handle_wipe_command",
    "handle_vectordb_providers_command",
    "handle_vectordb_status_command",
]


def _deprecated_warning():
    warnings.warn(
        "lsm.ingest.repl is deprecated, use lsm.gui.shell.ingest instead",
        DeprecationWarning,
        stacklevel=3,
    )

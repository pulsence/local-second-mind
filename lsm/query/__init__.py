"""Query subpackage with lazy exports to avoid heavy import-time dependencies."""

from __future__ import annotations

from typing import Any

__all__ = [
    "query",
    "query_sync",
    "QueryResult",
    "QueryProgress",
    "build_combined_context",
    "build_combined_context_async",
    "build_local_context",
    "build_remote_context",
    "build_context_block",
    "fallback_answer",
    "ContextResult",
    "SessionState",
    "Candidate",
    "prepare_local_candidates",
    "LocalQueryPlan",
    "QueryCache",
    "prefilter_by_metadata",
    "extract_tags_from_prompt",
    "QueryFields",
    "decompose_query",
    "remote",
]


def __getattr__(name: str) -> Any:
    if name in {"query", "query_sync", "QueryResult", "QueryProgress"}:
        from lsm.query import api as _api

        return getattr(_api, name)
    if name in {
        "build_combined_context",
        "build_combined_context_async",
        "build_local_context",
        "build_remote_context",
        "build_context_block",
        "fallback_answer",
        "ContextResult",
    }:
        from lsm.query import context as _context

        return getattr(_context, name)
    if name in {"SessionState", "Candidate"}:
        from lsm.query import session as _session

        return getattr(_session, name)
    if name in {"prepare_local_candidates", "LocalQueryPlan"}:
        from lsm.query import planning as _planning

        return getattr(_planning, name)
    if name == "QueryCache":
        from lsm.query.cache import QueryCache as _QueryCache

        return _QueryCache
    if name in {"prefilter_by_metadata", "extract_tags_from_prompt"}:
        from lsm.query import prefilter as _prefilter

        return getattr(_prefilter, name)
    if name in {"QueryFields", "decompose_query"}:
        from lsm.query import decomposition as _decomposition

        return getattr(_decomposition, name)
    if name == "remote":
        import lsm.remote as _remote
        if not hasattr(_remote, "wikipedia"):
            from lsm.remote.providers import wikipedia as _wikipedia

            setattr(_remote, "wikipedia", _wikipedia)

        return _remote
    raise AttributeError(f"module 'lsm.query' has no attribute '{name}'")

"""
Query subpackage.

Provides modular query system with provider abstraction.

Public API:
    query() - Async query execution
    query_sync() - Synchronous query execution
    QueryResult - Structured query result

Context Building:
    build_combined_context() - Build context from all sources
    build_combined_context_async() - Async version
    build_local_context() - Build context from local KB
    build_remote_context() - Build context from remote sources
    build_context_block() - Format candidates into LLM context
    fallback_answer() - Generate fallback when LLM unavailable
    ContextResult - Structured context result

Session Management:
    SessionState - Query session state
    Candidate - Retrieved chunk candidate

Planning:
    prepare_local_candidates() - Prepare local candidates
    LocalQueryPlan - Local query execution plan
"""

from lsm.query.api import query, query_sync, QueryResult
from lsm.query.context import (
    build_combined_context,
    build_combined_context_async,
    build_local_context,
    build_remote_context,
    build_context_block,
    fallback_answer,
    ContextResult,
)
from lsm.query.session import SessionState, Candidate
from lsm.query.planning import prepare_local_candidates, LocalQueryPlan

__all__ = [
    # Main API
    "query",
    "query_sync",
    "QueryResult",
    # Context building
    "build_combined_context",
    "build_combined_context_async",
    "build_local_context",
    "build_remote_context",
    "build_context_block",
    "fallback_answer",
    "ContextResult",
    # Session
    "SessionState",
    "Candidate",
    # Planning
    "prepare_local_candidates",
    "LocalQueryPlan",
]

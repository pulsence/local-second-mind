"""
Query subpackage.

Provides modular query system with provider abstraction.

Public API:
- cli: Query CLI entrypoint
- providers: LLM provider abstractions
- retrieval: Semantic search and filtering
- rerank: Local reranking strategies
- synthesis: Context building and source formatting
- session: Session state management
- repl: Interactive REPL (deprecated, use lsm.gui.shell.query)
"""

__all__ = [
    "run_query_cli",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name == "run_query_cli":
        from .cli import run_query_cli
        return run_query_cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

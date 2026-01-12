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
- repl: Interactive REPL
"""

from .cli import run_query_cli

__all__ = [
    "run_query_cli",
]

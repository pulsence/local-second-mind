"""Centralized table name registry for the unified database.

Every SQL table reference in production code should flow through a
``TableNames`` instance.  This eliminates scattered string literals
and enables the configurable ``table_prefix`` introduced in the
``DBConfig.table_prefix`` setting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VALID_PREFIX_RE = re.compile(r"^[a-zA-Z0-9_]*$")


@dataclass(frozen=True)
class TableNames:
    """Resolved table names for all application tables.

    Parameters:
        prefix: String prepended to every base table name.
                Must contain only alphanumeric characters and underscores.
                Defaults to ``"lsm_"``.
    """

    prefix: str = "lsm_"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not _VALID_PREFIX_RE.match(self.prefix):
            raise ValueError(
                f"table_prefix must contain only alphanumeric characters "
                f"and underscores, got {self.prefix!r}"
            )

    # ------------------------------------------------------------------
    # Core application tables (14)
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> str:
        return f"{self.prefix}chunks"

    @property
    def schema_versions(self) -> str:
        return f"{self.prefix}schema_versions"

    @property
    def manifest(self) -> str:
        return f"{self.prefix}manifest"

    @property
    def reranker_cache(self) -> str:
        return f"{self.prefix}reranker_cache"

    @property
    def agent_memories(self) -> str:
        return f"{self.prefix}agent_memories"

    @property
    def agent_memory_candidates(self) -> str:
        return f"{self.prefix}agent_memory_candidates"

    @property
    def agent_schedules(self) -> str:
        return f"{self.prefix}agent_schedules"

    @property
    def cluster_centroids(self) -> str:
        return f"{self.prefix}cluster_centroids"

    @property
    def graph_nodes(self) -> str:
        return f"{self.prefix}graph_nodes"

    @property
    def graph_edges(self) -> str:
        return f"{self.prefix}graph_edges"

    @property
    def embedding_models(self) -> str:
        return f"{self.prefix}embedding_models"

    @property
    def job_status(self) -> str:
        return f"{self.prefix}job_status"

    @property
    def stats_cache(self) -> str:
        return f"{self.prefix}stats_cache"

    @property
    def remote_cache(self) -> str:
        return f"{self.prefix}remote_cache"

    # ------------------------------------------------------------------
    # Migration bookkeeping (1)
    # ------------------------------------------------------------------

    @property
    def migration_progress(self) -> str:
        return f"{self.prefix}migration_progress"

    # ------------------------------------------------------------------
    # SQLite-only virtual tables (2)
    # ------------------------------------------------------------------

    @property
    def vec_chunks(self) -> str:
        return "vec_chunks"

    @property
    def chunks_fts(self) -> str:
        return "chunks_fts"

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def application_tables(self) -> tuple[str, ...]:
        """Return resolved core table names (excludes migration bookkeeping)."""
        return (
            self.chunks,
            self.schema_versions,
            self.manifest,
            self.reranker_cache,
            self.agent_memories,
            self.agent_memory_candidates,
            self.agent_schedules,
            self.cluster_centroids,
            self.graph_nodes,
            self.graph_edges,
            self.embedding_models,
            self.job_status,
            self.stats_cache,
            self.remote_cache,
        )


# Module-level default instance for convenience.
DEFAULT_TABLE_NAMES = TableNames()

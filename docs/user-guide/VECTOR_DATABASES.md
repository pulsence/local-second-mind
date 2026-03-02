Vector Databases
================

LSM supports pluggable vector database providers. The default provider is
SQLite + sqlite-vec, with PostgreSQL + pgvector available for larger deployments.

Configuration Overview
----------------------

Database settings live under the `db` section, with vector-specific settings
nested under `db.vector`.

Example (SQLite + sqlite-vec, default)
--------------------------------------

```json
{
  "db": {
    "table_prefix": "lsm_",
    "path": "Data",
    "vector": {
      "provider": "sqlite",
      "collection": "local_kb"
    }
  }
}
```

Example (PostgreSQL + pgvector)
-------------------------------

```json
{
  "db": {
    "path": "Data",
    "connection_string": "postgresql://user:pass@localhost:5432/lsm_vectors",
    "vector": {
      "provider": "postgresql",
      "collection": "local_kb",
      "index_type": "hnsw",
      "pool_size": 10
    }
  }
}
```

Provider Notes
--------------

SQLite + sqlite-vec
-------------------

- Default provider and fully supported in v0.8.0+.
- Stores vectors, metadata, full-text index (FTS5), manifest, graph, and agent state in a single `lsm.db`.
- Uses `db.path` as the directory containing `lsm.db`.
- Deployment defaults: WAL journal mode, 5-second busy_timeout, FK enforcement.

PostgreSQL + pgvector
---------------------

- Full feature parity with SQLite provider as of v0.8.1.
- Dependencies: `psycopg2-binary` and `pgvector` Python packages, PostgreSQL `vector` extension.
- Full-text search via native `tsvector`/`tsquery` with `ts_rank`.
- Knowledge graph tables (`lsm_graph_nodes`, `lsm_graph_edges`) with recursive CTE traversal.
- Embedding model registry (`lsm_embedding_models`) for fine-tuned model tracking.
- Useful when operating at larger scale or in shared/server environments.

ChromaDB (migration-only)
-------------------------

- ChromaDB is no longer a production provider in v0.8.0.
- Migration tooling remains available for moving legacy ChromaDB data.

Scale Guidance
--------------

### SQLite (up to ~250k chunks)

SQLite is the recommended provider for most users. At this scale:

- Single-file deployment, zero external dependencies.
- KNN queries complete in <100ms for typical corpora.
- FTS5 full-text search is instant for keyword queries.
- WAL mode allows concurrent reads during ingest writes.
- Memory footprint: ~1 KB per chunk (embedding + metadata).

**Estimated DB size**: embedding_dim x 4 bytes x chunk_count + metadata overhead.
For 100k chunks with 384-dim embeddings: ~150 MB.

### SQLite (250k-1M chunks)

At this scale SQLite still works well but consider:

- Run `lsm db prune` periodically to remove old chunk versions.
- Run `VACUUM` after large deletions to reclaim space.
- Set `retrieve_k` explicitly to avoid scanning too many candidates.
- Consider enabling clustering (`cluster_enabled=true`) for faster KNN pre-filtering.
- Monitor query latency; if >500ms, consider PostgreSQL migration.

### PostgreSQL (>1M chunks or shared access)

Migrate to PostgreSQL when:

- Corpus exceeds 1M chunks and query latency is a concern.
- Multiple users or services need concurrent access.
- You need server-side backup/replication infrastructure.
- HNSW index type is recommended for best recall/speed trade-off.

Use `lsm migrate` to move data between providers.

Migration & Enrichment
-----------------------

### Migration Commands

```
lsm migrate                                  # Auto-detect source, migrate, enrich
lsm migrate --from-db chroma --to-db sqlite  # Explicit backend migration
lsm migrate --from-version v0.7 --to-db sqlite --source-dir /path/to/legacy
lsm migrate --resume                         # Resume interrupted migration
lsm migrate --enrich                         # Enrich existing database (no copy)
lsm migrate --enrich --stage tier1           # Run only tier 1 enrichment
lsm migrate --enrich --stage graph           # Run only graph backfill
lsm migrate --skip-enrich                    # Copy only, skip enrichment
lsm migrate --rechunk                        # Auto-rechunk boundary-drifted files
lsm migrate --skip-rechunk                   # Skip rechunking prompt
```

Use `--stage` to re-run specific enrichment stages without repeating the entire
pipeline. This is useful after fixing a bug in a particular stage or when only
one stage needs updating on a large corpus. The flag may be repeated to select
multiple stages (e.g. `--stage tier1 --stage graph`). See
[CLI_USAGE.md](CLI_USAGE.md#enrichment-stage-reference) for the full stage
reference table.

### Post-Migration Enrichment

When migrating from an older version, chunks may lack metadata fields added
by newer ingest pipeline phases. The enrichment pipeline backfills these:

- **Tier 1** (automatic, in-place): simhash fingerprints, version/is_current
  defaults, node_type defaults, root/folder tags from current config.
- **Tier 2** (automatic if source files available): heading_path hierarchy,
  start_char/end_char positions, graph nodes/edges.
- **Tier 2b** (automatic if clustering enabled): cluster_id/cluster_size
  rebuild from existing embeddings.
- **Tier 3** (advisory): chunk boundary changes or missing section/file
  summaries. Run `lsm ingest --force-reingest-changed-config` after migration.

Enrichment runs automatically after backend migration unless `--skip-enrich`
is passed. Use `--enrich` to run enrichment standalone on an existing database.

### Database Health Check

LSM checks database health at startup (TUI and CLI). Detected issues:

- **missing**: No database file (first run, non-blocking).
- **legacy_detected**: `.chroma/` directory found, migration recommended.
- **mismatch**: Schema version differs from current config.
- **corrupt**: Database unreachable or missing required tables.
- **partial_migration**: Interrupted migration detected. Use `--resume`.
- **stale_chunks**: Chunks need enrichment (non-blocking).

Privacy Labels
--------------

| Feature | When | LLM call? | Data sent |
| --- | --- | --- | --- |
| Embedding generation | Ingest + Query | No (local model) | N/A |
| FTS5/BM25 search | Query | No | N/A |
| AI tagging | Ingest | Yes | Chunk text |
| Section/file summaries | Ingest | Yes | Section/file text |
| Query synthesis | Query | Yes | Retrieved context + question |
| HyDE generation | Query | Yes | Question |
| LLM reranking | Query | Yes | Candidate texts + question |
| Multi-hop decomposition | Query | Yes | Question |
| Graph construction | Ingest | No | N/A |
| Graph expansion | Query | No | N/A |
| Clustering | Offline (`lsm cluster build`) | No | N/A |
| Embedding fine-tuning | Offline (`lsm finetune train`) | No | N/A |

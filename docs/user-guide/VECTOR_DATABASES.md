Vector Databases
================

LSM supports pluggable vector database providers. The default provider is
SQLite + sqlite-vec, with PostgreSQL + pgvector available for larger deployments.

Configuration Overview
----------------------

Vector DB settings live under the `vectordb` section.

Example (SQLite + sqlite-vec, default)
--------------------------------------

```json
{
  "vectordb": {
    "provider": "sqlite",
    "path": "data",
    "collection": "local_kb"
  }
}
```

Example (PostgreSQL + pgvector)
-------------------------------

```json
{
  "vectordb": {
    "provider": "postgresql",
    "connection_string": "postgresql://user:pass@localhost:5432/lsm_vectors",
    "collection": "local_kb",
    "index_type": "hnsw",
    "pool_size": 10
  }
}
```

Provider Notes
--------------

SQLite + sqlite-vec
-------------------

- Default provider and fully supported in v0.8.0+.
- Stores vectors, metadata, full-text index (FTS5), manifest, graph, and agent state in a single `lsm.db`.
- Uses `vectordb.path` as the directory containing `lsm.db`.
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

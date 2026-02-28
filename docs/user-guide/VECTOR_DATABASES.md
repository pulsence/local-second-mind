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

- Default provider and fully supported in v0.8.0.
- Stores vectors, metadata, full-text index, manifest, and agent state in a single `lsm.db`.
- Uses `vectordb.path` as the directory containing `lsm.db`.

PostgreSQL + pgvector (scaffold)
--------------------------------

- Provider class is implemented.
- Dependencies `psycopg2` and `pgvector` are required.
- Useful when operating at larger scale or in shared/server environments.

ChromaDB (migration-only)
-------------------------

- ChromaDB is no longer a production provider in v0.8.0.
- Migration tooling remains available for moving legacy ChromaDB data.

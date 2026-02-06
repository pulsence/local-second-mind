Vector Databases
================

LSM supports pluggable vector database providers. The default provider is
ChromaDB, with a PostgreSQL + pgvector provider scaffold in progress.

Configuration Overview
----------------------

Vector DB settings live under the `vectordb` section.

Example (ChromaDB)
------------------

```json
{
  "vectordb": {
    "provider": "chromadb",
    "persist_dir": ".chroma",
    "collection": "local_kb",
    "chroma_hnsw_space": "cosine"
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

ChromaDB
--------

- Default provider and fully supported.
- Uses `persist_dir` for local storage.
- Example:

```json
{
  "vectordb": {
    "provider": "chromadb",
    "persist_dir": ".chroma",
    "collection": "local_kb",
    "chroma_hnsw_space": "cosine"
  }
}
```

PostgreSQL + pgvector (scaffold)
--------------------------------

- Provider class is implemented.
- Dependencies `psycopg2` and `pgvector` will be required.
- Schema, indexing, and migration tooling are provided.

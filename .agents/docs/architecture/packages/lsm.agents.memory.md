# lsm.agents.memory

Description: Persistent memory storage with SQLite/PostgreSQL backends, context builders, and API for agent memory lifecycle.
Folder Path: `lsm/agents/memory/`

## Modules

- [models.py](../../lsm/agents/memory/models.py): Memory and MemoryCandidate dataclasses
- [store.py](../../lsm/agents/memory/store.py): BaseMemoryStore, SQLiteMemoryStore, PostgreSQLMemoryStore implementations
- [api.py](../../lsm/agents/memory/api.py): Memory lifecycle operations and ranked retrieval helpers
- [context_builder.py](../../lsm/agents/memory/context_builder.py): Standing memory context builder for harness prompt injection
- [migrations.py](../../lsm/agents/memory/migrations.py): SQLite <-> PostgreSQL migration helpers

## Storage Model

- SQLite memory is stored in shared vectordb database tables (`lsm_agent_memories`, `lsm_agent_memory_candidates`) via injected sqlite connection.
- PostgreSQL memory uses provider-backed shared connection factories with the configured table prefix.
- `MemoryConfig` no longer includes a standalone `sqlite_path`; backend is selected with `storage_backend` (`auto`, `sqlite`, `postgresql`).

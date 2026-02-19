# lsm.agents.memory

Description: Persistent memory storage with SQLite/PostgreSQL backends, context builders, and API for agent memory lifecycle.
Folder Path: `lsm/agents/memory/`

## Modules

- [models.py](../../lsm/agents/memory/models.py): Memory and MemoryCandidate dataclasses
- [store.py](../../lsm/agents/memory/store.py): BaseMemoryStore, SQLiteMemoryStore, PostgreSQLMemoryStore implementations
- [api.py](../../lsm/agents/memory/api.py): Memory lifecycle operations and ranked retrieval helpers
- [context_builder.py](../../lsm/agents/memory/context_builder.py): Standing memory context builder for harness prompt injection
- [migrations.py](../../lsm/agents/memory/migrations.py): SQLite <-> PostgreSQL migration helpers

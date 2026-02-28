# v0.8.0 Research Plan: Embedding Retrieval Improvements

**Status**: In progress
**Version Target**: 0.8.0
**Source**: `TODO` v0.8.0 section + `.agents/future_plans/INGEST_FUTURE.md`
**Last Updated**: 2026-02-28 (post-v0.7.1 reanalysis)

> **Breaking Release**: v0.8.0 is a breaking release. Config file format changes are
> unconstrained — there is no obligation to maintain backward compatibility with v0.7.x
> config files. Migration code is provided for explicit data migration only.
> Old config files will need to be updated by the user with clear upgrade documentation.

This document records all design decisions and implementation considerations for v0.8.0.
All questions from the discovery phase are resolved. No open decisions remain.

---

## Table of Contents

1. [Scope Summary](#1-scope-summary)
2. [Current State Baseline (v0.7.1)](#2-current-state-baseline-v071)
3. [Part A: Unified Database Architecture](#3-part-a-unified-database-architecture)
   - 3.1 SQLite-vec: Replace ChromaDB
   - 3.2 Unified Database Schema
   - 3.3 Agent Data Consolidation (memories.db + schedules.json)
   - 3.4 SQLite-Backed Manifest
   - 3.5 Schema Versioning
   - 3.6 DB Completion — Incremental Corpus Updates
   - 3.7 Migration Paths
   - 3.8 Config Changes (vectordb)
   - 3.9 JSON Sidecar Audit (Non-Log, Non-Agent/Chat Artifacts)
4. [Part B: Ingest Pipeline Enhancements](#4-part-b-ingest-pipeline-enhancements)
   - 4.1 Document Headings: Configurable Depth
   - 4.2 Document Headings: Intelligent Selection (FileGraph Integration)
   - 4.3 heading_path Metadata
   - 4.4 Phase 2: Multi-Vector Representation (Ingest-Side)
   - 4.5 Phase 3: Graph Construction at Ingest Time
   - 4.6 Phase 3: Domain-Fine-Tuned Embeddings (Ingest-Side)
5. [Part C: Query Pipeline Overhaul](#5-part-c-query-pipeline-overhaul)
   - 5.1 Current Query Architecture
   - 5.2 Unified RetrievalPipeline Abstraction
   - 5.3 QueryRequest and Starting Prompt Handling
   - 5.4 ContextPackage and QueryResponse
   - 5.5 Retrieval Profiles
   - 5.6 Hybrid Retrieval: Dense + Sparse (BM25/FTS5) + RRF
   - 5.7 Cross-Encoder Reranking
   - 5.8 HyDE (Hypothetical Document Embeddings)
   - 5.9 Diversity and De-duplication
   - 5.10 Temporal-Aware Ranking
   - 5.11 Mode as Composition Preset
   - 5.12 Agent Integration: Unified Tool Surface
   - 5.13 Evaluation Harness
   - 5.14 Phase 2: Cluster-Aware Retrieval
   - 5.15 Phase 2: Multi-Vector Retrieval (Query-Side)
   - 5.16 Phase 3: Graph-Augmented Retrieval
   - 5.17 Phase 3: Multi-Hop Retrieval
6. [Cross-Cutting Concerns](#6-cross-cutting-concerns)
   - 6.1 TUI Startup Advisories for Offline Jobs
   - 6.2 Transactional Consistency
   - 6.3 VectorDB Provider Abstraction Updates
   - 6.4 LLM Provider Simplification
   - 6.5 Per-Provider Implementation Gaps
   - 6.6 Prompt Ownership Migration
   - 6.7 Tests
   - 6.8 SQLite Scale Guidance (User Guide)
   - 6.9 Privacy Model for Ingest-Time LLM Features
7. [Long-Term Design Principles](#7-long-term-design-principles)
8. [Dependency Map](#8-dependency-map)
9. [Resolved Decisions Summary](#9-resolved-decisions-summary)

---

## 1. Scope Summary

v0.8.0 targets all four TODO items plus all phases from INGEST_FUTURE.md:

| # | Item | Complexity |
|---|------|------------|
| A | Unified DB architecture (sqlite-vec, agent data consolidation, versioning, manifest) | High |
| B | Ingest pipeline enhancements (headings via FileGraph, multi-vector, graph construction) | High |
| C | Query pipeline overhaul (RetrievalPipeline, profiles, hybrid, cross-encoder, eval) | Very High |
| D | Document headings configurable depth + intelligent selection | Medium |

**Scope decision**: All of INGEST_FUTURE.md Phase 1, Phase 2, and Phase 3 features ship in
v0.8.0. Nothing in this plan is deferred to v0.8.x. The
SQLite-backed manifest and the replacement of ChromaDB with sqlite-vec are also included
as part of the DB overhaul. Agent memory and scheduling data are consolidated into the
unified database.

**Three-part structure**: The plan is organized around three major stages:
- **Part A — Database**: The unified `lsm.db` schema that all other work depends on.
- **Part B — Ingest Pipeline**: Enhancements to how documents are parsed, chunked, and stored.
- **Part C — Query Pipeline**: The retrieval pipeline overhaul, profiles, and evaluation.

---

## 2. Current State Baseline (v0.7.1)

This section captures the exact state of the codebase as of v0.7.1 to ground the plan in
reality. All references in subsequent sections note what exists vs. what is new.

### 2.1 Agent Architecture (v0.7.1 Foundation)

v0.7.1 established load-bearing patterns that v0.8.0 builds on:

- **`_run_phase()`**: The only approved method for agents to make LLM calls and tool
  executions. All 11 agents migrated. v0.8.0 must not introduce any path that bypasses
  `_run_phase()`.
- **`run_bounded()`**: Bounded LLM+tool loops with multi-context support via
  `context_label`. Budget enforcement across all contexts.
- **`query_knowledge_base` tool**: Replaced deprecated `query_embeddings`. Uses the full
  query pipeline (embeddings + reranking + LLM synthesis). This is the universal agent
  interface for knowledge retrieval. v0.8.0 replaces this with pipeline-backed tools.
- **Per-source `QueryRemoteTool` factory**: Each configured remote source gets its own
  named tool (`query_arxiv`, `query_pubmed`, etc.) via `remote_source_allowlist`.
  `QueryRemoteChainTool` handles named chains.
- **`ExtractSnippetsTool`**: Path-scoped semantic search (embed + retrieve within
  specified paths). Retained unchanged.
- **Workspace accessors**: `_workspace_root()`, `_artifacts_dir()`, `_logs_dir()`,
  `_memory_dir()` on `BaseAgent`. Agent memory folder used for `memories.db`.
- **`InteractionChannel` two-phase timeout**: Acknowledged requests wait indefinitely.

### 2.2 Current Query Pipeline

```
user query
   → embed_text() [retrieval.py:82]          # single-model dense embedding
   → retrieve_candidates() [retrieval.py:126] # kNN cosine ANN via ChromaDB HNSW
   → filter_candidates() [retrieval.py:196]   # path/ext post-filters
   → apply_local_reranking() [rerank.py:255]  # lexical reranking + diversity cap
   → build_context_block() [context.py]       # [S1]/[S2]/... context assembly
   → _synthesize_answer() [api.py:442]        # LLM synthesis
```

Key modules: `api.py` (485 lines, `QueryResult`, async `query()`), `planning.py` (319
lines, `LocalQueryPlan`), `session.py` (`Candidate`, `SessionState`), `rerank.py` (304
lines), `context.py`, `cost_tracking.py`, `decomposition.py`, `cache.py`, `notes.py`.

No retrieval profiles, no sparse index, no cross-encoder, no HyDE, no evaluation harness.
`QueryConfig.rerank_strategy = "hybrid"` is a misnaming — it is lexical reranking over
dense results, not a true dual-recall system.

### 2.3 Current System Prompt Handling

The "starting prompt" (system prompt for synthesis) is currently hard-coded in
`providers/helpers.py`:

- `get_synthesis_instructions(mode)` returns one of two templates:
  `SYNTHESIZE_GROUNDED_INSTRUCTIONS` or `SYNTHESIZE_INSIGHT_INSTRUCTIONS`.
- `format_user_content(question, context)` assembles the user message.
- `base.py:synthesize()` calls `_send_message(system=instructions, user=user_content)`.
- In chat mode, conversation history is prepended to the question payload as text
  (last 10 turns) and also passed as `conversation_history` kwarg.
- Provider-specific `_send_message()` maps `system` to `instructions` (OpenAI Responses
  API) or `system` (Anthropic Messages API).
- Server-side cache IDs are stored in `SessionState.llm_server_cache_ids` keyed by
  `{provider}:{model}:{mode}`.

**Problem**: The starting prompt is not configurable, not part of the request object, and
tightly coupled to `SessionState`. v0.8.0 adds `starting_prompt` to `QueryRequest`.

### 2.4 Current Agent Memory and Scheduling

- **Agent memory**: `SQLiteMemoryStore` uses a standalone `memories.db` file at
  `memory_config.sqlite_path` (default: `<agents_folder>/memories.db`). Tables:
  `memories`, `memory_candidates`. `PostgreSQLMemoryStore` uses tables with a
  configurable prefix (`agent_memory_memories`, `agent_memory_candidates`).
- **Agent scheduling**: `AgentScheduler` stores persistent state in
  `<agents_folder>/schedules.json` — last run times, status, error logs.
- **Problem**: These are separate data stores from the vector DB. v0.8.0 consolidates
  both into `lsm.db` (or the PostgreSQL database), eliminating the separate
  `memories.db` file and `schedules.json` file.

### 2.5 Current Vector DB Providers

- **`BaseVectorDBProvider`** (`vectordb/base.py:182 lines`): `add_chunks`, `get`, `query`,
  `delete_by_id`, `delete_by_filter`, `delete_all`, `count`, `get_stats`, `optimize`,
  `health_check`, `update_metadatas`.
- **`ChromaDBProvider`** (`vectordb/chromadb.py`): HNSW cosine, persistent storage.
- **`PostgreSQLProvider`** (`vectordb/postgresql.py`): pgvector, HNSW/IVFFlat indexes.
- **No `fts_query`**, no `prune_old_versions`, no graph methods on the base provider.

### 2.6 Current Manifest and Versioning

- **Manifest**: Flat JSON file (`manifest.py:36 lines`) — `load_manifest`, `save_manifest`,
  `get_next_version`.
- **Versioning**: Gated by `IngestConfig.enable_versioning` (default False). When enabled,
  old chunks are soft-deleted (`is_current=False`) instead of hard-deleted.
- **No schema versioning**: No tracking of embedding model, chunking params, or LSM version
  in the database.

### 2.7 Current Document Graph Tooling

`lsm/utils/file_graph.py` provides:
- **`GraphNode`** (frozen dataclass): `id`, `node_type`, `name`, `start_line`, `end_line`,
  `start_char`, `end_char`, `depth`, `parent_id`, `children`, `metadata`, `line_hash`.
  Node types: `"document"`, `"heading"`, `"paragraph"`, `"list"`, `"code_block"`, etc.
- **`FileGraph`** (frozen dataclass): `path`, `content_hash`, `nodes`, `root_ids`.
  Methods: `to_dict()`, `from_dict()`, `node_map()`, `with_path()`.
- **`stable_node_id()`**: Deterministic SHA1 node IDs.
- **Parser functions**: `build_markdown_graph()`, `build_text_graph()`,
  `build_html_graph()`, `build_code_graph()`, `build_docx_graph()`.
- **Supported extensions**: `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.cpp`,
  `.cs` (code); `.md`, `.txt`, `.rst`, `.docx` (text); `.html`, `.htm` (HTML).
- **Heading hierarchy**: Already tracks depth, parent/child relationships, and span
  boundaries. The `heading` node type has `metadata["level"]` for heading depth.

**This is the foundation for §4.1–4.3**: The heading improvement work should build on
`FileGraph` rather than reimplementing heading detection. `structure_chunking.py` needs
to accept `FileGraph` as input for heading-aware chunking decisions.

### 2.8 Current Config Models

Key dataclasses and their relevant fields (fields that will be modified or removed in
v0.8.0 are marked with ✗):

- **`VectorDBConfig`**: `provider` ("chromadb"✗/"postgresql"), `persist_dir`✗,
  `collection`, `chroma_hnsw_space`✗, PG connection fields, `index_type`, `pool_size`.
- **`IngestConfig`**: `roots`, `persist_dir`, `collection`, `chroma_flush_interval`✗,
  `manifest`, `extensions`, `chunk_size`, `chunk_overlap`, `chunking_strategy`,
  `enable_versioning`✗ (becomes always-on).
- **`QueryConfig`**: `k`, `retrieve_k`, `min_relevance`, `k_rerank`,
  `rerank_strategy`✗ ("none"/"lexical"/"llm"/"hybrid"), `no_rerank`, `local_pool`,
  `max_per_file`, `mode`, `path_contains`, `ext_allow`, `ext_deny`,
  `enable_query_cache`, `query_cache_ttl`, `query_cache_size`, `chat_mode`.
- **`ModeConfig`**: `synthesis_style`, `source_policy` (containing `LocalSourcePolicy`,
  `RemoteSourcePolicy`, `ModelKnowledgePolicy`).
- **`LocalSourcePolicy`**: `enabled`, `min_relevance`, `k`, `k_rerank`.
- **`MemoryConfig`**: `storage_backend` ("auto"/"sqlite"/"postgresql"),
  `sqlite_path`✗ (merges into lsm.db), `postgres_connection_string`,
  `postgres_table_prefix`, TTL fields.

### 2.9 Current Agent Tool Surface

| Tool | Purpose | Status in v0.8.0 |
|------|---------|-----------------|
| `query_knowledge_base` | Full pipeline query + synthesis | **Replaced** by `query_context` + `execute_context` + `query_and_synthesize` |
| `query_llm` | Direct LLM prompting | **Retained** — standalone LLM access still useful |
| `query_<provider>` | Per-source remote provider query | **Retained** unchanged |
| `query_remote_chain` | Named remote provider chain | **Retained** unchanged |
| `extract_snippets` | Path-scoped semantic search | **Retained** unchanged |
| `similarity_search` | Intra-corpus pairwise similarity | **Retained** unchanged |

---

## 3. Part A: Unified Database Architecture

### 3.1 SQLite-vec: Replace ChromaDB

**Decision**: ChromaDB is replaced entirely by sqlite-vec as the primary vector store.
v0.8.0 ships two providers:

| Provider | Default | Description |
|----------|---------|-------------|
| `sqlite` | **Yes** | sqlite-vec + FTS5 + all LSM tables in a single `lsm.db` file |
| `postgresql` | No | pgvector + native FTS + all LSM tables in a single PostgreSQL DB |

There is no longer a "ChromaDB provider." Users with existing ChromaDB data use the
migration command before upgrading (see §3.7).

The principle is: **one database technology at a time**. Either a single SQLite file
(`lsm.db`) or a single PostgreSQL database holds all LSM state — including vector data,
FTS index, manifest, schema versions, agent memories, schedule state, reranker cache,
and graph store.

#### 3.1.1 sqlite-vec Research Findings

**What it is**: sqlite-vec (`asg017/sqlite-vec`) is a SQLite extension that adds vector
similarity search via the `vec0` virtual table interface. Written in pure C with zero
external library dependencies. Mozilla-sponsored for local AI use cases.

**Current version**: v0.1.6 stable (November 2024). Actively maintained.

**ANN support**: sqlite-vec currently uses **brute-force KNN only**. No HNSW or IVF in
stable releases. ANN (planned: IVF + DiskANN — not HNSW, which is incompatible with
SQLite's page-based storage model) has no confirmed ship date. This is the primary
limitation.

**Brute-force KNN performance** (benchmarked on commodity hardware):

| Corpus size | Dimensions | Query latency |
|-------------|------------|---------------|
| 100k vectors | 384 (float32) | 15–68 ms |
| 100k vectors | 768 (float32) | 30–40 ms |
| 100k vectors | 1536 (float32) | ~105 ms |
| 1M vectors | 128 (float32) | ~33 ms |

For a personal knowledge base at 100k+ chunks using a 384-dim model (all-MiniLM-L6-v2),
query latency of 15–68ms is acceptable for interactive use.

**Python installation**: `pip install sqlite-vec` — zero runtime dependencies. Platform
wheels pre-compiled for Windows x86-64, Linux x86-64/ARM64, macOS x86-64/ARM64.

**Python API**:
```python
import sqlite3
import sqlite_vec

db = sqlite3.connect("lsm.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
```

**vec0 virtual table creation**:
```sql
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    chunk_id    TEXT PRIMARY KEY,
    embedding   FLOAT[384] distance_metric=cosine,
    is_current  INTEGER,
    node_type   TEXT,
    source_path TEXT
);
```

**Metadata filter operators** (v0.1.6+): `=`, `!=`, `<`, `<=`, `>`, `>=`, `IN (...)`,
`BETWEEN`, `IS NULL`, `IS NOT NULL`. Pre-filter (bitmap-indexed).

**FTS5 coexistence**: `vec0` virtual tables and FTS5 virtual tables coexist in the same
SQLite database with no conflicts.

**Batch insert throughput**: ~85,000 inserts/second at 384 dims (M1 Pro benchmark).

**SIMD acceleration**: AVX (x86-64) and NEON (ARM64) intrinsics for float32/int8.

#### 3.1.2 Concrete Benefits vs. ChromaDB

| Dimension | sqlite-vec | ChromaDB |
|-----------|-----------|----------|
| Python dependencies | **Zero** | 25+ packages (grpcio, posthog, hnswlib, etc.) |
| Install size | ~500KB–2MB | 20+ MB + transitive |
| DB files | **1 file** (`lsm.db`) | Multiple (chroma.sqlite3 + HNSW index files) |
| Transactional writes | **Yes — ACID across vectors + FTS5** | No cross-table transactions |
| BM25 sync problem | **Eliminated** (same transaction) | Present (separate writes) |
| Index type | Brute-force KNN | HNSW (approximate) |
| Backup | **One file to copy** | Multiple files, potential inconsistency |
| Native FTS5 | **Yes — same DB** | No (requires sidecar) |
| Telemetry | **None** | posthog telemetry (opt-out required) |
| Background processes | **None** | In-process HTTP server |

**Known limitation**: No ANN index. Brute-force KNN at 100k vectors/384 dims gives
15–68ms. Binary quantization (`BIT[N]`) reduces memory by 32× at some accuracy cost.

### 3.2 Unified Database Schema

**SQLite-vec provider — `lsm.db`**:

```sql
-- ═══════════════════════════════════════════════════════
-- Core chunk metadata (standard table)
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_chunks (
    chunk_id        TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    source_name     TEXT,
    chunk_text      TEXT NOT NULL,
    heading         TEXT,
    heading_path    TEXT,           -- JSON array, e.g. '["Introduction","Background"]'
    page_number     TEXT,
    paragraph_index INTEGER,
    mtime_ns        INTEGER,
    file_hash       TEXT,
    version         INTEGER,
    is_current      INTEGER DEFAULT 1,
    node_type       TEXT DEFAULT 'chunk',
    root_tags       TEXT,           -- JSON
    folder_tags     TEXT,           -- JSON
    content_type    TEXT,
    cluster_id      INTEGER,        -- Phase 2 (null until clustering job runs)
    cluster_size    INTEGER,        -- Phase 2 (null until clustering job runs)
    simhash         INTEGER         -- MinHash signature for near-dup detection
);

-- ═══════════════════════════════════════════════════════
-- Vector search (vec0 virtual table)
-- ═══════════════════════════════════════════════════════
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    chunk_id    TEXT PRIMARY KEY,
    embedding   FLOAT[384] distance_metric=cosine,
    is_current  INTEGER,
    node_type   TEXT,
    source_path TEXT,
    cluster_id  INTEGER
);

-- ═══════════════════════════════════════════════════════
-- Full-text search (FTS5 virtual table)
-- ═══════════════════════════════════════════════════════
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    chunk_text,
    heading,
    source_name,
    content=lsm_chunks,
    content_rowid=rowid
);

-- ═══════════════════════════════════════════════════════
-- Manifest (file-level ingest tracking)
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_manifest (
    source_path     TEXT PRIMARY KEY,
    mtime_ns        INTEGER,
    file_size       INTEGER,
    file_hash       TEXT,
    version         INTEGER,
    embedding_model TEXT,
    schema_version_id INTEGER REFERENCES lsm_schema_versions(id),
    updated_at      TEXT
);

-- ═══════════════════════════════════════════════════════
-- Schema version tracking
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_schema_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_version INTEGER,
    lsm_version     TEXT,
    embedding_model TEXT,
    embedding_dim   INTEGER,
    chunking_strategy TEXT,
    chunk_size      INTEGER,
    chunk_overlap   INTEGER,
    created_at      TEXT,
    last_ingest_at  TEXT
);

-- ═══════════════════════════════════════════════════════
-- Cross-encoder reranker cache
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_reranker_cache (
    cache_key   TEXT PRIMARY KEY,   -- hash(query_hash, chunk_id, model_version)
    score       REAL,
    created_at  TEXT
);

-- ═══════════════════════════════════════════════════════
-- Agent memory (replaces standalone memories.db)
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_agent_memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    scope TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    expires_at TEXT NULL,
    source_run_id TEXT NOT NULL
);

CREATE TABLE lsm_agent_memory_candidates (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL UNIQUE,
    provenance TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES lsm_agent_memories(id) ON DELETE CASCADE
);

CREATE INDEX idx_lsm_agent_memory_candidates_status
ON lsm_agent_memory_candidates(status);
CREATE INDEX idx_lsm_agent_memories_scope_type
ON lsm_agent_memories(scope, memory_type);
CREATE INDEX idx_lsm_agent_memories_expires_at
ON lsm_agent_memories(expires_at);

-- ═══════════════════════════════════════════════════════
-- Agent schedule state (replaces schedules.json)
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_agent_schedules (
    schedule_id     TEXT PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    last_run_at     TEXT,
    next_run_at     TEXT NOT NULL,
    last_status     TEXT DEFAULT 'idle',
    last_error      TEXT,
    queued_runs     INTEGER DEFAULT 0,
    updated_at      TEXT NOT NULL
);

-- ═══════════════════════════════════════════════════════
-- Phase 2: Cluster centroids
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_cluster_centroids (
    cluster_id  INTEGER PRIMARY KEY,
    centroid    BLOB,               -- serialized float32 array (or use vec0)
    size        INTEGER
);

-- ═══════════════════════════════════════════════════════
-- Phase 3: Graph store
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_graph_nodes (
    node_id     TEXT PRIMARY KEY,
    node_type   TEXT,               -- "file", "section", "chunk", "entity"
    label       TEXT,
    source_path TEXT,
    heading_path TEXT
);

CREATE TABLE lsm_graph_edges (
    edge_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id      TEXT REFERENCES lsm_graph_nodes(node_id),
    dst_id      TEXT REFERENCES lsm_graph_nodes(node_id),
    edge_type   TEXT,               -- "contains", "references", "same_author", "thematic_link"
    weight      REAL DEFAULT 1.0
);

-- ═══════════════════════════════════════════════════════
-- Phase 3: Embedding model registry
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_embedding_models (
    model_id    TEXT PRIMARY KEY,
    base_model  TEXT,
    path        TEXT,               -- local path or HuggingFace model ID
    dimension   INTEGER,
    created_at  TEXT,
    is_active   INTEGER DEFAULT 0
);

-- ═══════════════════════════════════════════════════════
-- Offline job status tracking (TUI startup advisories)
-- ═══════════════════════════════════════════════════════
CREATE TABLE lsm_job_status (
    job_name        TEXT PRIMARY KEY,
    status          TEXT NOT NULL,     -- "running", "completed", "failed"
    started_at      TEXT,
    completed_at    TEXT,
    corpus_size     INTEGER,
    metadata        TEXT               -- JSON
);
```

**PostgreSQL provider**: Equivalent tables in PostgreSQL with pgvector for the embedding
column and a `tsvector GENERATED ALWAYS AS` column on `lsm_chunks` for native FTS.

### 3.3 Agent Data Consolidation (memories.db + schedules.json)

**Current state**:
- `SQLiteMemoryStore` creates a standalone `memories.db` at `memory_config.sqlite_path`
  (typically `<agents_folder>/memories.db`). Tables: `memories`, `memory_candidates`.
- `AgentScheduler` writes persistent state to `<agents_folder>/schedules.json` —
  per-schedule `last_run_at`, `next_run_at`, `last_status`, `last_error`, `queued_runs`.

**Decision — Merge into `lsm.db`**: Both agent memory and schedule state move into the
unified database. The `lsm_agent_memories` and `lsm_agent_memory_candidates` tables in
`lsm.db` replace the standalone `memories.db`. The `lsm_agent_schedules` table replaces
`schedules.json`.

**Impact on agent code**:
- `SQLiteMemoryStore.__init__` no longer creates its own `sqlite3.connect()`. Instead it
  receives the shared `lsm.db` connection (or connection factory) from the provider.
- `AgentScheduler._save_state_locked()` and `_load_state()` use SQL instead of JSON I/O.
- The `agents_folder` no longer needs a `memories/` subdirectory or `schedules.json` file.
  Agents still have `artifacts/` and `logs/` directories under `agents_folder`.
- `MemoryConfig.sqlite_path` is removed. Memory storage is always co-located with the
  active vector DB backend.

**Migration**: Fully explicit via `lsm migrate`. Startup and ingest paths do not
auto-migrate legacy agent state. Migration imports legacy `memories.db` and
`schedules.json` only when the user runs `lsm migrate`.

**PostgreSQL path**: When `provider: "postgresql"`, agent memories and schedules use the
same PostgreSQL database with `lsm_agent_memories` / `lsm_agent_schedules` tables
(matching the current `PostgreSQLMemoryStore` pattern but with standardized table names).

### 3.4 SQLite-Backed Manifest

**Decision**: The flat JSON manifest (`manifest.json`) is replaced with the
`lsm_manifest` table in `lsm.db` (see §3.2).

**Rationale**: Corpus sizes of 100k+ files are expected. A 100k-file flat JSON manifest
(~20 MB) has ~200ms load/save time and no query capability. SQLite provides sub-
millisecond lookups, indexed queries, transactional writes, and in-place updates.

**Current code impact**: `manifest.py` (36 lines) — `load_manifest`, `save_manifest`,
`get_next_version` — is rewritten to use SQL. The public API remains the same but backed
by DB queries instead of JSON I/O.

**Migration**: Existing `manifest.json` files migrate only when the user runs
`lsm migrate`. There is no automatic migration at startup or ingest.

### 3.5 Schema Versioning

#### 3.5.1 Schema Version Concept

A "schema version" captures the provenance of the vector corpus:

| Dimension | Why It Matters |
|-----------|----------------|
| Embedding model name + dimension | Different models produce incompatible vector spaces |
| Chunking strategy + params | `structure` vs `fixed`, chunk_size, overlap |
| Metadata field set | Added fields don't backfill old chunks |
| Parser version | Parser behaviour changes affect chunk content |
| LSM version | Code version that produced the chunks |

Without schema versioning, any upgrade silently creates a mixed-generation corpus — old
and new chunks coexist with different vector spaces and metadata layouts.

#### 3.5.2 Storage

Schema version is stored only in `lsm_schema_versions` (§3.2). `lsm_manifest` stores
`schema_version_id` as a foreign key to the active schema version for each file record.
There is no JSON `_schema` sidecar in v0.8.0.

#### 3.5.3 Versioning Always On

**Decision**: The `enable_versioning` flag on `IngestConfig` is removed. Versioning is
the unconditional operating mode. The VectorDB provider gains a
`prune_old_versions(criteria: PruneCriteria) → int` method for intelligent cleanup of
soft-deleted chunks. An `lsm db prune` CLI command provides controlled cleanup.

#### 3.5.4 Migration / Upgrade Path

**Decision — Schema mismatch handling**: When the system detects a schema mismatch (e.g.,
config `embed_model` ≠ recorded `embedding_model`), it raises an error with clear
instructions on how to run `lsm migrate`. It does not proceed silently.

**Decision — Migration entry points**:
- `lsm migrate` — only migration command.
- The TUI Ingest screen surfaces migration warnings and provides a "Migrate" action.
- Migration never runs automatically at startup or ingest time.

**Migration strategies**:

| Strategy | Description | Trade-offs |
|----------|-------------|------------|
| **Full rebuild** | Wipe and re-ingest entire corpus | Simplest; expensive for large corpora |
| **Selective re-embed** | Re-embed only files using the old model | Requires per-file model provenance |
| **Incremental migration job** | Background job progressively re-embeds old chunks | Low disruption; requires per-chunk tracking |

### 3.6 DB Completion — Incremental Corpus Updates

#### 3.6.1 The Problem

"DB completion" means: given an existing vector DB built with LSM version X, bring it up
to date with version Y without a full rebuild. Cases:

1. **New file types**: New parsers (`.epub`, `.pptx`) — ingest only newly-supported types.
2. **New metadata fields**: `heading_path`, language tags — enrich without re-embedding.
3. **Changed chunking**: Must re-chunk and re-embed affected files.
4. **New ingest features**: AI tagging, language detection — re-tag without re-embedding.

#### 3.6.2 What Already Works

The current incremental system (`pipeline.py:620-647`) skips unchanged files via mtime →
size → hash. The gap is a file unchanged on disk whose vector representation or metadata
is stale by the codebase's definition — the manifest returns "skip" when it should
trigger re-ingest.

#### 3.6.3 Completion Modes

| Mode | Trigger | Scope | Cost |
|------|---------|-------|------|
| **Extension completion** | New `extensions` in config | Only new file types | Low |
| **Metadata enrichment** | New metadata field available | All files, no re-embedding | Medium |
| **Chunk boundary update** | Chunking params changed | All files for that strategy | High |
| **Embedding upgrade** | Embedding model changed | All files | Very high |
| **Selective re-ingest** | Manual `--force-file-pattern` | User-specified subset | Variable |

`lsm db complete` (or `--force-reingest-changed-config` on ingest) compares recorded
schema against current config and re-ingests only files whose chunks would differ.

### 3.7 Migration Paths

| From | To | Command |
|------|----|---------
| ChromaDB | SQLite-vec | `lsm migrate --from chroma --to sqlite` |
| ChromaDB | PostgreSQL | `lsm migrate --from chroma --to postgresql` |
| SQLite-vec | PostgreSQL | `lsm migrate --from sqlite --to postgresql` |
| PostgreSQL | SQLite-vec | `lsm migrate --from postgresql --to sqlite` |
| v0.7 sidecars | Unified DB state | `lsm migrate --from v0.7 --to v0.8` |

All migration paths copy vectors, chunk metadata, manifest entries, agent memories, and
schedule state. Schema versions are re-derived from the active config at migration time.

### 3.8 Config Changes (vectordb)

The `vectordb` config section is simplified (breaking change from v0.7.x):

```yaml
vectordb:
  provider: "sqlite"          # "sqlite" (default) or "postgresql"
  path: "./data"              # SQLite: directory for lsm.db; PostgreSQL: ignored
  collection: "lsm_chunks"   # table name prefix (optional)
  # PostgreSQL only:
  connection_string: ""       # or set LSM_POSTGRES_CONNECTION_STRING env var
```

The `persist_dir`, `chroma_hnsw_space`, and `chroma_flush_interval` config keys are
removed. No backward compatibility — config files must be updated.

Related ingest config change:
- `ingest.manifest` is removed in v0.8.0. Manifest state is DB-only (`lsm_manifest`).

### 3.9 JSON Sidecar Audit (Non-Log, Non-Agent/Chat Artifacts)

To enforce the DB-only manifest decision and reduce state fragmentation, v0.8.0 audits
all non-log, non-agent/chat file sidecars:

| Artifact | Current Behavior | v0.8.0 Decision |
|----------|------------------|-----------------|
| `.ingest/manifest.json` | Ingest state sidecar | **Removed**. Replaced by `lsm_manifest`. Migrated only via explicit `lsm migrate`. |
| `<vectordb_path>/stats_cache.json` | Cached ingest stats sidecar | **Move to DB** as runtime cache rows (SQLite table / PostgreSQL table) with TTL. |
| `remote` cache `*.json` blobs (`lsm/remote/storage.py`) | Provider response cache files | **Move to DB** (`lsm_remote_cache`) for unified backup/restore and transactional invalidation. |
| `.lsm_tags.json` in source folders | User-authored source metadata input | **Retained**. Treated as corpus input, not LSM runtime sidecar state. |

All runtime-sidecar migrations in this table run only via explicit `lsm migrate`.

Out of scope for this audit by explicit instruction: log/report artifacts (for example
ingest error reports) and products of agents/chats.

---

## 4. Part B: Ingest Pipeline Enhancements

### 4.1 Document Headings: Configurable Depth

**Decision — Implemented in v0.8.0.**

`max_heading_depth: Optional[int] = None` on `IngestConfig`. `None` = all heading levels
are chunk boundaries (current behaviour).

**Decision — Global + per-root overrides**: Set globally in `IngestConfig`, overridable
per `RootConfig`. Allows depth 2 for personal notes and depth 4 for code docs in the same
instance.

**Behaviour with `max_heading_depth=2`**:
```
# H1 Topic          ← chunk boundary (depth 1 ≤ 2)
## H2 Subtopic      ← chunk boundary (depth 2 ≤ 2)
### H3 Detail       ← NOT a boundary; flows into H2 chunk
#### H4 Sub-detail  ← NOT a boundary; flows into H2 chunk
```

**Applies only to `chunking_strategy: "structure"`.** The `fixed` strategy is unchanged.

### 4.2 Document Headings: Intelligent Selection (FileGraph Integration)

**Decision — Implemented in v0.8.0. Built on existing `lsm/utils/file_graph.py`.**

`intelligent_heading_depth: bool = False` on `IngestConfig`. When enabled, heading depth
is decided dynamically per section.

**FileGraph integration**: Instead of reimplementing heading detection in
`structure_chunking.py`, the ingest pipeline generates a `FileGraph` for each document
first, then uses the graph to make chunking decisions:

1. **Parse document → `FileGraph`**: Use `build_markdown_graph()`, `build_text_graph()`,
   `build_html_graph()`, or `build_docx_graph()` from `lsm/utils/file_graph.py`. The
   `FileGraph` contains all heading nodes with `depth`, `start_char`, `end_char`,
   `parent_id`, and `children`.
2. **Walk heading tree for chunk boundaries**: For each heading `GraphNode` in the
   `FileGraph`:
   - Compute section size from `end_char - start_char`.
   - If size ≤ `chunk_size`: keep as one chunk; sub-headings flow into body text.
   - If size > `chunk_size` AND has children of type `"heading"`: split at child heading
     boundaries, recurse.
   - Recursion depth bounded by `max_heading_depth` (if set) or by the tree depth.
3. **Extract chunks**: Use `GraphNode.start_char` / `end_char` spans to extract text
   segments from the raw document content. Attach `heading_path` metadata from the
   `GraphNode` hierarchy.

**Why FileGraph**: `file_graph.py` already implements the heading hierarchy, parent-child
tracking, depth, and span boundaries for Markdown, text, HTML, and DOCX. The `GraphNode`
already carries `metadata["level"]` for heading depth. Reusing it avoids duplicating the
heading detection logic that exists in `text_processing.py` and `file_graph.py`.

**Current `structure_chunking.py` change**: The function `structure_chunk_text()` gains
an optional `file_graph: Optional[FileGraph] = None` parameter. When provided, heading
detection uses the pre-built graph instead of regex-based heading detection. When not
provided (e.g., for formats without a graph builder), the existing regex path is used.

### 4.3 heading_path Metadata

**Decision**: A `heading_path: list[str]` field records the full hierarchy (e.g.,
`["Introduction", "Background", "Prior Work"]`), stored as a JSON array in
`lsm_chunks.heading_path`. The flat `heading` string field is retained for BM25 text
indexing.

**Construction from FileGraph**: Walk `parent_id` chain from the heading `GraphNode` up
to the document root, collecting `name` fields. Reverse to get root-to-leaf order.

**DB versioning impact**: Heading depth changes produce different `chunk_id` values.
Files with stale heading metadata are re-ingested automatically (not requiring manual
`lsm migrate`).

### 4.4 Phase 2: Multi-Vector Representation (Ingest-Side)

Multi-vector representation stores embeddings at multiple granularities:
- **Chunk-level** (`node_type = "chunk"`): current default; fine-grained semantic search.
- **Section-level** (`node_type = "section_summary"`): embedding of a section's summary.
- **File-level** (`node_type = "file_summary"`): embedding of a document summary.

All granularities stored in the same `vec_chunks` table, distinguished by `node_type`.

**Summary generation at ingest time**:
- **Section summary**: LLM summarises all text under a heading → embedded → stored with
  `node_type = "section_summary"` and the heading's `heading_path`.
- **File summary**: LLM summarises the document (or first N paragraphs + headings outline)
  → embedded → stored with `node_type = "file_summary"`.

Summary generation requires an LLM call per section/file. This is opt-in:
```python
enable_section_summaries: bool = False
enable_file_summaries: bool = False
```

### 4.5 Phase 3: Graph Construction at Ingest Time

Graph construction populates `lsm_graph_nodes` and `lsm_graph_edges` during ingest:

**From FileGraph / heading_path metadata** (ingest time): Each `heading_path` level
becomes a section node; parent-child relationships become `"contains"` edges. The
`FileGraph` from §4.2 provides the structural graph directly — `GraphNode.parent_id` and
`GraphNode.children` map to `"contains"` edges.

**From file references** (ingest time): Markdown `[[wikilinks]]`, `[text](path)` internal
links, and citation DOIs are parsed to create `"references"` edges.

**From entity extraction** (ingest time): NER or LLM-based entity extraction creates
entity nodes and `"mentioned_in"` edges. Requires LLM or `spacy`.

**Thematic links** (offline job): Cosine similarity above a threshold between chunk
embeddings creates `"thematic_link"` edges. Run as `lsm graph build-links`.

### 4.6 Phase 3: Domain-Fine-Tuned Embeddings (Ingest-Side)

**Training signal**: Heading-content pairs from the `FileGraph` — `(heading_text,
section_body)` → heading is the query, section body is the positive passage. User
interaction signals provide additional weak supervision.

**Fine-tuning CLI**:
```
lsm finetune embedding --base-model all-MiniLM-L6-v2 --epochs 3 --output ./models/finetuned
```

The `lsm_embedding_models` table (§3.2) tracks fine-tuned models. Switching models
triggers schema mismatch detection (§3.5.4).

---

## 5. Part C: Query Pipeline Overhaul

### 5.1 Current Query Architecture

See §2.2. The current pipeline is a linear chain of free functions in `retrieval.py`,
`rerank.py`, `planning.py`, `context.py`, and `api.py`. There is no stage abstraction,
no pluggable profiles, and no score breakdown.

### 5.2 Unified RetrievalPipeline Abstraction

**Framing**: The query pipeline is a **Context Builder**. Its job is to produce a
structured `ContextPackage` — local candidates with score breakdowns, remote sources, and
a pre-formatted LLM context block — that is then handed to an LLM for synthesis. Both
users (TUI/shell) and agents are doing the same thing: building a `ContextPackage` and
consuming the result. The pipeline is the single code path for both.

**Decision — Three-stage API**: The `RetrievalPipeline` exposes three composable steps
plus a high-level convenience entry point:

```python
class RetrievalPipeline:

    def build_sources(self, request: QueryRequest) -> ContextPackage:
        """
        Stage 1: Retrieve and rank local candidates + fetch remote sources.
        Runs: recall → fuse → rerank → diversify → remote fetch.
        Returns a ContextPackage with candidates, remote sources, and retrieval trace.
        No LLM synthesis call is made here.
        """

    def synthesize_context(self, package: ContextPackage) -> ContextPackage:
        """
        Stage 2: Format the ContextPackage into a context block for LLM consumption.
        Assigns [S1]/[S2]/... labels, builds source_labels map, applies any
        context-window trimming. Resolves starting_prompt using session caches.
        Returns an enriched ContextPackage ready for execute().
        No LLM call is made here — this is pure formatting.
        """

    def execute(self, package: ContextPackage) -> QueryResponse:
        """
        Stage 3: Send the formatted ContextPackage to the synthesis LLM.
        Reads package.starting_prompt as the instruction and package.context_block
        as the input. Calls provider.send_message() — the provider has no knowledge
        of synthesis semantics. Returns a QueryResponse.
        """

    def run(self, request: QueryRequest) -> QueryResponse:
        """
        High-level entry point: build_sources → synthesize_context → execute.
        Used by the TUI/shell user query path and agents that don't need
        to inspect the intermediate ContextPackage.
        """
        package = self.build_sources(request)
        package = self.synthesize_context(package)
        return self.execute(package)
```

**Why three stages**: Agents frequently need to inspect or modify the `ContextPackage`
between retrieval and synthesis. An agent may want to filter candidates, inject additional
context, or route the package to a different LLM. The three-stage design gives agents that
control without duplicating pipeline logic. For users, `run()` provides the single-call
interface.

### 5.3 QueryRequest and Starting Prompt Handling

**`QueryRequest`**: The unified input for both users and agents.

```python
@dataclass
class QueryRequest:
    query: str
    mode: Optional[ModeConfig] = None       # Full mode object (built-in or caller-composed)
    starting_prompt: Optional[str] = None   # Custom system prompt override
    filters: Optional[FilterSet] = None     # path_contains, ext_allow, ext_deny
    pinned_chunks: Optional[List[str]] = None
    context_documents: Optional[List[str]] = None
    context_chunks: Optional[List[str]] = None
    k_override: Optional[int] = None        # Override mode's final k
    conversation_id: Optional[str] = None   # LLM server-side cache / conversation thread ID
    prior_response_id: Optional[str] = None # Provider-specific ID for conversation chaining
    trace_enabled: bool = True
```

**Mode model decision**: `QueryRequest` carries exactly one mode field: `mode:
Optional[ModeConfig]`. Built-in modes (`grounded`, `insight`, `hybrid`) are prebuilt
`ModeConfig` objects exported from `lsm/config/models/modes.py`. Callers either pass one
of those built-ins or a custom `ModeConfig`; there is no separate `mode_name` +
`mode_config` dual path.
If `mode` is `None`, the pipeline resolves to `GROUNDED_MODE`.

**Starting prompt resolution**: The `starting_prompt` field on `QueryRequest` allows
callers to override the system prompt used for synthesis. The resolution order in
`synthesize_context()` is:

1. **Explicit `starting_prompt`** on `QueryRequest` — highest priority. If set, used
   directly as the instruction for `execute()`.
2. **Session cache lookup** — `synthesize_context()` receives session cache state (the
   same `llm_server_cache_ids` currently on `SessionState`) and determines whether a
   server-side cache continuation applies. If continuing a cached conversation, the
   starting prompt from the first turn is implicit (server-side) and no explicit prompt
   is set.
3. **Mode-derived default** — If neither explicit nor cached, the starting prompt is
   read from `request.mode.synthesis_instructions` (see §5.11). The synthesis prompt
   templates (currently `SYNTHESIZE_GROUNDED_INSTRUCTIONS` and
   `SYNTHESIZE_INSIGHT_INSTRUCTIONS` in `providers/helpers.py`) move to `ModeConfig`
   as default values. Each mode carries its own synthesis instructions.

The resolved starting prompt is stored in `ContextPackage.starting_prompt` so that
`execute()` can pass it to `provider.send_message(instruction=...)` without
re-resolving. The pipeline is stateless regarding prompts — all prompt state flows
through `QueryRequest` → `ContextPackage` → provider call.

**Conversation state migration**: `conversation_id` and `prior_response_id` on
`QueryRequest`/`QueryResponse` replace the `SessionState.llm_server_cache_ids` dict.
`SessionState` retains its role as a TUI-session artifact store (last candidates, last
answer, filter overrides) but is no longer the authoritative holder of conversation state.

### 5.4 ContextPackage and QueryResponse

**`ContextPackage`**: The first-class output of `build_sources()` and
`synthesize_context()`.

```python
@dataclass
class ContextPackage:
    query: str
    query_config: QueryConfig              # Resolved config (LLM, k, profile) for execute()
    local_candidates: List[Candidate]      # With ScoreBreakdown attached
    remote_sources: List[RemoteSource]     # From remote providers (if mode has remote enabled)
    context_block: str                     # Formatted [S1]/[S2]/... LLM context (after Stage 2)
    source_labels: Dict[str, Candidate]    # "S1" → Candidate (after Stage 2)
    starting_prompt: Optional[str]         # Resolved instruction for provider (after Stage 2)
    synthesis_style: str                   # "grounded" | "insight" — informational only after resolution
    model_knowledge_enabled: bool          # Whether to append model knowledge banner
    conversation_id: Optional[str]         # Carried from QueryRequest; passed to LLM provider
    prior_response_id: Optional[str]       # Carried from QueryRequest; for provider chaining
    retrieval_trace: RetrievalTrace        # Full trace (§7.3)
    retrieval_cost: Optional[CostEntry]    # Cost for embedding + HyDE generation (if any)
```

**`QueryResponse`**: The output of `execute()`.

```python
@dataclass
class QueryResponse:
    answer: str
    package: ContextPackage                # The ContextPackage that produced this answer
    citations: List[Citation]              # Resolved citations extracted from answer
    synthesis_cost: CostEntry             # LLM synthesis token cost
    conversation_id: Optional[str]         # Updated conversation ID
    response_id: Optional[str]             # Provider response ID for next turn
    total_cost: CostEntry                  # retrieval_cost + synthesis_cost
```

**Supporting types**:

- **`Candidate`**: Extended with `ScoreBreakdown` and its embedding vector (required for
  MMR diversity selection).
- **`ScoreBreakdown`**: Per-candidate scoring detail:
  - `dense_score`, `dense_rank`, `sparse_score`, `sparse_rank`, `fused_score`,
    `rerank_score`, `temporal_boost`.
- **`Citation`**: Resolved reference: `chunk_id`, `source_path`, `heading`,
  `page_number`, `url_or_doi`, `snippet`.

### 5.5 Retrieval Profiles

Five profiles, selectable via `retrieval_profile` config or at query time:

```yaml
retrieval_profiles:
  - dense_only           # baseline: single-model ANN only
  - hybrid_rrf           # dense ANN + BM25/FTS5 + RRF fusion  ← DEFAULT
  - hyde_hybrid          # HyDE embedding + hybrid_rrf pipeline
  - dense_cross_rerank   # dense ANN + cross-encoder reranking
  - llm_rerank           # dense ANN + LLM reranking (replaces legacy rerank_strategy: "llm")
```

**Decision — `hybrid_rrf` is the default profile**: Once the FTS5 index exists,
`hybrid_rrf` is the default. Before the first FTS5 index is built, the profile degrades
gracefully to `dense_only` with a log warning.

**Decision — Graceful degradation**:
- `hybrid_rrf` without FTS5 index → `dense_only` + warning.
- `dense_cross_rerank` without cross-encoder model → ANN results, no reranking.
- `hyde_hybrid` with LLM failure → direct query embedding.
- `temporal_boost` config absent → boost stage skipped.

**Decision — Hard-coded profiles, not a registry**: Five profiles are hard-coded for
v0.8.0. A registry adds indirection with no benefit for a fixed profile set.

Retrieval Profiles define the **mechanism** of retrieval. Modes define the **intent** —
which sources to query, how many results to keep, and how to synthesize. A Mode always
references one Retrieval Profile by name. See §5.11.

The legacy `rerank_strategy` config key is removed (breaking change).

### 5.6 Hybrid Retrieval: Dense + Sparse (BM25/FTS5) + RRF

With the unified SQLite-vec database (§3.1), both vector search and FTS5 search run
against the same `lsm.db` file. The RRF merge in Python is preferred for the
`RetrievalPipeline` abstraction, as it preserves stage separation and score tracking.

**Recall pool sizing**:
- `k_dense: int = 100` — candidates from vector ANN
- `k_sparse: int = 100` — candidates from BM25/FTS5

**RRF formula**: `score(d) = Σ w_i / (k + rank_i(d))` where `k=60` (standard default).
Per-channel weights configurable:
- `rrf_dense_weight: float = 0.7`
- `rrf_sparse_weight: float = 0.3`

**Score breakdown**: Every RRF-merged candidate carries `dense_rank`, `sparse_rank`, and
`fused_score` in its `ScoreBreakdown`.

**Provider abstraction**: `BaseVectorDBProvider` gains:
```python
def fts_query(self, text: str, top_k: int) -> VectorDBQueryResult: ...
```
- **SQLite-vec provider**: delegates to `chunks_fts` FTS5 virtual table.
- **PostgreSQL provider**: uses native `tsvector`/`tsquery` and `ts_rank`.

**Synchronisation**: With the unified database, vector writes (to `vec_chunks`) and FTS5
writes (to `chunks_fts`) happen in the **same SQLite transaction**. No sync problem.

### 5.7 Cross-Encoder Reranking

**Available models** (local, via `sentence-transformers`):

| Model | Size | MRR@10 on MS-MARCO |
|-------|------|---------------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 66M | 39.01 |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 127M | 39.97 |
| `cross-encoder/ms-marco-electra-base` | 109M | 39.96 |

**Decision — Lazy download**: Model downloaded on first use of `dense_cross_rerank`.

**Decision — CUDA/GPU support**: `CrossEncoder` honours the `device` parameter from
`GlobalConfig`.

**Decision — Reranker cache**: Persisted to the `lsm_reranker_cache` table in `lsm.db`
(§3.2). Cache key: `(query_hash, chunk_id, model_version)`. `lsm cache clear` provides
manual invalidation.

**LLM reranking**: The existing `rerank_strategy: "llm"` path becomes the `llm_rerank`
profile. The `rerank_strategy` config key is removed (breaking change).

**Prompt ownership**: The LLM rerank stage owns its prompt template
(`RERANK_INSTRUCTIONS`), candidate preparation (`prepare_candidates_for_rerank()`), and
response parsing (`parse_ranking_response()`). These move from `providers/helpers.py`
and `providers/base.py` into the query pipeline (see §6.6). The stage calls
`provider.send_message(instruction=rerank_prompt, input=payload, ...)` — the provider
has no knowledge of reranking semantics.

### 5.8 HyDE (Hypothetical Document Embeddings)

**Mechanism**:
1. LLM generates 1–3 hypothetical answers to the query (zero-shot).
2. Each is embedded with the same bi-encoder.
3. Embeddings are pooled (mean or max).
4. Pooled embedding replaces direct query embedding for ANN search.

**Decision — Profile-only**: HyDE is exclusively the `hyde_hybrid` profile. Disabled by
default.

**Decision — Observability**: Hypothetical documents are logged in
`ContextPackage.retrieval_trace` and viewable in the TUI debug/trace view.

**Tuning parameters**:
- `hyde_num_samples: int = 2`
- `hyde_temperature: float = 0.2`
- `hyde_pooling: str = "mean"` (options: `"mean"`, `"max"`)

### 5.9 Diversity and De-duplication

**Current state**: Exact hash dedup (`rerank.py:154`) + per-file cap (`rerank.py:204`).

**Decision — MinHash near-duplicate detection**: MinHash (Jaccard similarity on shingles)
replaces exact hash dedup. A MinHash signature is stored per chunk at ingest time (`simhash`
column in `lsm_chunks`). At retrieval time, candidates exceeding a configurable similarity
threshold are suppressed.

**Decision — Greedy MMR as default**: Maximal Marginal Relevance (MMR) is the default
post-rerank selection strategy. MMR iteratively picks the next candidate maximising
relevance while minimising similarity to already-selected results.

**Decision — Per-section cap default = 3**: `max_per_section: Optional[int] = 3` caps
chunks per heading-path prefix. Requires `heading_path` metadata from §4.3 to be populated.

### 5.10 Temporal-Aware Ranking

`mtime_ns` is already stored in chunk metadata (`pipeline.py:401`). In v0.8.0:

- **Recency boost**: Configurable decay. Default: `1.5×` for files modified within 30 days.
- **Time-range filter**: `WHERE mtime_ns BETWEEN ? AND ?` for queries like "notes from 2023".

```python
temporal_boost_enabled: bool = False
temporal_boost_days: int = 30
temporal_boost_factor: float = 1.5
```

### 5.11 Mode as Composition Preset

**Decision — A Mode is a first-class `ModeConfig` preset object.**
A Mode bundles five things:

1. **`retrieval_profile`**: Which `RetrievalPipeline` profile to run (§5.5).
2. **`local_policy`**: Source scope — `k` (final candidate count after reranking),
   `min_relevance` (gating threshold), `enabled`.
3. **`remote_policy`**: Which remote providers to query, `max_results`, `rank_strategy`,
   per-provider weight overrides.
4. **`model_knowledge_policy`**: Whether to append a model-knowledge banner.
5. **`synthesis_style`**: `"grounded"` or `"insight"` — forwarded to `execute()`.

**Updated `ModeConfig`** (breaking change from v0.7.x):

```python
@dataclass
class ModeConfig:
    retrieval_profile: str = "hybrid_rrf"          # one of the §5.5 profile names
    local_policy: LocalSourcePolicy = field(...)    # k, min_relevance, enabled
    remote_policy: RemoteSourcePolicy = field(...)  # providers, max_results, rank_strategy
    model_knowledge_policy: ModelKnowledgePolicy = field(...)
    synthesis_style: str = "grounded"              # "grounded" | "insight"
    synthesis_instructions: str = SYNTHESIZE_GROUNDED_INSTRUCTIONS  # prompt template
    chats: Optional[ModeChatsConfig] = None        # chat-mode overrides (auto_save, dir)
```

**`synthesis_instructions`**: The full system prompt template used for synthesis. Each
built-in mode carries its own default:
- `grounded` mode → `SYNTHESIZE_GROUNDED_INSTRUCTIONS` (cite sources, answer from context)
- `insight` mode → `SYNTHESIZE_INSIGHT_INSTRUCTIONS` (identify themes, contradictions, gaps)

These prompt templates currently live in `providers/helpers.py`. In v0.8.0 they move to
the mode definitions (see §6.6). Users can override `synthesis_instructions` per mode in
config to customise synthesis behaviour without modifying code.

The pipeline reads `synthesis_instructions` from the resolved `ModeConfig` and stores it
in `ContextPackage.starting_prompt` (unless overridden by `QueryRequest.starting_prompt`).
The provider receives it as a generic `instruction` parameter with no knowledge of
synthesis semantics.

`LocalSourcePolicy.k` is the **post-rerank final candidate count** (what was `k_rerank`
in v0.7.x). `k_rerank` is removed from `LocalSourcePolicy`. The pipeline's `k_dense` and
`k_sparse` (recall pool sizes, §5.6) are `QueryConfig`-level settings, not mode-level.

**Built-in mode objects** (declared in `lsm/config/models/modes.py`):

```python
GROUNDED_MODE = ModeConfig(...)
INSIGHT_MODE = ModeConfig(...)
HYBRID_MODE = ModeConfig(...)
BUILT_IN_MODES: Dict[str, ModeConfig] = {
    "grounded": GROUNDED_MODE,
    "insight": INSIGHT_MODE,
    "hybrid": HYBRID_MODE,
}
```

`QueryRequest.mode` accepts one of these objects (or a custom `ModeConfig`).
CLI/TUI mode names are resolved to these objects before constructing `QueryRequest`.
Default when unset: `GROUNDED_MODE`.

**Updated built-in mode presets**:

| Mode | retrieval_profile | k | min_relevance | remote | synthesis_style |
|------|-------------------|---|---------------|--------|-----------------|
| `grounded` | `hybrid_rrf` | 12 | 0.25 | off | grounded |
| `insight` | `hybrid_rrf` | 8 | 0.0 | on (max 5) | insight |
| `hybrid` | `hybrid_rrf` | 12 | 0.15 | on (max 5) | grounded |

**Decision — No automatic config migration**: v0.8.0 is a breaking release. Users update
their own config files. Upgrade docs cover key renames.

**Decision — Agent-composed modes**: Agents may pass a custom `ModeConfig` directly via
`QueryRequest.mode`. Before the pipeline accepts it, `AgentHarness` validates it
against global settings: the `retrieval_profile` must be listed in
`QueryConfig.retrieval_profiles`, and `remote_policy.enabled` requires
`allow_url_access=true` in the agent's sandbox config.

### 5.12 Agent Integration: Unified Tool Surface

The current agent tools (`query_knowledge_base`, `query_llm`, per-source `query_<provider>`,
`query_remote_chain`, `extract_snippets`, `similarity_search`) are reorganized:

**`query_context` (new — replaces `query_knowledge_base` for retrieval-only)**

```
execute(query, mode=None, filters=None, k=None) → ContextPackage
```

`mode` is resolved to a `ModeConfig` object before `QueryRequest` construction
(built-in preset or custom object).

Calls `RetrievalPipeline.build_sources(QueryRequest(...))`. Returns a serialized
`ContextPackage` that the agent can inspect, filter, or pass to `execute_context`.

**`execute_context` (new — replaces `query_llm` for pipeline synthesis)**

```
execute(question, context_package, synthesis_style=None) → QueryResponse
```

Calls `synthesize_context(package)` then `execute(package)`. The agent can pass an
unmodified `ContextPackage` from `query_context`, or one it has manually constructed.

**`query_and_synthesize` (convenience — single-call pattern)**

```
execute(query, mode=None) → QueryResponse
```

`mode` follows the same resolution path to a `ModeConfig` object.

Calls `RetrievalPipeline.run(QueryRequest(...))`. Single tool call for agents that don't
need to inspect the intermediate `ContextPackage`.

**Retained tools (unchanged)**:
- `query_llm` — standalone LLM prompting (not pipeline-backed). Calls
  `provider.send_message(instruction=..., input=...)` directly. Useful for agents that
  need direct LLM access without knowledge base context.
- `query_<provider>` — per-source remote provider tools (factory pattern from v0.7.1).
- `query_remote_chain` — named remote provider chains.
- `extract_snippets` — path-scoped semantic search.
- `similarity_search` — intra-corpus pairwise cosine similarity.

**Pipeline injection into the tool registry**:

```python
pipeline = RetrievalPipeline(db=sqlite_provider, embedder=embedder, config=lsm_config)
registry = create_default_tool_registry(
    config=lsm_config,
    collection=pipeline.db,
    embedder=pipeline.embedder,
    pipeline=pipeline,       # new injection point
    memory_store=memory_store,
)
```

The three pipeline tools (`query_context`, `execute_context`, `query_and_synthesize`) are
registered only when a `pipeline` is provided to `create_default_tool_registry`.

### 5.13 Evaluation Harness (lsm eval retrieval)

**Decision — CLI only.**

**Decision — BEIR benchmark format**: All eval output in BEIR format.

**Decision — Bundled synthetic dataset**: Ships with LSM.

**Decision — Versioned test corpus**: Dedicated versioned test corpus for consistent
regression comparisons.

**Decision — Named baselines**: `lsm eval save-baseline --name <name>`.

**CLI interface**:
```
lsm eval retrieval --profile hybrid_rrf
lsm eval retrieval --profile dense_only
lsm eval retrieval --profile dense_cross_rerank --compare baseline
lsm eval save-baseline --name dense_only_v0.8
lsm eval list-baselines
```

**Metrics**: `Recall@K`, `MRR`, `nDCG@K`, `Diversity@K`, `Latency`.

**Evaluation set sizing guidance**:
- **Minimum**: 50 queries — detects large effect sizes.
- **Recommended**: 200 queries — reliable for MRR and nDCG@10.
- **CI-grade**: 500+ queries — suitable for regression gates.

**Evaluation-first principle** (§7.2): Eval harness is implemented before hybrid retrieval
or cross-encoder features. No retrieval feature ships without measurable improvement.

### 5.14 Phase 2: Cluster-Aware Retrieval

Cluster-aware retrieval reduces the brute-force KNN search space by pre-assigning every
chunk to a cluster during an offline job. At query time, the query vector is matched
against cluster centroids first; only chunks in the top-N matching clusters are searched.

**Clustering algorithm**: k-means as default; HDBSCAN as option.

**Offline job**: `lsm cluster build` reads all chunk embeddings from `vec_chunks`, runs
clustering, writes `cluster_id` and `cluster_size` back to `lsm_chunks`.

**Centroid storage**: `lsm_cluster_centroids` table (§3.2).

**Query-time use**:
```sql
SELECT chunk_id, distance FROM vec_chunks
WHERE embedding MATCH ?
  AND cluster_id IN (3, 7, 12)
  AND is_current = 1
ORDER BY distance LIMIT 100;
```

**UMAP visualization**: `lsm cluster visualize` exports an HTML interactive plot.

**Config**:
```python
cluster_algorithm: str = "kmeans"
cluster_k: int = 50
cluster_top_n: int = 5
cluster_enabled: bool = False
```

### 5.15 Phase 2: Multi-Vector Retrieval (Query-Side)

Complements §4.4 (ingest-side multi-vector storage):

1. Run KNN at chunk level (k_chunk results).
2. Optionally run KNN at section level (k_section results).
3. Optionally run KNN at file level (k_file results).
4. RRF-fuse across granularities.
5. For file/section matches without a chunk match, expand to top-k chunks.

### 5.16 Phase 3: Graph-Augmented Retrieval

After initial vector retrieval returns candidates, graph expansion runs using recursive
CTEs against `lsm_graph_nodes` / `lsm_graph_edges`:

```sql
WITH RECURSIVE graph_walk(node_id, depth) AS (
    SELECT chunk_id, 0 FROM initial_candidates
    UNION ALL
    SELECT e.dst_id, gw.depth + 1
    FROM lsm_graph_edges e
    JOIN graph_walk gw ON e.src_id = gw.node_id
    WHERE gw.depth < 2
      AND e.edge_type IN ('contains', 'references', 'thematic_link')
)
SELECT DISTINCT node_id FROM graph_walk;
```

Expanded nodes are added to the candidate pool with a `graph_expansion_score` decaying
with hop count.

### 5.17 Phase 3: Multi-Hop Retrieval

**Two modes**:

**Mode A — Source Surfacing (`strategy: "parallel"`)**:
Decompose query → retrieve N sub-questions in parallel → merge packages → synthesize once.

**Mode B — Iterative Reasoning (`strategy: "iterative"`)**:
Run one sub-question per hop; the LLM's partial answer from hop N informs hop N+1.

**Decision**: `parallel` and `iterative` in v0.8.0.

**Conversation integration**: Uses the same `QueryRequest`/`QueryResponse` conversation
chain. Each hop's `QueryResponse.response_id` chains to the next hop's
`QueryRequest.prior_response_id`.

```python
@dataclass
class MultiHopRequest:
    query: str
    max_hops: int = 3
    strategy: str = "parallel"     # "parallel" | "iterative"
    mode: Optional[ModeConfig] = None
    conversation_id: Optional[str] = None
```

The existing meta-agent system provides the orchestration layer.

---

## 6. Cross-Cutting Concerns

### 6.1 TUI Startup Advisories for Offline Jobs

Several v0.8.0 features gate on an explicit offline build step. On TUI startup, LSM
inspects the DB and emits advisory messages for each offline job that is
configured-but-not-built or whose output is likely stale. Advisories are non-blocking.

**Job status table**: `lsm_job_status` (§3.2).

**Advisories per job**:

| Job | Config trigger | Advisory conditions | Message |
|-----|---------------|--------------------|---------
| `lsm cluster build` | `cluster_enabled: true` | Never run / stale (>20% corpus growth) | "Run: `lsm cluster build`" |
| `lsm graph build-links` | Active profile includes graph expansion | No thematic-link edges | "Run: `lsm graph build-links`" |
| `lsm finetune embedding` | `finetune_enabled: true` | No active fine-tuned model | "Run: `lsm finetune embedding`" |

Advisories are also emitted after `lsm ingest` on the CLI path.

### 6.2 Transactional Consistency

With the unified SQLite database, all writes in the ingest pipeline (`lsm_chunks`,
`vec_chunks`, `chunks_fts`, `lsm_manifest`, `lsm_agent_memories`, `lsm_agent_schedules`)
happen within the same SQLite transaction. A failed ingest leaves no partial state.

### 6.3 VectorDB Provider Abstraction Updates

`BaseVectorDBProvider` gains:

```python
def fts_query(self, text: str, top_k: int) -> VectorDBQueryResult: ...
def prune_old_versions(self, criteria: PruneCriteria) -> int: ...
def graph_insert_nodes(self, nodes: List[GraphNode]) -> None: ...
def graph_insert_edges(self, edges: List[GraphEdge]) -> None: ...
def graph_traverse(self, start_ids: List[str], max_hops: int) -> List[str]: ...
```

Providers:
- **`SQLiteVecProvider`** (new, default): Implements all methods against `lsm.db`.
- **`PostgreSQLProvider`** (existing, updated): Implements all methods against PostgreSQL.
- **`ChromaDBProvider`** (removed): Migration code only; no production use in v0.8.0.

### 6.4 LLM Provider Simplification

**Decision — Providers are pure transport. Domain logic moves to pipelines.**

`BaseLLMProvider` currently mixes two responsibilities:
1. **Generic LLM transport** (provider-specific): `_send_message()`,
   `_send_streaming_message()` — each provider implements these for its API.
2. **Domain-specific prompt/response logic** (shared in base class): `synthesize()`,
   `stream_synthesize()`, `rerank()`, `generate_tags()` — these construct domain prompts,
   call `_send_message()`, and parse responses.

The domain methods are already implemented entirely in the base class — no provider
overrides them. They just call `_send_message()` with the right prompt template. With
the `RetrievalPipeline` owning query logic and `ModeConfig` owning synthesis prompts,
these domain methods are no longer needed on providers.

**Changes to `BaseLLMProvider`**:

1. **Make transport methods public**: `_send_message()` → `send_message()`,
   `_send_streaming_message()` → `send_streaming_message()`.

2. **Updated signature** (breaking change):

```python
def send_message(
    self,
    input: str,                                      # was `user`
    instruction: Optional[str] = None,               # was `system` (always required)
    prompt: Optional[str] = None,                    # new — structured prompt for caching
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
    previous_response_id: Optional[str] = None,      # was in **kwargs
    prompt_cache_key: Optional[str] = None,           # was in **kwargs
    prompt_cache_retention: Optional[int] = None,     # new
    **kwargs,                                         # tools, json_schema, reasoning_effort, etc.
) -> str: ...

def send_streaming_message(
    self,
    input: str,
    instruction: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
    previous_response_id: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[int] = None,
    **kwargs,
) -> Iterator[str]: ...
```

Key signature changes:
- `system` → `instruction` — aligns with OpenAI Responses API field name. Optional:
  only included in the API request when non-`None`.
- `user` → `input` — aligns with OpenAI Responses API field name.
- `prompt` added — for structured prompt caching support.
- Caching parameters (`previous_response_id`, `prompt_cache_key`,
  `prompt_cache_retention`) are first-class named parameters, not hidden in `**kwargs`.
  No `enable_server_cache` gate — providers receive the values and decide how to map
  them to their API.

3. **Remove domain methods from base class**:
   - `synthesize()` → logic moves to `RetrievalPipeline.execute()` (§5.2)
   - `stream_synthesize()` → logic moves to `RetrievalPipeline.execute()` (streaming path)
   - `rerank()` → logic moves to `llm_rerank` pipeline stage (§5.7)
   - `generate_tags()` → logic moves to `lsm/ingest/tagging.py` (§6.6)

4. **Retained on base class**: `is_available()`, `name`, `model`,
   `supports_function_calling`, `list_models()`, `get_model_pricing()`,
   `_with_retry()`, `last_response_id`, health tracking.

5. **Provider support change**: Azure OpenAI is removed in v0.8.0. The
   `azure_openai.py` provider, factory registration, Azure-specific config fields, and
   Azure-specific test coverage are removed.

### 6.5 Per-Provider Implementation Gaps

Each provider's `send_message()` implementation needs a review pass. Current state
analysis (v0.7.1):

**OpenAI** (`openai.py`):
- `input` currently wrapped as `[{"role": "user", "content": user}]` (array of dicts) —
  should be plain string. The Responses API accepts `input` as a string directly.
- `instructions` always included even when empty — should be conditional.
- `previous_response_id` and `prompt_cache_key` are gated behind `enable_server_cache` —
  should be unconditional (the parameter presence is sufficient).
- `prompt` field: not implemented — needs to be added.
- `prompt_cache_retention`: not implemented — needs to be added.
- Streaming path: does NOT support `previous_response_id`, `prompt_cache_key`,
  `prompt`, tools, or JSON schema — needs parity with non-streaming.

**Anthropic** (`anthropic.py`):
- Uses cache_control headers (`{"type": "ephemeral"}`) on system content — different
  mechanism from OpenAI. `previous_response_id` is passed but silently ignored.
- `prompt` field: N/A for Anthropic API — map to appropriate alternative or no-op.
- Streaming path: no cache support, no tools, no JSON schema.
- `instruction` should map to `system` parameter in Anthropic Messages API.

**Gemini** (`gemini.py`):
- Non-tool path concatenates system + user into single prompt string — should use
  `system_instruction` config parameter consistently.
- All caching kwargs silently ignored — document as unsupported.
- `prompt`, `previous_response_id`, `prompt_cache_key`: N/A for Gemini API.

**OpenRouter** (`openrouter.py`):
- Uses cache_control headers on content (similar to Anthropic).
- `previous_response_id` and `prompt_cache_key` silently ignored.
- `prompt` field: N/A — map to cache_control mechanism or no-op.

**Local** (`local.py`):
- All kwargs completely ignored. Minimal Ollama API support.
- No caching of any kind — expected and acceptable.
- `instruction` maps to system role message.

**Cross-provider summary**:

| Parameter | OpenAI | Anthropic | Gemini | OpenRouter | Local |
|-----------|--------|-----------|--------|-----------|-------|
| `instruction` optional | Fix needed | OK | Fix needed | OK | OK |
| `input` as string | Fix needed | N/A (messages) | OK | N/A (messages) | N/A (messages) |
| `prompt` | Add | No-op | No-op | No-op | No-op |
| `previous_response_id` | Fix (remove gate) | No-op (log) | No-op | No-op | No-op |
| `prompt_cache_key` | Fix (remove gate) | Map to cache_control | No-op | Map to cache_control | No-op |
| `prompt_cache_retention` | Add | No-op | No-op | No-op | No-op |
| Streaming parity | Needs work | Needs work | Minimal | Needs work | Minimal |

Providers that cannot support a parameter log a debug-level message on first call
(using `UnsupportedParamTracker` pattern already in OpenAI) and ignore it. No exceptions
for unsupported parameters — graceful degradation.

### 6.6 Prompt Ownership Migration

Domain-specific prompt templates and response parsing move from `providers/` to the
pipeline stages and config objects that own the domain logic:

| Asset | Current Location | v0.8.0 Location | Owner |
|-------|------------------|-----------------|-------|
| `SYNTHESIZE_GROUNDED_INSTRUCTIONS` | `providers/helpers.py` | `ModeConfig.synthesis_instructions` default | Mode definition (§5.11) |
| `SYNTHESIZE_INSIGHT_INSTRUCTIONS` | `providers/helpers.py` | `ModeConfig.synthesis_instructions` default | Mode definition (§5.11) |
| `format_user_content()` | `providers/helpers.py` | `lsm/query/context.py` | Query pipeline `synthesize_context()` (§5.2) |
| `RERANK_INSTRUCTIONS` | `providers/helpers.py` | `lsm/query/stages/llm_rerank.py` | LLM rerank pipeline stage (§5.7) |
| `RERANK_JSON_SCHEMA` | `providers/helpers.py` | `lsm/query/stages/llm_rerank.py` | LLM rerank pipeline stage |
| `prepare_candidates_for_rerank()` | `providers/helpers.py` | `lsm/query/stages/llm_rerank.py` | LLM rerank pipeline stage |
| `parse_ranking_response()` | `providers/helpers.py` | `lsm/query/stages/llm_rerank.py` | LLM rerank pipeline stage |
| `TAG_GENERATION_TEMPLATE` | `providers/helpers.py` | `lsm/ingest/tagging.py` | Ingest pipeline |
| `TAGS_JSON_SCHEMA` | `providers/helpers.py` | `lsm/ingest/tagging.py` | Ingest pipeline |
| `get_tag_instructions()` | `providers/helpers.py` | `lsm/ingest/tagging.py` | Ingest pipeline |
| `generate_fallback_answer()` | `providers/helpers.py` | `lsm/query/pipeline.py` | Query pipeline `execute()` fallback |
| `parse_json_payload()` | `providers/helpers.py` | `lsm/providers/helpers.py` (retained) | Shared utility |

**What remains in `providers/helpers.py`**: Generic response parsing utilities
(`parse_json_payload()`, `UnsupportedParamTracker`) that have no domain knowledge.
Everything domain-specific moves to the stage or config that owns it.

**Ingest tagging change**: `lsm/ingest/tagging.py` already wraps `provider.generate_tags()`.
In v0.8.0, it constructs the tag prompt directly and calls
`provider.send_message(instruction=tag_instructions, input=text, ...)`,
owning the prompt template and response parsing.

### 6.7 Tests

| Feature | Test Type | Key Coverage |
|---------|-----------|-------------|
| SQLite-vec provider | Unit + Integration | vec0 insert, KNN query, metadata filters |
| FTS5 integration | Unit + Integration | Full-text insert, BM25 query, same-transaction sync |
| Agent memory consolidation | Unit + Integration | Migration from memories.db, CRUD in lsm.db |
| Agent schedule consolidation | Unit + Integration | Migration from schedules.json, CRUD in lsm.db |
| Schema version tracking | Unit | DB schema read/write, mismatch error |
| DB completion | Integration | Changed config → selective re-ingest |
| Migration (all paths) | Integration | ChromaDB→SQLite, SQLite→PG, PG→SQLite |
| SQLite-backed manifest | Unit + Integration | DB migration from JSON, query correctness |
| Manifest DB-only enforcement | Integration | Ingest/query paths do not read/write `.ingest/manifest.json` |
| Runtime cache DB migration | Unit + Integration | `stats_cache.json` + remote cache sidecars moved to DB tables |
| DB prune | Integration | Soft-delete cleanup, criteria enforcement |
| Heading depth (FileGraph) | Unit | FileGraph → chunk boundaries, heading_path |
| Heading depth (intelligent) | Unit | Section size estimation, recursion via FileGraph |
| RRF fusion | Unit | Score merging, rank preservation, weights |
| Cross-encoder reranking | Unit + Integration | Score improvement, CUDA path, cache |
| HyDE | Unit + Integration | Hypothetical generation, pooling, trace logging |
| Retrieval profiles | Integration | Each profile end-to-end, graceful degradation |
| MMR diversity | Unit | Diversity improvement, max_per_section cap |
| MinHash dedup | Unit | Near-duplicate suppression |
| Temporal ranking | Unit | Boost, time-range filter |
| Starting prompt handling | Unit | ModeConfig.synthesis_instructions → ContextPackage → execute |
| LLM provider simplification | Unit + Integration | send_message() signature, per-provider param mapping |
| Azure provider removal | Unit + Integration | `azure_openai` provider/factory/config path removed; Azure-specific tests removed |
| Provider caching | Integration | previous_response_id round-trip, prompt_cache_key, streaming parity |
| Prompt ownership migration | Unit | Rerank/tag prompts in pipeline stages, synthesis in ModeConfig |
| Agent pipeline tools | Integration | query_context, execute_context, query_and_synthesize |
| Cluster-aware retrieval | Unit + Integration | Centroid matching, cluster filter in KNN |
| Multi-vector retrieval | Integration | Section/file summary embeds, multi-granularity merge |
| Graph store | Unit + Integration | Node/edge insert, CTE traversal, expansion |
| Eval harness | Integration | BEIR format, metric computation, named baselines |

**Security**: FTS5 queries use parameterised queries. sqlite-vec queries use `?`
placeholders. Cross-encoder inputs are sanitised. Graph traversal depth is bounded.

### 6.8 SQLite Scale Guidance (User Guide)

sqlite-vec retrieval is brute-force KNN, so cost scales with corpus size and embedding
dimension. For user guidance, v0.8.0 documents practical operating bands and migration
cutovers:

| Corpus / workload | Recommendation |
|-------------------|----------------|
| Up to ~250k chunks @ 384 dims, single-user local desktop | SQLite default (`lsm.db`) |
| ~250k–1M chunks or heavy Phase 2/3 features (multi-vector + graph) | SQLite still viable, but enable cluster-aware retrieval and monitor p95 latency |
| >1M chunks, frequent concurrent writes, or shared multi-user server deployment | Prefer PostgreSQL provider |

User-guide performance/operations recommendations:
- Publish retrieval SLO targets: retrieval stage p95 <= 500ms, end-to-end query p95 <= 1.5s for local interactive use.
- Document vector memory estimate: `N_chunks * embedding_dim * 4 bytes` (float32), plus metadata and index overhead.
- For SQLite deployment defaults, set and document: `journal_mode=WAL`, `busy_timeout`, periodic `VACUUM`, and `ANALYZE`.
- Provide clear migration trigger guidance: if p95 retrieval latency breaches SLO for sustained workloads, run `lsm migrate --from sqlite --to postgresql`.

### 6.9 Privacy Model for Ingest-Time LLM Features

v0.8.0 features that call an LLM at ingest time (section/file summaries, entity extraction,
HyDE generation, optional tagging) follow a user-selected privacy model:
- If user selects a local provider, content remains local.
- If user selects a hosted provider, content is sent to that provider by user choice.
- Privacy level is therefore user-defined through provider/model selection.

User-guide requirement: clearly label which features invoke LLM calls and whether they run
during ingest, query, or offline jobs.

---

## 7. Long-Term Design Principles

### 7.1 Retrieval as Composable Pipeline

Each stage is independently testable and replaceable. Profiles are compositions of
stages, not monolithic paths.

### 7.2 Evaluation-First Development

No retrieval feature merges without measurable improvement. The eval harness is
implemented before hybrid retrieval or cross-encoder features.

### 7.3 Retrieval Traces as First-Class Artifacts

Every retrieval result carries a `RetrievalTrace` in `ContextPackage.retrieval_trace`.
The trace records: which stages executed, their timing, intermediate candidates,
`ScoreBreakdown` for every chunk, HyDE hypothetical documents (if applicable).

### 7.4 Local-First Foundation

All v0.8.0 features work without external service calls:
- Vector search and BM25 are local SQLite (sqlite-vec + FTS5) or local PostgreSQL.
- Cross-encoder runs locally via `sentence-transformers`.
- HyDE uses the configured LLM (local Ollama or hosted).
- Eval harness runs offline.

### 7.5 Single Database at a Time

The unified database principle is non-negotiable. No feature may introduce a new separate
database file or service. Agent memories, schedules, vectors, FTS, manifest, and graph
store all live in `lsm.db` or the PostgreSQL database.

### 7.6 Providers Are Transport, Pipelines Own Domain Logic

LLM providers expose generic `send_message()` / `send_streaming_message()` methods.
They have no knowledge of synthesis, reranking, tagging, or any domain concept.
Prompt templates, response parsing, and fallback strategies belong to the pipeline
stage or config object that owns the domain logic. This separation ensures that adding
a new provider never requires duplicating domain knowledge, and changing a prompt
template never requires touching provider code.

### 7.7 v0.7.1 Agent Invariants

The v0.7.1 agent architecture patterns are load-bearing and must be preserved:
- All agent LLM calls go through `_run_phase()`.
- Tool access is controlled via `tool_allowlist` and `remote_source_allowlist`.
- Budget enforcement via `run_bounded()`.
- Workspace isolation via accessor methods.
- `PhaseResult` carries operational output only.

---

## 8. Dependency Map

```
SQLite-vec Unified DB (§3.1-3.2)
  ├── Required by: Agent Data Consolidation (§3.3) ← memories + schedules in lsm.db
  ├── Required by: Manifest storage (§3.4)
  ├── Required by: Schema Versioning (§3.5)
  ├── Required by: BM25/FTS5 (§5.6) ← same DB, same transaction
  ├── Required by: Reranker cache (§5.7)
  ├── Required by: Phase 2 cluster centroids (§5.14)
  ├── Required by: Phase 3 graph store (§5.16)
  └── Enables: All providers use single-DB architecture

Schema Versioning (§3.5)
  ├── Required by: DB Completion (§3.6)
  ├── Required by: Heading Depth changes (§4.1-4.3) ← heading changes invalidate chunks
  └── Enables: Model registry foundation (§4.6)

FileGraph integration (§4.2)
  ├── Required by: heading_path metadata (§4.3)
  ├── Required by: Per-section diversity cap (§5.9)
  └── Required by: Phase 3 graph section edges (§4.5)

RetrievalPipeline abstraction (§5.2)
  ├── Required by: Retrieval Profiles (§5.5)
  ├── Required by: Hybrid RRF (§5.6)
  ├── Required by: Cross-encoder (§5.7)
  ├── Required by: HyDE (§5.8)
  ├── Required by: MMR Diversity & MinHash De-dup (§5.9)
  ├── Required by: Temporal Ranking (§5.10)
  ├── Required by: Agent Pipeline Tools (§5.12)
  ├── Required by: Cluster-aware retrieval (§5.14)
  ├── Required by: Multi-vector retrieval (§5.15)
  ├── Required by: Graph expansion (§5.16)
  └── Required by: Retrieval trace for eval

Eval Harness (§5.13)
  └── Must precede: Hybrid RRF, Cross-encoder, HyDE, Cluster-aware, Multi-vector validation

LLM Provider Simplification (§6.4)
  ├── Required by: RetrievalPipeline execute() (§5.2) ← calls provider.send_message()
  ├── Required by: LLM Rerank stage (§5.7) ← calls provider.send_message()
  ├── Required by: Ingest tagging (§6.6) ← calls provider.send_message()
  ├── Required by: Agent query_llm tool (§5.12) ← calls provider.send_message()
  └── Required by: Prompt Ownership Migration (§6.6)

Prompt Ownership Migration (§6.6)
  ├── Depends on: LLM Provider Simplification (§6.4)
  ├── Depends on: ModeConfig update (§5.11) ← synthesis_instructions field
  └── Required by: RetrievalPipeline execute() (§5.2)
```

**Critical path**:
SQLite-vec DB (§3.1) → Agent Data Consolidation (§3.3) → Schema Versioning (§3.5) →
LLM Provider Simplification (§6.4) → RetrievalPipeline (§5.2) → Eval Harness (§5.13) →
Hybrid RRF (§5.6) → Cross-encoder (§5.7) → HyDE (§5.8)

**Parallel paths**:
- FileGraph Heading Depth (§4.1-4.3) — independent of retrieval pipeline; shares DB only.
- Phase 2 Clustering (§5.14) — after RetrievalPipeline; parallel to cross-encoder work.
- Phase 3 Graph Store (§5.16) — after DB schema is locked; parallel to Phase 2.
- Prompt Ownership Migration (§6.6) — after LLM Provider Simplification; parallel to DB work.

---

## 9. Resolved Decisions Summary

| # | Topic | Decision |
|---|-------|----------|
| 1 | Mode config migration | No migration code. Breaking change. Users update config files. |
| 2 | Agent-composed ModeConfig validation | `AgentHarness` validates agent-supplied `ModeConfig` against global settings. |
| 3 | Pipeline API structure | Three stages: `build_sources → synthesize_context → execute`. High-level `run()`. |
| 4 | Conversation state | `QueryRequest` carries `conversation_id`/`prior_response_id`; `QueryResponse` returns updated IDs. Replaces `SessionState.llm_server_cache_ids`. |
| 5 | Starting prompt handling | `QueryRequest.starting_prompt` field. Resolution: explicit → session cache → mode-derived default. Stored in `ContextPackage.starting_prompt` for `execute()`. |
| 6 | Multi-hop strategy | Both `parallel` and `iterative` ship in v0.8.0. |
| 7 | Synthesis LLM selection | Set via resolved config + selected `ModeConfig`; no ad-hoc per-call selector field on `QueryRequest`. |
| 8 | Agent memory consolidation | `memories.db` merged into `lsm.db`. `MemoryConfig.sqlite_path` removed. Migration is explicit via `lsm migrate`. |
| 9 | Agent schedule consolidation | `schedules.json` merged into `lsm.db` (`lsm_agent_schedules` table). Migration is explicit via `lsm migrate`. |
| 10 | Document headings | Built on `lsm/utils/file_graph.py` (`FileGraph`, `GraphNode`). `structure_chunking.py` accepts optional `FileGraph`. |
| 11 | Plan structure | Three parts: DB (Part A), Ingest Pipeline (Part B), Query Pipeline (Part C). |
| 12 | LLM provider simplification | Providers are pure transport: `send_message()` / `send_streaming_message()`. Domain methods (`synthesize`, `rerank`, `generate_tags`) removed from base class. |
| 13 | Provider method signature | `system`→`instruction` (optional), `user`→`input`, add `prompt`. Caching params (`previous_response_id`, `prompt_cache_key`, `prompt_cache_retention`) are first-class named parameters. |
| 14 | Synthesis prompt ownership | `ModeConfig.synthesis_instructions` carries the synthesis prompt template. Each mode defines its own default. Pipeline reads from `ModeConfig`, not from `providers/helpers.py`. |
| 15 | Rerank prompt ownership | Rerank prompt, candidate preparation, and response parsing move from `providers/base.py` + `providers/helpers.py` to `lsm/query/stages/llm_rerank.py`. |
| 16 | Tag generation prompt ownership | Tag prompt, JSON schema, and response parsing move from `providers/base.py` + `providers/helpers.py` to `lsm/ingest/tagging.py`. |
| 17 | Manifest and sidecar policy | Manifest is fully DB-only. Non-log, non-agent/chat runtime sidecars are audited and moved to DB where applicable. |
| 18 | Azure provider support | Azure OpenAI provider is removed in v0.8.0, including related config paths and tests. |


---

*End of research document.*

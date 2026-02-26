# v0.8.0 Research Plan: Embedding Retrieval Improvements

**Status**: Fully Decided — Ready for Implementation Planning
**Version Target**: 0.8.0
**Source**: `TODO` v0.8.0 section + `.agents/future_plans/INGEST_FUTURE.md`

> **Breaking Release**: v0.8.0 is a breaking release. Config file format changes are
> unconstrained — there is no obligation to maintain backward compatibility with v0.7.x
> config files. Migration code is provided for data migration (vectors, manifest) only.
> Old config files will need to be updated by the user with clear upgrade documentation.

This document records all design decisions and implementation considerations for v0.8.0.
All questions from the discovery phase are resolved. No open decisions remain.

---

## Table of Contents

1. [Scope Summary](#1-scope-summary)
2. [DB Versioning with Migration/Upgrade](#2-db-versioning-with-migrationupgrade)
3. [DB Completion — Incremental Corpus Updates](#3-db-completion--incremental-corpus-updates)
4. [Architectural Overhaul: Retrieval Pipeline](#4-architectural-overhaul-retrieval-pipeline)
   - 4.1 Current Architecture
   - 4.2 Phase 1: Core Retrieval Features
   - 4.3 SQLite-vec: Unified Database Architecture
   - 4.4 Phase 2: Cluster-Aware Retrieval
   - 4.5 Phase 2: Multi-Vector Representation
   - 4.6 Phase 3: Graph-Augmented Retrieval
   - 4.7 Phase 3: Domain-Fine-Tuned Embeddings
   - 4.8 Phase 3: Multi-Hop Retrieval Agents
5. [Document Headings Improvement](#5-document-headings-improvement)
6. [Cross-Cutting Concerns](#6-cross-cutting-concerns)
   - 6.5 TUI Startup Advisories for Offline Jobs
7. [Long-Term Design Principles](#7-long-term-design-principles)
8. [Dependency Map](#8-dependency-map)

---

## 1. Scope Summary

v0.8.0 targets all four TODO items plus all phases from INGEST_FUTURE.md:

| # | Item | Complexity |
|---|------|------------|
| A | DB versioning with migration/upgrade | High |
| B | DB completion (incremental corpus updates) | High |
| C | Architectural overhaul: ingest and embedding retrieval (all three phases) | Very High |
| D | Document headings configurable depth + intelligent selection | Medium |

**Scope decision**: All of INGEST_FUTURE.md Phase 1, Phase 2, and Phase 3 features ship in
v0.8.0. Some may ship in point releases (v0.8.x) but all are designed in this plan. The
SQLite-backed manifest and the replacement of ChromaDB with sqlite-vec are also included
as part of the DB versioning and provider overhaul work.

---

## 2. DB Versioning with Migration/Upgrade

### 2.1 Current State and Required Changes

The codebase has a partial versioning foundation:

- **File-level versioning** (`manifest.py:23-36`): `get_next_version()` increments an
  integer per source file path across ingest runs.
- **Chunk-level soft-delete** (`pipeline.py:370-386`): Currently gated by
  `enable_versioning=True`.
- **Query-time version filter** (`planning.py:189`): Filters to `is_current=True`.
- **DB-to-DB migration** (`migrations/chromadb_to_postgres.py`): One-directional
  ChromaDB → PostgreSQL copy only.

**Decision — Versioning always on**: The `enable_versioning` flag is removed. Versioning
is the unconditional operating mode. The VectorDB provider gains a
`prune_old_versions(criteria: PruneCriteria) → int` method for intelligent cleanup of
soft-deleted chunks based on user-configurable criteria (age, version count, corpus size
limit).

**Decision — Soft-delete retained**: Old-version chunks are never automatically hard-
deleted on re-ingest. An `lsm db prune` CLI command (and equivalent in the TUI Ingest
screen) provides controlled cleanup with configurable criteria.

**Decision — Bidirectional migration**: The migration framework is extended to support all
provider-to-provider directions. See §4.3.5 for migration paths with the new SQLite-vec
provider.

### 2.2 Schema Version Concept

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

### 2.3 Where Schema Version Lives

**Decision**: Schema version is stored in the active database backend:

- **SQLite-vec provider**: A `lsm_schema_versions` table in `lsm.db` (the unified
  SQLite database — see §4.3).
- **PostgreSQL provider**: A `lsm_schema_versions` table in the same PostgreSQL database.

The manifest also retains a `_schema` header as a human-readable secondary copy.

### 2.4 Migration / Upgrade Path

**Decision — Schema mismatch handling**: When the system detects a schema mismatch (e.g.,
config `embed_model` ≠ recorded `embedding_model`), it raises an error with clear
instructions on how to run `lsm migrate`. It does not proceed silently.

**Decision — Migration entry points**:
- `lsm migrate` (or `lsm db upgrade`) — primary CLI command.
- The TUI Ingest screen surfaces migration warnings and provides a "Migrate" action.
- Migration does NOT happen automatically at ingest time.

**Migration strategies available**:

| Strategy | Description | Trade-offs |
|----------|-------------|------------|
| **Full rebuild** | Wipe and re-ingest entire corpus | Simplest; expensive for large corpora |
| **Selective re-embed** | Re-embed only files using the old model | Requires per-file model provenance |
| **Incremental migration job** | Background job progressively re-embeds old chunks | Low disruption; requires per-chunk tracking |

### 2.5 Manifest Schema Block

The manifest DB and `_schema` header capture the ingest provenance:

```json
{
  "_schema": {
    "manifest_version": 2,
    "lsm_version": "0.8.0",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "chunking_strategy": "structure",
    "chunk_size": 1800,
    "chunk_overlap": 200,
    "created_at": "2026-02-25T10:00:00Z",
    "last_ingest_at": "2026-02-25T10:00:00Z"
  }
}
```

---

## 3. DB Completion — Incremental Corpus Updates

### 3.1 The Problem

"DB completion" means: given an existing vector DB built with LSM version X, bring it up
to date with version Y without a full rebuild. Cases:

1. **New file types**: New parsers (`.epub`, `.pptx`) — ingest only newly-supported types.
2. **New metadata fields**: `heading_path`, language tags — enrich without re-embedding.
3. **Changed chunking**: Must re-chunk and re-embed affected files.
4. **New ingest features**: AI tagging, language detection — re-tag without re-embedding.

### 3.2 What Already Works

The current incremental system (`pipeline.py:620-647`) skips unchanged files via mtime →
size → hash. The gap is a file unchanged on disk whose vector representation or metadata
is stale by the codebase's definition — the manifest returns "skip" when it should
trigger re-ingest.

### 3.3 Completion Modes

| Mode | Trigger | Scope | Cost |
|------|---------|-------|------|
| **Extension completion** | New `extensions` in config | Only new file types | Low |
| **Metadata enrichment** | New metadata field available | All files, no re-embedding | Medium |
| **Chunk boundary update** | Chunking params changed | All files for that strategy | High |
| **Embedding upgrade** | Embedding model changed | All files | Very high |
| **Selective re-ingest** | Manual `--force-file-pattern` | User-specified subset | Variable |

`lsm db complete` (or `--force-reingest-changed-config` on ingest) compares recorded
schema against current config and re-ingests only files whose chunks would differ.

### 3.4 SQLite-Backed Manifest

**Decision**: The flat JSON manifest is replaced with a SQLite-backed manifest in v0.8.0,
co-located in `lsm.db` for the SQLite-vec provider or as a dedicated table set in
PostgreSQL.

**Rationale**: Corpus size of 100k+ files is expected. A 100k-file flat JSON manifest
(~20 MB) has ~200ms load/save time and no query capability. SQLite provides sub-
millisecond lookups, indexed queries, transactional writes, and in-place updates.

**Migration**: Existing `manifest.json` files are auto-migrated to the DB on first run
with v0.8.0.

---

## 4. Architectural Overhaul: Retrieval Pipeline

### 4.1 Current Retrieval Architecture

```
user query
   → embed_text() [retrieval.py:82]          # single-model dense embedding
   → retrieve_candidates() [retrieval.py:126] # kNN cosine ANN via vector DB
   → filter_candidates() [retrieval.py:196]   # path/ext post-filters
   → apply_local_reranking() [rerank.py:255]  # lexical reranking + diversity cap
   → context assembly                          # top-k chunks as LLM context
   → LLM synthesis
```

No retrieval profiles, no sparse index, no cross-encoder, no HyDE, no evaluation harness.
`QueryConfig.rerank_strategy = "hybrid"` is a misnaming — it is lexical reranking over
dense results, not a true dual-recall system.

---

### 4.2 Phase 1: Core Retrieval Features

#### 4.2.1 Unified RetrievalPipeline Abstraction

A `RetrievalPipeline` class with a `run(query: Query) → List[Candidate]` interface and
composable stages (`recall → fuse → rerank → diversify`). Each stage receives the prior
stage's output and adds score breakdowns. The current function chain across `api.py`,
`planning.py`, `context.py`, `retrieval.py`, `rerank.py` is replaced by this abstraction.

**Four standard objects**:

- **`Query`**: Wraps the user's question string plus retrieval parameters (profile, k,
  filters).
- **`Candidate`**: Extended with `ScoreBreakdown` and its embedding vector (required for
  MMR diversity selection).
- **`ScoreBreakdown`**: Per-candidate scoring detail:
  - `dense_score: float` — cosine similarity from vector ANN
  - `dense_rank: int` — rank position in dense results
  - `sparse_score: float` — BM25/FTS5 score
  - `sparse_rank: int` — rank position in sparse results
  - `fused_score: float` — RRF-combined score
  - `rerank_score: Optional[float]` — cross-encoder output
  - `temporal_boost: Optional[float]` — recency multiplier
- **`Citation`**: Resolved reference: `chunk_id`, `source_path`, `heading`,
  `page_number`, `url_or_doi`, `snippet`.

**Decision — Stage access to QueryConfig**: Stages have access to the full `QueryConfig`.
Many stages need cross-cutting parameters: diversity stage needs `max_per_file`;
temporal stage needs `temporal_boost_days`; rerank stage needs `cross_encoder_model`.

**Decision — Hard-coded profiles, not a registry**: Four profiles are hard-coded for
v0.8.0. A registry adds indirection with no benefit for a fixed profile set; the stage
abstraction is extensible to a registry in v0.9.0 if custom profiles become a requirement.

**Retrieval trace**: Every pipeline run emits a `retrieval_trace.json` to the workspace:
- Which stages executed and their timing
- Intermediate candidates at each stage
- `ScoreBreakdown` for every returned chunk
- HyDE hypothetical documents (if applicable)

#### 4.2.2 Retrieval Profiles

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
`hybrid_rrf` is the default. Before the first FTS5 index is built (e.g., on first run),
the profile degrades gracefully to `dense_only` with a log warning.

**Decision — Graceful degradation**:
- `hybrid_rrf` without FTS5 index → `dense_only` + warning.
- `dense_cross_rerank` without cross-encoder model → ANN results, no reranking.
- `hyde_hybrid` with LLM failure → direct query embedding.
- `temporal_boost` config absent → boost stage skipped.

`QueryConfig.mode` ("grounded"/"insight") is a synthesis mode, orthogonal to retrieval
profiles. The legacy `rerank_strategy` config key is removed (breaking change).

#### 4.2.3 Hybrid Retrieval: Dense + Sparse (BM25/FTS5) + RRF

With the unified SQLite-vec database (see §4.3), both vector search and FTS5 search run
against the same `lsm.db` file. Hybrid search can be expressed as a single SQL query
joining `vec_chunks` and `chunks_fts`, or run as two separate queries merged with RRF in
Python. The RRF merge in Python is preferred for the `RetrievalPipeline` abstraction, as
it preserves stage separation and score tracking.

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
- **SQLite-vec provider**: delegates to `chunks_fts` FTS5 virtual table in `lsm.db`.
- **PostgreSQL provider**: uses native `tsvector`/`tsquery` and `ts_rank`.

**FTS5 indexed fields**: `chunk_text`, `heading`, `source_name`, and other metadata text
fields.

**Synchronisation**: With the unified SQLite-vec database, vector writes (to `vec_chunks`)
and FTS5 writes (to `chunks_fts`) happen in the **same SQLite transaction**. The sync
problem described in earlier drafts (requiring `lsm db check` / `lsm db repair`) is
eliminated for the SQLite-vec provider. For the PostgreSQL provider, the vector write and
FTS update are a single `INSERT` to `lsm_chunks` (the `tsvector` column is generated
automatically). Both providers achieve transactional consistency natively.

#### 4.2.4 Cross-Encoder Reranking

**Approach**: Cross-encoders encode `(query, passage)` jointly. Typical pattern: dense
recall top-100, cross-encoder rerank to top-20.

**Available models** (local, via `sentence-transformers`):

| Model | Size | MRR@10 on MS-MARCO |
|-------|------|---------------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 66M | 39.01 |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 127M | 39.97 |
| `cross-encoder/ms-marco-electra-base` | 109M | 39.96 |

**Decision — Lazy download**: Model downloaded on first use of `dense_cross_rerank`.

**Decision — CUDA/GPU support**: `CrossEncoder` honours the `device` parameter.
When `GlobalConfig.device = "cuda"`, inference on 100 candidates drops from 2–10s (CPU)
to under 100ms. The `device` from `GlobalConfig` is passed at init time; CPU is the
fallback.

**Decision — Reranker cache**: Persisted to the `lsm_reranker_cache` table in `lsm.db`
(SQLite-vec provider) or a `lsm_reranker_cache` table in PostgreSQL. Cache key:
`(query_hash, chunk_id, model_version)`. `lsm cache clear` provides manual invalidation.

**LLM reranking**: The existing `rerank_strategy: "llm"` path becomes the `llm_rerank`
profile. The `rerank_strategy` config key is removed (breaking change).

#### 4.2.5 HyDE (Hypothetical Document Embeddings)

**Mechanism**:
1. LLM generates 1–3 hypothetical answers to the query (zero-shot).
2. Each is embedded with the same bi-encoder.
3. Embeddings are pooled (mean or max).
4. Pooled embedding replaces direct query embedding for ANN search.

**Decision — Profile-only**: HyDE is exclusively the `hyde_hybrid` profile. Disabled by
default.

**Decision — Observability**: Hypothetical documents are logged in `retrieval_trace.json`
and viewable in the TUI debug/trace view.

**Tuning parameters**:
- `hyde_num_samples: int = 2`
- `hyde_temperature: float = 0.2`
- `hyde_pooling: str = "mean"` (options: `"mean"`, `"max"`)
- Hypothetical generation prompt is domain-configurable per profile.

#### 4.2.6 Diversity and De-duplication

**Current state**: Exact hash dedup (`rerank.py:154`) + per-file cap (`rerank.py:204`).

**Decision — MinHash near-duplicate detection**: MinHash (Jaccard similarity on shingles)
replaces exact hash dedup. A MinHash signature is stored per chunk at ingest time. At
retrieval time, candidates exceeding a configurable similarity threshold are suppressed.
`diversity_strategy` param: `"exact"` (current), `"minhash"` (new default when not
"exact").

**Decision — Greedy MMR as default when not "exact"**: When `diversity_strategy != "exact"`,
Maximal Marginal Relevance (MMR) is the default post-rerank selection strategy. MMR
iteratively picks the next candidate maximising relevance while minimising similarity to
already-selected results. The `max_per_file` cap remains as a hard ceiling.

**Decision — Per-section cap default = 3**: `max_per_section: Optional[int] = 3` caps
chunks per heading-path prefix. The value `None` is valid to disable the cap entirely.
Benefit: a single long chapter cannot dominate results even when `max_per_file` is not
reached. Requires `heading_path` metadata from §5 to be populated.

#### 4.2.7 Temporal-Aware Ranking

`mtime_ns` is already stored in chunk metadata (`pipeline.py:401`). In v0.8.0:

- **Recency boost**: Configurable decay. Default: `1.5×` for files modified within 30 days.
- **Time-range filter**: `WHERE mtime_ns BETWEEN ? AND ?` for queries like "notes from 2023".

```python
temporal_boost_enabled: bool = False
temporal_boost_days: int = 30
temporal_boost_factor: float = 1.5
```

#### 4.2.8 Evaluation Harness (lsm eval retrieval)

**Decision — CLI only.**

**Decision — BEIR benchmark format**: All eval output in BEIR format.

**Decision — Bundled synthetic dataset**: A bundled synthetic query set ships with LSM.

**Decision — Versioned test corpus**: A dedicated versioned test corpus for consistent
regression comparisons across LSM versions.

**Decision — Named baselines**: `lsm eval save-baseline --name <name>`. When `--name` is
omitted, the default name `"baseline"` is used. `lsm eval list-baselines` manages stored
baselines. Regression comparison: `lsm eval retrieval --profile X --compare <name>`.

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
- **Minimum**: 50 queries — detects large effect sizes at 80% power.
- **Recommended**: 200 queries — detects medium effect sizes; reliable for MRR and nDCG@10.
- **CI-grade**: 500+ queries — detects small effects; suitable for regression gates.

**Evaluation-first principle** (§7.2): Eval harness is implemented before hybrid retrieval
or cross-encoder features. No retrieval feature ships without measurable improvement.

---

### 4.3 SQLite-vec: Unified Database Architecture

#### 4.3.1 Decision

**ChromaDB is replaced entirely by sqlite-vec as the primary vector store.** v0.8.0 ships
two providers:

| Provider | Default | Description |
|----------|---------|-------------|
| `sqlite` | **Yes** | sqlite-vec + FTS5 + all LSM tables in a single `lsm.db` file |
| `postgresql` | No | pgvector + native FTS + all LSM tables in a single PostgreSQL DB |

There is no longer a "ChromaDB provider." Users with existing ChromaDB data use the
migration command before upgrading (see §4.3.5).

The principle is: **one database technology at a time**. Either a single SQLite file
(`lsm.db`) or a single PostgreSQL database holds all LSM state.

#### 4.3.2 sqlite-vec Research Findings

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
query latency of 15–68ms is acceptable for interactive use. With OS page cache warm
(repeated queries), latency is toward the lower end. Cold queries (first query against
an uncached DB) are toward the upper end.

**Python installation**:
```bash
pip install sqlite-vec
```
Zero runtime dependencies. Platform wheels are pre-compiled for Windows x86-64, Linux
x86-64/ARM64, and macOS x86-64/ARM64.

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
    chunk_id    INTEGER PRIMARY KEY,
    embedding   FLOAT[384] distance_metric=cosine,
    is_current  INTEGER,        -- metadata filter column
    source_path TEXT,           -- metadata filter column (v0.1.6+ supports IN, BETWEEN)
    node_type   TEXT            -- "chunk", "section_summary", "file_summary"
);
```

**KNN query with metadata filter**:
```sql
SELECT chunk_id, distance
FROM vec_chunks
WHERE embedding MATCH ?
  AND is_current = 1
  AND node_type = 'chunk'
ORDER BY distance
LIMIT 100;
```

**Supported vector types**: `FLOAT[N]` (float32), `INT8[N]` (int8), `BIT[N]` (binary).
No hard dimension limit; 3072 dims confirmed working.

**Metadata filter operators** (v0.1.6+): `=`, `!=`, `<`, `<=`, `>`, `>=`, `IN (...)`,
`BETWEEN`, `IS NULL`, `IS NOT NULL`. Pre-filter (bitmap-indexed) — non-matching rows are
excluded before distance computation.

**FTS5 coexistence**: `vec0` virtual tables and FTS5 virtual tables coexist in the same
SQLite database with no conflicts. Both are virtual table implementations within SQLite's
extension system.

**Batch insert**: Use SQLite transactions for bulk loads:
```python
with db:
    db.executemany(
        "INSERT INTO vec_chunks(chunk_id, embedding, is_current, source_path) VALUES (?, ?, 1, ?)",
        [(chunk_id, embedding_bytes, source_path) for ...]
    )
```
Approximate throughput: ~85,000 inserts/second at 384 dims (M1 Pro benchmark).

**SIMD acceleration**: AVX (x86-64) and NEON (ARM64) intrinsics for float32/int8
distance computation. GPU acceleration is not applicable (brute-force on CPU).

#### 4.3.3 Concrete Benefits vs. ChromaDB

| Dimension | sqlite-vec | ChromaDB |
|-----------|-----------|----------|
| Python dependencies | **Zero** | 25+ packages (grpcio, posthog, hnswlib, uvicorn, opentelemetry, etc.) |
| Install size | ~500KB–2MB | 20+ MB + transitive |
| DB files | **1 file** (`lsm.db`) | Multiple (chroma.sqlite3 + HNSW index files) |
| Transactional writes | **Yes — ACID across vectors + FTS5** | No (ChromaDB has no cross-table transaction support) |
| BM25 sync problem | **Eliminated** (same transaction) | Present (separate write operations) |
| Index type | Brute-force KNN | HNSW (approximate) |
| Backup | **One file to copy** | Multiple files, potential inconsistency |
| Native FTS5 | **Yes — same DB** | No (requires sidecar) |
| Filter support | Yes (bitmap pre-filter) | Yes (Chroma where-filter) |
| Telemetry | **None** | posthog telemetry (opt-out required) |
| Background processes | **None** | In-process HTTP server |
| License | MIT / Apache-2.0 | Apache-2.0 |

**Key architectural benefit**: With sqlite-vec, the entire problem described in §6.2
(BM25 sync, drift detection, repair tools) is eliminated. Vectors, FTS5 index, manifest,
schema versions, and reranker cache are all written in the same SQLite transaction. A
failed ingest either fully completes or fully rolls back — no partial state.

**Known limitation**: No ANN index. Brute-force KNN at 100k vectors and 384 dims gives
15–68ms per query. This is acceptable for a personal knowledge base. When sqlite-vec ships
ANN support (IVF + DiskANN), LSM will gain it by updating the dependency. Binary
quantization (`BIT[N]`) reduces memory by 32× at the cost of some accuracy — a viable
interim strategy for users with very large corpora.

#### 4.3.4 Unified Database Schema

**SQLite-vec provider — `lsm.db`**:

```sql
-- Core chunk metadata (standard table)
CREATE TABLE lsm_chunks (
    chunk_id        TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
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

-- Vector search (vec0 virtual table)
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    chunk_id    TEXT PRIMARY KEY,
    embedding   FLOAT[384] distance_metric=cosine,
    is_current  INTEGER,
    node_type   TEXT,
    source_path TEXT
);

-- Full-text search (FTS5 virtual table)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    chunk_text,
    heading,
    source_name,
    content=lsm_chunks,
    content_rowid=rowid
);

-- Manifest (file-level ingest tracking)
CREATE TABLE lsm_manifest (
    source_path     TEXT PRIMARY KEY,
    mtime_ns        INTEGER,
    file_size       INTEGER,
    file_hash       TEXT,
    version         INTEGER,
    embedding_model TEXT,
    updated_at      TEXT
);

-- Schema version tracking
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

-- Cross-encoder reranker cache
CREATE TABLE lsm_reranker_cache (
    cache_key   TEXT PRIMARY KEY,   -- hash(query_hash, chunk_id, model_version)
    score       REAL,
    created_at  TEXT
);

-- Phase 3: Graph store (section 4.6)
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

-- Offline job status tracking (see §6.5 — TUI startup advisories)
CREATE TABLE lsm_job_status (
    job_name        TEXT PRIMARY KEY,  -- "cluster_build", "graph_build_links", "finetune_embedding"
    status          TEXT NOT NULL,     -- "running", "completed", "failed"
    started_at      TEXT,
    completed_at    TEXT,
    corpus_size     INTEGER,           -- is_current chunk count at completion (staleness detection)
    metadata        TEXT               -- JSON: job-specific details (model path, k value used, etc.)
);
```

**PostgreSQL provider**: Equivalent tables in PostgreSQL with pgvector for the embedding
column and a `tsvector GENERATED ALWAYS AS` column on `lsm_chunks` for native FTS.

#### 4.3.5 Migration Paths

| From | To | Command |
|------|----|---------|
| ChromaDB | SQLite-vec | `lsm migrate --from chroma --to sqlite` |
| ChromaDB | PostgreSQL | `lsm migrate --from chroma --to postgresql` |
| SQLite-vec | PostgreSQL | `lsm migrate --from sqlite --to postgresql` |
| PostgreSQL | SQLite-vec | `lsm migrate --from postgresql --to sqlite` |

All migration paths copy vectors, chunk metadata, and manifest entries. Schema versions
are re-derived from the active config at migration time.

#### 4.3.6 Config Changes

The `vectordb` config section is simplified (breaking change from v0.7.x):

```yaml
vectordb:
  provider: "sqlite"          # "sqlite" (default) or "postgresql"
  path: "./data"              # SQLite: directory for lsm.db; PostgreSQL: ignored
  collection: "lsm_chunks"   # table name prefix (optional)
  # PostgreSQL only:
  connection_string: ""       # or set LSM_POSTGRES_CONNECTION_STRING env var
```

The `persist_dir` config key is replaced by `path`. No backward compatibility — config
files must be updated.

---

### 4.4 Phase 2: Cluster-Aware Retrieval

#### 4.4.1 Overview

Cluster-aware retrieval reduces the brute-force KNN search space by pre-assigning every
chunk to a cluster during an offline job. At query time, the query vector is matched
against cluster centroids first; only chunks in the top-N matching clusters are searched.
This effectively provides approximate nearest-neighbor behaviour without a dedicated ANN
index — a valuable interim strategy given sqlite-vec's brute-force-only limitation.

#### 4.4.2 Clustering Algorithm

Two approaches from INGEST_FUTURE.md:

| Algorithm | Characteristics | Best For |
|-----------|----------------|----------|
| **k-means** | Fixed k clusters; each chunk assigned to nearest centroid; fast | Uniform corpus; known approximate cluster count |
| **HDBSCAN** | Density-based; variable cluster count; handles noise | Heterogeneous corpus; unknown structure |

**Decision path**: k-means is simpler to implement and deterministic. HDBSCAN produces
more natural clusters for varied personal knowledge bases. Both should be supported;
k-means as the default, HDBSCAN as an option.

#### 4.4.3 Implementation

**Offline clustering job**: `lsm cluster build` command. Reads all chunk embeddings from
`vec_chunks`, runs clustering, writes `cluster_id` and `cluster_size` back to
`lsm_chunks`.

**Storage**: `cluster_id INTEGER` and `cluster_size INTEGER` metadata columns on
`lsm_chunks`. These columns are defined in the v0.8.0 schema (§4.3.4) but populated as
`NULL` until the clustering job runs.

**Centroid storage**: `lsm_cluster_centroids` table in `lsm.db` (or PostgreSQL):
```sql
CREATE TABLE lsm_cluster_centroids (
    cluster_id  INTEGER PRIMARY KEY,
    centroid    BLOB,               -- serialized float32 array (or use vec0)
    size        INTEGER
);
```

**Query-time use**: Before KNN search, embed query vector → find top-N cluster centroids
by cosine similarity → filter `vec_chunks WHERE cluster_id IN (selected_clusters)` → run
KNN within that subset. With vec0's metadata filter, this translates to:
```sql
SELECT chunk_id, distance FROM vec_chunks
WHERE embedding MATCH ?
  AND cluster_id IN (3, 7, 12)
  AND is_current = 1
ORDER BY distance LIMIT 100;
```

**UMAP visualization**: A diagnostic `lsm cluster visualize` command exports a UMAP 2D
projection of chunk embeddings, coloured by cluster, as an HTML interactive plot. Requires
`umap-learn` as an optional dependency.

**Config**:
```python
cluster_algorithm: str = "kmeans"     # "kmeans" or "hdbscan"
cluster_k: int = 50                    # k-means only
cluster_top_n: int = 5                # clusters to search at query time
cluster_enabled: bool = False         # must run `lsm cluster build` first
```

#### 4.4.4 Dependency on sqlite-vec ANN

When sqlite-vec ships ANN support (IVF + DiskANN), the cluster-based approach becomes
less necessary for pure performance — IVF is essentially k-means-based ANN. At that point,
`cluster_enabled` can be deprecated in favour of native IVF. The cluster metadata remains
useful for UMAP visualization and graph-augmented retrieval even after ANN is available.

---

### 4.5 Phase 2: Multi-Vector Representation

#### 4.5.1 Overview

Multi-vector representation stores embeddings at multiple granularities:
- **Chunk-level** (`node_type = "chunk"`): current default; fine-grained semantic search.
- **Section-level** (`node_type = "section_summary"`): embedding of a section's summary;
  broader semantic coverage.
- **File-level** (`node_type = "file_summary"`): embedding of a document summary;
  highest-level semantic fingerprint.

At query time, search across all granularities and merge results. Chunk-level results
provide precise citations; section/file-level results provide thematic breadth.

#### 4.5.2 Storage in Unified DB

The `node_type` metadata column on `vec_chunks` and `lsm_chunks` distinguishes embedding
granularities within the same `vec_chunks` virtual table. No separate collections needed.

**Query with granularity selection**:
```sql
-- Section-level retrieval only
SELECT chunk_id, distance FROM vec_chunks
WHERE embedding MATCH ?
  AND node_type = 'section_summary'
  AND is_current = 1
ORDER BY distance LIMIT 20;
```

#### 4.5.3 Summary Generation at Ingest Time

Section and file summaries are generated during ingest:
- **Section summary**: LLM summarises all text under a heading → embedded → stored with
  `node_type = "section_summary"` and the heading's `heading_path`.
- **File summary**: LLM summarises the document (or first N paragraphs + headings outline)
  → embedded → stored with `node_type = "file_summary"`.

Summary generation requires an LLM call per section/file. This is opt-in:
```python
enable_section_summaries: bool = False
enable_file_summaries: bool = False
```

#### 4.5.4 Query-Time Multi-Granularity Retrieval

The `RetrievalPipeline` supports a multi-vector retrieval stage:
1. Run KNN at chunk level (k_chunk results).
2. Optionally run KNN at section level (k_section results).
3. Optionally run KNN at file level (k_file results).
4. RRF-fuse across granularities.
5. For file/section matches without a specific chunk match, expand to the top-k chunks
   from that section/file.

---

### 4.6 Phase 3: Graph-Augmented Retrieval

#### 4.6.1 Overview

Graph-augmented retrieval extends vector search with a knowledge graph stored locally.
After vector retrieval returns a set of candidate chunks, graph traversal expands coverage
to related content that may not have been retrieved by embedding similarity alone.

#### 4.6.2 Graph Store

**Decision**: The graph store lives in `lsm.db` as standard SQLite tables
(`lsm_graph_nodes`, `lsm_graph_edges` — defined in §4.3.4). SQLite recursive CTEs
provide graph traversal without a dedicated graph database dependency.

**Node types**:
- `"file"` — a source document
- `"section"` — a heading-delimited section within a file
- `"chunk"` — an individual chunk
- `"entity"` — a named entity (person, concept, term) extracted by NER or LLM

**Edge types**:
- `"contains"` — File → Section, Section → Chunk (structural)
- `"references"` — Chunk → File (citation/backlink)
- `"same_author"` — File → File
- `"thematic_link"` — Chunk/Section → Chunk/Section (similarity-derived)

#### 4.6.3 Graph Construction

**From heading_path metadata** (ingest time): Each `heading_path` level becomes a section
node; parent-child relationships become `"contains"` edges. The `heading_path` metadata
from §5 is the primary ingest-time artifact for section nodes.

**From file references** (ingest time): Markdown `[[wikilinks]]`, `[text](path)` internal
links, and citation DOIs are parsed to create `"references"` edges.

**From entity extraction** (ingest time): NER or LLM-based entity extraction creates
entity nodes and `"mentioned_in"` edges. Requires LLM or `spacy`.

**Thematic links** (offline job): Cosine similarity above a threshold between chunk
embeddings creates `"thematic_link"` edges. Run as `lsm graph build-links`.

#### 4.6.4 Query-Time Graph Traversal

After initial vector retrieval returns candidates, graph expansion runs:

```sql
-- Find all chunks within 2 hops of retrieved chunks
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

Expanded nodes are added to the candidate pool with a `graph_expansion_score` (decaying
with hop count) in their `ScoreBreakdown`.

#### 4.6.5 Integration with Meta-Agent System

The Phase 6/10 meta-agent system (already complete) provides the orchestration layer for
multi-hop graph retrieval agents. An agent can iteratively call `retrieve → graph_expand
→ analyze → retrieve` until a relevance threshold is met.

---

### 4.7 Phase 3: Domain-Fine-Tuned Embeddings

#### 4.7.1 Overview

Domain fine-tuning adapts the bi-encoder embedding model to the user's specific corpus,
improving semantic similarity for domain-specific terminology.

#### 4.7.2 Training Signal

Two sources of weak supervision without manual labelling:

- **Heading-content pairs**: `(heading_text, section_body)` → heading is the query,
  section body is the positive passage.
- **User interaction signals**: queries that led to accepted citations are positive pairs;
  queries with no clicks are hard negatives.

A synthetic triplet dataset `(query, positive_passage, hard_negative)` is derived
from the corpus at ingest time.

#### 4.7.3 Fine-Tuning Process

```
lsm finetune embedding --base-model all-MiniLM-L6-v2 --epochs 3 --output ./models/finetuned
```

Uses `sentence-transformers`' `MultipleNegativesRankingLoss` with the derived triplet
dataset. The fine-tuned model is saved locally and registered in the model registry.

#### 4.7.4 Model Registry

The `_schema.embedding_model` field in `lsm_schema_versions` is the foundation. A model
registry extends this:

```sql
CREATE TABLE lsm_embedding_models (
    model_id    TEXT PRIMARY KEY,
    base_model  TEXT,
    path        TEXT,               -- local path or HuggingFace model ID
    dimension   INTEGER,
    created_at  TEXT,
    is_active   INTEGER DEFAULT 0
);
```

Switching models triggers the schema mismatch detection (§2.4) and prompts `lsm migrate`.

---

### 4.8 Phase 3: Multi-Hop Retrieval Agents

#### 4.8.1 Overview

Multi-hop retrieval decomposes complex queries into sub-queries, retrieves iteratively,
and synthesises across multiple retrieval rounds. Uses the existing meta-agent system
as the orchestration layer.

#### 4.8.2 Foundation

`lsm/query/decomposition.py` already extracts keywords, author, title, DOI, and date
entities from queries. Phase 3 extends this to:

1. **Query decomposition**: LLM breaks a complex query into sub-questions.
2. **Iterative retrieval**: Each sub-question runs through the `RetrievalPipeline`.
3. **Context synthesis**: Partial answers from each retrieval round inform the next
   sub-question.
4. **Citation chain**: The `retrieval_trace.json` records each hop, producing a hop-wise
   citation trace.

#### 4.8.3 Implementation Sketch

```python
class MultiHopRetrievalAgent:
    def run(self, query: str, max_hops: int = 3) -> List[Candidate]:
        sub_questions = self.decompose(query)
        all_candidates = []
        for sub_q in sub_questions:
            candidates = self.pipeline.run(Query(sub_q))
            partial_answer = self.synthesise(candidates)
            all_candidates.extend(candidates)
            if self.is_sufficient(partial_answer):
                break
        return self.deduplicate(all_candidates)
```

The meta-agent `TaskGraph` system manages sub-agent lifecycle and result aggregation.

---

## 5. Document Headings Improvement

### 5.1 Current State

`structure_chunking.py` treats all heading levels identically as chunk boundaries. The
`heading` metadata field is a flat string.

Heading detection (`utils/text_processing.py:48-77`) recognises Markdown `#`-style
headings (levels 1–6), HTML `<h1>`–`<h6>`, bold-only lines, and plain-text markers.

### 5.2 Feature 1: Configurable Heading Depth

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

### 5.3 Feature 2: Intelligent Heading Depth Selection

**Decision — Implemented in v0.8.0.**

`intelligent_heading_depth: bool = False` on `IngestConfig`. When enabled, heading depth
is decided dynamically per section.

**Algorithm**:
1. Estimate section size (character count below heading, before next sibling/parent).
2. If size ≤ `chunk_size`: keep as one chunk; sub-headings flow into body text.
3. If size > `chunk_size` AND has sub-headings: split at sub-heading boundaries, recurse.
4. Recursion depth bounded to prevent pathological documents.

**Decision — `heading_path` as separate metadata**: A `heading_path: list[str]` field
records the full hierarchy (e.g., `["Introduction", "Background", "Prior Work"]`), stored
as a JSON array in `lsm_chunks.heading_path`. The flat `heading` string field is retained
for BM25 text indexing.

**Applies only to `chunking_strategy: "structure"`.** The `fixed` strategy is unchanged.

**DB versioning impact**: Heading depth changes produce different `chunk_id` values.
Files with stale heading metadata are re-ingested automatically (not requiring manual
`lsm db upgrade`).

---

## 6. Cross-Cutting Concerns

### 6.1 Config Changes (v0.8.0 — Breaking)

Since v0.8.0 is a breaking release, config keys are renamed and restructured freely.

**`vectordb` section** (breaking change):
```yaml
vectordb:
  provider: "sqlite"        # "sqlite" or "postgresql" (no "chroma")
  path: "./data"            # SQLite: directory for lsm.db
  collection: "lsm"        # table name prefix
```

**`IngestConfig` additions**:
```python
max_heading_depth: Optional[int] = None
intelligent_heading_depth: bool = False
enable_section_summaries: bool = False
enable_file_summaries: bool = False
cluster_algorithm: str = "kmeans"
cluster_k: int = 50
```

**`RootConfig` additions**:
```python
max_heading_depth: Optional[int] = None  # per-root override
```

**`QueryConfig` additions and changes**:
```python
retrieval_profile: str = "hybrid_rrf"
# Options: "dense_only", "hybrid_rrf", "hyde_hybrid", "dense_cross_rerank", "llm_rerank"
# rerank_strategy removed (breaking)
k_dense: int = 100
k_sparse: int = 100
rrf_k: int = 60
rrf_dense_weight: float = 0.7
rrf_sparse_weight: float = 0.3
cross_encoder_model: Optional[str] = None   # None = ms-marco-MiniLM-L-6-v2
hyde_num_samples: int = 2
hyde_temperature: float = 0.2
hyde_pooling: str = "mean"
diversity_strategy: str = "minhash"         # "exact", "minhash", "simhash"
max_per_section: Optional[int] = 3
temporal_boost_enabled: bool = False
temporal_boost_days: int = 30
temporal_boost_factor: float = 1.5
cluster_enabled: bool = False
cluster_top_n: int = 5
```

**New `EvalConfig`**:
```python
eval_queries_path: Optional[Path] = None
eval_output_path: Optional[Path] = None
eval_baselines_dir: Optional[Path] = None
metrics: List[str] = ["recall@k", "mrr", "ndcg@k", "diversity@k", "latency"]
```

### 6.2 Transactional Consistency

With the unified SQLite database, the sync problem between vector writes and FTS5 writes
is eliminated. All writes in the ingest pipeline (`lsm_chunks`, `vec_chunks`, `chunks_fts`,
`lsm_manifest`) happen within the same SQLite transaction. A failed ingest leaves no
partial state.

`lsm db check` remains as a diagnostic tool but is no longer necessary for sync repair.
For the PostgreSQL provider, the `tsvector` column is `GENERATED ALWAYS AS`, so FTS and
vector data are always consistent within a single `INSERT`.

### 6.3 Provider Abstraction

`BaseVectorDBProvider` is updated:

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

### 6.4 Tests

| Feature | Test Type | Key Coverage |
|---------|-----------|-------------|
| SQLite-vec provider | Unit + Integration | vec0 insert, KNN query, metadata filters |
| FTS5 integration | Unit + Integration | Full-text insert, BM25 query, sync in transaction |
| Schema version tracking | Unit | DB schema read/write, mismatch error |
| DB completion | Integration | Changed config → selective re-ingest |
| Migration (all paths) | Integration | ChromaDB→SQLite, SQLite→PG, PG→SQLite |
| SQLite-backed manifest | Unit + Integration | DB migration from JSON, query correctness |
| DB prune | Integration | Soft-delete cleanup, criteria enforcement |
| RRF fusion | Unit | Score merging, rank preservation, weights |
| Cross-encoder reranking | Unit + Integration | Score improvement, CUDA path, cache |
| HyDE | Unit + Integration | Hypothetical generation, pooling, trace logging |
| Heading depth (fixed) | Unit | Depth limit, heading_path metadata |
| Heading depth (intelligent) | Unit | Section size estimation, recursion |
| Retrieval profiles | Integration | Each profile end-to-end, graceful degradation |
| MMR diversity | Unit | Diversity improvement, max_per_section cap |
| MinHash dedup | Unit | Near-duplicate suppression |
| Temporal ranking | Unit | Boost, time-range filter |
| Cluster-aware retrieval | Unit + Integration | Centroid matching, cluster filter in KNN |
| Multi-vector representation | Integration | section/file summary embeds, multi-granularity merge |
| Graph store | Unit + Integration | Node/edge insert, CTE traversal, expansion |
| Eval harness | Integration | BEIR format, metric computation, named baselines |

**Security**: FTS5 queries use parameterised queries. sqlite-vec queries use the Python
`sqlite3` parameter binding (`?` placeholders), never string interpolation. Cross-encoder
inputs are sanitised against prompt injection. Graph traversal depth is bounded to prevent
runaway recursion.

### 6.5 TUI Startup Advisories for Offline Jobs

Several v0.8.0 features gate on an explicit offline build step that must be run by the user
after ingest. Since these steps are opt-in and not part of the normal ingest/query flow,
users who configure a feature risk never discovering the prerequisite command.

**Decision**: On TUI startup, LSM inspects the DB and emits advisory messages for each
offline job that is configured-but-not-built or whose output is likely stale. Advisories
are non-blocking info/warn messages displayed in the TUI startup log before the main screen
is shown. Startup is never delayed or blocked by advisory checks.

#### 6.5.1 Job Status Table

Advisory state is tracked in the `lsm_job_status` table (see §4.3.4). Each offline job
writes a row on start and updates it on completion or failure. This allows detecting three
distinct conditions:

| Condition | How detected |
|-----------|-------------|
| Never run | No row in `lsm_job_status` for the job name |
| Started but not completed | Row with `status = "running"` (interrupted or in progress) |
| Completed but stale | Row with `status = "completed"` AND current `is_current` chunk count > `corpus_size * 1.2` |
| Completed and fresh | Row with `status = "completed"` AND corpus size within 20% |

#### 6.5.2 Advisories per Job

| Job | Config trigger | Advisory conditions | Message |
|-----|---------------|--------------------|---------|
| `lsm cluster build` | `cluster_enabled: true` in `query` config | Never run, interrupted, or stale (>20% corpus growth) | "Cluster-aware retrieval is on but clusters are [missing / stale]. Run: `lsm cluster build`" |
| `lsm graph build-links` | Active retrieval profile includes graph expansion | No thematic-link edges in `lsm_graph_edges` (structural edges from ingest are not checked) | "Graph thematic links not built. Run: `lsm graph build-links`" |
| `lsm finetune embedding` | `finetune_enabled: true` in `ingest` config | No row with `is_active = 1` in `lsm_embedding_models` | "Embedding fine-tuning is configured but no active fine-tuned model found. Run: `lsm finetune embedding`" |

Each advisory includes the exact CLI command the user should run. No job is checked unless
its feature flag/profile is active — users who have not enabled a feature see no advisory
for it.

#### 6.5.3 Advisory Severity

| Condition | Level | TUI display |
|-----------|-------|-------------|
| Job never run + feature active | `warn` | Yellow text in startup log |
| Job interrupted (status = "running") | `warn` | Yellow — "job may have been interrupted" |
| Job stale (corpus grown >20%) | `info` | Dim — "consider re-running" |

#### 6.5.4 Implementation Location

Advisory checks run in the TUI startup sequence, after the DB connection is established
and schema version is validated, before the main screen renders. Implementation:

```python
# lsm/ui/tui/startup.py  (or equivalent startup hook)

def check_offline_job_advisories(db: SQLiteVecProvider, config: LSMConfig) -> list[Advisory]:
    advisories = []
    current_corpus_size = db.execute(
        "SELECT COUNT(*) FROM lsm_chunks WHERE is_current=1"
    ).fetchone()[0]

    if config.query.cluster_enabled:
        row = db.execute(
            "SELECT status, corpus_size FROM lsm_job_status WHERE job_name='cluster_build'"
        ).fetchone()
        if row is None:
            advisories.append(Advisory("warn", "Cluster-aware retrieval is on but no cluster index exists.", "lsm cluster build"))
        elif row["status"] == "running":
            advisories.append(Advisory("warn", "Cluster build job appears incomplete (may have been interrupted).", "lsm cluster build"))
        elif row["status"] == "completed" and current_corpus_size > row["corpus_size"] * 1.2:
            advisories.append(Advisory("info", "Cluster index may be stale — corpus has grown significantly.", "lsm cluster build"))

    # Similar checks for graph_build_links and finetune_embedding ...

    return advisories
```

Each `Advisory` carries `level`, `message`, and `command`. The TUI renders them as
styled log lines. The same function is also called at the end of `lsm ingest`, so the
terminal CLI path also surfaces reminders.

#### 6.5.5 Non-TUI (CLI) Path

After `lsm ingest` completes, `check_offline_job_advisories()` is called and any
active advisories are printed to stdout as `[INFO]` / `[WARN]` lines. This ensures
users running headless/scheduled ingests are also reminded without requiring a TUI
session.

---

## 7. Long-Term Design Principles

From `INGEST_FUTURE.md` — non-negotiable constraints for all v0.8.0 architectural
decisions.

### 7.1 Retrieval as Composable Pipeline

Each stage (ANN retrieval, BM25 retrieval, RRF fusion, cross-encoder reranking, diversity
filtering, temporal ranking, graph expansion) is independently testable and replaceable.
Profiles are compositions of stages, not monolithic paths.

### 7.2 Evaluation-First Development

No retrieval feature merges without a measurable improvement on the eval harness. The eval
harness (§4.2.8) is implemented before hybrid retrieval or cross-encoder reranking. All
profile changes track metrics against a named baseline.

### 7.3 Retrieval Traces as First-Class Artifacts

Every retrieval result carries a `retrieval_trace.json` written to the workspace. The
trace feeds debugging, eval comparison, and Phase 3 multi-hop citation traces.

### 7.4 Local-First Foundation

All v0.8.0 features work without external service calls:
- Vector search and BM25 are local SQLite (sqlite-vec + FTS5) or local PostgreSQL.
- Cross-encoder runs locally via `sentence-transformers`.
- HyDE uses the configured LLM (local Ollama or hosted).
- Eval harness runs offline.

### 7.5 Single Database at a Time

The unified database principle (one SQLite file or one PostgreSQL database) is a
non-negotiable architectural constraint. No feature may introduce a new separate database
file or service without first integrating it into `lsm.db` or the PostgreSQL schema.

---

## 8. Dependency Map

```
SQLite-vec Unified DB (§4.3)
  ├── Required by: DB Versioning schema storage (§2)
  ├── Required by: Manifest storage (§3.4)
  ├── Required by: BM25/FTS5 (§4.2.3) ← same DB, same transaction
  ├── Required by: Reranker cache (§4.2.4)
  ├── Required by: Phase 2 cluster centroids (§4.4)
  ├── Required by: Phase 3 graph store (§4.6)
  └── Enables: All providers use single-DB architecture

DB Versioning (§2)
  ├── Required by: DB Completion (§3)
  ├── Required by: Heading Depth changes (§5) ← heading changes invalidate chunks
  └── Enables: Model registry foundation (§4.7)

heading_path metadata (§5.3)
  ├── Required by: Per-section diversity cap (§4.2.6)
  └── Required by: Phase 3 graph section edges (§4.6.3)

RetrievalPipeline abstraction (§4.2.1)
  ├── Required by: Retrieval Profiles (§4.2.2)
  ├── Required by: Hybrid RRF (§4.2.3)
  ├── Required by: Cross-encoder (§4.2.4)
  ├── Required by: HyDE (§4.2.5)
  ├── Required by: MMR Diversity & MinHash De-dup (§4.2.6)
  ├── Required by: Temporal Ranking (§4.2.7)
  ├── Required by: Cluster-aware retrieval (§4.4)
  ├── Required by: Multi-vector retrieval (§4.5)
  ├── Required by: Graph expansion (§4.6)
  └── Required by: Retrieval trace for eval

Eval Harness (§4.2.8)
  └── Must precede: Hybrid RRF, Cross-encoder, HyDE, Cluster-aware, Multi-vector validation
```

**Critical path**:
SQLite-vec DB (§4.3) → DB Versioning (§2) → RetrievalPipeline (§4.2.1) → Eval Harness
(§4.2.8) → Hybrid RRF (§4.2.3) → Cross-encoder (§4.2.4) → HyDE (§4.2.5)

**Parallel paths**:
- Heading Depth (§5) — independent of retrieval pipeline; shares DB versioning only.
- Phase 2 Clustering (§4.4) — after RetrievalPipeline; parallel to cross-encoder work.
- Phase 3 Graph Store (§4.6) — after SQLite-vec DB schema is locked; parallel to Phase 2.

---

*End of research document. All decisions resolved. Ready for implementation planning.*

# v0.8.0 Research Plan: Embedding Retrieval Improvements

**Status**: Discovery Phase
**Version Target**: 0.8.0
**Source**: `TODO` v0.8.0 section + `.agents/future_plans/INGEST_FUTURE.md`

This document records design considerations, architectural options, open questions, and
dependencies for v0.8.0. It is not an implementation plan — that comes after these
questions are resolved. Clarification questions requiring user decisions are consolidated
at the end.

---

## Table of Contents

1. [Scope Summary](#1-scope-summary)
2. [DB Versioning with Migration/Upgrade](#2-db-versioning-with-migrationupgrade)
3. [DB Completion — Incremental Corpus Updates](#3-db-completion--incremental-corpus-updates)
4. [Architectural Overhaul: Retrieval Pipeline](#4-architectural-overhaul-retrieval-pipeline)
5. [Document Headings Improvement](#5-document-headings-improvement)
6. [Cross-Cutting Concerns](#6-cross-cutting-concerns)
7. [Long-Term Design Principles (INGEST_FUTURE.md)](#7-long-term-design-principles)
8. [Dependency Map](#8-dependency-map)
9. [Clarification Questions](#9-clarification-questions)

---

## 1. Scope Summary

The three TODO items for v0.8.0 are:

| # | TODO item | Complexity |
|---|-----------|------------|
| A | DB versioning with migration/upgrade between versions | High |
| B | DB completion (add new data without full rebuild) | High |
| C | Architectural overhaul of ingest and embedding retrieval (INGEST_FUTURE.md) | Very High |
| D | Document headings configurable depth + intelligent selection | Medium |

Items A and B are closely coupled: both require a schema-version concept that does not
currently exist. Item C is the largest body of work — INGEST_FUTURE.md defines three
phases of retrieval modernization; this document assesses what v0.8.0 should target.
Item D is self-contained but interacts with how the ingest pipeline produces chunks.

---

## 2. DB Versioning with Migration/Upgrade

### 2.1 What Exists Today

The codebase has a partial versioning foundation:

- **File-level versioning** (`manifest.py:23-36`): `get_next_version()` increments an
  integer per source file path across ingest runs.
- **Chunk-level soft-delete** (`pipeline.py:370-386`): When `enable_versioning=True`,
  old chunks are marked `is_current=False` instead of deleted.
  **User Response:** Enable versioning should not be a flag but versioning should just be how the system works. And then the VectorDB should have a clean method that cleans up old versions based upon some criteria.
- **Query-time version filter** (`planning.py:189`): Filters to `is_current=True` when
  versioning is enabled, hiding historical chunks.
- **DB-to-DB migration** (`migrations/chromadb_to_postgres.py`): Copies all vectors
  from ChromaDB to PostgreSQL via the provider interface.

What is **absent**: any notion of *schema version* — tracking what version of the
chunking algorithm, embedding model, or metadata layout produced a given set of vectors.
A freshly deployed codebase loads an existing vector DB and has no way to detect
incompatibility or know which chunks need updating.

### 2.2 What "Schema Version" Means Here

A "schema version" in this context needs to capture at minimum:

| Dimension | Why It Matters | Current State |
|-----------|---------------|---------------|
| Embedding model name + dimension | Different models produce incompatible vector spaces | Validated only at ingest time (`pipeline.py:279-287`); not recorded in DB or manifest |
| Chunking strategy + params | `structure` vs `fixed`, chunk_size, overlap | Not recorded |
| Metadata field set | Added fields (e.g., `heading`, `page_number`) don't backfill old chunks | Not recorded |
| Parser version | Parser behaviour changes affect chunk content | Not recorded |
| Ingest code version | LSM version that produced the chunks | Not recorded |

Without schema versioning, any upgrade to chunking or embedding silently creates a
mixed-generation corpus — old chunks and new chunks coexist in the same collection with
different vector spaces and metadata layouts.

### 2.3 Where Schema Version Should Live

There are two natural homes:

**Option A — In the manifest**: Add a top-level `"schema": {...}` block to
`manifest.json` capturing embedding model, chunking params, and code version. The
manifest already has a single-writer pattern, making this consistent.

**Option B — In the vector DB as collection metadata**: ChromaDB supports per-collection
metadata; PostgreSQL supports a metadata table. This keeps schema info co-located with
the data even if the manifest file is lost.

**Recommended exploration**: Use both. The manifest holds the authoritative schema record
(since it is already committed transactionally after successful writes in
`pipeline.py:349`). The vector DB holds a secondary copy as collection metadata for
self-describing data.

**User Response:** The schema version should live in SQLite if using Chroma and PG if using PG for vector store.

### 2.4 Migration / Upgrade Path

When the system detects a schema mismatch (e.g., user changes `embed_model` in config
or the code upgrades its default chunking), several strategies are possible:

| Strategy | Description | Trade-offs |
|----------|-------------|------------|
| **Full rebuild** | Wipe and re-ingest entire corpus | Simplest; expensive for large corpora |
| **Selective re-embed** | Re-embed only files whose chunks used the old model | Requires per-chunk model provenance; avoids full rebuild |
| **Dual-generation query** | Keep old vectors, query both old and new indices, RRF merge | Zero downtime; highest query complexity |
| **Incremental migration job** | Background job that progressively re-embeds old chunks | Low user disruption; requires per-chunk tracking |

The selective re-embed strategy requires the manifest to record `embedding_model` per
file entry (or per chunk via vector DB metadata). The dual-generation approach requires
multi-collection query support in `BaseVectorDBProvider`.

### 2.5 Key Open Questions Before Design

- How should schema mismatch be surfaced to the user (warning, error, or auto-migration)?
**User Response:** By an error with instruction on how to auto-migrate. Maintaining extra code to handle warnings adds too much code complexity.
- Is a `lsm migrate` CLI command the intended entry point, or should migration happen
  automatically at ingest time? **User Response:** Migrate should be a CLI command for ingest and handled by the Ingest screen in the TUI.
- Should the PostgreSQL provider expose collection metadata APIs to store schema info, or
  is the manifest sufficient? **User Response:** This data should be stored in DB.
- What is the expected corpus size? (Under 100k files: manifest can scale; over 500k:
  SQLite or DB-backed manifest becomes necessary — see §3.4.) **User Response:** Expect the corpus of chunks to be 100k+. DB backed manifest is necessary.

---

## 3. DB Completion — Incremental Corpus Updates

### 3.1 The Problem

"DB completion" means: given an existing vector DB that was built with version X of LSM,
bring it up to date with version Y without a full rebuild. Concretely this covers:

1. **New file types**: If a new parser (e.g., `.epub`, `.pptx`) is added in v0.8.0,
   old corpora should be completable by ingesting only the newly supported file types,
   not re-processing everything.

2. **New metadata fields**: If `heading` metadata or `page_number` tracking is added,
   existing chunks lack those fields. Can old chunks be enriched without re-embedding?

3. **Changed chunking**: If `structure` chunking is improved (e.g., heading depth
   configuration), old chunks have different boundaries. These can only be fixed by
   re-chunking.

4. **New ingest features**: If AI tagging (`enable_ai_tagging`) or language detection
   (`enable_language_detection`) is enabled on an existing corpus, old chunks lack those
   tags. Some enrichment is possible post-hoc (re-read file, re-tag, update metadata);
   re-embedding is not needed.

### 3.2 What Already Works

The current incremental system (`pipeline.py:620-647`) already skips unchanged files via
the three-level manifest check (mtime → size → hash). Files that are changed or new are
always re-ingested. This handles the "new files added to roots" case perfectly.

The gap is: a file that has *not* changed on disk but whose *vector representation or
metadata is now stale* by the codebase's definition. The manifest's hash-equality check
returns "skip" — but the file should be re-ingested.

### 3.3 Completion Modes to Research

| Mode | Trigger | Scope | Cost |
|------|---------|-------|------|
| **Extension completion** | New `extensions` added to config | Only files with new extensions | Low — only new file types |
| **Metadata enrichment** | New metadata field (tags, language) available | All files, no re-embedding | Medium — re-parse + metadata update |
| **Chunk boundary update** | Chunking params changed | All files for that strategy | High — re-chunk + re-embed all |
| **Embedding upgrade** | Embedding model changed | All files | Very high — full re-embed |
| **Selective re-ingest** | Manual via CLI flag `--force-file-pattern` | User-specified subset | Variable |

A `completion` command or a `--force-reingest-changed-config` flag could compare the
manifest's recorded schema against the current config and ingest only files whose chunks
would differ under the new schema.

### 3.4 Manifest Scaling

The current manifest is a flat JSON dict (`source_path → entry`). At scale:

| Corpus size | Manifest size | Load/save time |
|-------------|---------------|----------------|
| 10k files | ~2 MB | Negligible |
| 100k files | ~20 MB | ~200ms |
| 500k files | ~100 MB | ~1s+ |

For a personal knowledge base this is unlikely to be a bottleneck, but the manifest
should be considered for a future SQLite backing (see v0.9.0 TODO: "persist ingest state
and file hashes to DB"). For v0.8.0, adding a `schema` header block to the manifest
header is sufficient and does not require a format change that would invalidate existing
manifests.

**User Response:** Manifest should move to DB backed.

### 3.5 Manifest Schema to Record

A minimal v0.8.0 schema block in `manifest.json`:

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
  },
  "/path/to/file.md": {
    "mtime_ns": 1709000000000000000,
    "size": 12345,
    "file_hash": "abc123...",
    "version": 1,
    "updated_at": "2026-02-25T10:00:00Z"
  }
}
```

This is a backwards-compatible addition (existing code ignores unknown keys). Upgrade
detection reads `_schema.embedding_model` and compares to config.

---

## 4. Architectural Overhaul: Retrieval Pipeline

The full plan lives in `.agents/future_plans/INGEST_FUTURE.md`. It defines three phases.
This section assesses each sub-feature against the current state and identifies what
v0.8.0 should target.

### 4.1 Current Retrieval Architecture

```
user query
   → embed_text() [retrieval.py:82]        # single-model dense embedding
   → retrieve_candidates() [retrieval.py:126] # kNN cosine ANN via vector DB
   → filter_candidates() [retrieval.py:196]   # path/ext post-filters
   → apply_local_reranking() [rerank.py:255]  # lexical reranking + diversity cap
   → context assembly                          # top-k chunks as LLM context
   → LLM synthesis
```

There are no retrieval profiles, no sparse index, no cross-encoder, no HyDE, no
evaluation harness. The query config (`QueryConfig`) has `rerank_strategy` ("lexical",
"hybrid", "llm", "none") but "hybrid" currently means lexical-then-diversity, not
hybrid dense+sparse.

### 4.2 Feature Assessment: INGEST_FUTURE.md Phase 1

#### 4.2.1 Unified RetrievalPipeline Abstraction

**Current state**: Query pipeline is a linear function chain spread across `api.py`,
`planning.py`, `context.py`, `retrieval.py`, `rerank.py`. No base abstraction.

**What is needed**: A `RetrievalPipeline` class with a `run(query: Query) → List[Candidate]`
interface and composable stages (`recall → fuse → rerank → diversify`). Each stage
receives the output of the previous stage and can add score breakdowns.

INGEST_FUTURE.md specifies four standard objects that must be defined:
- **`Query`**: Wraps the user's question string plus retrieval parameters (profile, k, filters).
- **`Candidate`**: Exists already but needs extension — must carry a `ScoreBreakdown`.
- **`ScoreBreakdown`**: Per-candidate scoring detail, minimally containing:
  - `dense_score: float` — cosine similarity from vector ANN
  - `dense_rank: int` — rank position in dense results
  - `sparse_score: float` — BM25 score from FTS5
  - `sparse_rank: int` — rank position in sparse results
  - `fused_score: float` — RRF-combined score
  - `rerank_score: Optional[float]` — cross-encoder output, if applied
  - `temporal_boost: Optional[float]` — recency multiplier, if applied
- **`Citation`**: A resolved reference from a `Candidate`, distinct from the current
  ad-hoc citation string in `context.py`. Should carry: `chunk_id`, `source_path`,
  `heading`, `page_number`, `url_or_doi` (for remote sources), `snippet`.

The `retrieval_trace.json` emitted per query (per INGEST_FUTURE.md) requires stage-level
timing and the full `ScoreBreakdown` for every candidate that entered the pipeline — this
is only possible with a structured object model, not the current function chain.

**Why it matters**: Without an abstraction, adding hybrid retrieval, HyDE, or
cross-encoders requires modifying the `api.py` giant function. An abstraction makes
profiles switchable at runtime.

**Design considerations**:
- Should pipeline stages be registered by name (like tool registry) to allow config-driven
  profile construction? **User Response:** What are the concrete benefits of having registered names?
- Should stages have access to the full `QueryConfig` or only their stage-specific config? **User Response:** Would stages benefit having the extra data?
- `retrieval_trace.json` output requires stage-level timing and score breakdowns, which
  are only possible with an object model.

#### 4.2.2 Retrieval Profiles

**INGEST_FUTURE.md proposes**:
```yaml
retrieval_profiles:
  - dense_only
  - hybrid_rrf
  - hyde_hybrid
  - dense_cross_rerank
```

**Design considerations**:
- Profiles should be selectable at query time (per-request or config default).
- `QueryConfig.mode` currently has "grounded" and "insight" (synthesis modes, not
  retrieval modes). Retrieval profiles are a separate dimension.
- Profiles may have incompatible dependencies: `hyde_hybrid` requires an LLM at
  query time; `dense_cross_rerank` requires a cross-encoder model on disk.
- Should profiles degrade gracefully? E.g., if BM25 index does not exist, should
  `hybrid_rrf` fall back to `dense_only` with a warning? **User Response:** Yes

#### 4.2.3 Hybrid Retrieval: Dense + Sparse (BM25/FTS5) + RRF

**Current state**: No sparse index. "Hybrid" in `rerank_strategy` is misnaming — it is
lexical reranking over dense results, not a true dual-recall system.

**Recall pool sizing**: INGEST_FUTURE.md specifies separate limits for each channel:
- `K_dense`: number of candidates recalled from the vector ANN index
- `K_sparse`: number of candidates recalled from the BM25/FTS5 index
These should be independent config params (not the single `k` that currently drives
everything). Typical values: `K_dense=100, K_sparse=50`. The combined candidate pool
before RRF fusion can have up to `K_dense + K_sparse` entries, which is subsequently
fused and truncated to the final `k` result set.

**SQLite FTS5 approach** (most compatible with local-first constraint):
- SQLite is already an available dependency (used by the memory store).
- Create a sidecar SQLite DB at `<persist_dir>/bm25.db` with an FTS5 virtual table.
- Index fields: `chunk_text`, `heading`, `source_name`, metadata text fields.
- At ingest time: write chunks to both vector DB and FTS5 index (must stay in sync).
- At query time: run dense ANN (vector DB) and BM25 (FTS5) separately, merge with RRF.

**PostgreSQL FTS approach** (for users running PostgreSQL):
- PostgreSQL has native `tsvector`/`tsquery` full-text search with `ts_rank`.
- Add `chunk_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED`
  column to the `lsm_chunks` table.
- No separate index DB needed; both queries hit the same table.

**RRF formula**: `score(d) = Σ w_i / (k + rank_i(d))` where k=60 is the standard value
and `w_i` is a per-channel weight. INGEST_FUTURE.md specifically calls for configurable
per-channel weights — e.g., `w_dense=0.7, w_sparse=0.3` — allowing the user to tune
how much each retrieval channel contributes. The k parameter is not critically sensitive
(k=60 is the de-facto default in OpenSearch, Azure AI Search, and Chroma's implementation)
but the channel weights matter significantly for domain tuning.

**Score breakdown persistence**: Per INGEST_FUTURE.md, each RRF-merged candidate must
carry its full `ScoreBreakdown` (see §4.2.1) with `dense_rank`, `sparse_rank`, and
`fused_score`. This breakdown feeds both the `retrieval_trace.json` and the eval harness.

**Key design decisions**:
- Should the BM25 index be a separate sidecar, or extend the vector DB provider? **User Response:** When using CHROMA use separate SQLite, when using PG for vector use PG.
- `BaseVectorDBProvider` currently has no `fts_query()` method. Adding one would require
  all providers to implement it — or a new `HybridVectorDBProvider` subclass.
- At ingest time, the writer thread (`pipeline.py:325-459`) owns the vector DB; FTS5
  writes would need to happen in the same writer thread to keep sync.
- Chunk IDs must be consistent across both indices to enable the RRF join.

**Performance characteristics** (from research):
- SQLite FTS5 with BM25 handles hundreds of thousands of documents sub-second on
  commodity hardware when the query is a simple keyword match.
- `sqlite-vec` (SQLite vector extension) is an alternative to a separate ChromaDB for
  users who want a single-file database — may be worth researching as a future backend. **User Response:** Do further research on SQLite vector vs Chroma and how easy it is to add. I like removing dependancies.

**Acceptance criteria** (from INGEST_FUTURE.md): Hybrid recall improves Recall@20 vs
dense-only on the evaluation set. Retrieval trace shows full scoring breakdown per
candidate.

#### 4.2.4 Cross-Encoder Reranking

**Current state**: Only lexical (BM25-like token overlap) or LLM-based reranking.

**Cross-encoder approach**:
- A cross-encoder takes a `(query, passage)` pair and outputs a relevance score.
- Bi-encoders (used for dense recall) embed query and passage independently. Cross-
  encoders encode them jointly and achieve significantly better ranking at higher cost.
- Typical pattern: dense recall top-100, cross-encoder rerank to top-20.

**Available models** (all local, no API call):
| Model | Size | MRR@10 on MS-MARCO | Best For |
|-------|------|---------------------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 66M | 39.01 | Fast general-purpose |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 127M | 39.97 | Quality-balanced |
| `cross-encoder/ms-marco-electra-base` | 109M | 39.96 | Highest quality |

These models come from `sentence-transformers` and use `CrossEncoder` class, which is
already a dependency via `sentence-transformers`.

**LLM rerank reconciliation**: `QueryConfig` already has `rerank_strategy: "llm"` which
triggers the existing LLM reranking path. INGEST_FUTURE.md treats LLM reranking as an
"Optional LLM rerank mode: Disabled by default, Used only for justification or deep
reasoning profile." This needs to be reconciled with the profile system: the `llm`
reranking strategy should become a standalone profile (e.g., `llm_rerank`) rather than
an option inside existing profiles, since it has fundamentally different cost and latency
characteristics. The existing `rerank_strategy` config key would then become a legacy
field superseded by `retrieval_profile`.

**Reranker cache**: INGEST_FUTURE.md specifies caching with key
`(query_hash, chunk_id, model_version)` to avoid repeated scoring. For a personal
knowledge base with repetitive queries, this cache hit rate could be high.

**Cache invalidation**: INGEST_FUTURE.md explicitly requires cache invalidation "on corpus
rebuild." The trigger mechanism needs design: the cache key includes `model_version`, so
changing the cross-encoder model auto-invalidates. But a corpus rebuild (new chunks,
deleted chunks) also invalidates by changing `chunk_id` values — entries for old chunk
IDs become stale naturally. A rebuild that keeps the same chunk IDs (e.g., only metadata
enrichment) does not change IDs, so the cache remains valid in that case. An explicit
cache wipe command (`lsm cache clear`) should be available for manual invalidation.

**Acceptance criteria** (from INGEST_FUTURE.md): Reranked pipeline improves nDCG@10 vs
hybrid-only. Latency remains under a configurable threshold.

**Design decisions**:
- Should the cross-encoder model be downloaded at install time or lazily on first use? **User Response:** Lazily on first use.
- Should reranker cache be in-memory only (lost between sessions) or persisted (SQLite)? **User Response:** Persisted
- CPU-only cross-encoder inference on a 100-candidate pool is ~2-10s; is that acceptable? **User Response:** Not if it can be run on CUDA/GPU.

#### 4.2.5 HyDE (Hypothetical Document Embeddings)

**Mechanism**:
1. LLM generates 1-3 "hypothetical answers" to the user's query (zero-shot, no context).
2. Each hypothetical answer is embedded using the same bi-encoder model.
3. Embeddings are averaged (mean pooling).
4. The averaged embedding is used in place of the direct query embedding for ANN search.

**Why it works**: Direct query embeddings often land in a different vector space region
than answers. A "What is X?" query embedding is semantically closer to other questions
than to the answers. A hypothetical answer embedding is closer to real answers.

**Cost**: Requires one LLM call per query (before retrieval), adding latency. Number of
hypothetical documents (1-3) and temperature (~0.2) are key tuning parameters.

**Aggregation strategy**: INGEST_FUTURE.md allows "mean or max pooling" for combining the
hypothetical embeddings. Mean pooling (average each dimension across all hypothetical
embeddings) is the standard and produces a centroid in embedding space. Max pooling
(take the max value per dimension) is less common but can be more robust when hypothetical
answers are semantically very different from each other. Both should be supported.

**When HyDE helps most**: Abstract, conceptual, or exploratory queries ("What were my
notes about consciousness?"). For factual lookups ("page number of chapter 3") HyDE
may add noise.

**Acceptance criteria** (from INGEST_FUTURE.md): Abstract queries show improved Recall@20.
HyDE can be toggled per-profile. Generated hypothetical text is logged in the retrieval
trace.

**Design decisions**:
- HyDE should be gated by profile (`hyde_hybrid` profile only) and disabled by default.
- The hypothetical generation prompt needs to be tunable per domain (academic vs
  personal notes vs code).
- Should the hypothetical documents be logged in the retrieval trace and/or TUI debug view?

#### 4.2.6 Diversity & De-duplication

**Current state**: Two mechanisms exist in `rerank.py`:
- `deduplicate()` (line 154): removes exact duplicates using Python's `hash()` on
  normalized text — effective for identical or near-identical chunks but misses paraphrases.
- `enforce_diversity()` (line 204): caps results to `max_per_file` chunks per source file.

**What INGEST_FUTURE.md adds beyond current state**:

**Per-section cap (new)**: A cap based on *heading path*, not just file. If a query
pulls 8 chunks from a single large chapter, a per-section cap (e.g., `max_per_section=2`)
would limit chunks from `"Chapter 3::Background"` to 2, even if they come from different
files. This requires the heading path metadata from §5 to be populated at ingest time.

**Near-duplicate detection via SimHash/MinHash (new)**: The current hash-based
deduplication is exact — two chunks must normalize to identical text to be suppressed.
Near-duplicates (paraphrased sentences, slight rewording, overlapping overlap windows)
are missed. SimHash and MinHash are locality-sensitive hashing techniques that detect
approximate text similarity:
- **SimHash**: Generates a 64-bit fingerprint; two documents with ≤k differing bits are
  near-duplicates. Fast to compute; simple Hamming distance comparison.
- **MinHash**: Estimates Jaccard similarity between shingle sets. More accurate for
  short overlapping passages but higher compute cost.
Either can be stored as chunk metadata at ingest time (a `simhash: int` field) and
compared at query time during deduplication. This is a two-phase approach: compute at
ingest, compare at retrieval.

**Greedy diversity selection post-rerank (new)**: The current system caps by file count
after reranking, which is a filter not a selection. Greedy selection (Maximal Marginal
Relevance, MMR) would iteratively pick the next best candidate that maximizes relevance
while minimizing similarity to already-selected candidates. This produces diverse results
without hard per-file caps.

**Design considerations**:
- Per-section cap requires the heading path metadata feature (§5) to be in place first.
- SimHash/MinHash compute cost at ingest is negligible; storing a `simhash` int per chunk
  adds ~8 bytes to metadata.
- MMR greedy selection requires pairwise similarity computation over the candidate pool —
  with a pool of ~100 candidates this is trivial, but needs the candidate embeddings
  available at rerank time (currently not passed through to the reranking functions).
- The existing `max_per_file` config param becomes a subset of a broader diversity config.

**Acceptance criteria** (from INGEST_FUTURE.md): No more than the configured cap per
file (already met). Diversity@K metric improves without major drop in precision.

#### 4.2.7 Temporal-Aware Ranking

**Current state**: `mtime_ns` is already stored in chunk metadata (confirmed in
`pipeline.py:401`). It is never used at query time.

**What INGEST_FUTURE.md proposes**:
- Boost recent documents (configurable recency weight).
- Filter by time range ("what was I thinking in 2023?").

**Design considerations**:
- Recency boost needs a decay function. Common options: linear decay, exponential decay,
  step function (e.g., 2x boost for files touched in last 30 days).
- Time-range filtering is a pure metadata filter (already supported via `where_filter`
  in `retrieve_candidates()`).
- The mtime represents file modification time, not necessarily content creation time.
  Documents with stable `mtime` (e.g., PDFs copied from elsewhere) might be incorrectly
  penalised by recency boost.

#### 4.2.8 Evaluation Harness (lsm.eval.retrieval)

**Current state**: No evaluation infrastructure exists. There are no frozen test queries,
no golden answer sets, no metrics computation.

**Why this is critical**: INGEST_FUTURE.md states "Evaluation-first development" as a
long-term principle (see §7). Without an eval harness, it is impossible to know whether
hybrid retrieval, cross-encoder reranking, or HyDE actually improve results over the
baseline. Every new retrieval feature introduced in v0.8.0 is unverifiable without it.

**CLI interface** (per INGEST_FUTURE.md):
```
lsm eval retrieval --profile hybrid_rrf
lsm eval retrieval --profile dense_only   # run baseline
lsm eval retrieval --profile dense_cross_rerank --compare baseline.json
```

**Regression comparison**: INGEST_FUTURE.md explicitly requires output as a "regression
comparison vs baseline." This means the harness must:
1. Save a baseline run result to a file (e.g., `eval_baseline.json`).
2. On subsequent runs, load the baseline and print a diff table showing Δ per metric.
3. Flag regressions (metrics that worsen beyond a configurable threshold) as failures.
This makes the eval harness usable in CI-like workflows when tuning retrieval params.

**Proposed metrics**:
- **Recall@K**: What fraction of relevant chunks appear in top-K results?
- **MRR (Mean Reciprocal Rank)**: Reciprocal of the rank of the first relevant chunk.
- **nDCG@K**: Normalized Discounted Cumulative Gain — accounts for rank position.
- **Diversity@K**: Fraction of results from distinct source files.
- **Latency**: End-to-end query time per profile.

**Key challenge**: Building a golden evaluation set requires either manual annotation or
weak supervision. INGEST_FUTURE.md suggests deriving weak supervision from:
- Heading-content pairs as positives (query=heading text, positive=section body).
- Backlinks between files as positives.
- Near-miss retrieval failures (known-good chunk ranked low) as hard negatives.

A synthetic derivation pipeline would run at ingest time against the user's own corpus,
requiring no manual labelling — but quality is limited. Manual curation produces higher
quality but requires user effort.

**Design decisions**:
- Should eval be a CLI-only tool or also accessible from the TUI? **User Response:** CLI only
- How large does the evaluation set need to be for statistically meaningful comparisons?  **User Response:** Give the user suggestions.
- Should the eval harness export to a standard format (BEIR benchmark)?  **User Response:** yes
- How should the frozen query test set be versioned alongside the corpus?  **User Response:** Yes, but we will need to produce a versioned corpus to do builds on. This will help our testing as a whole, if we have a meaningful size corpus to do test runs on.

### 4.3 Phase 2 and Phase 3 Features (Out of Scope for v0.8.0)
**User Response:** v0.8.0 is another major version number. This is in the scope and should be researched as part of the whole version implementation.

Based on complexity, the following from INGEST_FUTURE.md should be deferred:

| Feature | Phase | Why Defer |
|---------|-------|-----------|
| Cluster-aware retrieval (k-means/HDBSCAN) | 2 | Requires offline clustering job; high ML complexity |
| Multi-vector representation (file/section summary embeddings) | 2 | Requires new DB schema columns for summary vectors |
| Graph-augmented retrieval | 3 | Graph store is a new infrastructure dependency |
| Domain-fine-tuned embeddings | 3 | Requires training pipeline; high research cost |
| Multi-hop retrieval agents | 3 | Requires integration with the meta-agent system |

These are v0.9.0+ territory. However, v0.8.0 design decisions must not foreclose them.

**Phase 2 forward-compatibility implications for v0.8.0**:

- **Cluster-Aware Retrieval** (Phase 2) will require `cluster_id: int` and
  `cluster_size: int` metadata fields per chunk, persisted at ingest time by an offline
  clustering job. The v0.8.0 schema should reserve these field names in documentation
  even if they are not yet populated. UMAP visualization export for clusters is also
  planned as a diagnostic.

- **Multi-Vector Representation** (Phase 2) requires file-level and section-level summary
  embeddings stored alongside chunk embeddings. The current vector DB schema stores one
  embedding per chunk. Supporting summary embeddings will either require: (a) a new
  collection per granularity level (file-collection, section-collection, chunk-collection)
  or (b) a `node_type` metadata field that distinguishes `"chunk"`, `"section_summary"`,
  `"file_summary"`. The v0.8.0 vectordb schema should be designed to accommodate option
  (b) without a breaking migration in v0.9.0.

**Phase 3 forward-compatibility implications for v0.8.0**:

- **Domain-Fine-Tuned Embeddings** (Phase 3) requires a model registry that versions
  embeddings and triggers re-embedding via CLI. This is directly related to the v0.8.0
  DB versioning work (§2): the embedding model version tracking in the manifest `_schema`
  block is the foundation the model registry will build on.

- **Multi-Hop Retrieval Agents** (Phase 3) will use the meta-agent system (Phase 6/10
  work, already complete) as its orchestration layer. The query decomposition strategies
  (key term expansion, named entity extraction, conceptual prerequisite discovery) have
  a partial foundation in `lsm/query/decomposition.py` — that module already extracts
  keywords, author, title, DOI, and date entities from queries. Phase 3 would extend it.

- **Graph-Augmented Retrieval** (Phase 3) proposes a local graph store with nodes
  (File, Section, Chunk, Entity) and edges (contains, references, same_author,
  thematic_link), plus heading tree ingestion as graph edges. The heading hierarchy
  metadata from §5 (heading path as `"H1::H2::H3"`) is the ingest-time artifact that
  Phase 3 will consume to build section-node edges in the graph.

---

## 5. Document Headings Improvement

### 5.1 Current State

The structure chunking system (`structure_chunking.py`) treats all heading levels
identically as chunk boundaries. A heading at level H1 has the same boundary-forcing
effect as H6. The heading text is stored in the chunk's `heading` metadata field for
retrieval context.

Heading detection (`utils/text_processing.py:48-77`) recognizes:
- Markdown `#`-style headings (levels 1-6)
- HTML `<h1>`–`<h6>` elements
- Bold-only lines (`**Heading**`) treated as level 1
- Plain-text section markers (optional, disabled for structure chunking)

### 5.2 The Two TODO Requirements

#### Requirement 1: Configurable Heading Depth

The user should be able to set `max_heading_depth: int` (e.g., 2) such that headings
deeper than that level are NOT treated as chunk boundaries — their text flows into the
body of the parent heading's chunk.

**Behavioral example with `max_heading_depth=2`**:
```
# H1 Topic                     ← chunk boundary (depth 1 ≤ 2)
## H2 Subtopic                 ← chunk boundary (depth 2 ≤ 2)
### H3 Detail                  ← NOT a boundary (depth 3 > 2); text flows into H2 chunk
#### H4 Sub-detail             ← NOT a boundary; flows into H2 chunk
Regular paragraph text
```

**Effect on chunk size**: Deeper headings that are normally short (headings often have
little text before the next sub-heading) would merge into their parent's chunk. Chunk
sizes increase but semantic coherence of the parent heading is better preserved.

**Heading metadata implication**: The `heading` field stored per chunk should reflect
the most recent *valid* heading (at depth ≤ max_heading_depth), not the deepest heading.
This affects how heading metadata is used in retrieval and BM25 indexing.

**Config location**: `IngestConfig` is the natural home:
```python
max_heading_depth: Optional[int] = None  # None = all levels are boundaries
```

#### Requirement 2: Intelligent Heading Depth Selection

When `max_heading_depth` is not explicitly set, the chunker should decide automatically
based on the following criteria from the TODO:

> When a heading is larger than the max chunk size and contains sub-headings, then the
> section should be chunked along the sub-heading boundaries. The metadata in the chunk
> should indicate the hierarchy of headings: `heading::sub-heading`. This process should
> repeat into nested subheadings so long as the current heading level is larger than the
> target max chunk size.

**Algorithm description**:
1. For each heading-delimited section, estimate its size (character count of all text
   below it, before the next sibling or parent heading).
2. If section size ≤ `chunk_size`: keep the whole section as one chunk; sub-headings
   within become part of that chunk's text (not boundaries).
3. If section size > `chunk_size` AND section has sub-headings: recursively split at
   sub-heading boundaries.
4. Sub-heading hierarchy is reflected in metadata: `"Introduction::Background::Prior Work"`.

**This is a fundamentally different algorithm** from the current one. Currently: headings
always split. Proposed: headings split only when the parent section would exceed
`chunk_size`.

**Key differences from fixed max_heading_depth**:

| Aspect | Fixed depth | Intelligent |
|--------|------------|-------------|
| Depth limit | Static per config | Dynamic per section size |
| Metadata | Shallowest valid heading | Full heading path |
| Small sub-sections | Always merged at depth>max | Merged when parent fits |
| Large top-level sections | Always splits at sub-heading | Splits to fit chunk_size |

### 5.3 Design Considerations

**Heading hierarchy in metadata**: The current `heading` field is a flat string. The
intelligent algorithm requires a path like `"Introduction::Background::Prior Work"`.
Changing the field format from a string to a path string has no schema migration impact
(it is still a string), but querying by heading prefix becomes important for retrieval.

**Interaction with fixed chunking**: The `fixed` chunking strategy ignores headings
entirely. Both heading improvements apply only to the `structure` strategy.

**Interaction with page segments**: PDF/DOCX page tracking (`page_segments`) is already
propagated through `structure_chunking.py`. The heading depth logic needs to preserve
page boundary tracking.

**Edge cases**:
- A document with only H1 headings and huge sections → intelligent mode would recurse
  to paragraph-level; max recursion depth needs a bound.
- A document with no headings → both modes fall through to sentence-boundary chunking,
  unchanged.
- A document where all content is under H6 headings → with `max_heading_depth=2`, the
  entire document might be a single chunk if H6 sections are small.

**Testing**: Heading depth changes produce different chunk boundaries and different
`chunk_id` values (which are derived from source_path + file_hash + chunk_index). Any
DB corpus built before heading depth changes becomes stale after — this is a DB
versioning concern (§2).

---

## 6. Cross-Cutting Concerns

### 6.1 Config Changes

All new features require config extensions. Key additions anticipated for `v0.8.0`:

**IngestConfig additions**:
```python
max_heading_depth: Optional[int] = None
intelligent_heading_depth: bool = False
```

**New `IngestSchemaConfig` (manifest schema tracking)**:
```python
embed_model: str  # duplicates global embed_model for schema tracking
chunking_strategy: str
chunk_size: int
chunk_overlap: int
manifest_version: int
```

**QueryConfig additions**:
```python
retrieval_profile: str = "dense_only"
# Options: "dense_only", "hybrid_rrf", "hyde_hybrid", "dense_cross_rerank"
k_dense: int = 100          # Recall limit for the dense ANN channel
k_sparse: int = 100         # Recall limit for the sparse BM25/FTS5 channel
rrf_k: int = 60             # RRF constant k (higher = less aggressive rank fusion)
rrf_dense_weight: float = 0.7   # Per-channel weight applied before RRF score sum
rrf_sparse_weight: float = 0.3  # Per-channel weight (should sum to 1.0)
cross_encoder_model: Optional[str] = None
# None = use default (ms-marco-MiniLM-L-6-v2); explicit = user-specified model path
hyde_enabled: bool = False
hyde_num_samples: int = 2
hyde_temperature: float = 0.2
hyde_pooling: str = "mean"  # Options: "mean", "max" — how to aggregate HyDE embeddings
diversity_strategy: str = "exact"
# Options: "exact" (current hash dedup), "simhash" (near-dup), "minhash" (near-dup)
max_per_section: Optional[int] = None
# None = no per-section cap; int = max chunks from same heading-path prefix
temporal_boost_enabled: bool = False
temporal_boost_days: int = 30
temporal_boost_factor: float = 1.5
```

**New `EvalConfig`**:
```python
eval_queries_path: Optional[Path] = None
eval_output_path: Optional[Path] = None
metrics: List[str] = ["recall@k", "mrr", "ndcg@k"]
```

### 6.2 BM25 Index Synchronization

The BM25/FTS5 sidecar index introduces a new synchronization problem: the vector DB and
the BM25 index can drift if one write succeeds and the other fails. Current design
considerations:

- The writer thread in `pipeline.py` is the sole owner of the vector DB.
- To keep sync, BM25 writes should happen in the same writer thread, not a separate one.
- Rollback semantics: ChromaDB has no transaction rollback; FTS5 does (SQLite transactions).
  If the vector DB write succeeds but FTS5 fails, the index is stale.
- A consistency check or repair tool (`lsm db check`) may be needed.

### 6.3 Provider Abstraction Impact

Adding hybrid retrieval (`hybrid_rrf` profile) challenges the current
`BaseVectorDBProvider` interface:

- `query()` currently returns vector search results only.
- BM25 query results need to be merged at the retrieval layer, not inside the provider.
- **Option A**: Keep BM25 as a separate module (`lsm.retrieval.bm25`) that runs
  parallel to vector DB query; merge with RRF in the new `RetrievalPipeline`.
- **Option B**: Add `fts_query(text: str, top_k: int) → VectorDBQueryResult` to
  `BaseVectorDBProvider`; implement it per-provider (SQLite FTS5 for ChromaDB backend;
  PostgreSQL FTS for PostgreSQL backend).

Option A keeps the provider interface clean but couples the retrieval layer to a specific
BM25 backend. Option B pushes the concern into providers but makes the retrieval
pipeline independent of backend. For PostgreSQL, native FTS is significantly more
powerful than a SQLite sidecar.

### 6.4 Incremental BM25 Index Maintenance

Unlike the vector DB (which supports upsert/delete), the FTS5 index also needs:

- **Insert**: On new chunk, add row to FTS5 table.
- **Delete**: On file re-ingest, delete all rows with that `source_path`.
- **Update**: On soft-delete versioning, mark FTS5 rows with `is_current` (FTS5 does not
  support partial updates — rows must be deleted and re-inserted).

This mirrors exactly what the writer thread already does for the vector DB.

### 6.5 Tests

All new features require tests following the existing patterns:

| Feature | Test Type | Key Coverage |
|---------|-----------|-------------|
| Schema version tracking | Unit | Manifest schema read/write, version detection |
| DB completion | Integration | Changed config → selective re-ingest, skips unchanged files |
| Migration/upgrade | Integration | Schema mismatch detection, upgrade path execution |
| BM25 index | Unit + Integration | Ingest→FTS5 sync, query returns results, sync drift detection |
| RRF fusion | Unit | Score merging, rank preservation, k parameter effect |
| Cross-encoder reranking | Unit + Integration | Score improvement, latency, cache behaviour |
| HyDE | Unit + Integration | Hypothetical generation, embedding aggregation, toggle |
| Heading depth (fixed) | Unit | Depth limit respected, metadata heading path correct |
| Heading depth (intelligent) | Unit | Section size estimation, recursion, metadata path |
| Retrieval profiles | Integration | Each profile end-to-end |
| Eval harness | Integration | Metric computation against known golden set |

Security considerations: The BM25/FTS5 index accepts user-controlled content (chunk
text). SQLite FTS5 queries should use parameterized queries, not string interpolation.
Cross-encoder inputs include user query + chunk text — both need sanitisation against
prompt injection if the model output influences tool execution.

---

## 7. Long-Term Design Principles

The following guiding principles are taken directly from the **Long-Term Direction**
section of `INGEST_FUTURE.md`. They constrain all architectural decisions in v0.8.0 and
should be treated as non-negotiable design requirements rather than aspirational goals.

### 7.1 Retrieval as Composable Pipeline

The retrieval system should be designed as a sequence of composable, replaceable stages
rather than a monolithic function. Each stage (ANN retrieval, BM25 retrieval, RRF
fusion, cross-encoder reranking, diversity filtering, temporal ranking) should be
independently testable and replaceable. The `RetrievalPipeline` abstraction in §4.2.1
is the direct implementation of this principle.

**Implication**: Do not design feature flags that bypass pipeline stages inside existing
functions. Instead, implement each stage as a unit and compose profiles from them.

### 7.2 Evaluation-First Development

No retrieval feature should be merged without a measurable improvement on the eval
harness. The eval harness (§4.2.8) is therefore a prerequisite for all other retrieval
features, not an afterthought. Metrics (Recall@K, MRR, nDCG@K, Diversity@K, Latency)
must be tracked against a saved baseline for every profile change.

**Implication**: Implement the eval harness before implementing hybrid retrieval or
cross-encoder reranking. The implementation order in the dependency map (§8) reflects
this.

### 7.3 Retrieval Traces as First-Class Artifacts

Every retrieval result should carry a complete `retrieval_trace.json` artifact recording:
- Which pipeline stages executed
- Intermediate candidates at each stage
- `ScoreBreakdown` for every returned chunk (§4.2.1)
- HyDE hypothetical documents (if applicable)
- Timing per stage

This trace is essential for debugging, eval comparison, and future Phase 3 graph-augmented
retrieval (where hop-wise citation traces are required). The trace should be written to
the workspace on every agent run that involves retrieval.

### 7.4 Local-First Foundation

All v0.8.0 retrieval features must work without external service calls:
- BM25/FTS5 is local SQLite — no external search API required.
- Cross-encoder runs locally via `sentence-transformers` — no hosted model endpoint needed.
- HyDE uses the configured LLM (which may be local via Ollama/OpenRouter).
- The eval harness runs entirely offline against a local golden set.

If a feature cannot degrade gracefully when the network is unavailable, it should not
be made a default or required component.

### 7.5 Graceful Degradation to Dense-Only Mode

Every advanced feature must have a clean fallback:
- If BM25 index is absent → fall back to `dense_only` retrieval silently (with a log warning).
- If cross-encoder model is missing → skip reranking, return fused results.
- If HyDE LLM call fails → fall back to direct query embedding.
- If temporal boost config is absent → skip boost stage.

This ensures that existing deployments are not broken when v0.8.0 config keys are
absent (they default to the safe, dense-only behavior).

---

## 8. Dependency Map

```
DB Versioning (§2)
  ├── Required by: DB Completion (§3)         ← can't do completion without version tracking
  ├── Required by: Heading Depth changes (§5) ← heading changes invalidate existing chunks
  └── Enables: Eval harness comparisons (§4.2.8)

Manifest Schema (§3.5)
  └── Required by: DB Completion triggers

RetrievalPipeline abstraction (§4.2.1)
  ├── Required by: Retrieval Profiles (§4.2.2)
  ├── Required by: Hybrid RRF (§4.2.3)
  ├── Required by: Cross-encoder (§4.2.4)
  ├── Required by: HyDE (§4.2.5)
  ├── Required by: Diversity & De-dup (§4.2.6)
  └── Required by: Retrieval trace for eval

BM25 Index (§4.2.3)
  └── Required by: Hybrid RRF

Eval Harness (§4.2.8)
  └── Needed before: Any retrieval profile can be validated
```

**Critical path for v0.8.0**: If retrieval improvements must be validated:
DB Versioning → Manifest Schema → BM25 Index → RetrievalPipeline → Hybrid RRF → Eval

---

## 9. Clarification Questions

The following decisions must be made by the user before implementation planning begins.
They are grouped by topic and ordered by architectural importance.

---

### Q-DB: DB Versioning and Migration

**Q-DB-1**: When the system detects a schema mismatch (e.g., config `embed_model` changed
vs. manifest `_schema.embedding_model`), what should happen?
- **(a)** Raise an error and require explicit user action to resolve.
- **(b)** Log a warning and ingest new files only (proceed with mixed-generation corpus).
- **(c)** Automatically trigger a full re-ingest.
- **(d)** Automatically trigger selective re-ingest of only mismatched files.
**User Response:** Option A

**Q-DB-2**: Should there be a `lsm migrate` / `lsm db upgrade` CLI command as the primary
entry point for schema migrations, or should migration happen automatically at ingest time?
**User Response:** There should be a migrate/ugrade command for the cli/tui ingest command. 

**Q-DB-3**: The current migration tool only moves data from ChromaDB to PostgreSQL. Is
bidirectional migration (PostgreSQL → ChromaDB) needed, or only one direction?
**User Response:** Bidirectional is needed.

**Q-DB-4**: Should old-version chunks be permanently deleted when a file is re-embedded,
or retained with `is_current=False` (the soft-delete approach in `enable_versioning=True`)?
Soft-delete grows the DB but allows "time-travel" queries; hard-delete keeps the DB lean.
**User Response:** No they should not be permanetly deleted, but instead an intelegent clean up tool should be created to prune soft-deltes.
---

### Q-HYBRID: Hybrid Retrieval Architecture

**Q-HYBRID-1**: For the BM25/FTS5 sparse index, which backend do you prefer?
- **(a)** SQLite FTS5 sidecar (works for both ChromaDB and PostgreSQL users; single
  additional file in `persist_dir`).
- **(b)** PostgreSQL native FTS (only for PostgreSQL users; BM25-equivalent via `ts_rank`;
  requires adding a `tsvector` column to the existing schema).
- **(c)** Both, selected by the active vectordb provider.
**User Response:** Option C

**Q-HYBRID-2**: Should hybrid retrieval (BM25 + dense + RRF) be the *new default*
retrieval mode, or should it remain opt-in (`retrieval_profile: "hybrid_rrf"`) while
`dense_only` stays the default?
**User Response:** What are the pros and cons of either option?

**Q-HYBRID-3**: For the `RetrievalPipeline` abstraction, should retrieval stages be
registered by name in a registry (like `ToolRegistry`) to allow config-driven composition,
or are a small number of hard-coded profiles sufficient?
**User Response:** See answers earlier.

---

### Q-RERANK: Cross-Encoder Reranking

**Q-RERANK-1**: Cross-encoder inference adds latency (~2-10s CPU for 100 candidates with
`ms-marco-MiniLM-L-6-v2`). Is this acceptable for interactive queries in the TUI, or
should cross-encoding be restricted to a non-interactive "deep search" mode?
**User Response:** Can this be ran via CUDA/GPU?

**Q-RERANK-2**: Should the cross-encoder model be downloaded at install time (add it
to `pyproject.toml` as a required asset) or downloaded lazily on first use of the
`dense_cross_rerank` profile?
**User Response:** Lazily on first use

**Q-RERANK-3**: Should the reranker cache be:
- **(a)** In-memory only (cleared between sessions).
- **(b)** Persisted to SQLite in `global_folder`.
- **(c)** Not implemented in v0.8.0.
**User Response:** Option B

---

### Q-HYDE: Hypothetical Document Embeddings

**Q-HYDE-1**: HyDE requires an LLM call before retrieval (adds latency + API cost for
hosted models). Should HyDE be:
- **(a)** A retrieval profile (`hyde_hybrid`) that users must explicitly opt into.
- **(b)** An optional flag per query (`--hyde`).
- **(c)** Out of scope for v0.8.0 (deferred to v0.9.0).
**User Response:** Option A

**Q-HYDE-2**: Should the hypothetical documents generated by HyDE be logged and visible
in the TUI debug view or only in `retrieval_trace.json`?
**User Response:** In `retrieval_trace.json` with the ability to view them in TUI

---

### Q-EVAL: Evaluation Harness

**Q-EVAL-1**: The eval harness requires a golden query set with known-good answers.
Where should this come from?
- **(a)** Manually curated by the user on their own corpus.
- **(b)** Synthetically generated from corpus headings/structure (weak supervision).
- **(c)** A bundled synthetic dataset for testing purposes only.
**User Response:** Option C

**Q-EVAL-2**: Should the eval harness be exposed as a TUI command, a CLI command, or
both?
**User Response:** CLI command

**Q-EVAL-3**: Is the BEIR benchmark format (standard IR evaluation format) of interest
for compatibility with external tooling, or should the harness use a simpler internal
format?
**User Response:** BEIR benchmark format

**Q-EVAL-4**: The eval harness should support regression comparison against a saved
baseline (e.g., compare `hybrid_rrf` vs. `dense_only` on the same query set). How should
baselines be stored and selected?
- **(a)** A single saved baseline file per corpus (overwritten on each `lsm eval save-baseline`).
- **(b)** Named baselines (e.g., `lsm eval save-baseline --name dense_only_v0.8`) stored
  as a directory of profiles.
- **(c)** Automatic — always compare against the last run result.
**User Response:** What are the pros and cons of each option?
---

### Q-CHUNK: Heading Depth and Intelligent Chunking

**Q-CHUNK-1**: The TODO describes two heading improvements: (a) *configurable*
`max_heading_depth` and (b) *intelligent* depth selection based on section size. Should
both be implemented in v0.8.0, or should only one be prioritized?
**User Response:** Both

**Q-CHUNK-2**: For the intelligent heading algorithm, what should the hierarchy separator
be in the `heading` metadata field? E.g., `"::"`  →  `"Introduction::Background::Prior Work"`.
Or should separate `heading_path: list[str]` metadata be added?
**User Response:** Separate heading_path metadata

**Q-CHUNK-3**: Should `max_heading_depth` be a global config (same for all roots) or
per-root (different roots might need different depths — e.g., code docs vs. personal
notes)?
**User Response:** Set global and allow overrides per-root

**Q-CHUNK-4**: When intelligent heading depth changes the chunk boundaries (changing
`chunk_id` values), should existing chunks in the DB for that file be treated as stale
and re-ingested on the next run, or should this only apply after a manual `lsm db
upgrade` command?
**User Response:** Treat as stale

---

### Q-DIVERSITY: Diversity and De-duplication

**Q-DIVERSITY-1**: The current de-duplication uses exact hash matching. INGEST_FUTURE.md
calls for SimHash or MinHash near-duplicate detection. Which approach is preferred?
- **(a)** SimHash (fast, single-hash similarity; good for short chunks ≤ 512 tokens).
- **(b)** MinHash + LSH (more accurate; standard for near-duplicate detection at scale).
- **(c)** Keep exact hash for v0.8.0; add near-dup detection in v0.9.0.
**User Response:** MinHash

**Q-DIVERSITY-2**: The per-section diversity cap uses the chunk's heading path
(e.g., `"Introduction::Background"`) as the grouping key. `max_per_section` sets the
maximum chunks returned from any single heading path prefix. What is the right default
for `max_per_section`?
- **(a)** None (no per-section cap; backward-compatible default).
- **(b)** 3 (opinionated default matching `max_per_file`).
- **(c)** Same as `max_per_file` (unified parameter).
**User Response:** What is the benefit of adding a `max_per_section` cap.

**Q-DIVERSITY-3**: The greedy MMR (Maximal Marginal Relevance) diversity selection is an
alternative to hard per-section caps. It selects the next result that maximizes a blend
of relevance and dissimilarity from already-selected results. Should MMR be:
- **(a)** An optional diversity strategy (`diversity_strategy: "mmr"`).
- **(b)** The default strategy when `diversity_strategy` is not "exact".
- **(c)** Out of scope for v0.8.0.
**User Response:** option b

---

### Q-SCOPE: v0.8.0 Scope Boundaries

**Q-SCOPE-1**: The INGEST_FUTURE.md has three phases. For v0.8.0, is the target:
- **(a)** All of Phase 1 (hybrid retrieval, cross-encoder, HyDE, diversity, temporal
  ranking, eval harness).
- **(b)** Phase 1 core only: hybrid retrieval + eval harness (cross-encoder and HyDE
  are v0.9.0).
- **(c)** DB versioning + heading improvements only (retrieval overhaul is v0.9.0).
- **(d)** Custom scope — specify which features.
**User Response:** Everything

**Q-SCOPE-2**: Should v0.8.0 also begin the sqlite-backed manifest (replacing the flat
JSON) as part of the DB versioning work, or defer that to v0.9.0's "persist ingest state
to DB" TODO item?
**User Response:** Include in v0.8.0

---

*End of research document. Implementation planning begins after Q-DB, Q-SCOPE, and the
most architecturally-impactful subset of questions above are resolved.*

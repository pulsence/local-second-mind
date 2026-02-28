# Phase 10: Retrieval Profiles and Hybrid RRF

**Status**: Pending

Implements the five retrieval profiles and the hybrid retrieval system combining dense
vector search with BM25/FTS5 sparse search via Reciprocal Rank Fusion (RRF).
Phase 9 (Evaluation Harness) must be complete first — capture `dense_only` baseline
before enabling `hybrid_rrf` as default and before finalizing RRF weight tuning.

Reference: [RESEARCH_PLAN.md §5.5, §5.6](../RESEARCH_PLAN.md#55-retrieval-profiles)

---

## 10.1: FTS5 Query Support

**Description**: Add `fts_query()` to the vector DB provider interface for BM25 full-text
search.

**Tasks**:
- Add `fts_query(text: str, top_k: int) -> VectorDBQueryResult` to `BaseVectorDBProvider`
  as an abstract method (default no-op for providers that don't support it)
- Implement in `SQLiteVecProvider`:
  - Query `chunks_fts` FTS5 virtual table
  - Return results with BM25 rank scores
  - Apply `is_current = 1` filter
- Ensure FTS5 queries use parameterized input (no injection risk)

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/vectordb/base.py` — `fts_query()` method
- `lsm/vectordb/sqlite_vec.py` — FTS5 implementation
- `tests/test_vectordb/test_sqlite_vec.py`:
  - Test: FTS5 query returns relevant results
  - Test: FTS5 ranking matches BM25 expectations
  - Test: FTS5 filters to is_current=1 only
  - Test: special characters in query are handled safely

**Success criteria**: `fts_query()` returns BM25-ranked results from FTS5. No SQL injection.

---

## 10.2: RRF Fusion and Retrieval Profiles

**Description**: Implement the RRF fusion logic and the five retrieval profiles.

**Tasks**:
- Capture `dense_only` baseline with eval harness (Phase 9) before enabling `hybrid_rrf`
  as default
- Record before/after metrics when changing `rrf_dense_weight`/`rrf_sparse_weight`
- Create `lsm/query/stages/rrf_fusion.py`:
  - `rrf_fuse(dense_results, sparse_results, dense_weight, sparse_weight, k=60)` →
    merged candidate list with `ScoreBreakdown` populated
  - `dense_rank`, `sparse_rank`, `fused_score` set on each candidate
- Create `lsm/query/stages/dense_recall.py`:
  - `dense_recall(query_embedding, db, top_k, filters)` → candidates with `dense_score`
- Create `lsm/query/stages/sparse_recall.py`:
  - `sparse_recall(query_text, db, top_k)` → candidates with `sparse_score`
- Define `VALID_PROFILES` constant tuple: `("dense_only", "hybrid_rrf", "hyde_hybrid",
  "dense_cross_rerank", "llm_rerank")`. Used by AgentHarness validation (Phase 13).
- Update `RetrievalPipeline.build_sources()` to route through profiles:
  - `dense_only` — dense recall only
  - `hybrid_rrf` — dense + sparse + RRF fusion (DEFAULT)
  - `hyde_hybrid` — HyDE embedding + hybrid_rrf (Phase 11)
  - `dense_cross_rerank` — dense + cross-encoder reranking (Phase 11)
  - `llm_rerank` — dense + LLM reranking (Phase 11)
- Add profile config to `QueryConfig`:
  - `retrieval_profile: str = "hybrid_rrf"`
  - `k_dense: int = 100`
  - `k_sparse: int = 100`
  - `rrf_dense_weight: float = 0.7`
  - `rrf_sparse_weight: float = 0.3`
- Remove legacy config fields from `QueryConfig` that are superseded by the profile system:
  - Remove `rerank_strategy` (replaced by retrieval profiles)
  - Remove `no_rerank` (replaced by `dense_only` profile)
  - Remove `local_pool` (replaced by `k_dense`)
  - Remove `max_per_file` (replaced by `max_per_section` in Phase 12)
- Implement graceful degradation:
  - `hybrid_rrf` without FTS5 → `dense_only` + log warning

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/rrf_fusion.py` — RRF merge
- `lsm/query/stages/dense_recall.py` — dense retrieval stage
- `lsm/query/stages/sparse_recall.py` — sparse retrieval stage
- `lsm/query/pipeline.py` — profile routing in `build_sources()`, `VALID_PROFILES`
- `lsm/config/models/query.py` — profile config fields, removed legacy fields
- `tests/test_query/test_rrf_fusion.py`:
  - Test: RRF produces correct fused scores
  - Test: candidates appearing in both lists get higher scores
  - Test: weight parameters affect scores correctly
  - Test: k parameter affects ranking correctly
- `tests/test_query/test_profiles.py`:
  - Test: each profile produces results
  - Test: graceful degradation (hybrid_rrf → dense_only)

**Success criteria**: `hybrid_rrf` fuses dense + sparse results. RRF scores are correct.
Profiles route to correct stages. Graceful degradation works. Legacy `rerank_strategy`,
`no_rerank`, `local_pool`, `max_per_file` removed. Metric deltas vs `dense_only` baseline
are captured via the eval harness.

---

## 10.3: Phase 10 Code Review and Changelog

**Tasks**:
- Review RRF formula implementation matches paper specification
- Review FTS5 query safety (parameterized queries)
- Review graceful degradation paths
- Review ScoreBreakdown population across all profiles
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CONFIGURATION.md` — document retrieval profiles
- Update `.agents/docs/architecture/development/QUERY.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`
- `.agents/docs/architecture/development/QUERY.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---

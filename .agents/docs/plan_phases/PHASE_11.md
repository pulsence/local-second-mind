# Phase 11: Cross-Encoder, LLM Reranking, and HyDE

**Status**: Pending

Implements the remaining three retrieval profiles: `dense_cross_rerank`, `llm_rerank`,
and `hyde_hybrid`.

Reference: [RESEARCH_PLAN.md §5.7, §5.8](../RESEARCH_PLAN.md#57-cross-encoder-reranking)

---

## 11.1: Cross-Encoder Reranking

**Description**: Implement the `dense_cross_rerank` profile using a local cross-encoder
model from `sentence-transformers`.

**Tasks**:
- Add `sentence-transformers` as an optional dependency in `pyproject.toml`
- Create `lsm/query/stages/cross_encoder.py`:
  - `CrossEncoderReranker.__init__(model_name, device, cache_conn)`:
    - Lazy model download on first use
    - Honour `device` from `GlobalConfig`
  - `rerank(query, candidates, top_k)` → reranked candidates with `rerank_score`
  - Cache integration: check `lsm_reranker_cache` before computing, store after
  - Cache key: `hash(query, chunk_id, model_version)`
- Update `RetrievalPipeline.build_sources()`:
  - `dense_cross_rerank` profile: dense recall → cross-encoder reranking
  - Graceful degradation: if model not available → use dense results + log warning
- Add config fields to `QueryConfig`:
  - `cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- Add CLI command: `lsm cache clear [--reranker] [--query]` for manual invalidation
  of the `lsm_reranker_cache` table and/or query result cache (§5.7)
- Run eval: `lsm eval retrieval --profile dense_cross_rerank --compare dense_only`

- Commit and push changes for this sub-phase.
**Files**:
- `pyproject.toml` — `sentence-transformers` optional dependency
- `lsm/query/stages/cross_encoder.py` — cross-encoder stage
- `lsm/query/pipeline.py` — profile integration
- `lsm/config/models/query.py` — cross-encoder config
- `tests/test_query/test_cross_encoder.py`:
  - Test: reranking changes candidate order
  - Test: cache hit avoids model inference
  - Test: GPU device parameter is honoured
  - Test: graceful degradation without model

**Success criteria**: Cross-encoder reranking improves retrieval quality (measured via eval
harness). Cache reduces repeated computation. Graceful degradation works.

---

## 11.2: LLM Reranking Profile

**Description**: Wire the LLM rerank logic (migrated to `llm_rerank.py` in Phase 6) into
the `llm_rerank` retrieval profile.

**Tasks**:
- Update `RetrievalPipeline.build_sources()`:
  - `llm_rerank` profile: dense recall → LLM reranking (via `llm_rerank()` from
    `lsm/query/stages/llm_rerank.py`)
- Ensure the LLM rerank stage uses `provider.send_message()` (not the removed
  `provider.rerank()`)
- Remove legacy `rerank_strategy: "llm"` code paths from `rerank.py` (if any remain)

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/pipeline.py` — llm_rerank profile routing
- `lsm/query/stages/llm_rerank.py` — verify integration
- `lsm/query/rerank.py` — remove legacy code
- `tests/test_query/test_llm_rerank.py`:
  - Test: llm_rerank profile uses send_message()
  - Test: rerank response parsing works correctly

**Success criteria**: `llm_rerank` profile works via the pipeline. No legacy rerank code
remains.

---

## 11.3: HyDE (Hypothetical Document Embeddings)

**Description**: Implement the `hyde_hybrid` profile. LLM generates hypothetical answers,
which are embedded and used for retrieval instead of the raw query.

**Tasks**:
- Create `lsm/query/stages/hyde.py`:
  - `generate_hypothetical_docs(query, provider, num_samples, temperature)` →
    list of hypothetical document strings
  - `pool_embeddings(embeddings, strategy)` → single embedding
    (strategy: `"mean"` or `"max"`)
  - `hyde_recall(query, provider, embedder, db, config)` → candidates
    using pooled hypothetical embedding
- Update `RetrievalPipeline.build_sources()`:
  - `hyde_hybrid` profile: HyDE embedding → hybrid_rrf pipeline
  - Graceful degradation: LLM failure → direct query embedding + log warning
- Add config fields to `QueryConfig`:
  - `hyde_num_samples: int = 2`
  - `hyde_temperature: float = 0.2`
  - `hyde_pooling: str = "mean"`
- Log hypothetical documents in `ContextPackage.retrieval_trace`
- Run eval: `lsm eval retrieval --profile hyde_hybrid --compare hybrid_rrf`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/hyde.py` — HyDE stage
- `lsm/query/pipeline.py` — profile integration
- `lsm/config/models/query.py` — HyDE config
- `tests/test_query/test_hyde.py`:
  - Test: hypothetical document generation
  - Test: embedding pooling (mean and max)
  - Test: HyDE recall returns candidates
  - Test: graceful degradation on LLM failure
  - Test: hypothetical docs appear in retrieval trace

**Success criteria**: HyDE generates hypothetical documents, pools embeddings, and uses them
for retrieval. Trace contains hypothetical documents. Graceful degradation works.

---

## 11.4: Phase 11 Code Review and Changelog

**Tasks**:
- Review cross-encoder cache key uniqueness
- Review HyDE fallback path
- Review eval results for all three new profiles
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CONFIGURATION.md` — document cross-encoder and HyDE config

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`

**Success criteria**: `pytest tests/ -v` passes. All five profiles functional. Changelog
and docs updated.

---

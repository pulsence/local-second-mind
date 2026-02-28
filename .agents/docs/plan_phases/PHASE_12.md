# Phase 12: Diversity, Dedup, and Temporal Ranking

**Status**: Pending

Adds MinHash near-duplicate detection, MMR diversity selection, per-section caps, and
temporal-aware ranking.

Reference: [RESEARCH_PLAN.md §5.9, §5.10](../RESEARCH_PLAN.md#59-diversity-and-de-duplication)

---

## 12.1: MinHash Near-Duplicate Detection

**Description**: Replace exact hash dedup with MinHash-based near-duplicate detection.

**Tasks**:
- Create `lsm/query/stages/dedup.py`:
  - `compute_minhash(text, num_perm=128)` → integer signature
  - `are_near_duplicates(sig_a, sig_b, threshold=0.8)` → bool
  - `deduplicate_candidates(candidates, threshold)` → deduplicated list
- Update ingest pipeline to compute and store `simhash` for each chunk:
  - Store in `lsm_chunks.simhash` column (already in schema from Phase 1)
- Update `RetrievalPipeline.build_sources()`:
  - After retrieval, before final selection, run MinHash dedup
- Add config: `dedup_threshold: float = 0.8` to `QueryConfig`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/dedup.py` — MinHash dedup
- `lsm/ingest/pipeline.py` — simhash computation at ingest time
- `lsm/query/pipeline.py` — dedup integration
- `lsm/config/models/query.py` — dedup config
- `tests/test_query/test_dedup.py`:
  - Test: near-duplicate pairs are detected
  - Test: distinct chunks are preserved
  - Test: threshold parameter works correctly

**Success criteria**: Near-duplicate chunks are suppressed. Distinct chunks with similar
content are preserved. Simhash is computed at ingest time.

---

## 12.2: MMR Diversity Selection

**Description**: Implement Maximal Marginal Relevance (MMR) as the default post-rerank
selection strategy.

**Tasks**:
- Create `lsm/query/stages/diversity.py`:
  - `mmr_select(candidates, lambda_param, k)` → selected candidates
    - MMR formula: `λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))`
    - Requires embedding vectors on candidates
  - `per_section_cap(candidates, max_per_section, heading_depth)` → capped candidates
    - Groups by `heading_path` prefix at `heading_depth`
    - Caps chunks per section group
- Update `RetrievalPipeline.build_sources()`:
  - After reranking, apply MMR selection
  - After MMR, apply per-section cap
- Add config fields to `QueryConfig`:
  - `mmr_lambda: float = 0.7`
  - `max_per_section: Optional[int] = 3`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/diversity.py` — MMR and section cap
- `lsm/query/pipeline.py` — diversity integration
- `lsm/config/models/query.py` — diversity config
- `tests/test_query/test_diversity.py`:
  - Test: MMR selects diverse candidates
  - Test: per-section cap limits chunks per heading group
  - Test: lambda parameter affects diversity-relevance trade-off

**Success criteria**: MMR selection reduces redundancy in results. Per-section cap prevents
any single section from dominating. Both are configurable.

---

## 12.3: Temporal-Aware Ranking

**Description**: Add recency boost and time-range filtering using `mtime_ns` metadata.

**Tasks**:
- Create `lsm/query/stages/temporal.py`:
  - `apply_temporal_boost(candidates, boost_days, boost_factor)` → boosted candidates
    - Candidates with `mtime_ns` within `boost_days` get `boost_factor` multiplier
    - Boost stored in `ScoreBreakdown.temporal_boost`
  - `filter_time_range(candidates, start_ns, end_ns)` → filtered candidates
- Update `RetrievalPipeline.build_sources()`:
  - Apply temporal boost after scoring, before final selection
- Add config fields to `QueryConfig`:
  - `temporal_boost_enabled: bool = False`
  - `temporal_boost_days: int = 30`
  - `temporal_boost_factor: float = 1.5`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/temporal.py` — temporal boost and filter
- `lsm/query/pipeline.py` — temporal integration
- `lsm/config/models/query.py` — temporal config
- `tests/test_query/test_temporal.py`:
  - Test: recent files get boost
  - Test: old files do not get boost
  - Test: time-range filter works
  - Test: boost factor is configurable

**Success criteria**: Recent documents get a configurable relevance boost. Time-range
filtering narrows results to a specific period.

---

## 12.4: Phase 12 Code Review and Changelog

**Tasks**:
- Review MMR implementation for numerical stability
- Review MinHash performance at scale (100k chunks)
- Review temporal boost interaction with other scores
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CONFIGURATION.md` — document diversity and temporal config

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---

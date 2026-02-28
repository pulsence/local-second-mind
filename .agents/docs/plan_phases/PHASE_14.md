# Phase 14: Phase 2 — Multi-Vector and Clustering

**Status**: Pending

Implements Phase 2 features from INGEST_FUTURE.md: multi-vector representation at
multiple granularities and cluster-aware retrieval.

Reference: [RESEARCH_PLAN.md §4.4, §5.14, §5.15](../RESEARCH_PLAN.md#44-phase-2-multi-vector-representation-ingest-side)

---

## 14.1: Multi-Vector Representation (Ingest-Side)

**Description**: Store embeddings at chunk, section, and file granularities using the
`node_type` field.

**Tasks**:
- Add config fields to `IngestConfig`:
  - `enable_section_summaries: bool = False`
  - `enable_file_summaries: bool = False`
- Update ingest pipeline:
  - When `enable_section_summaries` is True:
    - For each heading section: LLM summarize → embed → store with
      `node_type = "section_summary"` and heading's `heading_path`
  - When `enable_file_summaries` is True:
    - For each file: LLM summarize (first N paragraphs + headings outline) → embed →
      store with `node_type = "file_summary"`
- All granularities stored in the same `vec_chunks` / `lsm_chunks` tables
- Summary generation requires an LLM call per section/file — clearly document cost

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/config/models/ingest.py` — summary config fields
- `lsm/ingest/pipeline.py` — summary generation and storage
- `lsm/ingest/summarization.py` — new module for summary generation
- `tests/test_ingest/test_multi_vector.py`:
  - Test: section summaries are generated and stored
  - Test: file summaries are generated and stored
  - Test: node_type is correctly set for each granularity
  - Test: summaries are embeddings alongside chunks

**Success criteria**: Section and file summaries are generated, embedded, and stored
alongside chunks. node_type distinguishes granularities.

---

## 14.2: Cluster-Aware Retrieval

**Description**: Implement cluster assignment and cluster-filtered retrieval.

**Tasks**:
- Create `lsm/db/clustering.py`:
  - `build_clusters(conn, algorithm, k)`:
    - Read all embeddings from `vec_chunks`
    - Run k-means (default) or HDBSCAN
    - Write `cluster_id` and `cluster_size` to `lsm_chunks`
    - Write centroids to `lsm_cluster_centroids`
  - `get_top_clusters(query_embedding, conn, top_n)` → list of cluster IDs
- Add `umap-learn` and `hdbscan` as optional dependencies in `pyproject.toml`
- Add CLI commands:
  - `lsm cluster build [--algorithm kmeans|hdbscan] [--k 50]`
  - `lsm cluster visualize` — UMAP HTML plot export
- Update `RetrievalPipeline.build_sources()`:
  - When `cluster_enabled=True`: first match query to top-N clusters, then KNN within
    those clusters only
- Add config fields to `QueryConfig`:
  - `cluster_enabled: bool = False`
  - `cluster_algorithm: str = "kmeans"`
  - `cluster_k: int = 50`
  - `cluster_top_n: int = 5`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/clustering.py` — clustering logic
- `lsm/query/pipeline.py` — cluster-filtered retrieval
- `lsm/config/models/query.py` — cluster config
- CLI entry point — cluster commands
- `tests/test_query/test_clustering.py`:
  - Test: cluster build assigns cluster IDs to all chunks
  - Test: cluster-filtered retrieval returns results from top clusters
  - Test: cluster-filtered retrieval is faster than full scan (for large corpus)

**Success criteria**: `lsm cluster build` assigns clusters. Cluster-filtered retrieval
reduces search space while maintaining quality.

---

## 14.3: Multi-Vector Retrieval (Query-Side)

**Description**: Retrieve at multiple granularities and fuse results.

**Tasks**:
- Create `lsm/query/stages/multi_vector.py`:
  - `multi_vector_recall(query_embedding, db, config)`:
    - KNN at chunk level (`node_type = "chunk"`)
    - Optionally KNN at section level (`node_type = "section_summary"`)
    - Optionally KNN at file level (`node_type = "file_summary"`)
    - RRF-fuse across granularities
    - For file/section matches without a chunk match: expand to top-k chunks from
      that file/section
- Update `RetrievalPipeline.build_sources()` to use multi-vector recall when summaries
  are available in the database

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/multi_vector.py` — multi-granularity retrieval
- `lsm/query/pipeline.py` — integration
- `tests/test_query/test_multi_vector.py`:
  - Test: multi-vector retrieval finds results at all granularities
  - Test: RRF fusion across granularities
  - Test: chunk expansion from file/section matches

**Success criteria**: Multi-vector retrieval combines results from chunk, section, and file
levels. Expansion fills in missing chunk-level details.

---

## 14.4: Phase 14 Code Review and Changelog

**Tasks**:
- Review summary generation cost and performance
- Review clustering quality and retrieval impact via eval harness
- Review multi-vector fusion scoring
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CONFIGURATION.md` — document multi-vector and clustering config

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---

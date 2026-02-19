
## Phase 1: Retrieval Modernization

### Goals

Move LSM beyond naive cosine kNN retrieval into a hybrid, reranked,
diversity-aware system with measurable evaluation.

------------------------------------------------------------------------

### Architecture

-   Introduce a unified `RetrievalPipeline` abstraction:
    -   Standard stages: `recall → fuse → rerank → diversify → return`
    -   Standard objects: `Query`, `Candidate`, `ScoreBreakdown`,
        `Citation`
    -   Each query emits `retrieval_trace.json` for debugging
-   Add `lsm.eval.retrieval` evaluation harness:
    -   Frozen query test set
    -   Metrics:
        -   Recall@K
        -   MRR
        -   nDCG@K
        -   Diversity@K
        -   Latency
    -   CLI:
        -   `lsm eval retrieval --profile hybrid_rrf`
    -   Output regression comparison vs baseline
-   Add retrieval profiles in config:

``` yaml
retrieval_profiles:
  - dense_only
  - hybrid_rrf
  - hyde_hybrid
  - dense_cross_rerank
```

------------------------------------------------------------------------

### Hybrid Retrieval (Dense + Sparse + Fusion)

-   Add sparse index sidecar:
    -   Local BM25 engine (SQLite FTS5 or equivalent)
    -   Indexed fields:
        -   chunk_text
        -   headings
        -   file_name
        -   metadata text fields
-   Implement dual recall:
    -   Dense ANN (Chroma) → `K_dense`
    -   Sparse BM25 → `K_sparse`
-   Implement Reciprocal Rank Fusion (RRF):
    -   Configurable `rrf_k`
    -   Configurable per-channel weights
    -   Persist per-candidate breakdown:
        -   dense_rank
        -   sparse_rank
        -   fused_score

**Acceptance Criteria** - Hybrid recall improves Recall@20 vs dense-only
on evaluation set. - Retrieval trace shows full scoring breakdown.

------------------------------------------------------------------------

### Cross-Encoder Reranking

-   Add `lsm.retrieval.rerank` module:
    -   Cross-encoder model support
    -   Batch scoring
    -   CPU/GPU auto selection
-   Default rerank strategy:
    -   Recall top 100
    -   Cross-encoder rerank top 20
-   Optional LLM rerank mode:
    -   Disabled by default
    -   Used only for justification or deep reasoning profile
-   Add reranker cache:
    -   Key: `(query_hash, chunk_id, model_version)`
    -   Invalidate on corpus rebuild

**Acceptance Criteria** - Reranked pipeline improves nDCG@10 vs
hybrid-only. - Latency remains under configurable threshold.

------------------------------------------------------------------------

### HyDE (Hypothetical Document Embeddings)

-   Add HyDE query expansion stage:

    -   Generate 1--3 hypothetical answers
    -   Embed generated text
    -   Aggregate embeddings (mean or max pooling)

-   Support HyDE + Hybrid profile

-   Add configuration:

``` yaml
hyde:
  enabled: true
  num_samples: 2
  temperature: 0.2
```

-   Log generated hypothetical text in retrieval trace.

**Acceptance Criteria** - Abstract queries show improved Recall@20. -
HyDE can be toggled per-profile.

------------------------------------------------------------------------

### Diversity & De-duplication

-   Formalize per-file diversity cap:
    -   `max_chunks_per_file`
-   Optional per-section cap:
    -   Based on heading path metadata
-   Implement near-duplicate detection:
    -   SimHash or MinHash optional
-   Add greedy diversity selection post-rerank

**Acceptance Criteria** - No more than configured cap per file. -
Diversity@K metric improves without major drop in precision.

------------------------------------------------------------------------

### Temporal-Aware Ranking

You ingest mtime metadata already. You should expose:
- Boost recent
- Filter by time range
- “What was I thinking in 2023?”

------------------------------------------------------------------------

## Phase 2: Retrieval Structuring & Representation

### Cluster-Aware Retrieval

-   Offline clustering job:
    -   k-means or HDBSCAN
    -   Persist `cluster_id` and `cluster_size`
-   Add cluster-aware sampling:
    -   Prevent topic collapse
    -   Downweight overly dense clusters
-   Add diagnostic export:
    -   Cluster summaries
    -   Optional UMAP export for visualization

**Acceptance Criteria** - Reduced redundancy in top-K. - Improved
topical spread on broad queries.

------------------------------------------------------------------------

### Multi-Vector Representation

-   Generate file-level summary embeddings.

-   Generate section-level summary embeddings.

-   Store summary nodes in metadata layer.

-   Two-stage recall option:

    1.  Retrieve files/sections via summary embeddings
    2.  Retrieve chunks within selected scope

**Acceptance Criteria** - Long-form conceptual queries show improved
precision. - Reduced irrelevant chunk retrieval from large files.

------------------------------------------------------------------------

## Phase 3: Structural & SOTA Retrieval

### Graph-Augmented Retrieval

-   Add lightweight local graph store:
    -   Nodes:
        -   File
        -   Section
        -   Chunk
        -   Entity (optional)
    -   Edges:
        -   contains
        -   references
        -   same_author
        -   thematic_link
-   Retrieval pipeline:
    1.  Hybrid recall
    2.  Graph expansion (bounded depth)
    3.  Cross-encoder rerank
-   Add heading tree ingestion as graph edges.

**Acceptance Criteria** - Multi-hop queries retrieve relevant linked
documents. - Graph expansion trace visible in debug output.

------------------------------------------------------------------------

### Domain-Fine-Tuned Embeddings

-   Build training dataset:

    -   Positive pairs from headings, backlinks, citations
    -   Hard negatives from near-miss retrieval

-   Fine-tune contrastive embedding model offline.

-   Add model registry:

    -   Version embeddings
    -   Trigger re-embedding via CLI

**Acceptance Criteria** - Fine-tuned model outperforms base model on
evaluation harness. - Embedding model version tracked in metadata.

------------------------------------------------------------------------

### Multi-Hop Retrieval Agents

-   Implement iterative retrieval loop:
    -   Retrieve → summarize → decompose → retrieve → synthesize
-   Add query decomposition strategies:
    -   Key term expansion
    -   Named entity extraction
    -   Conceptual prerequisite discovery
-   Add strict budget controls:
    -   max_hops
    -   max_tokens
    -   max_latency
-   Persist hop-wise citation trace.

**Acceptance Criteria** - Multi-hop profile improves performance on
composite queries. - Agent retrieval trace auditable and reproducible.

------------------------------------------------------------------------

## Long-Term Direction

-   Retrieval as composable pipeline.
-   Evaluation-first development.
-   Retrieval traces as first-class artifacts.
-   Maintain strict local-first foundation.
-   Ensure all advanced features degrade gracefully to dense-only mode.

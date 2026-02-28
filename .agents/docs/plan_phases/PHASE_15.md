# Phase 15: Phase 3 — Graph, Multi-Hop, and Fine-Tuning

**Status**: Pending

Implements Phase 3 features: graph construction at ingest time, graph-augmented retrieval,
multi-hop retrieval, and domain-fine-tuned embeddings.

Reference: [RESEARCH_PLAN.md §4.5, §4.6, §5.16, §5.17](../RESEARCH_PLAN.md#45-phase-3-graph-construction-at-ingest-time)

---

## 15.1: Graph Construction at Ingest Time

**Description**: Populate `lsm_graph_nodes` and `lsm_graph_edges` during ingest.

**Tasks**:
- Add graph methods to `BaseVectorDBProvider`:
  - `graph_insert_nodes(nodes: List[GraphNode]) -> None`
  - `graph_insert_edges(edges: List[GraphEdge]) -> None`
  - `graph_traverse(start_ids: List[str], max_hops: int) -> List[str]`
- Implement in `SQLiteVecProvider`
- Create `lsm/ingest/graph_builder.py`:
  - **From FileGraph**: Convert heading hierarchy to graph nodes/edges
    (`"contains"` edge type)
  - **From file references**: Parse `[[wikilinks]]`, `[text](path)` internal links,
    citation DOIs → `"references"` edges
  - **From entity extraction** (optional): NER-based entity nodes and `"mentioned_in"` edges.
    Requires `spacy` — add as optional dependency in `pyproject.toml`
- Update ingest pipeline to call graph builder after chunking
- Add CLI command: `lsm graph build-links` — builds thematic links offline using
  cosine similarity above threshold between chunk embeddings

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/vectordb/base.py` — graph methods
- `lsm/vectordb/sqlite_vec.py` — SQLite graph implementation
- `lsm/ingest/graph_builder.py` — graph construction
- `lsm/ingest/pipeline.py` — graph builder integration
- CLI entry point — graph commands
- `tests/test_ingest/test_graph_builder.py`:
  - Test: heading hierarchy produces correct nodes and edges
  - Test: wikilinks produce reference edges
  - Test: entity extraction produces entity nodes
- `tests/test_vectordb/test_graph.py`:
  - Test: graph insert/traverse works
  - Test: CTE traversal respects max_hops

**Success criteria**: Graph nodes and edges are created during ingest. Structural
(heading hierarchy), reference (links), and entity relationships are captured.

---

## 15.2: Graph-Augmented Retrieval

**Description**: After initial vector retrieval, expand the candidate set using graph
traversal.

**Tasks**:
- Create `lsm/query/stages/graph_expansion.py`:
  - `expand_via_graph(candidates, db, max_hops, edge_types)` → expanded candidate list
  - Uses `db.graph_traverse()` with recursive CTE
  - Expanded nodes get `graph_expansion_score` decaying with hop count
- Update `RetrievalPipeline.build_sources()`:
  - After standard retrieval, optionally run graph expansion
  - Graph expansion merges with existing candidates (dedup by chunk_id)
- Add config: `graph_expansion_hops: int = 2`, `graph_expansion_enabled: bool = False`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/graph_expansion.py` — graph expansion stage
- `lsm/query/pipeline.py` — integration
- `lsm/config/models/query.py` — graph config
- `tests/test_query/test_graph_expansion.py`:
  - Test: graph expansion adds related chunks
  - Test: expansion score decays with hops
  - Test: deduplication with existing candidates

**Success criteria**: Graph expansion enriches retrieval with structurally and semantically
related chunks.

---

## 15.3: Multi-Hop Retrieval

**Description**: Implement parallel and iterative multi-hop retrieval strategies.

**Tasks**:
- Create `lsm/query/multi_hop.py`:
  - `MultiHopRequest` dataclass: `query`, `max_hops`, `strategy`, `mode`, `conversation_id`
  - `parallel_multi_hop(request, pipeline)`:
    - Decompose query into sub-questions (LLM)
    - Retrieve N sub-questions in parallel via pipeline
    - Merge ContextPackages
    - Synthesize once with merged context
  - `iterative_multi_hop(request, pipeline)`:
    - Run one sub-question per hop
    - LLM partial answer from hop N informs hop N+1
    - Each hop chains via `prior_response_id`
    - Return final synthesized answer
- Add CLI/TUI support for multi-hop queries

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/multi_hop.py` — multi-hop logic
- `lsm/query/decomposition.py` — update for sub-question generation
- `tests/test_query/test_multi_hop.py`:
  - Test: parallel strategy decomposes and merges
  - Test: iterative strategy chains hops
  - Test: max_hops is respected

**Success criteria**: Both parallel and iterative multi-hop strategies produce comprehensive
answers by combining multiple retrieval passes.

---

## 15.4: Domain-Fine-Tuned Embeddings

**Description**: Implement embedding model fine-tuning using heading-content pairs from
the corpus.

**Tasks**:
- Create `lsm/finetune/` package:
  - `lsm/finetune/embedding.py`:
    - Extract training pairs from `FileGraph`: `(heading_text, section_body)`
    - Fine-tune a `sentence-transformers` model using contrastive learning
    - Save to `./models/finetuned/`
  - `lsm/finetune/registry.py`:
    - Register fine-tuned models in `lsm_embedding_models` table
    - Track active model
- Add `finetune_enabled: bool = False` to `GlobalConfig` (or appropriate config section).
  This is the config trigger checked by TUI startup advisories (Phase 16.2, §6.1).
- Add CLI command: `lsm finetune embedding --base-model <model> --epochs 3 --output <path>`
- Model switch triggers schema mismatch detection (§3.5)

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/finetune/__init__.py` — package init
- `lsm/finetune/embedding.py` — fine-tuning logic
- `lsm/finetune/registry.py` — model registry
- CLI entry point — finetune command
- `tests/test_finetune/test_embedding.py`:
  - Test: training pair extraction from FileGraph
  - Test: model registry CRUD
  - Test: model switch triggers schema mismatch

**Success criteria**: `lsm finetune embedding` produces a fine-tuned model. Model is
registered in DB. Switching to fine-tuned model triggers re-ingest via schema versioning.

---

## 15.5: Phase 15 Code Review and Changelog

**Tasks**:
- Review graph traversal CTE for performance (bounded depth)
- Review multi-hop for runaway LLM calls (max_hops enforcement)
- Review fine-tuning data extraction quality
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CONFIGURATION.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---

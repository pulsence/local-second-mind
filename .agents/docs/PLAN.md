# v0.8.0 Implementation Plan: Embedding Retrieval Improvements

**Version**: 0.8.0
**Research Plan**: [RESEARCH_PLAN.md](./RESEARCH_PLAN.md)

> **Breaking Release**: v0.8.0 is a breaking release. Config file format changes are
> unconstrained — there is no obligation to maintain backward compatibility with v0.7.x
> config files.

---

## Phases

| Phase | Title | Status | Plan |
|-------|-------|--------|------|
| 1 | SQLite-vec Provider and Unified Schema | Completed | [PHASE_1.md](./plan_phases/PHASE_1.md) |
| 2 | Agent Data, Manifest, and Sidecar Consolidation | Completed | [PHASE_2.md](./plan_phases/PHASE_2.md) |
| 3 | Schema Versioning and DB Completion | Completed | [PHASE_3.md](./plan_phases/PHASE_3.md) |
| 4 | Migration System | Completed | [PHASE_4.md](./plan_phases/PHASE_4.md) |
| 5 | LLM Provider Simplification | Pending | [PHASE_5.md](./plan_phases/PHASE_5.md) |
| 6 | Prompt Ownership Migration and ModeConfig | Pending | [PHASE_6.md](./plan_phases/PHASE_6.md) |
| 7 | FileGraph Heading Enhancements | Pending | [PHASE_7.md](./plan_phases/PHASE_7.md) |
| 8 | RetrievalPipeline Core | Pending | [PHASE_8.md](./plan_phases/PHASE_8.md) |
| 9 | Evaluation Harness | Pending | [PHASE_9.md](./plan_phases/PHASE_9.md) |
| 10 | Retrieval Profiles and Hybrid RRF | Pending | [PHASE_10.md](./plan_phases/PHASE_10.md) |
| 11 | Cross-Encoder, LLM Reranking, and HyDE | Pending | [PHASE_11.md](./plan_phases/PHASE_11.md) |
| 12 | Diversity, Dedup, and Temporal Ranking | Pending | [PHASE_12.md](./plan_phases/PHASE_12.md) |
| 13 | Agent Pipeline Tools | Pending | [PHASE_13.md](./plan_phases/PHASE_13.md) |
| 14 | Phase 2 — Multi-Vector and Clustering | Pending | [PHASE_14.md](./plan_phases/PHASE_14.md) |
| 15 | Phase 3 — Graph, Multi-Hop, and Fine-Tuning | Pending | [PHASE_15.md](./plan_phases/PHASE_15.md) |
| 16 | PostgreSQL Parity and Cross-Cutting | Pending | [PHASE_16.md](./plan_phases/PHASE_16.md) |
| 17 | Final Code Review and Release | Pending | [PHASE_17.md](./plan_phases/PHASE_17.md) |

---

## Phase Summaries

### [Phase 1: SQLite-vec Provider and Unified Schema](./plan_phases/PHASE_1.md)
Replaces ChromaDB with sqlite-vec as the default vector store. Creates the unified `lsm.db` database file. Sub-tasks: 1.1–1.4.

### [Phase 2: Agent Data, Manifest, and Sidecar Consolidation](./plan_phases/PHASE_2.md)
Merges agent memory, schedules, manifest, and runtime sidecars into `lsm.db`. Sub-tasks: 2.1–2.5.

### [Phase 3: Schema Versioning and DB Completion](./plan_phases/PHASE_3.md)
Schema version tracking, always-on versioning with prune, and incremental corpus updates. Sub-tasks: 3.1–3.4.

### [Phase 4: Migration System](./plan_phases/PHASE_4.md)
`lsm migrate` CLI for cross-backend and v0.7→v0.8 legacy state migration. Sub-tasks: 4.1–4.3.

### [Phase 5: LLM Provider Simplification](./plan_phases/PHASE_5.md)
Providers become pure transport (`send_message`/`send_streaming_message` only). Removes domain methods and Azure provider. Sub-tasks: 5.1–5.4.

### [Phase 6: Prompt Ownership Migration and ModeConfig](./plan_phases/PHASE_6.md)
Domain prompts move from `providers/helpers.py` to owning modules. ModeConfig gets `retrieval_profile` and `synthesis_instructions`. Sub-tasks: 6.1–6.3.

### [Phase 7: FileGraph Heading Enhancements](./plan_phases/PHASE_7.md)
Configurable heading depth, intelligent heading selection via FileGraph, and `heading_path` metadata. Sub-tasks: 7.1–7.4.

### [Phase 8: RetrievalPipeline Core](./plan_phases/PHASE_8.md)
Central three-stage API (`build_sources → synthesize_context → execute`) with `QueryRequest`, `ContextPackage`, `QueryResponse`. Sub-tasks: 8.1–8.4.

### [Phase 9: Evaluation Harness](./plan_phases/PHASE_9.md)
Retrieval evaluation system with BEIR format, metrics, baselines, and CLI. Must precede all retrieval feature phases. Sub-tasks: 9.1–9.2.

### [Phase 10: Retrieval Profiles and Hybrid RRF](./plan_phases/PHASE_10.md)
Five retrieval profiles, FTS5 query support, RRF fusion, and legacy config field removal. Sub-tasks: 10.1–10.3.

### [Phase 11: Cross-Encoder, LLM Reranking, and HyDE](./plan_phases/PHASE_11.md)
Remaining three profiles: `dense_cross_rerank`, `llm_rerank`, `hyde_hybrid`. Sub-tasks: 11.1–11.4.

### [Phase 12: Diversity, Dedup, and Temporal Ranking](./plan_phases/PHASE_12.md)
MinHash near-duplicate detection, MMR diversity, per-section caps, and temporal-aware ranking. Sub-tasks: 12.1–12.4.

### [Phase 13: Agent Pipeline Tools](./plan_phases/PHASE_13.md)
Three pipeline-backed agent tools replace `query_knowledge_base`. Agent mode validation and `run_bounded` context tracking. Sub-tasks: 13.1–13.3.

### [Phase 14: Multi-Vector and Clustering](./plan_phases/PHASE_14.md)
Multi-vector representation at chunk/section/file granularities and cluster-aware retrieval. Sub-tasks: 14.1–14.4.

### [Phase 15: Graph, Multi-Hop, and Fine-Tuning](./plan_phases/PHASE_15.md)
Graph construction at ingest, graph-augmented retrieval, multi-hop strategies, and domain-fine-tuned embeddings. Sub-tasks: 15.1–15.5.

### [Phase 16: PostgreSQL Parity and Cross-Cutting](./plan_phases/PHASE_16.md)
PostgreSQL provider parity, TUI startup advisories, documentation, and scale guidance. Sub-tasks: 16.1–16.4.

### [Phase 17: Final Code Review and Release](./plan_phases/PHASE_17.md)
Comprehensive review, integration testing, architecture docs, and release preparation. Sub-tasks: 17.1–17.4.

---

*See individual phase files for detailed task blocks, file lists, and success criteria.*

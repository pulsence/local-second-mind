# lsm.query

Description: Query retrieval, reranking, synthesis, session management, and citation export.
Folder Path: `lsm/query/`

## Modules

- [pipeline.py](../lsm/query/pipeline.py): RetrievalPipeline three-stage API (build_sources → synthesize_context → execute)
- [pipeline_types.py](../lsm/query/pipeline_types.py): Pipeline data types (QueryRequest, ContextPackage, QueryResponse, etc.)
- [api.py](../lsm/query/api.py): High-level query API — delegates to RetrievalPipeline
- [session.py](../lsm/query/session.py): SessionState and Candidate dataclasses
- [retrieval.py](../lsm/query/retrieval.py): Vector search and candidate filtering
- [rerank.py](../lsm/query/rerank.py): Local reranking strategies
- [planning.py](../lsm/query/planning.py): Shared query planning (candidate retrieval, filtering, reranking)
- [context.py](../lsm/query/context.py): Context block building, remote source fetching, fallback answers
- [prompts.py](../lsm/query/prompts.py): Synthesis prompt templates (grounded, insight)
- [cache.py](../lsm/query/cache.py): Query result caching
- [cost_tracking.py](../lsm/query/cost_tracking.py): Cost tracking for LLM queries
- [notes.py](../lsm/query/notes.py): Note generation
- [citations.py](../lsm/query/citations.py): Citation export
- [prefilter.py](../lsm/query/prefilter.py): Query prefiltering
- [decomposition.py](../lsm/query/decomposition.py): Query decomposition
- [stages/](../lsm/query/stages/): Pipeline stage implementations
  - [llm_rerank.py](../lsm/query/stages/llm_rerank.py): LLM-based reranking stage

# lsm.query

Description: Query retrieval, reranking, synthesis, session management, and citation export.
Folder Path: `lsm/query/`

## Modules

- [retrieval.py](../lsm/query/retrieval.py): Vector search and candidate filtering
- [rerank.py](../lsm/query/rerank.py): Reranking strategies
- [synthesis.py](../lsm/query/synthesis.py): LLM answer generation
- [session.py](../lsm/query/session.py): SessionState and Candidate dataclasses
- [notes.py](../lsm/query/notes.py): Note generation
- [citations.py](../lsm/query/citations.py): Citation export
- [planning.py](../lsm/query/planning.py): Shared query planning (candidate retrieval, filtering, reranking)
- [context.py](../lsm/query/context.py): Query context management
- [prefilter.py](../lsm/query/prefilter.py): Query prefiltering
- [decomposition.py](../lsm/query/decomposition.py): Query decomposition
- [cache.py](../lsm/query/cache.py): Query caching
- [cost_tracking.py](../lsm/query/cost_tracking.py): Cost tracking for LLM queries
- [api.py](../lsm/query/api.py): High-level query API

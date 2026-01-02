# Tools
- LlammaIndex for building embedings (https://developers.llamaindex.ai/python/framework/)
    - https://developers.llamaindex.ai/python/framework/understanding/rag/
    - https://developers.llamaindex.ai/python/examples/low_level/oss_ingestion_retrieval/
- Watchdog for updating embeddings as files changes (https://python-watchdog.readthedocs.io/en/stable/index.html)
- Chroma for vector DB (https://docs.trychroma.com/docs/overview/introduction)

# Corpus Prep
1. Ingestion + orchestration framework
    - LlamaIndex (Python) for ingestion pipelines + query engines + RAG patterns
2. Embeddings (local)
    - Sentence Transformers (local embeddings; avoids sending raw text to any cloud for indexing)
    - Can also use ChatGPT for embeddings
3. Embeddings are stored in Vector DB
4. Incremental updates
    - A file watcher (e.g., watchdog) that triggers re-indexing for changed files and deletes removed files from the index
    - On change:
        - re-parse the file
        - re-chunk
        - re-embed
        - upsert into Chroma
    - On delete:
        - delete all chunks with that file’s doc_id metadata
5. Reranking (high impact)
    - Add a reranker step (cross-encoder or LLM-based) so retrieval quality stays high when your corpus grows
    - A strong baseline retrieval pipeline:
        - Vector search (top 30–60 chunks)
        - Rerank to top 8–15 chunks
    - Reranking options:
        - Local reranker (cross-encoder) for better relevance
        - Or LLM rerank (more expensive)
6. ChatGPT integration (OpenAI Responses API)
  - Use OpenAI’s Responses API for the reasoning/synthesis step; OpenAI provides official migration guidance and positioning of Responses vs older Chat Completions-style workflows. 
  - Key policy you should enforce in your prompt layer
       - “Answer using only provided context; cite file paths and headings.”
       - “If context is insufficient, say what is missing and propose a retrieval query to find it.”

# Query Flow
1. You ask: “Where do my notes on X conflict with my draft chapter on Y?”
2. LlamaIndex:
    - embeds the query
    - retrieves top-K chunks from the vector store
    - reranks them
    - assembles a compact context packet with citations (file path + heading + chunk id)
3. The OpenAI model (via Responses API) receives:
    - your question
    - the retrieved packet
    - a system prompt enforcing citation use and “no-claim-without-source” rules
4. Output:
    - synthesis + contradictions + suggested outline + links back to exact passages

# Output Modes
1. Grounded Q&A mode
    - Model must cite retrieved chunks; if not found, it says so.
2. Corpus insight mode
    - “Map themes,” “find repeated assumptions,” “surface open questions,” “compare drafts over time.”

# Todo Later:
- Can ChatGPT Plus account be linked with API account, so I can pull the Plus context into my local context
- Detect when OCR needs to be done on PDF and do so
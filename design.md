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

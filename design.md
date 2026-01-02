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
## Infrastructure
- Better organization to config file
- Refactor our the config reading into DataClasses in a single module (lsm.commands)
- Add some kind of GUI over tooling
- Wrap OpenAI with generic wrapper to enable other service providers (Super long term)
## Ingest
- Can ChatGPT Plus account be linked with API account, so I can pull the Plus context into my local context
- Detect when OCR needs to be done on PDF and do so
- Detect authors and store that data
- embed file position information for text
- And post-processing step where the chunks are "pre-taged"
- Change CLI to interactive and enable the following interactions:
    - info on current vector db
    - explorer for vector db
    - wipe
    - build (refresh or start clean)
    - exit out to main terminal or exit program over all
- Add embeddings model selection
## Query
- Refactor components out into files
- Make clearer use of rerank modes
- Make clearer Grounded Q&A and Corpus Insight Mode

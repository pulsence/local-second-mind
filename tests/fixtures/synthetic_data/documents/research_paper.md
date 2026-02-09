# Retrieval-Augmented Knowledge Work in Local-First Systems

## Abstract
We evaluate a local-first retrieval augmented generation pipeline for note-intensive research workflows. The study examines parsing fidelity, chunk quality, embedding stability, and citation-grounded synthesis under realistic constraints [1].

## 1. Introduction
Modern knowledge workers process heterogeneous corpora that include plain text, markdown, HTML, and scanned documents. Traditional search fails when terminology varies across sources, while naive semantic search can over-index stylistic similarity [2].

### 1.1 Research Questions
- How does structure-aware chunking affect retrieval precision in mixed document sets?
- Can local embedding models maintain stable neighborhood structure across incremental ingests?
- What citation patterns improve user trust in synthesized answers?

## 2. Related Work
Prior studies on RAG focus on benchmark question answering and cloud-hosted model stacks. Fewer works address local-first constraints such as offline availability, deterministic persistence, and transparent provenance [3].

## 3. Methodology
### 3.1 Corpus Design
The corpus contains theoretical essays, technical manuals, edge-case files, and controlled duplicates. This composition supports robustness testing for ingestion pipelines and filter-aware retrieval.

### 3.2 Pipeline Definition
```python
def run_pipeline(docs, embedder, vectordb):
    chunks = parse_and_chunk(docs, strategy="structure")
    vectors = embedder.encode(chunks.texts)
    vectordb.add_chunks(chunks.ids, chunks.texts, chunks.metadatas, vectors)
    return vectordb.count()
```

### 3.3 Evaluation Metrics
- Retrieval precision at k for question sets with known supporting passages.
- Citation validity rate based on source traceability.
- Chunk boundary quality for heading and paragraph preservation.
- Incremental ingest consistency for unchanged document hashes.

## 4. Experimental Notes
### 4.1 Observation
Run 1 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.2 Observation
Run 2 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.3 Observation
Run 3 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.4 Observation
Run 4 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.5 Observation
Run 5 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.6 Observation
Run 6 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.7 Observation
Run 7 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.8 Observation
Run 8 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.9 Observation
Run 9 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.10 Observation
Run 10 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.11 Observation
Run 11 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.12 Observation
Run 12 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.13 Observation
Run 13 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.14 Observation
Run 14 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.15 Observation
Run 15 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.16 Observation
Run 16 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.17 Observation
Run 17 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.18 Observation
Run 18 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.19 Observation
Run 19 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.20 Observation
Run 20 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.21 Observation
Run 21 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.22 Observation
Run 22 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.23 Observation
Run 23 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.24 Observation
Run 24 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.25 Observation
Run 25 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.26 Observation
Run 26 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.27 Observation
Run 27 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.28 Observation
Run 28 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.29 Observation
Run 29 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.30 Observation
Run 30 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.31 Observation
Run 31 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.32 Observation
Run 32 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.33 Observation
Run 33 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.34 Observation
Run 34 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.35 Observation
Run 35 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.36 Observation
Run 36 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.37 Observation
Run 37 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.38 Observation
Run 38 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.39 Observation
Run 39 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.40 Observation
Run 40 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.41 Observation
Run 41 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.42 Observation
Run 42 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.43 Observation
Run 43 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.44 Observation
Run 44 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.45 Observation
Run 45 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.46 Observation
Run 46 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.47 Observation
Run 47 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.48 Observation
Run 48 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.49 Observation
Run 49 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.50 Observation
Run 50 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.51 Observation
Run 51 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.52 Observation
Run 52 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.53 Observation
Run 53 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.54 Observation
Run 54 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.55 Observation
Run 55 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.56 Observation
Run 56 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.57 Observation
Run 57 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.58 Observation
Run 58 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.59 Observation
Run 59 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.60 Observation
Run 60 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.61 Observation
Run 61 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.62 Observation
Run 62 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.63 Observation
Run 63 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.64 Observation
Run 64 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.65 Observation
Run 65 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.66 Observation
Run 66 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.67 Observation
Run 67 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.68 Observation
Run 68 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.69 Observation
Run 69 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.70 Observation
Run 70 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.71 Observation
Run 71 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.72 Observation
Run 72 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.73 Observation
Run 73 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.74 Observation
Run 74 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.75 Observation
Run 75 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.76 Observation
Run 76 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.77 Observation
Run 77 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.78 Observation
Run 78 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.79 Observation
Run 79 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.80 Observation
Run 80 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.81 Observation
Run 81 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.82 Observation
Run 82 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.83 Observation
Run 83 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.84 Observation
Run 84 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.85 Observation
Run 85 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.86 Observation
Run 86 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.87 Observation
Run 87 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.88 Observation
Run 88 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.89 Observation
Run 89 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

### 4.90 Observation
Run 90 combined lexical and semantic retrieval before reranking. The ranked set favored passages that matched intent-level semantics while maintaining explicit citation anchors. Failure cases appeared when highly generic chunks shared topical vocabulary but lacked direct evidentiary content. We mitigated this by increasing metadata-aware filtering and tightening maximum candidates per file [1][2].

## 5. Discussion
Results suggest that local-first RAG quality depends less on single-model sophistication and more on disciplined data handling. In particular, chunk integrity and provenance metadata had outsized impact on synthesis reliability [4].

## 6. Conclusion
A robust local pipeline can deliver practical, transparent synthesis when parsing, embeddings, and storage interfaces are validated with realistic corpora rather than minimal toy data.

## References
- [1] Chen, R. and Patel, M. (2024). Grounded Synthesis in Practical RAG Systems.
- [2] Ibarra, L. (2023). Failure Modes in Dense Retrieval under Domain Shift.
- [3] Miller, A. (2025). Local-First AI Infrastructure for Knowledge Management.
- [4] Osei, D. and Novak, J. (2022). Citation-Centric Evaluation for Hybrid Search.

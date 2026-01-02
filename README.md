# Local Second Mind (LSM)

Local Second Mind is a **local-first retrieval-augmented reasoning system** designed to help you search, analyze, and synthesize your own writing, research, and notes using modern embedding and LLM tooling — without surrendering control of your corpus.

It is optimized for **heterogeneous, evolving document collections** (PDFs, Word documents, Markdown, HTML, plain text, etc.) stored on a local machine, with incremental updates and strong provenance.

---

## Core Goals

* Maintain a **living knowledge base** from local files
* Support **incremental ingest** as documents change
* Enable **high-quality semantic + lexical retrieval**
* Produce **LLM-generated answers with citations** back to source files
* Keep embeddings local while optionally using LLM APIs for reranking and synthesis

---

## High-Level Architecture

```
Local Files
   ↓
Parse → Chunk → Embed (SentenceTransformers)
   ↓
ChromaDB (persistent vector store)
   ↓
Query
   ├─ Semantic retrieval
   ├─ Lexical rerank (optional)
   ├─ LLM rerank (optional, OpenAI)
   └─ Answer synthesis with citations
```

---

## Features

### Ingest Pipeline

* Recursive crawl of one or more root directories
* File-type aware parsing (PDF, DOCX, HTML, TXT, MD)
* Stable, deterministic chunking
* Local embeddings using **sentence-transformers**
* Persistent vector storage via **ChromaDB**
* Incremental ingest using file hashes + modification times
* Threaded parsing for performance

### Query Pipeline

* Semantic search using the same embedding model as ingest
* Configurable `k` retrieval depth
* Optional lexical reranking
* Optional LLM reranking (OpenAI Responses API)
* Deduplication and per-file diversity controls
* Grounded answers with inline citations

### Operational Design

* Single unified configuration file (`config.json` or `config.yaml`)
* CLI-first interface suitable for scripting
* Windows-friendly paths and filesystem behavior
* Clear separation between ingest, query, and orchestration layers

---

## Installation

### Python Version

Python **3.10+** is recommended.

### Dependencies

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

Key libraries include:

* sentence-transformers
* chromadb
* PyMuPDF
* python-docx
* beautifulsoup4
* openai

---

## Command-Line Interface

The project exposes a single CLI entrypoint:

```bash
python -m lsm
```

### Ingest Command

```bash
python -m lsm ingest --config config.json
```

**Purpose:**
Indexes or updates your local knowledge base.

**Behavior:**

* Walks configured directories
* Parses supported file types
* Chunks and embeds content
* Skips unchanged files automatically

---

### Query Command

```bash
python -m lsm query --config config.json
```

**Purpose:**
Interactively query your local knowledge base.

**Behavior:**

* Prompts for a natural-language question
* Retrieves semantically relevant chunks
* Optionally reranks results
* Produces a synthesized answer with citations

---

## Configuration

All behavior is controlled via a **single unified config file**.

Supported formats:

* `config.json`
* `config.yaml`

### Example Configuration

```json
{
  "roots": [
    "C:/Users/User/Documents/Research",
    "C:/Users/User/Documents/Writing"
  ],

  "persist_dir": ".chroma",
  "chroma_flush_interval": 2000,

  "collection": "local_kb",
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "device": "cpu",
  "batch_size": 32,

  "manifest": ".ingest/manifest.json",

  "extensions": [".txt", ".md", ".pdf", ".docx", ".html"],
  "override_extensions": false,

  "exclude_dirs": [".cache"],
  "override_excludes": false,

  "dry_run": false,

  "openai": {
    "api_key": "INSERT_YOUR_KEY"
  },

  "query": {
    "k": 12,
    "k_rerank": 6,
    "no_rerank": false,
    "max_per_file": 2,
    "local_pool": 36,

    "model": "gpt-5.2",

    "min_relevance": 0.25,

    "path_contains": ["\\Research\\"],
    "ext_allow": [".md", ".pdf"],
    "ext_deny": [".docx"],

    "retrieve_k": 36
  }
}
```

## Metadata & Citations

Each chunk stored in Chroma includes:

* `source_path`
* `source_name`
* `ext`
* `mtime_ns`
* `file_hash`
* `chunk_index`
* `ingested_at`

Citations in answers are derived from this metadata, allowing you to trace claims backto exact source files.

## Typical Workflow

1. Configure paths and models in config.json
2. Run ingest to build or update the knowledge base
3. Run query to explore your corpus interactively
4. Re‑run ingest whenever files change

## Design Philosophy

Local Second Mind intentionally:

- Keeps embeddings local
- Treats LLMs as reasoning layers, not memory
- Optimizes for long‑term personal knowledge growth
- Prioritizes transparency and debuggability

## Roadmap

TBD

## License

Private / internal project. License to be defined.

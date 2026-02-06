# LSM Architecture Overview

This document provides a high-level overview of Local Second Mind's architecture and design principles.

## Design Principles

### 1. Local-First
- **Core functionality works offline** - Embeddings and retrieval don't need internet
- **Data ownership** - All your documents stay on your machine
- **Privacy** - Only LLM synthesis calls go to external APIs
- **No vendor lock-in** - Standard formats (ChromaDB, JSON config)

### 2. Modular & Extensible
- **Provider abstraction** - Easy to add new LLM providers (see ADDING_PROVIDERS.md)
- **Plugin-like architecture** - Remote sources, custom modes
- **Clear interfaces** - Well-defined contracts between components

### 3. Grounded by Default
- **Citation-first** - Every claim must be cited
- **Source transparency** - Users see exactly where information comes from
- **Quality over quantity** - Better to say "I don't know" than hallucinate

### 4. User Control
- **Explicit configuration** - No hidden magic
- **Interactive tools** - TUI for exploration
- **Multiple interfaces** - CLI (single-shot) and TUI (interactive)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │   CLI    │  │  Unified │  │   TUI    │  │  Single   │  │
│  │  (lsm)   │  │  Shell   │  │(Textual) │  │   Shot    │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
┌───────────────▼───────────┐   ┌──────────▼──────────────┐
│    Ingest Pipeline        │   │    Query Pipeline        │
│  ┌─────────────────────┐  │   │  ┌──────────────────┐   │
│  │ Document Scanning   │  │   │  │   Retrieval      │   │
│  │ File Parsing        │  │   │  │   Reranking      │   │
│  │ Text Chunking       │  │   │  │   Synthesis      │   │
│  │ Embedding           │  │   │  │   Formatting     │   │
│  │ Storage             │  │   │  └──────────────────┘   │
│  └─────────────────────┘  │   └─────────────────────────┘
└───────────────────────────┘
                │                             │
                │                             │
┌───────────────▼───────────────┐  ┌─────────▼──────────────┐
│    Storage Layer              │  │  Provider Layer        │
│  ┌─────────────────────────┐  │  │  ┌──────────────────┐  │
│  │ ChromaDB (Vectors)      │  │  │  │ LLM Providers    │  │
│  │ Manifest (Metadata)     │  │  │  │ - OpenAI         │  │
│  │ Config (Settings)       │  │  │  │ - Anthropic (*)  │  │
│  └─────────────────────────┘  │  │  │ - Local (*)      │  │
└───────────────────────────────┘  │  └──────────────────┘  │
                                   │  ┌──────────────────┐  │
                                   │  │ Remote Providers │  │
                                   │  │ - Brave Search   │  │
                                   │  │ - Wikipedia (*)  │  │
                                   │  └──────────────────┘  │
                                   └─────────────────────────┘

(*) = Planned but not yet implemented
```

## Core Components

### Ingest Pipeline

**Purpose:** Transform documents into searchable chunks with embeddings.

**Flow:**
1. **Scanner** - Recursively finds files in configured roots
2. **Parser** - Extracts text from various formats (PDF, DOCX, MD, HTML)
3. **Chunker** - Splits text into overlapping chunks (~1800 chars)
4. **Embedder** - Creates vector embeddings (sentence-transformers)
5. **Storage** - Stores in ChromaDB with metadata

**Key Files:**
- `lsm/ingest/pipeline.py` - Main orchestration
- `lsm/ingest/parsers.py` - File format parsers
- `lsm/ingest/chunking.py` - Text chunking logic
- `lsm/ingest/fs.py` - File system scanning
- `lsm/ingest/manifest.py` - Incremental update tracking

**See:** [Ingest Pipeline Details](INGEST.md)

### Query Pipeline

**Purpose:** Find relevant chunks and synthesize grounded answers.

**Flow:**
1. **Embed Query** - Convert question to vector
2. **Retrieve** - Find top-k candidates by similarity
3. **Filter** - Apply user filters (path, extension, etc.)
4. **Rerank** - Lexical + LLM reranking for quality
5. **Synthesize** - Generate answer with citations via LLM
6. **Format** - Present with source metadata

**Key Files:**
- `lsm/query/retrieval.py` - Vector search
- `lsm/query/rerank.py` - Reranking strategies
- `lsm/query/synthesis.py` - Answer generation
- `lsm/query/repl.py` - Interactive commands
- `lsm/query/session.py` - State management

**See:** [Query Pipeline Details](QUERY.md)

### Provider System

**Purpose:** Abstract LLM interactions for multi-provider support.

**Interface:**
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def rerank(self, question, candidates, k) -> List[Dict]

    @abstractmethod
    def synthesize(self, question, context, mode) -> str

    @abstractmethod
    def is_available(self) -> bool

    @property
    @abstractmethod
    def name(self) -> str

    @property
    @abstractmethod
    def model(self) -> str
```

**Current Providers:**
- **OpenAI** - GPT models (gpt-5.2, gpt-4o-mini)

**Planned Providers:**
- Anthropic Claude
- Local models (Ollama)
- Azure OpenAI

**See:** [Provider System Details](PROVIDERS.md) and [Adding Providers](../ADDING_PROVIDERS.md)

### Mode System

**Purpose:** Configure different query behaviors and source blending.

**Built-in Modes:**

| Mode | Local Sources | Remote Sources | Model Knowledge | Use Case |
|------|--------------|----------------|-----------------|----------|
| **grounded** | ✅ | ❌ | ❌ | Factual Q&A with strict citations |
| **insight** | ✅ | ❌ | ❌ | Thematic analysis, pattern finding |
| **hybrid** | ✅ | ✅ | ✅ | Comprehensive research |

**Custom Modes:** Users can define their own modes in `config.json`.

**See:** [Mode System Details](MODES.md)

### Remote Sources

**Purpose:** Augment local knowledge with web search and APIs.

**Current Providers:**
- **Brave Search** - Web search API

**Planned:**
- Wikipedia API
- Custom APIs (user-defined)

**See:** [Remote Source Configuration](../user-guide/REMOTE_SOURCES.md)

### Notes System

**Purpose:** Save query sessions for future reference.

**Features:**
- Markdown format
- Includes query, sources, and answer
- Editable in external editor
- Customizable filename formats

**See:** [Notes System Guide](../user-guide/NOTES.md)

## Data Flow

### Ingest Flow

```
Documents → Parse → Chunk → Embed → Store
    │         │       │        │       │
    ├─ PDFs   ├─ Text ├─ 1800  ├─ Vec  └─ ChromaDB
    ├─ DOCX   │       │   chars │   384      +
    ├─ MD     │       ├─ Over  └─ dims  Manifest
    ├─ HTML   │       │   lap
    └─ TXT    │       └─ 200
              │
              └─ Metadata
                 (path, title, author, etc.)
```

### Query Flow

```
Question
   │
   ├─ Embed (local)
   │
   ├─ Retrieve (ChromaDB)
   │     ↓
   │  Candidates (k=12-36)
   │
   ├─ Filter (optional)
   │     ↓
   │  Filtered candidates
   │
   ├─ Rerank
   │   ├─ Lexical (local)
   │   └─ LLM (API)
   │     ↓
   │  Top candidates (k=6)
   │
   ├─ Synthesize (LLM)
   │   ├─ Build context
   │   ├─ Choose mode
   │   └─ Generate answer
   │     ↓
   │  Answer + Citations
   │
   └─ Format & Display
```

## Configuration System

**Hierarchical Structure:**

```
LSMConfig
├─ IngestConfig
│  ├─ roots: List[Path]
│  ├─ extensions: List[str]
│  ├─ exclude_dirs: Set[str]
│  ├─ chunk_size: int
│  ├─ enable_ocr: bool
│  └─ enable_ai_tagging: bool
│
├─ QueryConfig
│  ├─ mode: str
│  ├─ k: int
│  ├─ k_rerank: int
│  ├─ rerank_strategy: str
│  ├─ min_relevance: float
│  └─ filters: Optional
│
├─ LLMConfig
│  ├─ provider: str
│  ├─ model: str
│  ├─ api_key: str
│  ├─ temperature: float
│  ├─ max_tokens: int
│  ├─ query: Optional[LLMConfig]
│  ├─ tagging: Optional[LLMConfig]
│  └─ ranking: Optional[LLMConfig]
│
├─ ModeConfig (custom modes)
│  ├─ synthesis_style: str
│  └─ source_policy: SourcePolicyConfig
│
└─ RemoteProviderConfig
   ├─ type: str
   ├─ weight: float
   ├─ api_key: str
   └─ max_results: int
```

**See:** [Configuration Guide](../user-guide/CONFIGURATION.md)

## Performance Characteristics

### Ingest
- **Speed:** ~100-500 chunks/second (CPU/GPU)
- **Memory:** ~2GB for embedding model
- **Storage:** ~100-200 bytes per chunk (vectors + metadata)
- **Incremental:** Only processes new/modified files

### Query
- **Retrieval:** <100ms for 100K chunk collection
- **Reranking:** ~1-2s for lexical + LLM
- **Synthesis:** ~3-10s depending on LLM
- **Total:** Typically 5-15 seconds per query

## Technology Stack

### Core Dependencies
- **sentence-transformers** - Local embeddings (all-MiniLM-L6-v2)
- **chromadb** - Vector database
- **openai** - LLM API client
- **PyMuPDF** - PDF parsing
- **python-docx** - DOCX parsing
- **beautifulsoup4** - HTML parsing
- **requests** - HTTP client for remote sources

### Development Dependencies
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **ruff** - Linting and formatting

## Extension Points

LSM is designed for extensibility:

1. **New LLM Providers** - Implement `BaseLLMProvider`
2. **New Remote Sources** - Implement `BaseRemoteProvider`
3. **Custom Modes** - Define in `config.json`
4. **Custom File Parsers** - Add to `parsers.py`
5. **Custom Reranking** - Implement in `rerank.py`

## Security Considerations

### Data Privacy
- ✅ **Documents stay local** - Never sent to external APIs
- ✅ **Embeddings computed locally** - No external calls
- ⚠️ **Queries sent to LLM** - Question + context go to OpenAI/etc
- ⚠️ **Remote sources** - Web searches reveal query intent

### API Keys
- ✅ **Environment variables** - Preferred method
- ⚠️ **Config file** - Don't commit with keys
- ✅ **Per-provider keys** - Different keys for different services

### File System
- ✅ **Read-only operations** - Ingest never modifies source files
- ✅ **Safe paths** - Path traversal protection
- ⚠️ **Excluded directories** - Configure to avoid system folders

## Future Architecture

### Planned Enhancements
1. **Multi-provider LLMs** - Anthropic, local models
2. **Advanced reranking** - Cross-encoder models
3. **Multi-modal** - Image and audio support
4. **Collaborative** - Shared knowledge bases
5. **Version control** - Track document evolution

### Not Planned (For Now)
- Cloud deployment
- Web UI (CLI-first approach)
- Real-time sync
- Mobile clients

## References

- [Ingest Pipeline](INGEST.md)
- [Query Pipeline](QUERY.md)
- [Provider System](PROVIDERS.md)
- [Mode System](MODES.md)
- [Adding Providers](../ADDING_PROVIDERS.md)
- [Configuration](../user-guide/CONFIGURATION.md)

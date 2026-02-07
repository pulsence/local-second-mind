# Getting Started with Local Second Mind

This guide will help you get up and running with Local Second Mind (LSM) in just a few minutes.

## What is Local Second Mind?

Local Second Mind is a **local-first RAG (Retrieval-Augmented Generation)** system that helps you:

- 📚 **Index your documents** - PDFs, Markdown, DOCX, HTML, and more
- 🔍 **Search semantically** - Find information by meaning, not just keywords
- 💬 **Ask questions** - Get AI-generated answers with citations
- 🔒 **Keep it private** - All your data stays local (only LLM calls go to APIs)
- 🎯 **Stay grounded** - Every claim is cited with sources

## Prerequisites

- **Python 3.10+**
- **OpenAI API key** (or other LLM provider)
- **~500MB disk space** for models and dependencies

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-second-mind.git
cd local-second-mind
```

### 2. Install Dependencies

```bash
pip install -e .
```

This installs LSM and all required dependencies.

### 3. Set Up Configuration

Create a `config.json` file in the project root:

```bash
cp example_config.json config.json
```

Edit `config.json` and update:

```json
{
  "ingest": {
    "roots": [
      "C:\\Users\\YourName\\Documents",
      "D:\\Research"
    ]
  },
  "llms": [
    {
      "provider_name": "openai",
      "query": { "model": "gpt-5.2" },
      "tagging": { "model": "gpt-5-nano" },
      "ranking": { "model": "gpt-5-nano" }
    }
  ]
}
```

**Security Tip:** Put API keys in a `.env` file (see `.env.example`) or export them in your shell:

```bash
# Windows
set OPENAI_API_KEY=your-key-here

# Linux/Mac
export OPENAI_API_KEY=your-key-here
```

## First Run: Ingest Your Documents

The first step is to ingest your documents into LSM's knowledge base.

### Option 1: TUI Ingest (Recommended)

```bash
lsm
```

Open the Ingest tab and run:

```
/build
/stats
/explore
```

### Option 2: Single-Shot Ingest

```bash
lsm ingest build
```

This runs the ingest once and exits. Use this for automation.

### What Happens During Ingest?

1. **Scans** your configured directories for documents
2. **Parses** each document (extracting text from PDFs, DOCX, etc.)
3. **Chunks** the text into manageable pieces
4. **Embeds** each chunk using a local model (all-MiniLM-L6-v2)
5. **Stores** embeddings in ChromaDB for fast retrieval

## First Query: Ask Questions

Now that your documents are indexed, you can query them!

### Option 1: TUI (Recommended)

```bash
lsm
```

Open the Query tab and ask a question:

```
What are the main themes in my research notes?
```

### Option 2: Single-Shot Query

```bash
lsm query "What is the capital of France?"
```

Asks a single question and exits. Perfect for scripts.

## Understanding Query Results

LSM provides rich, cited answers:

```
Answer: The capital of France is Paris [S1]. Paris has been
the capital since 987 AD [S2] and is home to over 2 million
residents [S3].

Sources:
[S1] geography/europe.md (chunk 5)
    Relevance: 0.95
    "France's capital city is Paris, located in the..."

[S2] history/french_history.pdf (chunk 12)
    Relevance: 0.87
    "Paris became the permanent capital in 987..."

[S3] demographics/cities.txt (chunk 3)
    Relevance: 0.82
    "Paris population: 2.1 million (city proper)..."
```

### Key Features

- **[S#] Citations** - Every claim is cited
- **Relevance Scores** - See how relevant each source is
- **Source Snippets** - Preview the actual text
- **Metadata** - File paths, chunk indices, titles, authors

## Useful Commands (TUI)

### Query Tab

```bash
/help           # Show all commands
/show S2        # Show full text of source 2
/open S2        # Open source file in default app
/mode grounded  # Switch to strict citation mode
/mode insight   # Switch to thematic analysis mode
/note           # Save this query as an editable note
/debug          # Show retrieval diagnostics
/exit           # Exit TUI
```

### Ingest Tab

```bash
/info           # Show collection statistics
/stats          # Detailed statistics
/explore        # Browse indexed files
/build          # Run ingest pipeline
/build --force  # Force full rebuild
/wipe           # Clear collection (with confirmation)
/exit           # Exit TUI
```

## Next Steps

### Customize Your Configuration

See the [Configuration Guide](CONFIGURATION.md) to learn about:
- File type filtering
- Chunk size tuning
- Query modes (grounded vs insight vs hybrid)
- Remote source integration (web search)
- Per-feature LLM settings

### Learn About Query Modes

LSM supports different query modes for different needs:

- **Grounded** - Strict citations, factual Q&A
- **Insight** - Thematic analysis, pattern finding
- **Hybrid** - Local + web + model knowledge

See [Query Modes](QUERY_MODES.md) for details.

### Add Remote Sources

Integrate web search for hybrid queries:

See [Remote Sources](REMOTE_SOURCES.md) to set up Brave Search API.

### Save Your Queries

Use the notes system to save queries with sources:

```bash
What are my action items?
[answer appears]

/note
[Opens in editor for you to refine]
Note saved to: notes/20260112-143022.md
```

See [Notes System](NOTES.md) for details.

## Troubleshooting

### "Config file not found"

Make sure `config.json` exists in the project root:

```bash
cp example_config.json config.json
```

### "ChromaDB directory not found"

You need to run ingest first:

```bash
lsm ingest build
```

### "Collection is empty"

No documents were ingested. Check:
1. Your `ingest.roots` in `config.json` point to existing directories
2. Those directories contain supported file types (.txt, .md, .pdf, .docx, .html)
3. Files aren't in excluded directories (node_modules, .cache, etc.)

### "API key not configured"

Set your OpenAI API key:

```bash
# In config.json
"llms": [
  {
    "provider_name": "openai",
    "api_key": "sk-...",
    "query": { "model": "gpt-5.2" },
    "tagging": { "model": "gpt-5-nano" },
    "ranking": { "model": "gpt-5-nano" }
  }
]

# Or environment variable
export OPENAI_API_KEY=sk-...
```

### Slow Ingest

Ingest can be slow for large collections. Tips:
- Use `lsm ingest build --dry-run` to test without writing
- Exclude large directories in `ingest.exclude_dirs`
- Use GPU with `"device": "cuda"` in the `"global"` section of config

### Poor Search Results

Try adjusting retrieval parameters in `config.json`:

```json
{
  "query": {
    "k": 20,              // Retrieve more candidates
    "min_relevance": 0.20  // Lower threshold
  }
}
```

## Getting Help

- **Documentation:** Check other guides in `docs/`
- **Issues:** Open a GitHub issue
- **Examples:** See `example_config.json` for all options

## What's Next?

- 📖 Read the [Configuration Guide](CONFIGURATION.md)
- 🎯 Learn about [Query Modes](QUERY_MODES.md)
- 📝 Set up the [Notes System](NOTES.md)
- 🌐 Enable [Remote Sources](REMOTE_SOURCES.md)
- 🔧 Explore [CLI Usage](CLI_USAGE.md) for advanced features

Happy querying! 🎉

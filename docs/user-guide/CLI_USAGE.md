# LSM CLI Usage

The Local Second Mind (LSM) command-line interface provides multiple interactive modes for managing your knowledge base, including a TUI, basic shell, and single-shot commands.

## Modes of Operation

### 1. TUI (Textual User Interface)

For a rich terminal experience with tabbed navigation, keyboard shortcuts, and a modern interface:

```python
from lsm.gui.shell.tui import run_tui
from lsm.config import load_config_from_file

config = load_config_from_file("config.json")
run_tui(config)
```

The TUI provides:
- **Tabbed interface** - Switch between Query, Ingest, and Settings tabs
- **Keyboard shortcuts** - Navigate with Ctrl+Q/I/S, build with Ctrl+B, etc.
- **Autocomplete** - Tab completion for commands (press Tab)
- **Command history** - Up/Down arrow navigation through previous commands
- **Status bar** - Real-time mode, cost, and chunk count display
- **Results panel** - Scrollable results with expandable citations
- **File browser** - Directory tree for exploring indexed files (Ingest tab)

**TUI Screens:**

| Screen | Description |
|--------|-------------|
| Query | Search your knowledge base with natural language queries |
| Ingest | Manage document ingestion, build pipeline, view statistics |
| Settings | Configure query modes, view provider status, session info |

**TUI Widgets:**

| Widget | Description |
|--------|-------------|
| ResultsPanel | Displays query results with expandable citations (click to select) |
| CommandInput | Input with command history (Up/Down) and Tab completion |
| StatusBar | Shows current mode, chunk count, session cost, provider status |

**TUI Keyboard Shortcuts:**

| Shortcut | Action |
|----------|--------|
| Ctrl+I | Switch to Ingest tab |
| Ctrl+Q | Switch to Query tab |
| Ctrl+S | Switch to Settings tab |
| F1 | Show help modal |
| Ctrl+C | Quit application |
| Ctrl+B | Run build (Ingest) |
| Ctrl+T | Run tagging (Ingest) |
| Ctrl+R | Refresh stats (Ingest) |
| Ctrl+E | Expand selected citation (Query) |
| Ctrl+O | Open source file (Query) |
| Tab | Autocomplete command |
| Up/Down | Navigate command history |
| Escape | Clear input |

### 2. Unified Interactive Shell

When you run `lsm` without any arguments, you enter the unified interactive shell where you can switch between ingest and query contexts:

```bash
lsm
```

This starts an interactive session where you can:
- Type `/ingest` or `/i` to switch to ingest context
- Type `/query` or `/q` to switch to query context
- Type `/help` to see context-specific commands
- Type `/exit` or `/quit` to exit

The prompt shows your current context:
- `>` - No context selected (select ingest or query)
- `[ingest] >` - Ingest context (manage knowledge base)
- `[query] >` - Query context (ask questions)

### 2. Single-Shot Commands

For automation and scripting, you can run commands directly without entering interactive mode.

#### Ingest Command

Run ingest pipeline once and exit:

```bash
lsm ingest
```

Options:
- `--dry-run` - Simulate ingest without writing to database
- `--force` - Force re-ingest all files (ignore manifest)
- `--interactive` or `-i` - Start interactive ingest REPL only

Examples:
```bash
# Single-shot ingest
lsm ingest

# Force full rebuild
lsm ingest --force

# Interactive ingest management
lsm ingest --interactive
```

#### Query Command

Ask a single question and exit:

```bash
lsm query "What is the capital of France?"
```

Without a question, starts interactive query REPL:

```bash
lsm query
```

Options:
- `--interactive` or `-i` - Start interactive REPL (ignores question)
- `--mode {grounded,insight,hybrid}` - Set query mode
- `--model MODEL` - Override LLM model from config
- `--no-rerank` - Skip LLM reranking step
- `-k K` - Number of chunks to retrieve

Examples:
```bash
# Single-shot query
lsm query "Explain quantum computing"

# Interactive query with mode override
lsm query --mode hybrid --interactive

# Single-shot query with custom settings
lsm query "What is RAG?" --mode grounded -k 10
```

## Unified Shell Commands

### Global Commands (Available in All Contexts)

- `/ingest`, `/i` - Switch to ingest context
- `/query`, `/q` - Switch to query context
- `/help` - Show context-specific help
- `/exit`, `/quit` - Exit shell

### Ingest Context Commands

When in ingest context (`[ingest] >`):

- `/info` - Show collection information
- `/stats` - Show detailed statistics
- `/explore [query]` - Browse indexed files
- `/show <path>` - Show chunks for a file
- `/search <query>` - Search metadata
- `/build [--force]` - Run ingest pipeline
- `/tag [--max N]` - Run AI tagging on chunks
- `/tags` - Show all tags in collection
- `/wipe` - Clear collection (requires confirmation)

### Query Context Commands

When in query context (`[query] >`):

- Just type your question to query the knowledge base
- `/show S#` - Show cited chunk (e.g., /show S2)
- `/expand S#` - Show full chunk text
- `/open S#` - Open source file in default app
- `/models` - List available models
- `/model` - Show current model
- `/model <name>` - Set model for this session
- `/providers` - List available LLM providers
- `/provider-status` - Show provider health and recent stats
- `/mode` - Show current query mode
- `/mode <name>` - Switch query mode
- `/note` - Save last query as note (opens editor)
- `/load <path>` - Pin document for forced inclusion
- `/load clear` - Clear pinned documents
- `/set <key> <value>` - Set session filter
- `/clear` - Clear session filters
- `/debug` - Show debug information

## Configuration

All modes use the same configuration file (default: `config.json`).

Override config file location:
```bash
lsm --config /path/to/config.json
```

Enable verbose logging:
```bash
lsm --verbose
```

Set log level:
```bash
lsm --log-level DEBUG
```

## Examples

### Typical Workflow

1. Start unified shell:
   ```bash
   lsm
   ```

2. Switch to ingest context and build knowledge base:
   ```
   > /ingest
   [ingest] > /build
   ```

3. Switch to query context and ask questions:
   ```
   [ingest] > /query
   [query] > What are the main topics in my documents?
   ```

4. Exit when done:
   ```
   [query] > /exit
   ```

### Automation / Scripting

Run ingest and query in a script:
```bash
# Build knowledge base
lsm ingest

# Ask specific questions
lsm query "Summary of recent notes?" > output.txt
lsm query "What are my action items?" >> output.txt
```

### Mixed Interactive and Single-Shot

Use interactive shell for exploration, single-shot for specific tasks:
```bash
# Explore knowledge base interactively
lsm ingest --interactive

# Then in another terminal, query while exploring
lsm query "Find all references to project X"
```

# LSM CLI Usage

The Local Second Mind (LSM) CLI provides a TUI for all interactive workflows and single-shot commands for automation.

## TUI (Textual User Interface)

Launch the TUI with:

```bash
lsm
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

## Single-Shot Commands

For automation and scripting, you can run commands directly without entering the TUI.

### Ingest Commands

Build or update the collection:

```bash
lsm ingest build
```

Options:
- `--dry-run` - Simulate ingest without writing to database
- `--force` - Force re-ingest all files (clears manifest)
- `--skip-errors` - Continue ingest when parsing errors occur

Tag untagged chunks:

```bash
lsm ingest tag --max 200
```

Wipe the collection (destructive):

```bash
lsm ingest wipe --confirm
```

### Query Command

Ask a single question and exit:

```bash
lsm query "What is the capital of France?"
```

Options:
- `--mode {grounded,insight,hybrid}` - Set query mode
- `--model MODEL` - Override LLM model from config
- `--no-rerank` - Skip LLM reranking step
- `-k K` - Number of chunks to retrieve

If you want interactive query, use the TUI (`lsm`).

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

1. Start the TUI:
   ```bash
   lsm
   ```

2. Build the knowledge base from the Ingest tab:
   ```
   /build
   ```

3. Switch to Query tab and ask questions:
   ```
   What are the main topics in my documents?
   ```

### Automation / Scripting

Run ingest and query in a script:
```bash
# Build knowledge base
lsm ingest build

# Ask specific questions
lsm query "Summary of recent notes?" > output.txt
lsm query "What are my action items?" >> output.txt
```

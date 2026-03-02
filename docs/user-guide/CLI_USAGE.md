# LSM CLI Usage

The Local Second Mind (LSM) CLI provides a TUI for all interactive workflows and single-shot commands for automation.

## TUI (Textual User Interface)

Launch the TUI with:

```bash
lsm
```

The TUI provides:
- **Tabbed interface** - Switch between Query, Ingest, Remote, Agents, and Settings tabs
- **Keyboard shortcuts** - Navigate with Ctrl+Q/N/R/G/S, build with Ctrl+B, etc.
- **Autocomplete** - Tab completion for commands (press Tab)
- **Command history** - Up/Down arrow navigation through previous commands
- **Status bar** - Real-time mode, cost, and chunk count display
- **Results panel** - Scrollable results with expandable citations
- **File browser** - Directory tree for exploring indexed files (Ingest tab)
- **Interactive agents** - Run multiple agents, approve/deny interaction requests, and stream logs in real time (Agents tab)

**TUI Screens:**

| Screen | Description |
|--------|-------------|
| Query | Search your knowledge base with natural language queries |
| Ingest | Manage document ingestion, build pipeline, view statistics |
| Remote | Test and inspect remote providers |
| Agents | Launch and manage concurrent agent runs, interactions, schedules, and memory candidates |
| Settings | Edit config sections with live updates and section save/reset |

**TUI Keyboard Shortcuts:**

| Shortcut | Action |
|----------|--------|
| Ctrl+N | Switch to Ingest tab |
| Ctrl+Q | Switch to Query tab |
| Ctrl+R | Switch to Remote tab |
| Ctrl+G | Switch to Agents tab |
| Ctrl+S | Switch to Settings tab |
| F1 | Show help modal |
| Ctrl+C | Quit application |
| Ctrl+B | Run build (Ingest) |
| Ctrl+T | Run tagging (Ingest) |
| Ctrl+Shift+R | Refresh stats (Ingest) |
| Ctrl+E | Expand selected citation (Query) |
| Ctrl+O | Open source file (Query) |
| Tab | Autocomplete command |
| Up/Down | Navigate command history |
| Escape | Clear input |

**Agents Tab Shortcuts:**

| Shortcut | Action |
|----------|--------|
| Enter (Topic input) | Launch selected agent with current topic |
| F6 / F7 | Select previous/next running agent |
| F8 | Approve pending interaction |
| F9 | Approve pending interaction for session |
| F10 | Deny pending interaction |
| F11 | Reply to clarification/feedback interaction |

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

### Querying

Query workflows are currently TUI-based. Start the app with:

```bash
lsm
```

Then use the Query tab for interactive questions, citations, mode switching,
notes, and chat mode.

### Agents and Interaction Commands

Agent runtime and interaction commands are available in interactive mode:

```bash
lsm
```

Common commands:

- `/agent start <name> <topic>`
- `/agent list`
- `/agent status [agent_id]`
- `/agent pause [agent_id]`
- `/agent resume [agent_id] [message]`
- `/agent stop [agent_id]`
- `/agent log [agent_id]`
- `/agent interact [agent_id]`
- `/agent approve <agent_id>`
- `/agent deny <agent_id> [reason]`
- `/agent approve-session <agent_id>`
- `/agent reply <agent_id> <message>`
- `/agent queue [agent_id] <message>`
- `/agent select <agent_id>`

### Evaluation Commands

Run retrieval evaluation against a test dataset:

```bash
lsm eval retrieval --profile dense_only
```

Compare against a saved baseline:

```bash
lsm eval retrieval --profile hybrid_rrf --compare my-baseline
```

Save current results as a named baseline:

```bash
lsm eval save-baseline --name my-baseline --profile dense_only
```

List saved baselines:

```bash
lsm eval list-baselines
```

Options for `eval retrieval` and `eval save-baseline`:
- `--profile <name>` - Retrieval profile to evaluate (default: `dense_only`)
- `--dataset <path>` - Path to custom evaluation dataset directory (default: bundled dataset)
- `--compare <name>` - Compare against a saved baseline (retrieval only)

### Migration & Enrichment Commands

Migrate data between backends or run enrichment on an existing database:

```bash
lsm migrate                                  # Auto-detect source, migrate, enrich
lsm migrate --from-db chroma --to-db sqlite  # Explicit backend migration
lsm migrate --from-version v0.7 --to-db sqlite --source-dir /path/to/legacy
lsm migrate --resume                         # Resume interrupted migration
lsm migrate --enrich                         # Enrich existing database (no copy)
lsm migrate --enrich --stage tier1           # Run only tier 1 enrichment stages
lsm migrate --enrich --stage graph           # Run only graph backfill
lsm migrate --enrich --stage tier1 --stage graph  # Combine multiple stages
lsm migrate --skip-enrich                    # Copy only, skip enrichment
lsm migrate --rechunk                        # Auto-rechunk boundary-drifted files
lsm migrate --skip-rechunk                   # Skip rechunking prompt
```

Options:
- `--from-db <backend>` - Source backend (`chroma`, `sqlite`, `postgresql`)
- `--to-db <backend>` - Target backend (`sqlite`, `postgresql`)
- `--from-version <ver>` - Source schema version (e.g. `v0.7`)
- `--resume` - Resume an interrupted migration
- `--enrich` - Run enrichment on existing database (no backend copy)
- `--skip-enrich` - Skip post-migration enrichment
- `--stage <name>` - Run only specified enrichment stage(s). Requires `--enrich`. May be repeated
- `--rechunk` - Automatically rechunk boundary-drifted files
- `--skip-rechunk` - Skip rechunking of boundary-drifted files
- `--source-path`, `--source-collection`, `--source-connection-string` - Source overrides
- `--target-path`, `--target-collection`, `--target-connection-string` - Target overrides
- `--batch-size <n>` - Vector migration batch size (default: 1000)

#### Enrichment Stage Reference

| Stage Name | Alias For | Description |
|------------|-----------|-------------|
| `tier1` | All tier 1 stages | simhash, defaults, node_type, tags |
| `tier2` | All tier 2 stages | heading_path, positions, graph |
| `tier2b` | `clusters` | Cluster rebuild from embeddings |
| `tier3` | `gap_detection` | Boundary drift and missing summaries |
| `simhash` | `enrich_tier1_simhash` | Compute simhash fingerprints |
| `defaults` | `enrich_tier1_defaults` | Backfill version/is_current |
| `node_type` | `enrich_tier1_node_type` | Backfill node_type defaults |
| `tags` | `enrich_tier1_tags` | Backfill root/folder tags |
| `heading_path` | `enrich_tier2_heading_path` | Rebuild heading hierarchy |
| `positions` | `enrich_tier2_positions` | Backfill start_char/end_char |
| `graph` | `enrich_tier2_graph` | Build graph nodes/edges |
| `clusters` | `enrich_tier2_clusters` | Rebuild cluster assignments |
| `gap_detection` | `enrich_tier3_gap_detection` | Detect gaps and drift |

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

# Query from the TUI Query tab
lsm
```

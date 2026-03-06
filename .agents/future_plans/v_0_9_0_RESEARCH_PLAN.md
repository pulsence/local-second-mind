# v0.9.0 UI Rework — Research Plan

## Overview

v0.9.0 is a significant UX overhaul with four distinct concerns:

1. **General housekeeping** — consolidate duplicated logging/path modules and clarify module boundaries.
2. **Web UI** — introduce a browser-based primary interface backed by an HTTP server.
3. **TUI simplification** — strip the TUI down to an admin/operator-focused command-prompt experience.
4. **CLI + TUI unification** — eliminate the separate `lsm.ui.shell` layer; its contents fold into `lsm.ui.tui`.

All decisions from the clarifications phase are recorded in **Section 9** and reflected throughout.
No import-path shims or backwards-compatibility aliases are introduced. Where explicitly documented
(legacy `notes` and remote-chain config keys), one-cycle read-tolerance is kept to avoid hard startup
breaks during migration.

---

## 1. General Housekeeping

### 1.1 Unified Logging Architecture

**Decision:** Replace both `lsm/logging.py` and `lsm/utils/logger.py` with a single, unified logging
system that routes records to multiple sinks: console, rotating file, and an in-process event buffer
consumed by the TUI and Web UI. The logging module **stays at `lsm/logging.py`** — moving it to
`lsm/utils/` was considered but rejected: since the Python logger hierarchy root must be `"lsm"`, the
file belongs at the root of the `lsm` package where its module path matches its purpose.

**Current state — two parallel systems:**

| Module | Purpose | Problem |
|--------|---------|---------|
| `lsm/logging.py` | Python `logging`-based system. Used everywhere via `get_logger`, `setup_logging`. Supports console + optional file output. | No automatic global-folder log routing; file handler is a plain `FileHandler` with no rotation. |
| `lsm/utils/logger.py` | `PlainTextLogger` / `LogVerbosity` helpers currently used by agent log-formatting utilities (`lsm/agents/log_formatter.py`) and re-exported by `lsm/utils/__init__.py`. | Parallel severity model duplicates logging semantics and bypasses the Python `logging` pipeline used elsewhere in the app/UI. |

**Target architecture:** A single Python named logger `"lsm"` with centralized redaction and three handler types:

```
"lsm" (named logger, root of all lsm.* loggers)
├── RedactingFilter       — mandatory sanitization of message + structured extras before any sink
├── ConsoleHandler        — coloured output to stdout (existing, refined)
├── TimedRotatingFileHandler — auto-routed to <GLOBAL_FOLDER>/Logs/lsm-YYYY-MM-DD.log
└── EventBufferHandler    — in-memory ring buffer + subscriber callbacks for UI consumers
```

**Why the logger hierarchy root is `"lsm"` and the file stays in root:**

Python's `logging` module uses dotted-name hierarchy for propagation. When any module anywhere in `lsm.*`
calls `logging.getLogger(__name__)`, the resulting logger (e.g. `lsm.ingest.pipeline`) automatically
propagates records up through `lsm.ingest` → `lsm`. Attaching handlers to the `"lsm"` named logger
captures all records from all subpackages. If the handler were attached to `"lsm.utils"`, records from
`lsm.ingest`, `lsm.query`, `lsm.agents`, etc. would not propagate to it.

Since the named logger root must be `"lsm"`, the file that configures it belongs at `lsm/logging.py`.
Moving it to `lsm/utils/logging.py` would create a mismatch between the file's package path (`lsm.utils`)
and the logger hierarchy root it manages (`lsm`), which is misleading. The file stays in root where
its module name matches the logger hierarchy it owns.

All existing callers keep their imports unchanged: `from lsm.logging import get_logger, setup_logging`.

**`EventBufferHandler` design:**

```python
class EventBufferHandler(logging.Handler):
    """
    In-memory ring buffer that delivers log records to registered callbacks.

    Used by the TUI CommandScreen (live log tail) and the Web UI SSE log
    stream (/api/admin/logs). Subscribers register a callable that receives
    a logging.LogRecord and can format or forward it as needed.
    """
    def __init__(self, maxlen: int = 2000):
        super().__init__()
        self.buffer: deque[logging.LogRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[logging.LogRecord], None]] = []

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self.buffer.append(record)
            for callback in list(self._subscribers):
                try:
                    callback(record)
                except Exception:
                    pass

    def subscribe(self, callback: Callable[[logging.LogRecord], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[logging.LogRecord], None]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass
```

**Updated `setup_logging()` signature:**

```python
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    global_folder: Optional[Path] = None,
    event_buffer_maxlen: int = 2000,
) -> EventBufferHandler:
    """
    Configure logging for LSM.

    When global_folder is provided and log_file is not, automatically routes
    to <global_folder>/Logs/lsm-YYYY-MM-DD.log using TimedRotatingFileHandler
    (midnight rotation, 30-day retention).

    Returns the EventBufferHandler instance so callers can subscribe to records.
    """
```

**Module-level singleton:** A module-level `_event_buffer: EventBufferHandler` is created at import time
so that callers can `from lsm.logging import get_event_buffer` to subscribe before `setup_logging()`
is called. The handler is registered into the logger tree on `setup_logging()`.

**Centralized log redaction (mandatory):**

Redaction happens inside `lsm/logging.py` before any handler emits a record. This guarantees parity
across all sinks (console, file, TUI/web event buffer) and avoids sink-specific redaction drift.

```python
class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # redact formatted message
        sanitized = redact_secrets(record.getMessage())
        record.msg = sanitized
        record.args = ()

        # redact known structured extras if present
        for key in ("prompt", "raw_response", "action_arguments", "extra_json"):
            if hasattr(record, key):
                setattr(record, key, redact_secrets(str(getattr(record, key))))
        return True
```

Attach `RedactingFilter` to the `"lsm"` logger during `setup_logging()` so every downstream handler
receives already-redacted records.

**`lsm/utils/logger.py` deletion:** Code-verified grep confirms that `PlainTextLogger` and
`create_plaintext_logger` are **dead code** — zero callers anywhere in the codebase. Only
`LogVerbosity` and `normalize_verbosity` are actively imported (by `lsm/agents/log_formatter.py`).
The re-exports in `lsm/utils/__init__.py` export all four symbols but the unused two have no
consumers. Migration steps are:
1. Replace `LogVerbosity` and `normalize_verbosity` usage in `lsm/agents/log_formatter.py` with standard logging levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`) and direct level comparisons.
2. Remove all logger re-exports (`LogVerbosity`, `PlainTextLogger`, `create_plaintext_logger`, `normalize_verbosity`) from `lsm/utils/__init__.py`.
3. Delete `lsm/utils/logger.py`.

No migration needed for `PlainTextLogger` or `create_plaintext_logger` — they are unused dead code.

**Log persistence:** `ensure_global_folders()` (now in `lsm/utils/paths.py`) also creates
`<GLOBAL_FOLDER>/Logs/`. `setup_logging()` is called early in both startup paths (`lsm` web-server mode
and `lsm cli` TUI mode), passing `global_folder` from config so the file sink is always active.

**`configure_logging_from_args()` removal:**

The existing `configure_logging_from_args(verbose, log_level, log_file)` wrapper in `lsm/logging.py`
is a convenience shim around `setup_logging()`. With the updated `setup_logging()` signature (adding
`global_folder` and `event_buffer_maxlen`), `configure_logging_from_args` would need to grow matching
parameters — but at that point it adds no value. Instead, replace all callsites of
`configure_logging_from_args` with direct `setup_logging()` calls and delete the wrapper.

**Callsite update in `lsm/__main__.py`:**
```python
# Before (line 509)
configure_logging_from_args(verbose=args.verbose, log_level=args.log_level, log_file=args.log_file)

# After
level = "DEBUG" if args.verbose else (args.log_level or "INFO")
setup_logging(level=level, log_file=args.log_file, global_folder=config.global_folder)
```

**Migration impact:**
- Update `lsm/logging.py` in-place: add `EventBufferHandler`, `TimedRotatingFileHandler`, updated `setup_logging()` signature. No file move.
- Delete `configure_logging_from_args()` from `lsm/logging.py`; update `lsm/__main__.py` to call `setup_logging()` directly.
- Add logger-level `RedactingFilter` in `lsm/logging.py`; all sinks consume already-redacted records.
- Update `lsm/agents/log_formatter.py` to stop importing from `lsm.utils.logger`
- Remove logger re-exports from `lsm/utils/__init__.py`
- Delete `lsm/utils/logger.py`
- Add log-file smoke test: assert `<GLOBAL_FOLDER>/Logs/` is created and written to

---

### 1.2 Path Consolidation

**Decision:** Merge all path utilities into `lsm/utils/paths.py`. Delete `lsm/paths.py` entirely.
All import sites update to `from lsm.utils.paths import ...`. No alias, no shim.

**Current split:**

| Module | Functions |
|--------|-----------|
| `lsm/paths.py` | `get_global_folder`, `get_chats_folder`, `get_mode_chats_folder`, `get_notes_folder`, `resolve_relative_path`, `ensure_global_folders` |
| `lsm/utils/paths.py` | `resolve_path`, `resolve_paths`, `canonical_path`, `safe_filename` |

**After merge — `lsm/utils/paths.py`** contains all ten functions. (`get_notes_folder` is retained
until the notes convergence step removes it — see §7 step 12.)

**Known import sites for `lsm.paths` (code-verified):**
- `lsm/config/models/lsm_config.py` — `ensure_global_folders`
- `lsm/config/models/global_config.py` — `get_global_folder`
- `lsm/remote/storage.py` — `get_global_folder`
- `lsm/remote/oauth.py` — `get_global_folder`
- `lsm/query/api.py` — `get_mode_chats_folder`
- `lsm/query/session.py` — `get_chats_folder`
- `lsm/query/notes.py` — `get_notes_folder`
- `tests/test_paths.py` — all legacy `lsm.paths` imports

A global grep (`from lsm.paths` and `import lsm.paths`) confirms the full list before migration.

**`ensure_global_folders()` expansion:**
After the merge, `ensure_global_folders()` also creates the `Logs/` subdirectory:

```python
def ensure_global_folders(global_folder: Optional[str | Path] = None) -> None:
    root = get_global_folder(global_folder)
    for folder in (
        root,
        root / "Chats",
        root / "Logs",
    ):
        folder.mkdir(parents=True, exist_ok=True)
```

---

### 1.3 `lsm.vectordb` — Dependency Boundary Fix (rename cancelled)

**Revised decision:** The previously-planned rename `lsm.vectordb` → `lsm.db.vectordb` is **cancelled**.
Deep investigation of the actual import graph reveals that `lsm.db` imports from `lsm.vectordb` in
four distinct places for different reasons. Nesting `vectordb` inside `lsm.db` would worsen
the problem; the correct fix is to eliminate each upward import individually.

**Intended dependency direction:**

```
lsm.vectordb  ──imports──▶  lsm.db   (providers use connection, schema, transaction utilities)
lsm.db        ─────────────────────   (never imports from lsm.vectordb — zero upward dependency)
```

`lsm.vectordb` is a **peer** of `lsm.db`, not a child. `lsm.db` is the foundational SQLite layer;
`lsm.vectordb` is an application layer that builds on top of it.

**Why the rename makes it worse:** Placing `lsm.vectordb` inside `lsm.db` does not remove the
circular references — it collapses both sides of the circle into a single package where modules
import each other within the same namespace, which is harder to reason about and harder to untangle
in the future.

---

**Deep analysis — the four upward dependencies in `lsm.db`:**

#### Site 1: `lsm/db/__init__.py` — convenience re-export

```python
# lsm/db/__init__.py line 3
from lsm.vectordb.factory import create_vectordb_provider
```

**Why it exists:** `lsm.db.__init__` acts as a facade, re-exporting `create_vectordb_provider` so
callers can write `from lsm.db import create_vectordb_provider` instead of the longer
`from lsm.vectordb.factory import create_vectordb_provider`.

**Impact:** This is a pure convenience re-export — `lsm.db` does not actually *use* the factory
itself. It only re-exports it for consumer convenience.

**Fix:** Remove the re-export. Update all callsites that relied on `from lsm.db import
create_vectordb_provider` to import directly from `lsm.vectordb.factory`. This is a mechanical
grep-and-replace with no logic change.

---

#### Site 2: `lsm/db/connection.py` — isinstance guard in `_is_provider_instance()`

```python
# lsm/db/connection.py line 115 (inside function body)
def _is_provider_instance(vectordb: Any) -> bool:
    try:
        from lsm.vectordb.base import BaseVectorDBProvider
        if isinstance(vectordb, BaseVectorDBProvider):
            return True
    except ImportError:
        pass
    return hasattr(vectordb, "name") and hasattr(vectordb, "config")
```

**Why it exists:** `connection.py` needs to distinguish provider instances from config objects. It
tries an `isinstance` check and falls back to duck-typing if the import fails.

**Key observation:** The code *already has* a duck-type fallback that is sufficient. `BaseVectorDBProvider`
is abstract; all concrete implementations have `.name` and `.config` attributes. The `isinstance`
check adds no safety that the duck-type check does not already provide.

**Fix:** Remove the try/except block entirely. Keep only the duck-typing line:

```python
def _is_provider_instance(vectordb: Any) -> bool:
    return hasattr(vectordb, "name") and hasattr(vectordb, "config")
```

No behavior change intended — the duck-type path already exists as fallback.

---

#### Site 3: `lsm/db/connection.py` — indirect factory import through `lsm.db` facade

```python
# lsm/db/connection.py lines 15-19
def create_vectordb_provider(config: Any) -> Any:
    from lsm.db import create_vectordb_provider as _create_vectordb_provider
    return _create_vectordb_provider(config)
```

**Why it exists:** `connection.py` centralizes "provider from config" behavior and forwards to the
historical `lsm.db` facade.

**Impact:** Once the facade re-export is removed (Site 1 fix), this indirection becomes fragile and
keeps `connection.py` coupled to `lsm.db.__init__`.

**Fix:** Either remove the wrapper and import the factory directly at callsites, or keep the wrapper
but switch it to a direct lazy import from `lsm.vectordb.factory`:

```python
def create_vectordb_provider(config: Any) -> Any:
    from lsm.vectordb.factory import create_vectordb_provider as _factory
    return _factory(config)
```

---

#### Site 4: `lsm/db/migration.py` — provider instantiation during migration

```python
# lsm/db/migration.py line 22 (module level)
from lsm.vectordb import create_vectordb_provider

# Used inside:
def _provider_from_source(source, source_config):
    if source == MigrationSource.CHROMA:
        from lsm.vectordb.chromadb import ChromaDBProvider  # already lazy!
        ...
    return create_vectordb_provider(config)

def _provider_from_target(target, target_config):
    return create_vectordb_provider(config)
```

**Why it exists:** The migration system needs to create vector database provider instances for both
the source and target backends. This is a genuine functional use — unlike the re-export above, the
migration module actually calls `create_vectordb_provider()`.

**Key observation:** The Chroma import (line 902) is already done lazily inside the function body
as a conditional import. The top-level `create_vectordb_provider` import is the odd one out — it
violates the pattern the migration module itself already uses for the Chroma case.

**Fix:** Convert to a lazy import inside each function that needs it, matching the existing Chroma
pattern:

```python
# lsm/db/migration.py — remove top-level import, use lazy imports instead

def _provider_from_source(source, source_config):
    if source == MigrationSource.CHROMA:
        from lsm.vectordb.chromadb import ChromaDBProvider
        config = _to_vectordb_config(source_config, provider_hint="chromadb")
        return ChromaDBProvider(config)
    from lsm.vectordb.factory import create_vectordb_provider  # lazy
    return create_vectordb_provider(_to_vectordb_config(source_config))

def _provider_from_target(target, target_config):
    from lsm.vectordb.factory import create_vectordb_provider  # lazy
    return create_vectordb_provider(_to_vectordb_config(target_config))
```

Lazy imports at function scope are imported once per process (Python caches modules). There is no
performance cost beyond the first call. The pattern is already used throughout the codebase.

---

**Net result of all four fixes:**

| Module | Before | After |
|--------|--------|-------|
| `lsm/db/__init__.py` | Module-level re-export of `create_vectordb_provider` | Re-export removed; callers import from `lsm.vectordb.factory` directly |
| `lsm/db/connection.py` | (a) try/except `isinstance` check using `lsm.vectordb.base`; (b) factory helper imports `lsm.db` facade | (a) Duck-type only; (b) direct lazy import from `lsm.vectordb.factory` (or direct callsite import) |
| `lsm/db/migration.py` | Module-level import of `create_vectordb_provider` | Lazy function-scope imports; matches existing Chroma pattern |

`lsm.vectordb` is unchanged in location. `lsm.db` becomes a pure foundational layer with no
upward imports. The circular dependency is eliminated.

**Migration steps:**
1. `lsm/db/__init__.py` — delete the `create_vectordb_provider` import and `__all__` entry.
2. Grep `from lsm.db import create_vectordb_provider` across the entire codebase; update each
   callsite to `from lsm.vectordb.factory import create_vectordb_provider` (currently includes
   `lsm/ui/tui/app.py`).
3. `lsm/db/connection.py` — replace `_is_provider_instance()` body with the duck-type line only.
4. `lsm/db/connection.py` — update `create_vectordb_provider()` wrapper to import from
   `lsm.vectordb.factory` directly (or inline factory import at each callsite and remove wrapper).
5. `lsm/db/migration.py` — delete top-level import; add lazy imports inside `_provider_from_source()`
   and `_provider_from_target()`.
6. Run full test suite; verify no `ImportError` or circular import warnings.

**Integrated user feedback — if circular dependencies are removed, why not move `vectordb` under `lsm.db` anyway?**

Even after cycle removal, package placement still communicates architecture and ownership:
- `lsm.db` is the foundational SQL/storage utility layer.
- `lsm.vectordb` is an application abstraction layer that *depends on* `lsm.db`.
- Nesting `vectordb` under `lsm.db` implies the inverse (or same-layer ownership), which is misleading.
- Keeping them as peers preserves a clean dependency rule: `vectordb` may import `db`, but `db` must never import `vectordb`.

---

### 1.4 `lsm.finetune` — Resource Requirements Documentation

`lsm.finetune` stays at `lsm.finetune` — no rename, no move.

It provides:
- `extract_training_pairs(conn)` — extracts heading→chunk pairs from the corpus.
- `finetune_embedding_model(pairs, base_model, output_path, epochs)` — Sentence-Transformers fine-tuning
  with `MultipleNegativesRankingLoss`.
- `register_model`, `set_active_model`, `get_active_model`, `list_models`, `delete_model` — SQLite model registry (`lsm/finetune/registry.py`).

Exposed via `lsm finetune train|list|activate|inventory` CLI subcommands.

**Training pair extraction — critical requirement:**

Training pairs are `(heading, chunk_text)` tuples. The system only extracts pairs where:
- `is_current = 1` (current version)
- `node_type = 'chunk'`
- `heading` is non-null and non-empty
- Chunk content length ≥ 50 characters

**If your corpus has no chunks with headings, you get zero training pairs and the run fails.**
Headings are set by the structure chunker when documents have explicit section headings (Markdown `#`,
Word heading styles, PDF headings detected by font size). Fixed-chunking strategy rarely produces headings.
Documents with heavy structure (technical docs, manuals, reports) produce the most pairs.

**Minimum data requirements research (guideline ranges, not hard thresholds):**

| Pairs | Expected outcome |
|-------|-----------------|
| < 20  | Model trains but likely overfits severely. Results worse than base model for out-of-sample queries. Not recommended. |
| 20–100 | Marginal domain adaptation. May help for very narrow jargon-heavy corpora. High overfitting risk. Use more epochs (8–10) and smaller batch size (8). |
| 100–500 | Meaningful domain adaptation for specialised corpora. Sweet spot for personal knowledge bases. 3–5 epochs, batch size 16. |
| 500–2000 | Good generalisation. Expected measurable improvement on retrieval eval benchmarks. Standard settings (3 epochs, batch size 16). |
| 2000+ | Full-quality fine-tune. Benefit plateaus above ~5000 pairs for MiniLM-scale models. |

**Compute requirements:**

| Hardware | ~500 pairs, 3 epochs | ~2000 pairs, 3 epochs |
|----------|---------------------|----------------------|
| CPU (modern) | 2–5 minutes | 10–25 minutes |
| GPU (RTX 3060 or equivalent) | 15–30 seconds | 1–3 minutes |
| GPU (T4 / Colab free tier) | 20–45 seconds | 2–4 minutes |

CPU training works and is the expected default for a local tool. For users with a GPU, sentence-transformers
automatically uses CUDA if available (respects the `device` config setting).

**Practical guidance to document:**

1. Run `lsm ingest build` with `structure` chunking strategy (default) before fine-tuning.
2. Check pair count with `lsm finetune train --dry-run` (to be added if not already present).
3. If < 100 pairs: add more structured documents or switch to structure chunking strategy.
4. If overfitting is suspected (retrieval eval degrades after fine-tune): reduce epochs or discard model.
5. Fine-tuned model applies only to the embedding step — the LLM synthesis step is unaffected.
6. After `lsm finetune activate`, run `lsm eval retrieval` and compare against saved baseline to confirm improvement.

**Action:** Create `docs/user-guide/FINETUNE.md` documenting these requirements and the recommended workflow.
The CLI `lsm finetune train` help text should include a one-line hint about required pair count.

### 1.5 Embedding Model Inventory Visibility

Deep gap analysis identified a remaining visibility problem: current model surfaces focus on either
`global.embed_model` (single configured model) or `lsm_embedding_models` (fine-tuned registry rows),
but there is no single "what embedding models are available on this machine right now" view.

Required inventory sources for v0.9:
- **Configured model**: `config.global_settings.embed_model` (currently selected runtime model).
- **Fine-tuned registry**: all rows in `lsm_embedding_models` (`model_id`, `base_model`, `path`,
  `dimension`, `is_active`).
- **Well-known catalog**: `WELL_KNOWN_EMBED_MODELS` (dimension-known candidates users can pick).
- **Local install/cached presence**: best-effort filesystem detection for local model paths and
  Hugging Face cache entries (no network calls).

Inventory row contract:
- `model_id`
- `source` (`configured`, `registry`, `catalog`)
- `dimension` (nullable when unknown)
- `is_active_registry`
- `is_configured_default`
- `is_installed_locally`
- `local_path` (nullable)
- `load_check` (`ok`, `missing`, `error`, `unknown`)

Operational rules:
- Inventory checks are **offline-only** (filesystem + DB); no remote model listing calls.
- Missing paths for active/registered models are clearly flagged.
- Active registry model and configured model can diverge; UI shows both states explicitly.
- `Set Active` remains registry-scoped; changing configured default remains config-scoped.
- Registry cleanup is explicit: stale fine-tuned rows can be removed via
  `DELETE /api/admin/finetune/models/{model_id}`.

---

## 2. Web UI

### 2.1 Goals

- **Primary everyday interface** for query and agent interaction. The browser replaces the TUI for daily use.
- Standardised chat layout: scrollable conversation panel, fixed input at bottom, collapsible sidebar.
- Six top-level sections: **Query** (chat), **Ingest**, **Agents**, **Settings**, **Admin**, **Help**.
- Server exposes a REST + SSE API usable by a future Obsidian plugin.
- Local-first defaults: binds to `127.0.0.1` (localhost) by default, with optional LAN exposure via `server.expose_to_lan`.
- Authentication posture in v0.9.0: no user-account system; LAN mode requires access-token protection on non-GET API routes.
- No JavaScript build toolchain — everything is server-rendered HTML with HTMX for interactivity.
- **Dark mode** with system-preference detection as the default, user-overrideable via a toggle,
  preference persisted in `localStorage`.
- **Markdown rendering** in chat responses — assistant messages render formatted output (headings,
  bold, lists, code blocks, inline code).
- **In-browser documentation** — all user guide docs from `docs/` are accessible via the Help screen,
  rendered as HTML.

---

### 2.2 Confirmed Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Backend | **FastAPI** | Async-native (matches existing asyncio patterns), automatic OpenAPI docs for Obsidian plugin, first-class `StreamingResponse` for SSE. |
| Server | **Uvicorn** (standard extras) | ASGI server required by FastAPI. |
| Templates | **Jinja2** | Tight FastAPI integration via `Jinja2Templates`; enables server-rendered HTML fragments for HTMX. |
| Interactivity | **HTMX** | No build toolchain, vendor-able as a single JS file, handles form submission, out-of-band swaps, and SSE connections. |
| Streaming | **Server-Sent Events (SSE)** | One-directional (server → client), HTTP/1.1 compatible, browser-native `EventSource`, sufficient for LLM token streaming and log tailing. No WebSockets in v0.9.0. |
| CSS | **Custom hand-written CSS** with custom properties | Local tool with one user — no Tailwind/Bootstrap dependency. CSS custom properties enable dark mode theming without JS rewriting styles. |
| Markdown (server) | **`mistune` ≥ 3.0** | Pure-Python markdown renderer; used to pre-render stored conversation messages and docs pages. |
| Markdown (streaming) | **`marked.js`** (vendored) | Client-side markdown rendering for progressive streaming output — renders tokens as they arrive without a round-trip. |

**New `pyproject.toml` dependencies:**
```toml
"fastapi>=0.115",
"uvicorn[standard]>=0.30",
"jinja2>=3.1",
"python-multipart>=0.0.9",   # Required for FastAPI form parsing
"mistune>=3.0",              # Server-side markdown rendering
"nh3>=0.2.14",               # HTML sanitisation after markdown rendering (bleach is deprecated)
```

HTMX, its SSE extension, and `marked.js` are **vendored** as static files under `lsm/ui/web/static/js/`.
This avoids CDN dependencies for a local tool and pins versions.

---

### 2.3 HTMX + Jinja2 Architecture

#### 2.3.1 Template Inheritance Structure

```
lsm/ui/web/templates/
├── base.html          # Shell: <html>, <head>, sidebar nav, dark-mode toggle, main content slot
├── query.html         # extends base.html — chat conversation UI
├── ingest.html        # extends base.html — ingest controls
├── agents.html        # extends base.html — agent management
├── settings.html      # extends base.html — config form
├── admin.html         # extends base.html — power-user admin panel
├── help.html          # extends base.html — documentation browser
└── fragments/         # Partial templates returned by HTMX-targeted endpoints
    ├── query_submitted.html
    ├── ingest_progress.html
    ├── agent_card.html
    ├── health_table.html
    ├── stats_table.html
    └── doc_page.html
```

`base.html` defines:
- `<head>` with HTMX script tag, SSE extension script, CSS link, theme-init script (inline, see §2.7).
- Left sidebar with navigation links and active-page highlighting.
- Dark mode toggle button (sun/moon icon) in the sidebar footer.
- `{% block main %}{% endblock %}` content slot.
- A global notification area for out-of-band status messages (`id="notifications"`).

Each screen template extends `base.html` and fills the `main` block.

#### 2.3.2 HTMX Core Patterns Used

**Pattern 1 — Form submission returning an HTML fragment:**
```html
<form hx-post="/api/ingest/build"
      hx-target="#ingest-output"
      hx-swap="innerHTML"
      hx-indicator="#ingest-spinner">
  <button type="submit">Build Index</button>
</form>
<div id="ingest-output"></div>
<span id="ingest-spinner" class="htmx-indicator">Building...</span>
```
The POST endpoint returns an HTML fragment (not a full page), which HTMX inserts into `#ingest-output`.

**Pattern 2 — SSE streaming via POST → fragment-with-SSE-div:**

The query form POSTs to an endpoint that:
1. Creates a `query_id` and stores the pending query.
2. Returns an HTML fragment containing the user message bubble AND an HTMX-SSE div:

```python
@router.post("/api/query", response_class=HTMLResponse)
async def submit_query(query: str = Form(...), mode: str = Form("grounded")):
    query_id = str(uuid4())
    _pending_queries[query_id] = (query, mode)
    return templates.TemplateResponse("fragments/query_submitted.html", {
        "request": request, "query_id": query_id, "query": query
    })
```

`fragments/query_submitted.html`:
```html
<div class="message user-message">{{ query | e }}</div>
<div class="message assistant-message"
     hx-ext="sse"
     sse-connect="/api/query/stream/{{ query_id }}"
     sse-swap="message"
     id="response-{{ query_id }}">
  <span class="streaming-cursor">▌</span>
</div>
<script>
  // Invoked when event: done fires — remove cursor, trigger marked.js render
  document.getElementById("response-{{ query_id }}")
    .addEventListener("htmx:sseClose", function(e) {
      const el = e.target;
      el.querySelector(".streaming-cursor")?.remove();
      el.innerHTML = marked.parse(el.dataset.rawContent || el.innerText);
    });
</script>
```

HTMX mounts the fragment into the conversation `div`, automatically opens the SSE connection, and appends
incoming `message` events to `#response-{query_id}`.

**Pattern 3 — Out-of-band swaps for secondary elements:**
FastAPI includes `HX-Trigger` response headers to update secondary elements (cost display, chunk count in
sidebar) without a separate request:
```python
response.headers["HX-Trigger"] = json.dumps({"updateCost": {"value": "$0.0032"}})
```

**Pattern 4 — Polling for agent status:**
The Agents screen refreshes its agent list every 5 seconds using `hx-trigger="every 5s"`:
```html
<div hx-get="/api/agents"
     hx-trigger="every 5s"
     hx-target="#agent-list"
     hx-swap="innerHTML">
```

#### 2.3.3 SSE Event Format

All SSE endpoints use a consistent event format:

```
# Token streaming (query)
data: {"type":"token","content":"The "}
data: {"type":"token","content":"answer "}

# Citation emitted during response
data: {"type":"citation","id":"abc123","title":"mydoc.pdf","page":"3","score":0.87}

# Completion with metadata
data: {"type":"done","cost":{"tokens_in":512,"tokens_out":120,"cost_usd":0.0032}}
event: done
data: {}

# Error
event: error
data: {"message": "LLM provider timeout"}

# Progress (ingest/eval/migrate)
data: {"type":"progress","current":150,"total":1234,"message":"Processing file.pdf"}
data: {"type":"progress","current":1234,"total":1234,"message":"Done"}
event: done
data: {}
```

All streaming endpoints close with `event: done\ndata: {}\n\n`. Clients listen for this to finalise UI state.

#### 2.3.4 Static Assets Layout

```
lsm/ui/web/static/
├── css/
│   └── main.css           # All custom styles (sidebar, chat bubbles, cards, dark/light themes)
└── js/
    ├── htmx.min.js        # Vendored HTMX (currently 2.x)
    ├── htmx-sse.js        # Vendored HTMX SSE extension
    └── marked.min.js      # Vendored marked.js for client-side markdown rendering
```

FastAPI serves these via:
```python
app.mount("/static", StaticFiles(directory=static_dir), name="static")
```

---

### 2.4 Web Server Lifecycle

#### 2.4.1 Entry Points

| Command | Effect |
|---------|--------|
| `lsm` (no args) | Starts the FastAPI web server. Prints `Listening on http://{host}:{port}` once ready. |
| `lsm cli` | Starts the Textual admin TUI. |
| `lsm ingest build` | Single-shot CLI batch operation (unchanged). |
| `lsm db prune` | Single-shot CLI batch operation (unchanged). |
| `lsm eval retrieval` | Single-shot CLI batch operation (unchanged). |
| *(all other subcommands)* | Single-shot CLI operations (unchanged). |

`lsm` with no args is now the web server entry point. `run_tui()` is only called from `lsm cli`.

#### 2.4.2 Server Startup Sequence

```python
# In __main__.py — no-subcommand handler becomes:
if not args.command:
    from lsm.ui.web.server import run_server
    return run_server(config)
```

`run_server()` in `lsm/ui/web/server.py`:
1. Call `setup_logging(global_folder=config.global_folder)` — establishes log file and event buffer.
2. Create the FastAPI app via `create_app(config)`.
3. Print `Starting LSM server...` to stdout.
4. Start Uvicorn: `uvicorn.run(app, host=config.server.host, port=config.server.port, log_level=config.server.log_level)`.
5. On bind success, Uvicorn's startup event fires — hook it to print `Listening on http://{host}:{port}`.

The TUI and web server are always **separate OS processes** — no shared in-process state. They coordinate
exclusively through the shared SQLite database file and the shared config file on disk.

#### 2.4.3 Server Config Object

New `ServerConfig` dataclass added to `lsm/config/models/server.py`:

```python
@dataclass
class ServerConfig:
    """HTTP server configuration."""
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "info"   # Uvicorn log level string
    expose_to_lan: bool = False
    require_access_token: bool = True
    access_token: str = ""    # generated on first LAN-enabled startup if empty

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError(f"server.port must be 1-65535, got {self.port}")
        valid_levels = {"critical", "error", "warning", "info", "debug", "trace"}
        if self.log_level not in valid_levels:
            raise ValueError(f"server.log_level must be one of {valid_levels}")
        if self.expose_to_lan and self.host == "127.0.0.1":
            # If LAN exposure is enabled, bind all interfaces by default.
            self.host = "0.0.0.0"
        if not self.expose_to_lan:
            # Access token only enforced for LAN mode.
            self.require_access_token = False
```

`LSMConfig` gains a new field:
```python
server: ServerConfig = field(default_factory=ServerConfig)
```

`config.json` example:
```json
"server": {
    "host": "127.0.0.1",
    "port": 8080,
    "log_level": "info",
    "expose_to_lan": false,
    "require_access_token": true,
    "access_token": ""
}
```

Config loader follows the existing `load_*` pattern: `raw.get("server", {})` → `ServerConfig(**raw_server)`
with defaults.

When `expose_to_lan=true`, startup logs print both bind address and detected LAN URL candidates
(for example `http://192.168.1.24:8080`) so users can open the UI from other local devices.
If `require_access_token=true` and `access_token` is empty, startup generates a random token, persists it,
and logs a one-time "copy this token" message for local-network clients.

**LAN-mode safety requirement (new):**
- In LAN mode, all non-GET API routes require `X-LSM-Access-Token` (or `Authorization: Bearer ...`).
- Browser HTMX requests include token via a base-template `hx-headers` hook.
- SSE/EventSource requests are GET-based and cannot set custom headers in the standard browser API; v0.9
  keeps read endpoints unauthenticated while requiring token protection for all state-changing routes.
- This mitigates same-network drive-by writes and browser CSRF-style form posts when server is exposed.

---

### 2.5 Chat History — Database-Backed Storage

**Recommendation: Move to SQLite DB storage.**

The user flagged that conversations will eventually be included in the query pipeline (indexed for
retrieval). DB storage is the only approach that makes this tractable.

**Comparison:**

| Criterion | Flat Markdown Files (current) | SQLite DB (proposed) |
|-----------|------------------------------|---------------------|
| Human-readable without tooling | Yes | No |
| Queryable | No — requires grep/parsing | Yes — full SQL + FTS |
| RAG indexable | Requires separate parse step | Direct embed of message content |
| Retry variants | Hard/impossible to model | First-class variant rows with active-selection |
| Archive/search | Ad-hoc file naming only | Explicit archive flags + indexed search |
| Atomic writes | No — file append is not atomic | Yes — SQLite transactions |
| Consistent with architecture | Partial (everything else is DB) | Fully consistent |
| Scale for hundreds of convos | Moderate (many small files) | Excellent |

**Schema:**

```sql
CREATE TABLE lsm_conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT,              -- Auto-generated from first query or user-set
    mode        TEXT NOT NULL,     -- grounded/insight/hybrid/chat
    llm_provider TEXT,             -- optional per-conversation override
    llm_model    TEXT,             -- optional per-conversation override
    branched_from_conversation_id TEXT REFERENCES lsm_conversations(id),
    branched_from_message_id TEXT,
    is_archived INTEGER NOT NULL DEFAULT 0, -- 0=false, 1=true
    archived_at TEXT,              -- ISO-8601, nullable
    created_at  TEXT NOT NULL,     -- ISO-8601
    updated_at  TEXT NOT NULL
);

CREATE TABLE lsm_messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES lsm_conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content         TEXT NOT NULL,   -- raw markdown (not pre-rendered)
    parent_user_message_id TEXT REFERENCES lsm_messages(id) ON DELETE CASCADE,
        -- for assistant messages, points at the user message they answer
    variant_group_id TEXT,           -- non-null for assistant variants
    variant_index    INTEGER,        -- 1,2,3... within a variant group
    is_active_variant INTEGER NOT NULL DEFAULT 1, -- exactly one active assistant variant per group
    edited_at        TEXT,           -- ISO-8601, nullable (for user/assistant edits)
    sources_json    TEXT,            -- JSON array of {id, title, file_path, page, score}
    cost_info_json  TEXT,            -- JSON: {tokens_in, tokens_out, cost_usd}
    created_at      TEXT NOT NULL
);

CREATE INDEX lsm_messages_conversation_id ON lsm_messages(conversation_id);
CREATE INDEX lsm_messages_parent_user_id ON lsm_messages(parent_user_message_id);
CREATE INDEX lsm_messages_variant_group ON lsm_messages(variant_group_id);
CREATE INDEX lsm_conversations_updated_at ON lsm_conversations(updated_at DESC);
CREATE INDEX lsm_conversations_archived_updated
    ON lsm_conversations(is_archived, updated_at DESC);

CREATE UNIQUE INDEX lsm_messages_one_active_variant
    ON lsm_messages(variant_group_id, is_active_variant)
    WHERE role = 'assistant' AND variant_group_id IS NOT NULL AND is_active_variant = 1;

CREATE UNIQUE INDEX lsm_messages_variant_group_index_unique
    ON lsm_messages(variant_group_id, variant_index)
    WHERE role = 'assistant' AND variant_group_id IS NOT NULL;

CREATE VIRTUAL TABLE lsm_messages_fts USING fts5(
    conversation_id UNINDEXED,
    content,
    sources_text
);

CREATE VIRTUAL TABLE lsm_conversations_fts USING fts5(
    conversation_id UNINDEXED,
    title,
    mode
);

-- Keep FTS tables synchronized with lsm_messages / lsm_conversations writes.
-- (via INSERT/UPDATE/DELETE triggers in migration DDL)
```

Message `content` is stored as raw markdown. When displayed in the Web UI, server-side `mistune` renders
it to HTML before embedding it in the Jinja2 template. This keeps the DB content portable (not tied to
any HTML structure) and allows re-rendering if the markdown library or CSS changes.

**Mutation semantics for new chat controls:**
- Retry latest assistant response creates a new assistant row with the same `parent_user_message_id` and
  `variant_group_id`, increments `variant_index`, and makes the new row active (`is_active_variant=1`).
- Variant retention has no cap in v0.9 (single-user local system; storage impact acceptable).
- Only the active assistant variant is included when building next-turn context.
- Delete latest message is hard delete (no soft-delete tombstone in v0.9):
  - If latest is user: delete user row; cascading delete removes assistant variants tied to it.
  - If latest is assistant: delete the entire assistant variant group for that user turn.
- Edit latest user message updates that user row, hard-deletes assistant variants tied to it, then
  automatically regenerates a fresh assistant response.
- Edit latest assistant message updates that assistant row in-place (`edited_at` set) and is used in
  future context immediately.
- Branch chat creates a new conversation copied from the current chat up to a chosen message (inclusive),
  sets `branched_from_*` metadata, and starts an independent continuation path.

**Conversation search + archive behavior:**
- Sidebar default query: only `is_archived=0` conversations.
- Archive sets `is_archived=1`, `archived_at=now`; unarchive clears both.
- Sidebar has a dedicated `Archived` view (`/api/conversations?view=archived`) in addition to search.
- Search queries union-rank `lsm_messages_fts` + `lsm_conversations_fts` and include:
  - message content
  - conversation title
  - mode
  - cited-source metadata text
  Archived and non-archived chats are both returned, with archived items visually tagged.
- Delete conversation is permanent row deletion from `lsm_conversations`; `ON DELETE CASCADE` removes
  all messages and variants.

**Migration from flat files:** A one-shot migration reads existing `<GLOBAL_FOLDER>/Chats/` markdown
transcripts, parses them into conversation/message records, and inserts them into the new tables. This
runs automatically on first startup if the new tables are empty but the Chats folder has content (or
triggered explicitly via `lsm migrate --chats` flag on the existing `migrate` subcommand). The flat
files remain on disk as read-only backup until the user explicitly removes them.

The existing `ChatsConfig` is retained for the folder path (migration source), but new conversations
write only to the DB.

#### 2.5.1 Provider conversation-state research ("infinite chat" feasibility)

Research conclusion: "infinite conversation" cannot rely on provider memory alone. LSM must keep a
canonical local transcript in SQLite and treat provider-side state as an optimization.

| Provider | API conversation state | Practical implication for LSM |
|----------|------------------------|-------------------------------|
| OpenAI Responses + Conversations APIs | Supports `previous_response_id` chaining and durable conversation objects; provider-side state is model/provider specific and controlled by provider retention/store semantics | Use OpenAI state IDs as acceleration when available, but keep full local transcript canonical for portability, branch/edit/delete correctness, and provider fallback |
| Anthropic Messages API | Stateless: full message history is sent each request | Always build prompt context from local transcript; optionally use prompt caching for stable prefixes |
| Gemini API (`generateContent` + Interactions) | `generateContent` remains stateless; Interactions API can continue state with `previous_interaction_id` and provider retention windows | Keep local transcript canonical; optionally persist Gemini interaction IDs per active branch and fall back to local replay when IDs are missing/expired |
| OpenRouter Responses API | Explicitly stateless: include prior messages each request; compatible request fields do not guarantee durable server-side memory | Treat as stateless; rely on local transcript and compaction |

Current code partially aligns with this direction:
- `lsm/providers/openai.py` conditionally sends `previous_response_id`.
- `lsm/providers/anthropic.py`, `gemini.py`, `openrouter.py`, and `local.py` explicitly ignore
  `previous_response_id` as unsupported.
- Gemini Interactions-style continuation IDs are not yet wired in current provider adapters.

References:
- OpenAI conversation state guide: <https://platform.openai.com/docs/guides/conversation-state>
- OpenAI Responses API reference (`previous_response_id`): <https://platform.openai.com/docs/api-reference/responses/create>
- OpenAI Conversations API reference (durable conversation objects): <https://platform.openai.com/docs/api-reference/conversations>
- Anthropic Messages examples (stateless model): <https://docs.anthropic.com/en/api/messages-examples>
- Anthropic prompt caching: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>
- Gemini text generation docs: <https://ai.google.dev/gemini-api/docs/text-generation>
- Gemini Interactions docs (`previous_interaction_id`, retention): <https://ai.google.dev/gemini-api/docs/interactions>
- OpenRouter Responses API overview (stateless behavior): <https://openrouter.ai/docs/api-reference/responses-api/overview>

#### 2.5.2 Compaction migration: `lsm.agents` -> `lsm.providers` primitive

Current state:
- `AgentHarness._prepare_messages_from_history()` performs a local heuristic compaction inline:
  `"fresh"` keeps last 6 messages; default path replaces older history with a synthetic system summary.
- Web query paths and future long-running chat flows need the same logic, but currently do not share a
  single primitive.

Planned architecture:
- Add `lsm/providers/compaction.py` as the shared primitive module.
- Introduce provider-facing contracts:

```python
@dataclass
class CompactionPolicy:
    strategy: str              # "fresh" | "summary" | "provider"
    keep_recent: int = 8
    max_messages: int = 12

@dataclass
class CompactionResult:
    messages: list[dict[str, str]]
    summary_text: str | None
    provider_state_id: str | None

def compact_messages(
    *,
    provider: BaseLLMProvider | None,
    messages: list[dict[str, str]],
    policy: CompactionPolicy,
    previous_response_id: str | None = None,
) -> CompactionResult: ...
```

- `BaseLLMProvider` gains an optional hook for provider-native compaction/state continuation:
  `compact_conversation_state(...) -> Optional[str]` (default `None`).
- OpenAI provider can implement this hook against Responses compaction/state APIs.
- Other providers return `None` and fall back to local summarization strategy.

Adoption points:
- `lsm/agents/harness.py`: replace `_prepare_messages_from_history()` logic with the new primitive.
- Web query server (new route/service layer): call the same primitive before LLM invocation.
- Any future background chat workers reuse the same primitive (single policy registry, single tests).

#### 2.5.3 Peer chat UI divergence analysis (ChatGPT + Open WebUI)

| Feature | ChatGPT (documented) | Open WebUI (documented) | LSM v0.9 target |
|---------|-----------------------|--------------------------|-----------------|
| Search chats | Yes (sidebar search; archived chats still searchable) | Yes (history search across titles/content/tags) | Add `Ctrl/Cmd+K` style search + `/api/conversations/search` |
| Archive chats | Yes (hide from main list, retain data) | Yes (archive and archive-all features) | Add archive/unarchive endpoints + dedicated archived view |
| Delete chats | Yes (delete is irreversible from UI perspective) | Yes (delete individual/all chats) | Permanent hard delete |
| Retry response | Yes ("Regenerate"/retry controls) | Yes (regenerate response + regeneration history) | Retry latest assistant response and keep variants |
| Branching | Yes (branch conversation from a message into new chat) | Yes (chat branches/overview) | Support explicit branch-to-new-chat from selected message |
| Message edit/delete controls | Available in product with client variance | Explicit permissions for edit/delete/regenerate | Latest-message edit/delete in v0.9 (deterministic scope) |

Divergence summary:
- LSM is intentionally narrower than Open WebUI RBAC-heavy controls for v0.9 (single-user localhost
  assumptions), but should match core chat ergonomics: retry, edit-last, delete-last, archive, search.
- LSM supports both in-thread assistant variants and explicit branch-to-new-chat so users can choose
  either lightweight rerolls or fully separated conversation trees.

References:
- ChatGPT search history help: <https://help.openai.com/en/articles/10056348-how-do-i-search-my-chat-history-in-chatgpt>
- ChatGPT archive/delete help: <https://help.openai.com/en/articles/8809935-how-chat-retention-works-in-chatgpt>
- ChatGPT release notes (branch conversations): <https://help.openai.com/en/articles/6825453-chatgpt-release-notes>
- Open WebUI history/search docs: <https://docs.openwebui.com/features/chat-conversations/chat-features/history-search>
- Open WebUI features overview: <https://docs.openwebui.com/features>
- Open WebUI permissions (edit/delete/regenerate): <https://docs.openwebui.com/features/access-security/rbac/permissions/>

#### 2.5.4 Server-side cache implications (variants, edits, deletes, branches)

The new mutation features do not remove the value of provider-side caching/chaining, but they require
cache-invalidation rules because provider state is linear while the local transcript can branch.

**Key finding:** local SQLite transcript remains canonical; provider cache IDs are per-path accelerators.
In current provider implementations this is tracked as `previous_response_id` (not
`previous_conversation_id`).

| Operation | Provider-chain impact | Required behavior |
|-----------|------------------------|-------------------|
| Normal linear turn | Safe to continue with prior chain ID (`previous_response_id` where supported) | Keep current fast path |
| Retry variant for same user turn | Creates divergence from same parent context | Generate using parent anchor state; store chain ID per variant |
| Switch active variant | Future turns now follow a different branch | Update conversation active chain pointer to chosen variant |
| Edit latest user | Invalidates downstream assistant variants and chain IDs after that point | Delete downstream variants; reset chain pointer to pre-edit anchor |
| Edit latest assistant | Changes downstream context semantics | Invalidate chain pointer for subsequent turns; next call rebuilds from transcript |
| Delete latest user/assistant group | Removes context nodes | Invalidate chain pointer to nearest surviving ancestor |
| Branch chat | Starts a second lineage | New conversation gets independent chain state map |

Implementation note:
- Add lightweight provider-state table keyed by message variant, e.g.
  `lsm_message_provider_state(message_id, provider, model, state_kind, provider_state_id, parent_state_id, created_at)`.
  (`state_kind` examples: `openai_response_id`, `openai_conversation_id`, `gemini_interaction_id`)
- Query service resolves the active branch, picks matching provider/model state when valid, and falls
  back to transcript replay when not valid.
- For providers without chain IDs, only local compaction/prompt caching applies.

#### 2.5.5 Notes System Convergence (Chats -> Export)

User feedback indicates the standalone Notes flow is now redundant with chat-first workflows. For v0.9:
- Notes folder generation and dedicated note-save workflow are removed from the core flow.
- Users export either:
  - entire conversation transcript, or
  - a single message (user/assistant) with metadata.
- Export replaces previous "save to notes" intent while keeping context-rich outputs tied to chats.

Export formats in scope:
- `markdown` (default, human-readable)
- `json` (structured, machine-readable)

This keeps data model simpler (chat as source of truth) and avoids parallel content systems.

**Removal scope (code-verified):**

The notes system is spread across multiple files. Complete removal requires:
1. **`lsm/query/notes.py`** (424 lines) — the main notes module. Contains `generate_note_content()`,
   `write_note()`, `edit_note_in_editor()`, `resolve_notes_dir()`, template support, wikilinks/backlinks.
   Entire file is deleted.
2. **`NotesConfig`** in `lsm/config/models/modes.py` (lines 148–176) — interleaved with other config
   dataclasses. Surgical removal of the class and its `notes: NotesConfig` field from `LSMConfig`.
3. **`get_notes_folder()`** in `lsm/utils/paths.py` (after path consolidation merge) — remove function.
4. **`lsm/query/notes.py` import of `get_notes_folder`** — already gone once notes.py is deleted.
5. **Config loader** — stop loading `"notes"` section into `NotesConfig`; add backward-tolerant read
   path that logs a deprecation warning if `"notes"` key exists in config.
6. **Config serializer** — stop writing `"notes"` section.
7. **TUI settings** — remove notes-related fields from settings view-model/widgets.
8. **`/note`/`/notes` command handlers** — replace with explicit "use chat export" guidance message.

**Sequencing with path consolidation (§1.2):**

Path consolidation (step 3) merges `get_notes_folder` into `lsm/utils/paths.py`. Notes removal
(step 12) deletes the function. Between steps 3 and 12, the function exists in the merged file and
is used by `lsm/query/notes.py`. This ordering is safe — `get_notes_folder` is included in the
merge and removed later when the notes module itself is deleted.

`ensure_global_folders()` stops creating `Notes/` in step 3 (not step 12), since new installations
should not create the folder. Existing users retain their Notes folder on disk; it is simply no
longer auto-created on startup.

#### 2.5.6 Export Contract and Safety

To avoid ambiguity and data-loss bugs, export payloads are standardized:
- Conversation export includes:
  - conversation metadata (`id`, `title`, `mode`, `llm_provider`, `llm_model`, timestamps, branch lineage)
  - ordered messages on active branch path
  - citation/source metadata per assistant message
- Message export includes:
  - message metadata (`id`, `role`, `created_at`, `edited_at`, variant metadata if assistant)
  - message content
  - attached source/cost metadata if present

File naming:
- Conversation: `chat-{title-or-id}-{timestamp}.{md|json}`
- Message: `chat-message-{message_id}-{timestamp}.{md|json}`

Safety behavior:
- Query parameter `redact=true|false` (default `true`) controls whether export output is passed through
  the same redaction pipeline used by logger sinks.
- `redact=false` is allowed only for local trusted workflows and should show a warning in UI.

#### 2.5.7 Transaction Boundaries for Context Mutations

Retry/edit/delete/variant-switch/branch operations must be atomic to prevent split context state under
concurrent requests (double-clicks, duplicate browser tabs, or slow network retries).

Required server behavior:
- Wrap each mutation in `BEGIN IMMEDIATE ... COMMIT` so only one writer mutates a conversation at a time.
- Validate latest-message preconditions inside the same transaction (not before it).
- Use optimistic conflict checks (`conversation.updated_at` or equivalent version token). If mismatch,
  rollback and return `409 Conflict` with UI refresh hint.
- Update `lsm_messages`, FTS triggers, and provider-state rows in the same transaction.
- For retry, enforce `(variant_group_id, variant_index)` uniqueness and retry the index increment on
  conflict rather than allowing duplicate variant indices.

---

### 2.6 Screen-by-Screen Design

#### 2.6.1 Query Screen — Primary Chat Interface

**URL:** `/` and `/query`

**Layout:**
```
+--sidebar-----------+----main panel---------------------------------+
| [LSM]              |  [conversation history — scrollable]          |
|                    |  ┌────────────────────────────────────────┐   |
| ● Query            |  │ User:  what is X?                      │   |
|   Ingest           |  ├────────────────────────────────────────┤   |
|   Agents           |  │ LSM:   **Answer heading**              │   |
|   Settings         |  │        Body text with *italics* and    │   |
|   Admin            |  │        `code snippets` rendered as HTML│   |
|   Help             |  │ Sources:                               │   |
|                    |  │  ▸ doc-title.pdf  p.3                  │   |
| ─────────────────  |  │  ▸ another-doc.md  §Introduction       │   |
| Mode: grounded ▾   |  └────────────────────────────────────────┘   |
| Cost: $0.0032      |                                               |
|                    |  ─────────────────────────────────────────── |
| [+ New Chat]       |  [input textarea                      ] [▶]  |
| ─────────────────  +-----------------------------------------------+
| Past conversations |
| • 2026-03-01 AI... |
| • 2026-02-28 How.. |
+--------------------+
```

**Features:**
- Mode selector in sidebar — `hx-put="/api/query/mode"` on change; updates sidebar label via OOB swap.
- Model selector in sidebar — per-conversation provider/model override via
  `hx-put="/api/conversations/{id}/model"`.
- Conversation list in sidebar: past conversations loaded via `hx-get="/api/conversations"` on page load.
- Conversation search box in sidebar: `hx-get="/api/conversations/search"` with debounce (`q` query param).
  Searches title/mode/message content/citation metadata.
- Dedicated `Archived` toggle in sidebar: `hx-get="/api/conversations?view=archived"`.
- Query submit: `hx-post="/api/query"` on the form. Returns HTML fragment with user message bubble +
  SSE-connected response div. Fragment is appended to the conversation history via `hx-swap="beforeend"`.
- SSE response streaming: HTMX SSE extension appends tokens to the response div. During streaming,
  raw content is accumulated in a `data-raw-content` attribute. On `event: done`, `marked.js` renders
  the full accumulated markdown to HTML in-place. The streaming cursor is removed.
- **Markdown rendering in streaming:** While streaming, raw text is appended to the div — this provides
  instant visual feedback. On completion, `marked.js` replaces it with properly formatted HTML. The
  brief "raw then formatted" transition is acceptable (similar to how ChatGPT renders post-stream).
- **Markdown in stored messages:** When loading past conversation history, server-side `mistune` renders
  each `content` field to HTML before embedding in the Jinja2 template. Past messages are always
  fully formatted.
- Citation cards: initially collapsed `<details>` elements with chunk title, file path, page number,
  score. Expand inline.
- Cost display: updated per-response via `HX-Trigger` out-of-band swap to `#cost-display` in sidebar.
- New conversation: POSTs to `/api/conversations/new`, clears history panel.
- Latest assistant bubble includes:
  - `Retry` button: requests a new assistant variant for the same preceding user message.
  - Variant picker (`v1`, `v2`, ...): switches active variant for next-turn context.
  - `Edit` button: inline edit of latest assistant response; edited content is used in future context.
  - `Delete` button: permanently deletes the entire latest assistant variant group for that user turn.
- Any message bubble (user or assistant) includes:
  - `Branch` button: creates a new chat branch from that message and switches UI to the new chat.
  - `Export` button: downloads the single message (`markdown`/`json`).
- Latest user bubble includes:
  - `Edit` button: inline edit of latest user message, then auto-regenerate assistant response.
  - `Delete` button: permanently deletes latest user message and all assistant variants attached to it.
- Guardrails for deterministic context mutation:
  - Retry/edit/delete controls are rendered only for the latest mutable message(s).
  - Server rejects stale mutations (409) if the conversation changed after the UI loaded.
  - Next-turn context is built from chronological messages, but for each assistant variant group only the
    `is_active_variant=1` row is included.
- Chat-level controls in sidebar list row:
  - `Archive`: move chat into dedicated archived list view and keep it searchable.
  - `Delete`: permanent chat deletion with confirmation modal.
- Conversation header actions include `Export Chat` (`markdown`/`json`) for whole-thread export.
- Responsive/mobile behavior:
  - Desktop (`>=1024px`): persistent left sidebar + main chat panel.
  - Tablet (`640px-1023px`): collapsible sidebar drawer.
  - Mobile (`<640px`): full-width chat; sidebar opens as overlay; touch-friendly 44px minimum targets.
  - Sticky bottom composer on mobile with safe-area padding and no horizontal overflow.

**Markdown rendering — supported elements:**

| Element | Rendering |
|---------|-----------|
| `**bold**`, `*italic*` | Strong, emphasis tags |
| `# Heading` | h1–h4 (capped at h4 to fit sidebar width) |
| `` `code` `` | Inline code with monospace styling |
| ` ```code block``` ` | Pre-formatted block with optional language label |
| `- list items` | Unordered list |
| `1. numbered` | Ordered list |
| `> blockquote` | Blockquote styling |
| Links `[text](url)` | Rendered as hyperlinks (external links `target="_blank"`) |
| Tables | Styled HTML table |

HTML sanitisation: `mistune` is configured with an allow-list sanitiser to strip any `<script>` tags
or event handler attributes from LLM-generated content before insertion into the DOM. This prevents
prompt-injection XSS attacks where an LLM response contains malicious HTML.

---

#### 2.6.2 Ingest Screen — All Ingest Commands

**URL:** `/ingest`

All ingest-related operations are available on this screen:

| Operation | Button / Form | Backend endpoint | CLI equivalent |
|-----------|--------------|-----------------|----------------|
| View stats | [Refresh] | `GET /api/ingest/stats` | *(new in web)* |
| Build index | [Build Index] | `POST /api/ingest/build` | `lsm ingest build` |
| Build — dry run | ☐ checkbox | `POST /api/ingest/build {dry_run: true}` | `--dry-run` |
| Build — force reingest | ☐ checkbox | `POST /api/ingest/build {force: true}` | `--force` |
| Build — changed config | ☐ checkbox | `{force_reingest_changed_config: true}` | `--force-reingest-changed-config` |
| Build — file pattern | text input | `{force_file_pattern: "*.pdf"}` | `--force-file-pattern` |
| Build — skip errors | ☐ checkbox | `{skip_errors: true}` | `--skip-errors` |
| Run AI tagging | [Run Tagging] | `POST /api/ingest/tag` | `lsm ingest tag` |
| Tag — max chunks | number input | `{max: 500}` | `--max` |
| Wipe collection | [Wipe] + confirm | `POST /api/ingest/wipe` | `lsm ingest wipe --confirm` |
| DB prune | [Prune] | `POST /api/db/prune` | `lsm db prune` |
| DB prune — max versions | number input | `{max_versions: 3}` | `--max-versions` |
| DB prune — older than | number input | `{older_than_days: 30}` | `--older-than-days` |
| DB complete | [Complete] | `POST /api/db/complete` | `lsm db complete` |
| DB complete — pattern | text input | `{force_file_pattern: "*.pdf"}` | `--force-file-pattern` |
| Clear reranker cache | [Clear Cache] | `POST /api/cache/clear` | `lsm cache clear --reranker` |

All long-running operations (build, tag) return an HTML fragment with an SSE div that streams progress
events matching the `{"type":"progress","current":N,"total":M,"message":"..."}` format.

Wipe requires the user to type `wipe` into a confirmation input before the button activates.
Server-side also verifies the `confirm` flag is set.

---

#### 2.6.3 Agents Screen

**URL:** `/agents`

**Layout:**
```
+--sidebar--+---main panel----------------------------------------+
|   Agents  | Agents                                               |
|           |                                                       |
|           | Running ──────────────────────────────────────────── |
|           | ┌─────────────────────────────────────────────────┐ |
|           | │ research-agent  [running]  [Pause] [Stop]       │ |
|           | │ weekly-digest   [scheduled: Sun 08:00] [Run Now]│ |
|           | └─────────────────────────────────────── [Refresh] ┘ |
|           |                                                       |
|           | Log Stream ─ research-agent ─────────────────────── |
|           | ┌─────────────────────────────────────────────────┐ |
|           | │ [INFO] Starting research task...                │ |
|           | │ [INFO] Tool: search_web("AI safety 2026")       │ |
|           | │ [WARN] Rate limit hit, retrying...              │ |
|           | └──────────────────────────────────── (SSE live) ──┘ |
|           |                                                       |
|           | Interaction Requests ──────────────────────────────  |
|           | ┌─────────────────────────────────────────────────┐ |
|           | │ research-agent requests:                        │ |
|           | │   delete_file("sensitive.txt")                  │ |
|           | │   [Approve] [Deny]                              │ |
|           | └─────────────────────────────────────────────────┘ |
+--sidebar--+-----------------------------------------------------+
```

**API endpoints:**
- `GET /api/agents` → HTML fragment: agent list (running + scheduled).
- `POST /api/agents/{name}/start` → HTML fragment: updated agent card.
- `POST /api/agents/runs/{agent_id}/stop` → HTML fragment: updated agent card.
- `POST /api/agents/runs/{agent_id}/pause` → HTML fragment: updated agent card.
- `GET /api/agents/runs/{agent_id}/logs` → SSE log stream for a specific run.
- `GET /api/agents/interactions` → HTML fragment: current pending interaction queue across runs.
- `POST /api/agents/runs/{agent_id}/interactions/{request_id}/ack` → idempotent acknowledgment when a
  prompt is first rendered in the browser.
- `POST /api/agents/runs/{agent_id}/interactions/{request_id}/respond` → post approve/deny/reply for
  the specific pending request.

Agent list auto-refreshes every 5 seconds via `hx-trigger="every 5s"`.

#### 2.6.3.1 WebUI agent tool-request prompting model

The current TUI/shell interaction flow is already defined by `InteractionRequest`,
`InteractionResponse`, `InteractionChannel`, and `AgentRuntimeManager`. The Web UI should reuse that
contract directly rather than inventing a second approval system.

**Key code-verified constraints from the current runtime:**
- A single `InteractionChannel` holds **at most one pending request per agent run**.
- Multiple runs may each have one pending request, so the Web UI needs a queue of cards keyed by
  `(agent_id, request_id)`.
- Requests already carry the exact display fields the Web UI needs:
  `request_type`, `tool_name`, `risk_level`, `reason`, `args_summary`, `prompt`, `timestamp`.
- The two-phase timeout is only activated correctly when the UI calls
  `InteractionChannel.acknowledge_request(request_id)` after rendering, matching existing TUI behavior.

**Decision:** the Web UI gets an "interaction inbox" panel driven by HTML polling, not a separate
JavaScript prompt engine. HTMX polls `GET /api/agents/interactions` every 2 seconds and swaps the
interaction list fragment. This is fast enough for approval UX, keeps the state server-owned, and
avoids duplicating the existing SSE channel already reserved for log streaming.

**Acknowledgment lifecycle (required):**
1. The browser polls `/api/agents/interactions` and receives cards for pending requests.
2. When a card first appears, HTMX immediately posts
   `POST /api/agents/runs/{agent_id}/interactions/{request_id}/ack`.
3. The server forwards that to `AgentRuntimeManager.acknowledge_interaction(agent_id, request_id)`.
4. Repeated `ack` posts are harmless and must be treated as no-ops.

Without this endpoint, the Web UI would leave requests in the "unacknowledged" timeout phase even
after the user can see the prompt, which would diverge from TUI semantics.

**Timeout policy decision:**
- Default agent interaction wait is **infinite** once a request is pending.
- The system still retains configurable fixed wait support for deployments that want automatic expiry.
- In config terms, this means the effective default should be:
  - `agents.interaction.timeout_seconds = 0` (or equivalent "wait forever" sentinel for the initial phase)
  - `agents.interaction.acknowledged_timeout_seconds = 0`
- If the implementation keeps the current validator requirement that `timeout_seconds > 0`, it should be
  relaxed so `0` explicitly means "no timeout" rather than forcing operators to choose an arbitrary large
  number.

**Rationale:** a browser-first approval UI should not silently deny or auto-approve a tool request just
because the user stepped away. Infinite wait is the safer default for human-in-the-loop agent actions,
while fixed wait remains useful for unattended or highly automated deployments.

**Prompt rendering rules by request type:**

| `request_type` | UI controls | Backend decision |
|----------------|------------|------------------|
| `permission` | `Approve`, `Deny`, optional deny reason | `approve` or `deny` |
| `confirmation` | `Approve`, `Deny`, optional deny reason | `approve` or `deny` |
| `clarification` | reply textarea + `Send Reply` + optional `Deny` | `reply` or `deny` |
| `feedback` | reply textarea + `Send Reply` + optional `Deny` | `reply` or `deny` |

**Decision:** `approve_session` is **not exposed in the Web UI for v0.9**.

Reason: the current session-approval cache lives on `AgentRuntimeManager._session_tool_approvals`,
which is process-wide, not browser-session scoped. In the TUI this is acceptable because there is
one operator session. In the Web UI, especially with LAN exposure enabled, surfacing "approve for
session" would ambiguously grant future approvals to every browser connected to that server process.
Until the web stack has first-class authenticated user/session identity, the safe contract is:
- Web UI: `approve`, `deny`, `reply`
- TUI/shell: keep existing `approve_session`

**Stale-response handling:**
- The `respond` route must validate that the current pending request still matches `request_id`.
- If the request was already answered or timed out, return `409 Conflict` and re-render the interaction
  panel with a "request is no longer pending" message.
- This matches the current in-memory channel semantics and keeps multi-tab behavior deterministic.

**Agent run identity correction:**

The original draft used `{name}` for pause/stop/log routes, but current runtime code is run-centric:
`AgentRuntimeManager.start()` creates a unique `agent_id`, and interaction/log lookups already use that
identifier. Therefore the Web UI must key all live operations by `agent_id`, not by `agent_name`.

This also requires the cross-process runtime table to store runs, not agent classes:

```sql
CREATE TABLE lsm_agent_runtime_state (
    agent_id      TEXT PRIMARY KEY,
    agent_name    TEXT NOT NULL,
    topic         TEXT NOT NULL,
    status        TEXT NOT NULL CHECK(status IN ('running', 'paused', 'stopped', 'scheduled')),
    pid           INTEGER,
    started_at    TEXT,
    updated_at    TEXT NOT NULL,
    error_message TEXT
);
```

#### 2.6.3.2 Communication assistant launchers

The existing `email_assistant` and `calendar_assistant` agents do not want a free-form "topic"
string for most useful operations. Their `run()` paths already accept structured JSON payloads via
`_extract_payload()` and route behavior from keys like `action`, `provider`, `filters`, `draft`,
`event`, and `updates`.

**Decision:** the Agents screen includes two specialized launcher cards in addition to the generic
"start agent" control:
- **Email Assistant launcher**:
  - provider selector filtered to `gmail`, `microsoft_graph_mail`, `imap`
  - actions: `summary`, `draft`, `send`
  - summary fields: query, from, to, unread-only, folder, time window, max results
  - draft/send fields: recipients, subject, body, optional thread id
- **Calendar Assistant launcher**:
  - provider selector filtered to `google_calendar`, `microsoft_graph_calendar`, `caldav`
  - actions: `summary`, `suggest`, `create`, `update`, `delete`
  - summary/suggest fields: query, date window, duration, workday bounds, max suggestions
  - mutation fields: event id, title, start/end, location, attendees, description

The launcher serializes the form to the exact JSON payload shape the current agents already parse,
then starts the run normally through `POST /api/agents/{name}/start`.

This avoids asking the user to hand-type JSON into a generic topic field and keeps the Web UI aligned
with the existing assistant implementation instead of requiring new agent-side parsing rules.

**Cross-process agent runtime coordination:**

The TUI and web server run as separate OS processes. Agent runtime state (`AgentRuntimeManager`) is
in-memory, so a running agent started from the web server is invisible to a concurrent TUI process
and vice versa. To solve this, agent runtime state is persisted in a shared SQLite table so both
processes see a consistent view:

```sql
CREATE TABLE lsm_agent_runtime_state (
    agent_id      TEXT PRIMARY KEY,
    agent_name    TEXT NOT NULL,
    topic         TEXT NOT NULL,
    status        TEXT NOT NULL CHECK(status IN ('running', 'paused', 'stopped', 'scheduled')),
    pid           INTEGER,             -- OS process ID of the owning process
    started_at    TEXT,                -- ISO-8601
    updated_at    TEXT NOT NULL,       -- ISO-8601; heartbeat timestamp
    error_message TEXT                 -- last error, if any
);
```

**Coordination protocol:**
- On agent start: insert/update row for the new `agent_id` with `status='running'`, current `pid`,
  `topic`, and `started_at`.
- Heartbeat: the owning process updates `updated_at` every 10 seconds while the agent runs.
- On agent stop/pause: update `status` accordingly.
- Stale detection: if `updated_at` is older than 30 seconds and `status='running'`, the agent is
  considered crashed. The reader (TUI or web) marks it `stopped` and clears the row.
- The TUI `/agent list` command reads this table to show runs started by the web server (and vice
  versa). Stop/pause/log/interaction actions resolve against `agent_id`, not `agent_name`.
- The existing `lsm_agent_schedules` table (used by the agent scheduler) is unchanged.

---

#### 2.6.4 Settings Screen

**URL:** `/settings`

A single long-form page with anchor-linked sections. Both the Web UI settings screen and the TUI
settings screen read and write the same `config.json` file. Communication provider connection state
(OAuth tokens) is not stored in `config.json`; it stays in the existing encrypted token store under
`<global_folder>/oauth_tokens/<provider>/`.

| Section | Config object | Representative fields |
|---------|--------------|----------------------|
| Global | `GlobalConfig` | global_folder, embed_model, device, batch_size |
| LLM | `LLMRegistryConfig` | providers[], services{} |
| Vector DB | `DBConfig` | path, collection, table_prefix |
| Query | `QueryConfig` | mode, top_k, candidates_k, reranker |
| Modes | `modes{}` | per-mode: system_prompt, retrieval settings |
| Ingest | `IngestConfig` | roots[], chunk_size, chunking_strategy |
| Chats | `ChatsConfig` | dir, auto_save, format |
| Remote Providers | `remote_providers[]` | name, type, api_key |
| Remote Chains | `lsm_remote_chains` tables (DB-backed) | chain metadata, ordered links, revisions |
| Server | `ServerConfig` | host, port, log_level, expose_to_lan, require_access_token, access_token |

**Behaviour:**
- Page loads current config from `GET /api/config` (JSON). Jinja2 pre-populates form fields.
- On Save: `hx-put="/api/config"` submits the entire form (`application/x-www-form-urlencoded`).
  Server calls `save_config_to_file()` and returns an HTML fragment (updated form or inline errors)
  for HTMX swaps.
- Programmatic clients may also `PUT /api/config` with JSON; response is JSON success/error payload.
- On invalid field: response includes `HX-Trigger` to highlight the offending input.
- Remote chains are edited through dedicated DB-backed endpoints (not saved inline in `config.json`).

**Single endpoint vs split endpoint (`PUT /api/config`)**

| Option | Pros | Cons |
|--------|------|------|
| Single endpoint (dual-mode) | One canonical write path; less duplicated validation/save logic; easier to keep TUI/Web/API behavior consistent | Handler complexity (content-type negotiation, response negotiation) |
| Split endpoints (`/api/config` JSON + `/api/config/form`) | Simpler per-handler behavior; clearer contracts per caller type | Duplicated wiring and higher drift risk between form and JSON paths |

**Recommendation:** Keep the single endpoint with explicit content-type handling and response negotiation.
Use one shared validation/save function underneath to avoid drift.

#### 2.6.4.1 Communication providers (email + calendar) in the Web UI

The repo already contains six communication providers:
- Email: `gmail`, `microsoft_graph_mail`, `imap`
- Calendar: `google_calendar`, `microsoft_graph_calendar`, `caldav`

The missing design work is not provider implementation from scratch; it is how the Web UI configures,
authorizes, and launches flows that use those existing providers.

**Decision:** communication providers live in two places in the Web UI:
- **Settings screen**: configuration, OAuth connect/disconnect, health/test status
- **Agents screen**: per-run provider selection for `email_assistant` and `calendar_assistant`

There is **no separate v0.9 email page or calendar page**. The provider backends are consumed through
the assistant agents already in-tree, and the Web UI should expose those assistants cleanly rather than
creating a parallel feature surface.

**Settings form structure:**
- `remote_providers[]` remains the canonical config home.
- The Settings page groups the communication subset into a dedicated "Communication Providers" section
  with provider-specific fieldsets.
- The assistant defaults stay under `agents.agent_configs.email_assistant.provider` and
  `agents.agent_configs.calendar_assistant.provider`; the Settings page links these defaults to the
  configured communication providers.

**Provider field groups:**

| Provider type | Web UI fields | Notes |
|---------------|--------------|-------|
| `gmail` / `google_calendar` | name, redirect URI, client id, client secret, scopes, timeout, max results, calendar id (calendar only) | OAuth-backed; connect/disconnect/test controls |
| `microsoft_graph_mail` / `microsoft_graph_calendar` | name, redirect URI, client id, client secret, scopes, timeout, max results | OAuth-backed; connect/disconnect/test controls |
| `imap` | name, host, port, username, password, folder, drafts folder, SMTP host/port/user/password, SSL flags | password-bearing config; no OAuth |
| `caldav` | name, calendar URL, username, password, timeout | password-bearing config; no OAuth |

**Secret-handling rule for the Settings form:**

Do not echo stored secrets back into HTML. For `client_secret`, IMAP/SMTP passwords, and CalDAV
passwords:
- render the input blank with helper text "Leave blank to keep existing value"
- on save, blank means "preserve current secret" rather than "erase"
- show adjacent status text such as "secret configured" / "not configured"

This matters more for communication providers because their configs include full account credentials,
not just API keys.

**OAuth flow redesign for browser use:**

Current provider implementations call `OAuth2Client.get_access_token(allow_interactive=True)` inside
request methods. That is acceptable in CLI/TUI because `OAuth2Client.authorize()` spins up a temporary
`OAuthCallbackServer` and waits for the browser redirect locally. It is the wrong abstraction for the
Web UI because the web server already owns the browser request/response cycle.

**Decision:** Web UI OAuth uses the existing token store and token exchange logic, but not the
temporary callback server. Add a browser-native OAuth service layer:

```python
class WebOAuthSessionStore:
    # pending state -> provider name, redirect target, created_at
    ...

def begin_provider_oauth(provider_cfg: RemoteProviderConfig) -> RedirectResponse:
    # build auth URL from OAuth2Client.build_authorization_url(...)
    # persist state in WebOAuthSessionStore
    # redirect browser to provider auth URL

def finish_provider_oauth(provider_name: str, code: str, state: str) -> None:
    # validate pending state
    # call OAuth2Client.exchange_code(code)
    # token persists into existing OAuthTokenStore
```

**New Web UI routes for communication providers:**
- `POST /api/providers/communication/{name}/connect` → starts browser OAuth; returns redirect
- `GET /api/providers/oauth/{name}/callback` → validates `state`, exchanges code, stores token, redirects
  back to `/settings#provider-{name}`
- `POST /api/providers/communication/{name}/disconnect` → deletes stored OAuth token
- `POST /api/providers/communication/{name}/test` → lightweight connectivity check + status fragment

**Non-interactive runtime rule:**

Agent runs and Web UI background operations must never attempt to open interactive OAuth during normal
execution. Instead:
- Web UI "Connect" establishes/refreshes tokens ahead of time
- provider instances created for Web UI agents use `allow_interactive=False`
- missing/expired token produces a deterministic error like "Provider 'gmail' is not connected"
- the error is surfaced with an action link back to the Settings connect button

This requires a small provider/OAuth refactor so communication providers do not hardcode
`allow_interactive=True` in `_headers()`; they need an injected policy or constructor flag.

**Why this is the least-risk design:**
- Reuses the existing provider classes, token store, and assistant parsing logic
- Keeps OAuth state transitions in the foreground request cycle where browser redirects belong
- Prevents background agent threads from blocking on local callback servers
- Avoids creating a second communication abstraction alongside `remote_providers[]`

---

#### 2.6.5 Admin Screen — All Power-User Operations

**URL:** `/admin`

Consolidates all operations that are not day-to-day query/ingest: system health, evaluation, migration,
clustering, knowledge graph, fine-tuning, statistics, and live log tailing.

**Sections:**

**System Health** — `GET /api/health`
```
Database       [OK]   SQLite 1,234,567 chunks
Embedding      [OK]   all-MiniLM-L6-v2 (384d), device=cpu
LLM Provider   [OK]   openai/gpt-4o
Remote         [OK]   3 providers configured
Agents         [OK]   0 running, 2 scheduled
Server         [OK]   http://127.0.0.1:8080
Logs           [OK]   ~/Local Second Mind/Logs/lsm-2026-03-02.log
```
`[Refresh Health]` button triggers `hx-get="/api/health"` swapping the table.

**Evaluation** — runs retrieval quality tests:
- Profile selector, dataset selector, compare-vs-baseline field.
- `[Run Retrieval Eval]` → `POST /api/admin/eval/retrieval` → SSE progress stream.
- `[Save as Baseline]` → `POST /api/admin/eval/baseline`.
- `[List Baselines]` → `GET /api/admin/eval/baselines` → HTML table.

**Migration** — backend migration between DB formats/versions:
- From/To selectors, resume/enrich/rechunk checkboxes, batch size, source/target path overrides.
- `[Run Migration]` → `POST /api/admin/migrate` → SSE progress stream.

**Clustering** — embedding cluster management:
- Algorithm (kmeans/hdbscan), K value.
- `[Build Clusters]` → `POST /api/admin/cluster/build` → SSE progress stream.
- `[Visualize]` → `POST /api/admin/cluster/visualize` → file download (HTML plot).

**Knowledge Graph** — thematic link building:
- Threshold, batch size fields.
- `[Build Links]` → `POST /api/admin/graph/build-links` → SSE progress stream.

**Fine-Tuning** — embedding model fine-tuning:
- Base model, epochs, max pairs, output path.
- Pair count estimate displayed before running (calls `GET /api/admin/finetune/pair-count`).
- `[Train]` → `POST /api/admin/finetune/train` → SSE progress stream.
- `[List Models]` → `GET /api/admin/finetune/models` → HTML table.
- Activate model: model_id input + `[Set Active]` → `PUT /api/admin/finetune/active`.
- Unregister stale model row: `DELETE /api/admin/finetune/models/{model_id}`.

**Embedding Models** — system inventory and availability:
- `[Refresh Inventory]` → `GET /api/admin/embeddings/models` (HTML table + JSON option).
- Table includes configured model, active registry model, all registered fine-tuned models, and
  well-known catalog entries with local-install status.
- Filters:
  - `installed_only=true`
  - `source=configured|registry|catalog`
- Optional `[Validate Load]` action on a row runs `POST /api/admin/embeddings/validate` for a
  lightweight loadability check (no full ingest).

**Statistics** — `GET /api/admin/stats`
```
Chunks: 1,234,567      Files indexed: 2,345
Tags applied: 89%      Cluster coverage: 78%
Conversations: 142     Messages: 890
Graph edges: 45,678    Finetune models: 2
Embedding models: 14 total (installed: 6, registry: 2, active: finetuned-abc123)
```

**System Logs** — live log tail:
- Level filter selector (DEBUG/INFO/WARNING/ERROR).
- `[Start Live Log]` → SSE connection to `GET /api/admin/logs` (streams from `EventBufferHandler`).
- Clicking `[Stop]` closes the SSE connection.
- Alternatively, `[Load Recent]` → `GET /api/admin/logs?last=100` returns last N buffered records as HTML.

**All admin API endpoints:**

| Operation | Endpoint |
|-----------|----------|
| System health | `GET /api/health` |
| Eval run | `POST /api/admin/eval/retrieval` (SSE) |
| Eval save baseline | `POST /api/admin/eval/baseline` |
| Eval list baselines | `GET /api/admin/eval/baselines` |
| Migration | `POST /api/admin/migrate` (SSE) |
| Cluster build | `POST /api/admin/cluster/build` (SSE) |
| Cluster visualize | `POST /api/admin/cluster/visualize` (file download) |
| Graph build-links | `POST /api/admin/graph/build-links` (SSE) |
| Finetune pair count | `GET /api/admin/finetune/pair-count` |
| Finetune train | `POST /api/admin/finetune/train` (SSE) |
| Finetune list models | `GET /api/admin/finetune/models` |
| Finetune activate | `PUT /api/admin/finetune/active` |
| Finetune delete model | `DELETE /api/admin/finetune/models/{model_id}` |
| Embedding inventory | `GET /api/admin/embeddings/models` |
| Embedding validate | `POST /api/admin/embeddings/validate` |
| Statistics | `GET /api/admin/stats` |
| Live log stream | `GET /api/admin/logs` (SSE) |
| Recent logs (buffered) | `GET /api/admin/logs?last=100` |

---

#### 2.6.6 Help & Docs Screen

**URL:** `/help` (index), `/help/{doc_slug}` (individual doc)

**Rationale:** The browser is the ideal medium for documentation — scrollable, searchable via Ctrl+F,
linkable, and can render markdown as formatted HTML. Serving the existing `docs/` files through the Web
UI makes the documentation accessible without leaving the application and without requiring a separate
docs site. The same `mistune` library used for chat markdown does double duty here. No new dependencies.

**Layout:**
```
+--sidebar-----------+----main panel---------------------------------+
| [LSM]              | Getting Started                              |
|                    |                                               |
|   Query            | # Getting Started with Local Second Mind      |
|   Ingest           |                                               |
|   Agents           | ## What is Local Second Mind?                |
|   Settings         |                                               |
|   Admin            | Local Second Mind is a local-first RAG...    |
| ● Help             |                                               |
|                    | ## Prerequisites                              |
| ─────────────────  |                                               |
| Documentation      | - Python 3.10+                               |
|                    | - 8GB RAM minimum                            |
| ▶ Getting Started  |                                               |
|   Configuration    | ## Installation                              |
|   CLI Usage        |                                               |
|   Setup            | ```bash                                       |
| ─────────────────  | pip install lsm                              |
| User Guides        | ```                                           |
| ▶ Query Modes      |                                               |
|   Agents           |                                               |
|   Chat Exports     |                                               |
|   Remote Sources   |                                               |
|   Local Models     |                                               |
|   Vector Databases |                                               |
|   Integrations     |                                               |
+--------------------+-----------------------------------------------+
```

**Navigation structure (sidebar):**

The docs sidebar is hardcoded (not auto-discovered) to match the actual docs structure with
human-readable labels. This avoids filename-derived labels like "GETTING_STARTED" being shown:

```python
DOCS_NAV = [
    {"label": "Getting Started", "slug": "getting-started", "file": "user-guide/GETTING_STARTED.md"},
    {"label": "Configuration",   "slug": "configuration",   "file": "user-guide/CONFIGURATION.md"},
    {"label": "CLI Usage",       "slug": "cli-usage",       "file": "user-guide/CLI_USAGE.md"},
    {"label": "Setup (Dev)",     "slug": "setup",           "file": "user-guide/SETUP.md"},
    {"label": "Query Modes",     "slug": "query-modes",     "file": "user-guide/QUERY_MODES.md"},
    {"label": "Agents",          "slug": "agents",          "file": "user-guide/AGENTS.md"},
    {"label": "Chat Exports",    "slug": "chat-exports",    "file": "user-guide/CHAT_EXPORTS.md"},
    {"label": "Remote Sources",  "slug": "remote-sources",  "file": "user-guide/REMOTE_SOURCES.md"},
    {"label": "Local Models",    "slug": "local-models",    "file": "user-guide/LOCAL_MODELS.md"},
    {"label": "Vector Databases","slug": "vector-databases","file": "user-guide/VECTOR_DATABASES.md"},
    {"label": "Integrations",    "slug": "integrations",    "file": "user-guide/INTEGRATIONS.md"},
    {"label": "Fine-Tuning",     "slug": "finetune",        "file": "user-guide/FINETUNE.md"},
]
```

Docs migration note:
- Replace/retire `docs/user-guide/NOTES.md` with `docs/user-guide/CHAT_EXPORTS.md` to match v0.9
  feature direction and avoid stale UI labels.

**Docs bundling:**

`setuptools` `package-data` only applies reliably to files inside discovered packages. Since the
project-level `docs/` folder is outside `lsm/`, a robust wheel strategy is:

1. Keep top-level `docs/user-guide/` as the source of truth.
2. Add a packaged mirror at `lsm/ui/web/docs/user-guide/`.
3. Add a sync script (e.g. `scripts/sync_user_guide_docs.py`) run before release/build.
4. Package the mirrored docs with the web assets:

```toml
[tool.setuptools.package-data]
"lsm.ui.web" = [
    "templates/**/*.html",
    "static/**/*",
    "docs/user-guide/*.md",
]
```

**Docs path resolution:**

```python
def find_docs_root() -> Path:
    """
    Locate the bundled docs/user-guide/ directory.
    First preference: packaged mirror under lsm/ui/web/docs.
    Dev fallback: repository docs/user-guide.
    """
    candidates = [
        Path(__file__).resolve().parents[1] / "docs",         # packaged mirror
        Path(__file__).resolve().parents[4] / "docs",         # dev repo root
    ]
    for p in candidates:
        if p.is_dir() and (p / "user-guide").is_dir():
            return p
    raise FileNotFoundError(
        "docs/user-guide/ not found. Run docs sync and verify package data."
    )
```

**Rendering:**

Each doc page is rendered by reading the markdown file and converting it with `mistune`. The rendered
HTML is embedded in the `fragments/doc_page.html` fragment, which HTMX swaps into the main content area
when a sidebar link is clicked — no full page reload:

```html
<!-- In help.html -->
<nav id="docs-nav">
  {% for doc in docs_nav %}
  <a hx-get="/help/{{ doc.slug }}"
     hx-target="#doc-content"
     hx-swap="innerHTML"
     class="{% if doc.slug == active_slug %}active{% endif %}">
    {{ doc.label }}
  </a>
  {% endfor %}
</nav>
<div id="doc-content">
  {{ rendered_html | safe }}
</div>
```

**API endpoints:**
- `GET /help` → full HTML page with first doc (Getting Started) pre-rendered
- `GET /help/{slug}` → HTML fragment (doc content only, HTMX swap target)

**New route file:** `lsm/ui/web/routes/docs.py`

**New template:** `lsm/ui/web/templates/help.html` (full page) and `fragments/doc_page.html` (partial)

**Considerations:**
- CONFIGURATION.md (807 lines) renders to a large HTML page — add a sticky "Back to top" link.
- Internal markdown links (`[something](./OTHER.md)`) should be rewritten to `/help/{slug}` during
  rendering so they navigate within the app rather than triggering a 404.
- Code blocks in docs use syntax highlighting classes; apply a simple CSS-based highlight (no JS
  highlighting library needed for a single user tool).

---

### 2.7 Dark Mode Architecture

**Default behaviour:** follows `prefers-color-scheme` CSS media query — system setting is honoured
automatically with no JS required for the default case.

**Override:** user clicks a sun/moon toggle in the sidebar. Preference stored in `localStorage` as
key `"lsm-theme"` with values `"system"` (default), `"light"`, or `"dark"`.

**Anti-flash strategy:** Theme is applied before the page renders. A tiny inline script in `<head>`
(no external file, no render-blocking) reads `localStorage` and applies the correct `data-theme`
attribute to `<html>` before the browser paints:

```html
<head>
  <!-- Inline: runs synchronously, before first paint to prevent theme flash -->
  <script>
    (function() {
      var t = localStorage.getItem("lsm-theme") || "system";
      if (t === "dark" || (t === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
        document.documentElement.setAttribute("data-theme", "dark");
      }
    })();
  </script>
  <link rel="stylesheet" href="/static/css/main.css">
</head>
```

**CSS implementation — custom properties:**

All colour, surface, and border values are defined as CSS custom properties on `:root` and `[data-theme="dark"]`:

```css
/* main.css — light mode defaults */
:root {
  --color-bg:           #ffffff;
  --color-bg-secondary: #f5f5f5;
  --color-surface:      #fafafa;
  --color-border:       #e0e0e0;
  --color-text:         #1a1a1a;
  --color-text-muted:   #6b6b6b;
  --color-accent:       #2563eb;
  --color-accent-hover: #1d4ed8;
  --color-user-msg:     #eff6ff;
  --color-asst-msg:     #f9fafb;
  --color-code-bg:      #f4f4f5;
  --color-success:      #16a34a;
  --color-warning:      #d97706;
  --color-error:        #dc2626;
}

/* Dark mode overrides — applied when data-theme="dark" on <html> */
[data-theme="dark"] {
  --color-bg:           #0f0f10;
  --color-bg-secondary: #1a1a1b;
  --color-surface:      #1e1e1f;
  --color-border:       #2e2e30;
  --color-text:         #e8e8ea;
  --color-text-muted:   #9d9da0;
  --color-accent:       #3b82f6;
  --color-accent-hover: #60a5fa;
  --color-user-msg:     #1e2a3a;
  --color-asst-msg:     #1e1e1f;
  --color-code-bg:      #27272a;
  --color-success:      #4ade80;
  --color-warning:      #fbbf24;
  --color-error:        #f87171;
}

/* All element styles use only custom property references, never hard-coded colours */
body { background: var(--color-bg); color: var(--color-text); }
```

**Toggle behaviour (small inline JS, no framework):**

```html
<!-- In base.html sidebar -->
<button id="theme-toggle" aria-label="Toggle dark mode" onclick="lsmToggleTheme()">
  <span class="icon-sun">☀</span>
  <span class="icon-moon">☽</span>
</button>

<script>
function lsmToggleTheme() {
  var current = localStorage.getItem("lsm-theme") || "system";
  var next = { "system": "dark", "dark": "light", "light": "system" }[current];
  localStorage.setItem("lsm-theme", next);
  // Re-evaluate and apply
  var isDark = (next === "dark") ||
    (next === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
  // Update toggle button label
  document.getElementById("theme-toggle").dataset.theme = next;
}
// Sync toggle icon to current state on page load
document.addEventListener("DOMContentLoaded", function() {
  var t = localStorage.getItem("lsm-theme") || "system";
  document.getElementById("theme-toggle").dataset.theme = t;
});
</script>
```

**System preference change listener:**

```javascript
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function(e) {
  if ((localStorage.getItem("lsm-theme") || "system") === "system") {
    document.documentElement.setAttribute("data-theme", e.matches ? "dark" : "light");
  }
});
```

**No server-side state:** Theme preference is entirely client-side (localStorage). The server never
needs to know the user's theme. This avoids cookies and session management complexity.

---

### 2.8 Markdown Rendering Architecture

Markdown rendering has two contexts with different requirements:

**Context 1 — Streaming chat (client-side):**

During a query response, tokens arrive over SSE one or a few words at a time. Full markdown parsing
cannot happen per-token (incomplete syntax). Strategy:

1. Accumulate raw tokens in a hidden `data-raw-content` attribute of the response div.
2. Display raw text in a `<span>` during streaming — readable but unformatted.
3. On `event: done` (SSE closes): call `marked.parse(rawContent)` to render the full markdown to HTML.
4. Replace the `<span>` with the rendered HTML.
5. Remove the streaming cursor.

The brief period where raw markdown is visible (e.g., `**bold**` before rendering) is acceptable for
a local tool and avoids complex incremental rendering.

**Context 2 — Stored messages and docs (server-side):**

When loading past conversations or rendering doc pages, `mistune` renders markdown to HTML on the
server before embedding in Jinja2 templates.

```python
# In lsm/ui/web/rendering.py
import mistune

def create_renderer() -> mistune.Markdown:
    """
    Create a mistune renderer with HTML sanitisation.
    Strips <script> tags and event handler attributes.
    Rewrites relative .md links to /help/{slug} equivalents.
    """
    from mistune.plugins.table import table
    renderer = mistune.HTMLRenderer(escape=True)
    md = mistune.create_markdown(
        renderer=renderer,
        plugins=["table", "strikethrough"],
    )
    return md

_renderer = create_renderer()

def render_markdown(text: str) -> str:
    """Render markdown string to sanitised HTML."""
    return _renderer(text)
```

`render_markdown()` is registered as a Jinja2 filter:
```python
templates.env.filters["render_markdown"] = render_markdown
```

Used in templates as:
```html
<div class="message assistant-message">
  {{ message.content | render_markdown | safe }}
</div>
```

**Security consideration — prompt injection via LLM output:**

An LLM response could theoretically include `<script>alert(1)</script>` or `javascript:` URLs. Since
`mistune` is configured with `escape=True`, raw HTML in markdown is escaped. However the rendered HTML
(headings, links) could still carry attributes. Post-process rendered HTML with a strict allow-list
using `nh3.clean(...)` before marking it `| safe` in Jinja2. (`nh3` is the actively-maintained
Rust-backed replacement for the deprecated `bleach` library.)

**Vendored `marked.js`:**

Pin to a specific stable version (e.g., 12.x) at vendor time. The file is minified and checked in.
Update policy: only update when there is a security patch or required feature. Document the version in
`lsm/ui/web/static/js/README.txt`.

---

### 2.9 Complete REST + SSE API Surface

LAN auth note for this surface:
- When `server.expose_to_lan=true` and `server.require_access_token=true`, all non-GET `/api/*` routes
  require `X-LSM-Access-Token` (or bearer token) and return 401/403 on missing/invalid token.
- GET/SSE endpoints remain read-only and unauthenticated in v0.9.

```
# Pages (HTML — full page, HTMX entry points)
GET    /                              → redirect to /query
GET    /query
GET    /ingest
GET    /agents
GET    /settings
GET    /admin
GET    /help
GET    /help/{slug}                   → HTML fragment: rendered doc page

# Query
POST   /api/query                     → HTML fragment (user msg + SSE div)
GET    /api/query/stream/{id}         → SSE: token stream + citations + done
PUT    /api/query/mode                → sets active mode; OOB swap

# Conversations / Messages
GET    /api/conversations             → HTML fragment: conversation list (supports `?view=active|archived`)
GET    /api/conversations/search      → HTML fragment: filtered conversation list (includes archived)
GET    /api/conversations/{id}        → HTML fragment: full conversation
POST   /api/conversations/new         → redirects to /query with empty session
PUT    /api/conversations/{id}/model     → HTML fragment: update provider/model for conversation
POST   /api/conversations/{id}/archive   → HTML fragment: updated conversation row/list
POST   /api/conversations/{id}/unarchive → HTML fragment: updated conversation row/list
DELETE /api/conversations/{id}           → HTML fragment or JSON: conversation removed
GET    /api/conversations/{id}/export    → file download (`?format=markdown|json&redact=true|false`)
POST   /api/messages/{id}/branch         → creates new conversation from selected message path
POST   /api/messages/{id}/retry          → HTML fragment: new assistant variant + variant controls
PUT    /api/messages/{id}/variant        → HTML fragment: active assistant variant switched
PATCH  /api/messages/{id}                → HTML fragment: edited latest message (user or assistant)
DELETE /api/messages/{id}                → HTML fragment: latest message permanently deleted
GET    /api/messages/{id}/export         → file download (`?format=markdown|json&redact=true|false`)

# Ingest
POST   /api/ingest/build              → HTML fragment + SSE div
GET    /api/ingest/build/stream/{id}  → SSE: progress events
POST   /api/ingest/tag                → HTML fragment + SSE div
GET    /api/ingest/tag/stream/{id}    → SSE: progress events
POST   /api/ingest/wipe               → HTML fragment: result
GET    /api/ingest/stats              → HTML fragment: stats table

# DB / Cache
POST   /api/db/prune                  → HTML fragment: result
POST   /api/db/complete               → HTML fragment + SSE div
GET    /api/db/complete/stream/{id}   → SSE: progress events
POST   /api/cache/clear               → HTML fragment: result

# Agents
GET    /api/agents                    → HTML fragment: agent list
POST   /api/agents/{name}/start       → HTML fragment: updated card
POST   /api/agents/runs/{agent_id}/stop  → HTML fragment: updated card
POST   /api/agents/runs/{agent_id}/pause → HTML fragment: updated card
GET    /api/agents/runs/{agent_id}/logs  → SSE: filtered log stream
GET    /api/agents/interactions          → HTML fragment: pending interaction queue
POST   /api/agents/runs/{agent_id}/interactions/{request_id}/ack
                                       → acknowledge rendered interaction
POST   /api/agents/runs/{agent_id}/interactions/{request_id}/respond
                                       → approve/deny/reply interaction

# Communication Providers
POST   /api/providers/communication/{name}/connect    → redirect into OAuth flow or 400 for non-OAuth providers
GET    /api/providers/oauth/{name}/callback           → exchange code, store token, redirect back to settings
POST   /api/providers/communication/{name}/disconnect → revoke local token/cache and refresh status fragment
POST   /api/providers/communication/{name}/test       → HTML fragment or JSON: provider connectivity status

# Config
GET    /api/config                    → JSON: full config
PUT    /api/config                    → accepts HTMX form payload or JSON; returns HTML fragment (HTMX) or JSON (API client)

# Remote Chains
GET    /api/remote-chains             → JSON/HTML list: DB-backed remote chains
POST   /api/remote-chains             → create chain
PUT    /api/remote-chains/{id}        → update chain + links
DELETE /api/remote-chains/{id}        → delete chain
POST   /api/remote-chains/{id}/import → import chain JSON
GET    /api/remote-chains/{id}/export → export chain JSON
GET    /api/remote-chains/defaults    → list packaged default chains

# Health
GET    /api/health                    → JSON: per-subsystem health status

# Admin
POST   /api/admin/eval/retrieval      → HTML fragment + SSE div
GET    /api/admin/eval/retrieval/stream/{id} → SSE: progress events
POST   /api/admin/eval/baseline       → JSON: result
GET    /api/admin/eval/baselines      → HTML fragment: baseline list
POST   /api/admin/migrate             → HTML fragment + SSE div
GET    /api/admin/migrate/stream/{id} → SSE: progress events
POST   /api/admin/cluster/build       → HTML fragment + SSE div
GET    /api/admin/cluster/build/stream/{id} → SSE: progress events
POST   /api/admin/cluster/visualize   → file download: clusters.html
POST   /api/admin/graph/build-links   → HTML fragment + SSE div
GET    /api/admin/graph/stream/{id}   → SSE: progress events
GET    /api/admin/finetune/pair-count → JSON: {count, has_enough}
POST   /api/admin/finetune/train      → HTML fragment + SSE div
GET    /api/admin/finetune/train/stream/{id} → SSE: progress events
GET    /api/admin/finetune/models     → HTML fragment: model list
PUT    /api/admin/finetune/active     → JSON: result
DELETE /api/admin/finetune/models/{model_id} → JSON: unregister fine-tuned model
GET    /api/admin/embeddings/models   → HTML fragment or JSON: embedding model inventory
POST   /api/admin/embeddings/validate → JSON: loadability check for a specific embedding model
GET    /api/admin/stats               → HTML fragment: stats table
GET    /api/admin/logs                → SSE: live log stream
GET    /api/admin/logs?last={n}       → HTML fragment: recent log entries
```

---

### 2.10 FastAPI Application Structure

```
lsm/ui/web/
├── __init__.py
├── app.py               # create_app(config: LSMConfig) -> FastAPI
├── server.py            # run_server(config: LSMConfig) -> int
├── dependencies.py      # FastAPI Depends: get_config, get_db_conn, get_event_buffer
├── rendering.py         # render_markdown(), create_renderer() — mistune wrapper
├── streaming.py         # make_sse_stream(), StreamingJobRunner, event format helpers
├── services/
│   ├── __init__.py
│   ├── conversations.py # chat persistence, model override, retry/edit/delete/archive/search/branch/export ops
│   ├── query.py         # orchestration for retrieval + synthesis + SSE jobs
│   ├── agents.py        # runtime-manager adapters, interaction ack/respond, run-card shaping
│   ├── communication.py # provider filtering, connect/test/disconnect, browser OAuth orchestration
│   ├── remote_chains.py # DB CRUD/import/export for remote chains
│   └── embeddings.py    # embedding model inventory + loadability checks
└── routes/
    ├── __init__.py
    ├── pages.py         # GET /query, /ingest, /agents, /settings, /admin (HTML pages)
    ├── query.py         # POST /api/query, /api/query/stream/{id}, /api/query/mode
    ├── conversations.py # /api/conversations/*, /api/messages/* (model/retry/edit/delete/archive/search/branch/export)
    ├── ingest.py        # POST /api/ingest/*, /api/db/*, /api/cache/*
    ├── agents.py        # /api/agents/*
    ├── providers.py     # /api/providers/communication/*, /api/providers/oauth/*
    ├── config.py        # GET/PUT /api/config
    ├── remote_chains.py # /api/remote-chains/*
    ├── health.py        # GET /api/health
    ├── admin.py         # /api/admin/* (eval, migrate, cluster, graph, finetune, stats, logs)
    └── docs.py          # GET /help, GET /help/{slug}
```

**`app.py` — application factory:**
```python
def create_app(config: LSMConfig) -> FastAPI:
    app = FastAPI(title="Local Second Mind", docs_url="/docs", redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["render_markdown"] = render_markdown
    app.state.config = config
    app.state.templates = templates

    # Embedding model preload — runs as a background task AFTER the server is
    # listening, so the bind/ready message prints immediately and the model
    # loads without blocking incoming requests.
    #
    # NOTE: `load_embedding_model()` is a NEW function to be created during
    # implementation. Currently, embedding model loading is done inline in
    # `lsm/ingest/pipeline.py` (lines 424-426) via:
    #     from sentence_transformers import SentenceTransformer
    #     model = SentenceTransformer(embed_model_name, device=device)
    # This logic must be extracted into a standalone function in a new module
    # `lsm/ingest/embedding.py` so the web server can preload the model at
    # startup without importing the full pipeline machinery.  The function
    # signature should accept config (or embed_model_name + device) and return
    # the loaded SentenceTransformer instance.
    @app.on_event("startup")
    async def startup():
        import asyncio
        async def _preload():
            from lsm.ingest.embedding import load_embedding_model
            app.state.embedding_model = await asyncio.to_thread(
                load_embedding_model, config
            )
        asyncio.create_task(_preload())

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    for router in [pages_router, query_router, conversations_router, ingest_router,
                   agents_router, providers_router, config_router, remote_chains_router,
                   health_router, admin_router, docs_router]:
        app.include_router(router)
    return app
```

**`streaming.py` — shared SSE helper:**
```python
async def event_stream_response(
    generator: AsyncGenerator[dict, None],
) -> StreamingResponse:
    """
    Wrap an async generator of dicts into a proper SSE StreamingResponse.
    Automatically appends 'event: done' on completion.
    """
    async def stream():
        try:
            async for item in generator:
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

The `X-Accel-Buffering: no` header prevents nginx/proxies from buffering SSE — important if the user
later puts LSM behind a reverse proxy.

**Long-running job pattern (ingest, eval, migrate):**

Long-running operations run in a `asyncio.create_task()` background task. The POST endpoint:
1. Creates a `job_id`.
2. Launches the background task which writes progress events to an `asyncio.Queue`.
3. Returns the HTML fragment (with SSE div pointing at `.../stream/{job_id}`).

The SSE stream endpoint consumes from the queue until the job completes.

---

### 2.11 Dependency Injection

```python
# dependencies.py

def get_config(request: Request) -> LSMConfig:
    return request.app.state.config

def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates

def get_db_conn(config: LSMConfig = Depends(get_config)):
    """Opens a per-request SQLite connection with integrity and lock-handling pragmas."""
    conn = sqlite3.connect(str(config.db.path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()

def get_event_buffer() -> EventBufferHandler:
    from lsm.logging import get_event_buffer as _get
    return _get()

def get_docs_root() -> Path:
    from lsm.ui.web.routes.docs import find_docs_root
    return find_docs_root()  # raises FileNotFoundError if packaged/dev docs are missing
```

The embedding model is preloaded as a background task after the server starts listening (so the
startup message prints immediately). Routes that need the embedding model (query, ingest) should
check `app.state.embedding_model` and return a 503 with "Model loading, please retry" if it is
not yet available. This matches the existing TUI pattern of creating providers once in `AppState`.

---

### 2.12 Remote Chains Persistence Research (Config -> DB)

Feedback requested evaluation of moving `remote_provider_chains` out of `config.json` into DB-native
storage with UI design tooling.

**Conclusion:** This is reasonable and recommended. Remote chains are structured, sizable, and
iterative; SQLite-backed storage improves maintainability and keeps `config.json` focused on runtime
settings rather than user-authored graph payloads.

| Option | Pros | Cons |
|--------|------|------|
| Keep chains only in `config.json` | Simple startup loading, no migration needed | Large JSON payloads, difficult manual editing, poor versioning/search, weak UX for chain design |
| Move chains to DB (recommended) | Better CRUD/UI tooling, versionable revisions, searchable, cleaner config | Requires migration path + import/export and default seeding logic |

**Proposed schema:**

```sql
CREATE TABLE lsm_remote_chains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    is_default INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE lsm_remote_chain_links (
    id TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL REFERENCES lsm_remote_chains(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    provider_name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE lsm_remote_chain_revisions (
    id TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL REFERENCES lsm_remote_chains(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

**Migration/boot behavior:**
1. On first run, seed DB with packaged default chains.
2. If `remote_provider_chains[]` or `remote.chains` bootstrap selections exist in `config.json`,
   import/resolve once into DB and mark them user-managed.
3. Keep config field as backward-compatible **import-only** source for one release; DB is source of truth
   immediately after migration completes.
4. Expose JSON import/export per chain and bulk export for backup/portability.

**Split-brain prevention between config and DB (required):**
- After successful DB migration, config serialization stops writing `remote_provider_chains`.
- Loader still tolerates legacy `remote_provider_chains` in config, but only uses it for first-time import
  when DB has no chains yet.
- A migration marker table/flag prevents repeated imports on each startup.
- TUI/Web settings for remote chains read/write DB only.

**API surface additions:**
- `GET /api/remote-chains`
- `POST /api/remote-chains`
- `PUT /api/remote-chains/{id}`
- `DELETE /api/remote-chains/{id}`
- `POST /api/remote-chains/{id}/import` (JSON)
- `GET /api/remote-chains/{id}/export` (JSON)
- `GET /api/remote-chains/defaults` (packaged defaults list)

This aligns with the planned chain design tooling and avoids config-file bloat.

---

## 3. TUI Simplification

### 3.1 Design Direction

The simplified TUI is a **keyboard-first admin shell** — a secondary tool used alongside the web server
for operator-level access. It is no longer the primary user interface.

Three-zone layout:
```
┌──────────────────────────────────┐
│  Output / content area           │
│  (scrollable, read-only)         │
│                                  │
├──────────────────────────────────┤
│  [Command prompt]                │
├──────────────────────────────────┤
│  Shortcuts for current screen    │
└──────────────────────────────────┘
```

All interaction through the command prompt. No mouse buttons anywhere.

### 3.2 Tab Structure: 5 → 2

| Old Tab | Fate |
|---------|------|
| Query | Removed — queries happen in Web UI only |
| Ingest | Folded into unified `CommandScreen` |
| Remote | Folded into unified `CommandScreen` |
| Agents | Folded into unified `CommandScreen` |
| Settings | **Retained** as-is (button cleanup, keyboard-nav improvements) |

Keybindings: `Ctrl+H` → `CommandScreen`. `Ctrl+S` → `SettingsScreen`.

### 3.3 `CommandScreen` — The Unified REPL

**New file:** `lsm/ui/tui/screens/command.py`

```python
class CommandScreen(ManagedScreenMixin, Widget):
    """
    Unified keyboard-first command REPL.
    Replaces the separate Query, Ingest, Remote, and Agents screens.
    No query commands — the Web UI is the only query interface.
    """
    BINDINGS = [
        Binding("ctrl+s", "switch_to_settings", "Settings"),
    ]
```

All TUI commands unified into a single dispatch table. **No query or remote-query commands — queries
happen exclusively in the Web UI.** Users who attempt a query-like interaction receive a message
directing them to the web interface:

```
$ what is X?
This is the admin shell. For queries, open http://127.0.0.1:8080 in your browser.
Type /help for available commands.
```

**Full command list (post-feedback):**

| Command prefix | Examples | Origin |
|----------------|---------|--------|
| `/build` | `/build`, `/build --force`, `/build --dry-run` | IngestScreen |
| `/tag` | `/tag`, `/tag --max 500` | IngestScreen |
| `/wipe` | `/wipe` | IngestScreen |
| `/prune` | `/prune`, `/prune --max-versions 3` | new in TUI |
| `/cache` | `/cache clear` | new in TUI |
| `/remote` | `/remote list` — **list only, no query** | RemoteScreen (query removed) |
| `/agent` | `/agent list`, `/agent start <name>`, `/agent stop <name>` | AgentsScreen |
| `/cluster` | `/cluster build`, `/cluster visualize` | new in TUI |
| `/graph` | `/graph build-links` | new in TUI |
| `/health` | `/health` | new |
| `/log` | `/log`, `/log 50`, `/log --level WARNING` | new |
| `/embed-models` | `/embed-models`, `/embed-models --installed-only` | new |

**Removed commands (not in TUI, belong in Web UI):**

| Removed command | Reason |
|----------------|--------|
| Plain text → query | Queries happen in Web UI only |
| `/query [--mode] <text>` | Web UI only |
| `/mode <name>` | Mode selection is a Web UI concept |
| `/remote query <text>` | Remote queries go through Web UI query flow |

The bottom shortcut bar updates per-command to show the most relevant bindings for the currently
displayed content.

### 3.4 New TUI Commands

#### `/health`

```
$ /health
System Health
=============
Database       [OK]   SQLite @ ~/Local Second Mind/lsm.db (1,234,567 chunks)
Embedding      [OK]   all-MiniLM-L6-v2 (384d), device=cpu
LLM Provider   [OK]   openai/gpt-4o
Remote         [OK]   3 providers configured
Agents         [OK]   0 running, 2 scheduled
Server         [OK]   http://127.0.0.1:8080 (responding)
Logs           [OK]   ~/Local Second Mind/Logs/lsm-2026-03-02.log
```

The **Server** row polls `GET http://{config.server.host}:{config.server.port}/api/health`
with a 2-second timeout. Reports `[OK]` + URL if 200, `[DOWN]` + HTTP error if non-200, or
`[NOT RUNNING]` if connection refused (common when the web server is not started).

#### `/log`

```
$ /log           # last 20 lines from current log file
$ /log 50        # last 50 lines
$ /log --level WARNING   # filter to WARNING+
$ /log --follow  # tail -f style polling (Ctrl+C to stop)
```

Reads from `<GLOBAL_FOLDER>/Logs/lsm-YYYY-MM-DD.log`. The `--follow` mode uses an existing
`_ManagedTimer` (1-second poll) that re-reads the file tail, comparing byte offset to detect new content.

#### `/embed-models`

```
$ /embed-models
$ /embed-models --installed-only
```

Displays the same embedding inventory contract as the Admin screen: configured model, active
fine-tuned model (if any), registered fine-tuned models, and well-known catalog entries with
local-install availability flags.

### 3.5 Button Audit and Removal

All `Button` widgets removed from TUI screens. Replacements:

| Screen | Button | Replacement |
|--------|--------|-------------|
| Ingest | Build | `/build` command in CommandScreen |
| Ingest | Tag | `/tag` command |
| Ingest | Wipe | `/wipe` command |
| Settings | Save | `Enter` key or `/save` command |
| Settings | Reset | `/reset` command |
| Agents | Start | `/agent start <name>` |
| Agents | Pause | `/agent pause <name>` |
| Agents | Stop | `/agent stop <name>` |

### 3.6 Shell → TUI Module Migration

`lsm.ui.shell` is deleted as a package. Contents move to `lsm.ui.tui`:

| Old path | New path |
|----------|----------|
| `lsm/ui/shell/cli.py` | `lsm/ui/tui/cli.py` |
| `lsm/ui/shell/commands/agents.py` | `lsm/ui/tui/commands/agents.py` |
| `lsm/ui/shell/commands/__init__.py` | `lsm/ui/tui/commands/__init__.py` |
| `lsm/ui/shell/__init__.py` | *(deleted)* |

`lsm/__main__.py` import updates (e.g.):
```python
# Before
from lsm.ui.shell.cli import run_ingest, run_db, run_cache, run_migrate, run_cluster, run_finetune, run_graph
# After
from lsm.ui.tui.cli import run_ingest, run_db, run_cache, run_migrate, run_cluster, run_finetune, run_graph
```

**`lsm.ui.helpers` import update (required):**

`lsm/ui/helpers/commands/query.py` has lazy imports from the shell package that must be updated:
```python
# Before (lines 353, 356)
from lsm.ui.shell.commands.agents import handle_agent_command
from lsm.ui.shell.commands.agents import handle_memory_command
# After
from lsm.ui.tui.commands.agents import handle_agent_command
from lsm.ui.tui.commands.agents import handle_memory_command
```

`lsm.ui.helpers` (`helpers/commands/common.py`, `helpers/commands/query.py`) stays in place — it is
shared parsing consumed by both TUI commands and Web UI route handlers.

**Test files requiring import path updates (`tests/test_ui/shell/` → `tests/test_ui/tui/`):**
- `tests/test_ui/shell/test_cli.py`
- `tests/test_ui/shell/test_cli_routing.py`
- `tests/test_ui/shell/test_agents_commands.py`
- `tests/test_ui/shell/test_agent_interaction_commands.py`
- `tests/test_ui/shell/test_memory_commands.py`
- `tests/test_ui/shell/test_multi_agent_manager.py`
- `tests/test_ui/shell/test_meta_commands.py`
- `tests/test_ui/shell/test_schedule_commands.py`
- `tests/test_ui/shell/test_imports.py`

All `from lsm.ui.shell` imports in these files update to `from lsm.ui.tui`. The test directory
itself may be renamed or merged at the implementer's discretion.

---

## 4. CLI + TUI Unification

### 4.1 Entry Point Change Summary

| Invocation | v0.8.x | v0.9.0 |
|-----------|--------|--------|
| `lsm` | Launches Textual TUI | Starts FastAPI web server |
| `lsm cli` | *(did not exist)* | Launches Textual TUI |
| `lsm ingest build` | CLI batch | CLI batch (unchanged) |
| `lsm db prune` | CLI batch | CLI batch (unchanged) |
| `lsm eval retrieval` | CLI batch | CLI batch (unchanged) |
| *(all other subcommands)* | CLI batch | CLI batch (unchanged) |

**`__main__.py` changes:**
1. Add `subparsers.add_parser("cli", help="Start the interactive TUI admin shell")`.
2. No-subcommand handler: `from lsm.ui.web.server import run_server; return run_server(config)`.
3. `args.command == "cli"` handler: `from lsm.ui.tui.app import run_tui; return run_tui(config)`.

### 4.2 No More Interactive Shell Layer

The `lsm.ui.shell` REPL is removed. The TUI's `CommandScreen` is the only interactive shell.
There is no non-TUI REPL mode.

---

## 5. Module Structure After v0.9.0

```
lsm/
├── __main__.py              # Entry: no-args → web server; "cli" → TUI; batch subcommands
├── logging.py               # UPDATED in-place: + EventBufferHandler, TimedRotatingFileHandler
│                            #   (lsm/utils/logger.py DELETED; file stays at lsm/logging.py)
├── utils/
│   ├── paths.py             # MERGED: all from lsm/paths.py + original lsm/utils/paths.py
│   │                        #   (lsm/paths.py DELETED)
│   ├── text_processing.py   # (unchanged)
│   └── file_graph.py        # (unchanged)
├── config/
│   ├── models/
│   │   ├── lsm_config.py    # + server: ServerConfig field; imports from lsm.utils.paths
│   │   ├── global_config.py # imports from lsm.utils.paths (not lsm.paths)
│   │   ├── server.py        # NEW: ServerConfig dataclass
│   │   └── ...              # NotesConfig removed; all other models mostly unchanged
│   └── loader.py            # + load_server_config(); imports from lsm.utils.paths
├── ingest/
│   ├── embedding.py         # NEW: extracted load_embedding_model() for web server preload
│   └── ...                  # existing ingest modules unchanged
├── query/                   # note-export flow replaces standalone query/notes.py usage
├── providers/
│   ├── compaction.py        # NEW: shared conversation compaction primitive
│   └── ...                  # existing provider backends
├── remote/                  # (unchanged)
├── db/
│   ├── __init__.py          # CLEANED: no longer imports from lsm.vectordb
│   ├── connection.py        # CLEANED: conditional BaseVectorDBProvider import removed
│   ├── migration.py         # CLEANED: create_vectordb_provider import removed
│   ├── health.py            # (unchanged)
│   ├── enrichment.py        # (unchanged)
│   └── ...                  # (other db modules unchanged)
├── vectordb/                # UNCHANGED — stays as lsm.vectordb (rename cancelled)
│   ├── base.py              # (unchanged)
│   ├── factory.py           # (unchanged)
│   ├── sqlite_vec.py        # (unchanged — correctly imports from lsm.db)
│   ├── postgresql.py        # (unchanged)
│   └── chromadb.py          # (unchanged)
├── agents/                  # log_formatter no longer depends on lsm.utils.logger
├── eval/                    # (unchanged — has its own eval/cli.py)
├── finetune/                # (unchanged — stays at lsm.finetune)
└── ui/
    ├── web/
    │   ├── __init__.py
    │   ├── app.py           # create_app(config) -> FastAPI
    │   ├── server.py        # run_server(config) -> int
    │   ├── dependencies.py  # Depends: get_config, get_db_conn, get_event_buffer, get_docs_root
    │   ├── rendering.py     # render_markdown() — mistune wrapper + sanitiser
    │   ├── streaming.py     # event_stream_response(), SSE event format helpers
	    │   ├── services/
	    │   │   ├── __init__.py
	    │   │   ├── conversations.py  # retry/edit/delete/archive/search orchestration
	    │   │   ├── query.py          # query orchestration for SSE + context wiring
	    │   │   ├── remote_chains.py  # DB-backed remote-chain CRUD + import/export
	    │   │   └── embeddings.py     # embedding inventory + validation service
    │   ├── routes/
    │   │   ├── __init__.py
    │   │   ├── pages.py     # GET /query /ingest /agents /settings /admin /help
    │   │   ├── query.py     # POST /api/query, /api/query/stream/{id}, /api/query/mode
    │   │   ├── conversations.py # /api/conversations/* and /api/messages/*
    │   │   ├── ingest.py    # POST /api/ingest/*, /api/db/*, /api/cache/*
    │   │   ├── agents.py    # /api/agents/*
    │   │   ├── config.py    # GET/PUT /api/config
    │   │   ├── remote_chains.py # /api/remote-chains/*
    │   │   ├── health.py    # GET /api/health
    │   │   ├── admin.py     # /api/admin/*
    │   │   └── docs.py      # GET /help, GET /help/{slug}
    │   ├── templates/
    │   │   ├── base.html    # + dark-mode toggle, theme-init inline script
    │   │   ├── query.html
    │   │   ├── ingest.html
    │   │   ├── agents.html
    │   │   ├── settings.html
    │   │   ├── admin.html
    │   │   ├── help.html    # NEW: docs browser
    │   │   └── fragments/
    │   │       ├── query_submitted.html   # + marked.js rendering on SSE close
    │   │       ├── ingest_progress.html
    │   │       ├── agent_card.html
    │   │       ├── health_table.html
    │   │       ├── stats_table.html
    │   │       └── doc_page.html          # NEW: rendered doc content
    │   ├── docs/
    │   │   └── user-guide/
    │   │       ├── GETTING_STARTED.md
    │   │       ├── CONFIGURATION.md
    │   │       └── ...                    # packaged mirror synced from top-level docs/
    │   └── static/
    │       ├── css/main.css               # + CSS custom properties for dark/light themes
    │       └── js/
    │           ├── htmx.min.js
    │           ├── htmx-sse.js
    │           └── marked.min.js          # NEW: vendored marked.js
    ├── tui/
    │   ├── app.py           # Simplified: 2 tabs (Command + Settings), Ctrl+H / Ctrl+S
    │   ├── cli.py           # MOVED from lsm/ui/shell/cli.py
    │   ├── screens/
    │   │   ├── __init__.py
    │   │   ├── base.py      # ManagedScreenMixin (unchanged)
    │   │   ├── command.py   # NEW: unified REPL (replaces query/ingest/remote/agents)
    │   │   ├── settings.py  # Retained, buttons removed
    │   │   ├── help.py      # Retained
    │   │   └── main.py      # DELETED (was unused)
    │   ├── commands/
    │   │   ├── __init__.py
    │   │   └── agents.py    # MOVED from lsm/ui/shell/commands/agents.py
    │   ├── widgets/         # Existing widgets, mostly unchanged
    │   ├── state/           # AppState (unchanged)
    │   ├── presenters/      # Simplified (fewer screens)
    │   └── styles/
    │       ├── command.tcss # NEW (for CommandScreen)
    │       └── settings.tcss
    └── helpers/
        └── commands/        # UI-agnostic parsing — unchanged, shared by TUI + Web
            ├── common.py
            └── query.py

# DELETED packages / files:
#   lsm/paths.py              (merged into lsm/utils/paths.py)
#   lsm/utils/logger.py       (removed; functionality absorbed into lsm/logging.py and agent-local formatting helpers)
#   lsm/query/notes.py        (notes flow converged into chat/message export)
#   lsm/ui/shell/             (entire directory — moved to lsm/ui/tui/)
#   lsm/ui/tui/screens/query.py   (absorbed into command.py)
#   lsm/ui/tui/screens/ingest.py  (absorbed into command.py)
#   lsm/ui/tui/screens/remote.py  (absorbed into command.py)
#   lsm/ui/tui/screens/agents.py  (absorbed into command.py)
#   lsm/ui/tui/screens/main.py    (runtime-unused placeholder; tests updated accordingly)

# NOT DELETED (rename cancelled):
#   lsm/vectordb/             (stays as lsm.vectordb — circular dep fix in lsm.db instead)
```

---

## 6. Testing Considerations

### 6.1 Web UI Tests

**Framework:** `httpx.AsyncClient` with `ASGITransport` for async handler tests. FastAPI `TestClient`
for synchronous smoke tests.

**Unit tests (per route module):**
- `tests/web/test_query_routes.py` — mock query provider, verify HTML fragment returned with SSE div;
  verify SSE event sequence for a streaming response; verify per-conversation model override is passed
  into provider selection.
- `tests/web/test_conversation_routes.py` — verify archive/unarchive/delete conversation flows; verify
  retry variant creation + active variant switching; verify branch endpoint clones selected lineage;
  verify latest-only edit/delete guards and 409 behavior on stale mutations; verify conversation/message
  export endpoints return expected `markdown` and `json` payloads.
- `tests/web/test_ingest_routes.py` — mock ingest runner, verify progress SSE format; verify wipe
  requires confirm flag.
- `tests/web/test_agents_routes.py` — mock `AgentRuntimeManager`, verify agent card HTML, run-id based
  stop/pause/log routes, interaction queue fragment, required `ack` forwarding, stale `request_id`
  409 behavior, and permission-vs-reply control rendering by `request_type`.
- `tests/web/test_provider_routes.py` — verify communication provider filtering, OAuth connect redirect,
  callback state validation, disconnect behavior, non-OAuth rejection for `connect`, provider-test
  success/error fragments, and blank-secret preservation semantics on Settings saves.
- `tests/web/test_config_routes.py` — GET returns valid JSON; PUT supports both JSON and HTMX form
  payloads; invalid config returns structured error payloads.
- `tests/web/test_health_routes.py` — verify JSON structure matches documented health schema.
- `tests/web/test_admin_routes.py` — mock eval/migrate/cluster/graph/finetune runners, verify SSE events;
  verify finetune activate/delete-model behavior and embedding inventory/validate endpoint filters.
- `tests/web/test_docs_routes.py` — mock `find_docs_root()` to point at a temp dir with fake markdown
  files; verify `/help` renders index, `/help/{slug}` renders correct content; verify clear error
  behavior when docs are missing (no silent fallback state); verify `chat-exports` slug resolves.
- `tests/web/test_remote_chains_routes.py` — verify DB-backed CRUD, defaults listing, and JSON
  import/export behavior.

**SSE test helper:**
```python
async def collect_sse_events(async_client, url: str, max_events: int = 50) -> list[dict]:
    events = []
    async with async_client.stream("GET", url) as r:
        async for line in r.aiter_lines():
            if line.startswith("data:") and line != "data: {}":
                events.append(json.loads(line[5:].strip()))
            if line.startswith("event: done"):
                break
            if len(events) >= max_events:
                break
    return events
```

**Contract tests:** Verify API response schemas match the documented API surface. Useful for ensuring
Obsidian plugin compatibility.

**Conversation mutation integration tests:**
- Retry latest assistant twice, switch active variant, send next user turn, and assert only selected
  variant is included in prompt context.
- Edit latest user message and assert old assistant variants are removed before regeneration.
- Edit latest assistant response and assert edited content is used in next-turn prompt context.
- Delete latest assistant with multiple variants and assert the entire variant group is removed.
- Branch from a mid-thread message and assert the new conversation history is truncated at the branch
  point and diverges independently.
- Export a full conversation and a single message; verify content, metadata, and format selection.
- Export redaction test: `redact=true` masks known secret patterns; `redact=false` preserves raw text.

**Responsive UI tests:**
- Playwright viewport tests (`390x844`, `768x1024`, `1440x900`) verify sidebar collapse behavior,
  composer stickiness, and no horizontal overflow.

**Server exposure tests:**
- Config validation test for `server.expose_to_lan`; startup test ensures bind host resolves to
  `0.0.0.0` when enabled.
- LAN token test: non-GET API requests fail with 401/403 when token missing/invalid in LAN mode, and
  succeed when valid token is provided.
- LAN read-path test: representative GET/SSE routes remain reachable without token in v0.9 (documented
  read-only behavior).

**Template render tests:** Render each Jinja2 template with known context; assert key HTML elements
are present (sidebar links, form targets, SSE attributes, `data-theme` attribute path).

**Dark mode tests:**
- Render `base.html` and assert the inline theme-init `<script>` block is present in `<head>`.
- Assert `#theme-toggle` element exists in sidebar.
- Assert CSS file includes `[data-theme="dark"]` custom property overrides.

**Markdown rendering tests:**
- Unit test `render_markdown()`: verify `**bold**` → `<strong>bold</strong>`, code blocks, tables.
- Security test: assert `render_markdown("<script>alert(1)</script>")` does not return a `<script>` tag.
- Security test: assert `render_markdown("[x](javascript:alert(1))")` does not produce a `javascript:` href.

### 6.2 TUI Tests

- Existing `tui_fast` pattern (fake widget doubles) adapts to the new `CommandScreen`.
- `CommandScreen` dispatch table is the primary test surface — each command class is unit-tested
  independently.
- `health` command: mock `check_db_health` + `requests.get` (for server poll); verify output table
  rows and status indicators.
- `log` command: mock `Path.open` with synthetic log content; verify tail and level-filter behaviour.
- `embed-models` command: verify inventory formatting, installed-only filter behavior, and active/configured markers.
- After tab collapse: update existing `tui_slow` and `tui_integration` tests that reference old tab
  structure.
- Verify that plain text input returns the "use the web UI" message and does not attempt a query.
- Verify that `/query` and `/mode` commands are not registered in the dispatch table.

### 6.3 Logging Tests

- After `setup_logging(global_folder=tmp_path)`: assert `<tmp_path>/Logs/` created; emit a record;
  assert file written.
- `EventBufferHandler`: subscribe a mock callback; emit a record; verify callback invoked with record.
- Thread-safety test: emit records from 10 threads simultaneously; verify no `deque` corruption.
- Redaction test: emit known secret patterns and assert console/file/event-buffer payloads are redacted.
- Redaction parity test: same log call produces identical redacted content across all sinks.
- Verify `from lsm.utils.logger import PlainTextLogger` raises `ModuleNotFoundError` (module deleted).
- Verify `from lsm.logging import setup_logging, get_logger, get_event_buffer` works correctly.
- Verify records from `lsm.ingest.*`, `lsm.query.*`, `lsm.agents.*` all propagate to the `"lsm"` root
  logger handler (confirms root logger name and file location are consistent).

### 6.4 Path Tests

- All existing path-helper tests continue to pass against the merged `lsm/utils/paths.py`.
- Smoke import test: `import lsm.paths` raises `ModuleNotFoundError`.
- `ensure_global_folders` test: assert `Chats/` and `Logs/` subdirectories are created; `Notes/` is not
  created by default in v0.9.

### 6.5 Conversation DB Tests

- Schema test: create tables, insert a conversation + messages, verify cascade delete, verify index.
- Variant test: create assistant siblings in same `variant_group_id`; verify only one active variant is
  allowed by the unique partial index.
- Variant index test: verify `(variant_group_id, variant_index)` uniqueness prevents duplicate numbering
  under concurrent retry attempts.
- Variant retention test: create high-count variants (e.g. 100) and verify no pruning occurs.
- Archive test: archive/unarchive conversation and verify default list excludes archived rows.
- Search test: populate message/citation text plus title/mode and verify multi-field search returns
  archived + non-archived rows.
- Branch test: create branch conversation and verify `branched_from_*` metadata is persisted.
- Export test: generate conversation/message export payloads and verify markdown/json serialization with
  stable metadata fields.
- Migration test: write synthetic markdown transcripts to a temp folder, run migration routine, verify
  correct records in DB, verify flat files untouched.
- Query test: retrieve conversation list ordered by `updated_at DESC`.
- Concurrency test: two concurrent latest-message mutations on same conversation produce one success and
  one deterministic `409 Conflict` (no partial writes).

### 6.6 Provider Compaction Tests

- Unit test `lsm.providers.compaction.compact_messages()` for `fresh`, `summary`, and provider-first modes.
- OpenAI provider test: mock provider-native compaction/state continuation path and verify fallback when
  provider hook returns `None` or raises unsupported error.
- Cross-caller parity test: same input history produces equivalent compacted output when called from
  agent harness and from web query service.
- Cache invalidation test: retry/edit/delete/branch operations invalidate or re-anchor provider chain IDs
  as expected.

### 6.7 Vectordb Circular Dependency Tests

- Import test: `import lsm.db; import lsm.vectordb` — verify no circular import error.
- Verify `lsm.db.__init__` module does not import from `lsm.vectordb` (grep or `importlib` introspection).
- Integration test: create a `SqliteVecProvider`, perform an insert and search — confirms `lsm.vectordb`
  still correctly uses `lsm.db` utilities after the cleanup.

### 6.8 Remote Chain DB Tests

- Migration test: import legacy `remote_provider_chains[]` config into DB tables once and mark migration
  complete.
- Bootstrap test: import `remote.chains` preconfigured selections into DB when DB is empty.
- CRUD test: create/update/delete chain with ordered links and revision rows.
- Import/export round-trip test: JSON export then import yields equivalent chain structure.
- Split-brain test: after migration marker is set, config `remote_provider_chains[]` edits are ignored
  and DB remains authoritative.

### 6.9 Notes Convergence Compatibility Tests

- Backward config test: config containing legacy `notes` object still loads without hard failure
  (warning allowed) while notes-save actions are disabled.
- TUI settings test: chat-settings tab migration removes notes fields without breaking settings save flow.
- Legacy command test: invoking `/note`/`/notes` in command parsers returns explicit "replaced by export"
  guidance.

### 6.10 Embedding Inventory Tests

- Inventory service test: includes configured model, active registry model, registered fine-tuned
  models, and catalog entries in one normalized response.
- Local availability test: `is_installed_locally` and `load_check` correctly reflect existing/missing
  local paths and cache entries without network access.
- Filter test: `installed_only=true` and `source=` filters produce deterministic subsets.
- Unregister test: `DELETE /api/admin/finetune/models/{model_id}` removes only registry row metadata
  and does not delete filesystem model artifacts.
- Validation endpoint test: `POST /api/admin/embeddings/validate` returns clear status for valid model,
  missing model path, and load error.

---

## 7. Implementation Sequencing

Order minimises breakage: low-blast-radius infrastructure first, high-blast-radius changes last.

1. **Update `lsm/logging.py` in-place** — add `EventBufferHandler`, `TimedRotatingFileHandler`, and a
   logger-level `RedactingFilter`; update `setup_logging()` signature to return `EventBufferHandler`;
   add `get_event_buffer()` module-level accessor; delete `configure_logging_from_args()` and update
   `lsm/__main__.py` to call `setup_logging()` directly. File stays at `lsm/logging.py`. Run full
   test suite.

2. **Retire `lsm/utils/logger.py` safely** — migrate `lsm/agents/log_formatter.py` off
   `LogVerbosity`/`normalize_verbosity`, remove logger re-exports from `lsm/utils/__init__.py`, then
   delete `lsm/utils/logger.py`. Run full test suite.

3. **Path consolidation** — merge `lsm/paths.py` into `lsm/utils/paths.py`; update all imports; add
   `Logs/` to `ensure_global_folders()`; delete `lsm/paths.py`. Run full test suite.

4. **Fix `lsm.db` circular dependency** — four targeted changes:
   (a) `lsm/db/__init__.py`: remove `create_vectordb_provider` re-export; grep callers → update to
   `from lsm.vectordb.factory import create_vectordb_provider`.
   (b) `lsm/db/connection.py`: replace `_is_provider_instance()` body with duck-type check only
   (remove try/except vectordb import — fallback was already sufficient).
   (c) `lsm/db/connection.py`: replace the `create_vectordb_provider()` wrapper import so it no longer
   depends on `lsm.db` facade re-exports.
   (d) `lsm/db/migration.py`: delete top-level `from lsm.vectordb import create_vectordb_provider`;
   add lazy function-scope imports inside `_provider_from_source()` and `_provider_from_target()`,
   matching the existing Chroma lazy import pattern already in place.
   Run full test suite. Confirm with circular-dependency smoke test.

5. **`ServerConfig` dataclass** — add to config models; add `server` field to `LSMConfig`; include
   `expose_to_lan`; add LAN token enforcement fields (`require_access_token`, `access_token`);
   enforce non-GET `/api/*` token checks in LAN mode; update config loader and serialiser; write parser tests.

6. **`lsm.ui.shell` → `lsm.ui.tui`** — move `cli.py` and `commands/agents.py`; update all imports in
   `__main__.py` and test files; delete `lsm/ui/shell/`. Run full test suite.

7. **`__main__.py` entry point** — no-args → `run_server()`; add `cli` subcommand → `run_tui()`. Write
   new startup path tests.

8. **Web server scaffold** — `app.py`, `server.py`, `dependencies.py`, `streaming.py`, `rendering.py`,
   stub page routes, `GET /api/health`. Server starts and `http://127.0.0.1:8080` is reachable.
   Vendor HTMX, htmx-sse, and marked.js into `static/js/`. Also create `lsm/ingest/embedding.py`
   with a `load_embedding_model(config) -> SentenceTransformer` function extracted from the inline
   loading logic in `lsm/ingest/pipeline.py` (lines 424-426). This function is needed by the web
   server startup preload and keeps pipeline.py from being imported at server boot time.

9. **Dark mode CSS** — write `main.css` with CSS custom properties; embed theme-init `<script>` in
   `base.html`; implement toggle. All pages respect dark mode preference.

10. **Chat history DB schema v2** — create/extend `lsm_conversations` + `lsm_messages` with archive
   flags, assistant variant-group fields, edit metadata, branch metadata, model override fields, and
   multi-table FTS search schema (including variant-index uniqueness); write migration from flat files;
   update conversation read/write in query provider.

11. **Conversation services + routes** — implement `lsm/ui/web/services/conversations.py` and
   `/api/conversations/*`, `/api/messages/*` endpoints for archive/unarchive/delete, retry variants,
   variant selection, edit-last (user + assistant), delete-last semantics, branch creation, and
   per-conversation model selection, plus conversation/message export endpoints. Ensure all context-mutating
   operations run in `BEGIN IMMEDIATE` transactions with latest-state conflict checks (`409` on stale).

12. **Notes convergence cleanup** — full removal of standalone notes system (see §2.5.5 removal
    scope for the complete file list). Includes:
    - delete `lsm/query/notes.py` (424 lines),
    - remove `NotesConfig` class from `lsm/config/models/modes.py` and `notes` field from `LSMConfig`,
    - remove `get_notes_folder()` from `lsm/utils/paths.py` (merged in step 3),
    - remove notes fields from TUI settings view-model/widgets,
    - replace `/note`/`/notes` command handlers with export guidance,
    - update config loader to tolerate old `"notes"` key (deprecation warning) but stop writing it,
    - ensure chat/message export is the canonical replacement.

13. **Compaction primitive extraction** — move compaction logic from `AgentHarness` into
    `lsm.providers.compaction`; wire both agent harness and web query service to this shared primitive.

14. **Web UI Query screen (advanced chat controls)** — full SSE query flow, conversation list + search
    in sidebar, archived view, retry/variant picker/edit/delete on latest assistant, branch from any
    message,
    edit/delete on latest user, model selector, citation cards, streaming markdown via `marked.js`,
    stored message rendering via `mistune`, responsive mobile behavior, and export actions.

15. **Web UI Ingest screen** — all ingest operations, SSE progress streaming, wipe confirmation.

16. **Web UI Agents screen** — run-id based agent list/start/stop/pause, log SSE stream, interaction
    inbox with explicit acknowledge/respond flow, and specialized email/calendar assistant launchers.

17. **Web UI Settings screen** — all config sections, read/write, inline validation, LAN exposure
    controls, communication-provider secret handling, and browser-native OAuth connect/test/disconnect
    flows for email/calendar providers.

18. **Remote chains DB migration + API** — seed defaults, one-time import from config chains (ships in
    v0.9.0), CRUD +
    revision tracking, JSON import/export endpoints, settings/admin integration, and split-brain
    prevention rules so config no longer rewrites chain definitions post-migration.

19. **Web UI Admin screen** — health, eval, migrate, cluster, graph, finetune (with pair-count
    preview), finetune unregister endpoint, embedding-model inventory/validate endpoints + panel,
    `lsm finetune inventory` CLI parity output, stats, log tail.

20. **Web UI Help & Docs screen** — `find_docs_root()`, `DOCS_NAV` registry, `docs.py` route,
    `help.html` template, markdown rendering with link rewriting, packaged docs mirror under
    `lsm/ui/web/docs/user-guide/`, docs-sync build script, and docs content updates: rename
    `NOTES.md` → `CHAT_EXPORTS.md`, create new `FINETUNE.md` (both referenced by `DOCS_NAV`).
    Write docs route tests.

21. **TUI simplification** — implement `CommandScreen` REPL (collapsing 4 tabs → 1); `Ctrl+H`
    keybinding; `/health`, `/log`, and `/embed-models` commands; remove query/mode/remote-query
    commands; remove all buttons from TUI screens. Update `lsm/ui/tui/screens/__init__.py` lazy `__getattr__` imports:
    remove `MainScreen`, `QueryScreen`, `IngestScreen`, `RemoteScreen`, `AgentsScreen` entries;
    add `CommandScreen` entry. Update `__all__` list to match. Update
    `lsm/ui/tui/screens/help.py`: its `ContextType` literal and `_CONTEXT_LABELS` dict reference
    the old 5-tab structure (`"query"`, `"ingest"`, `"remote"`, `"agents"`, `"settings"`); update
    to match the new 2-screen layout (`"command"`, `"settings"`). Update global shortcut text in
    `_GLOBAL_SHORTCUTS` to reflect `Ctrl+H`/`Ctrl+S` bindings. Update TUI tests.

---

## 8. Risk Areas

| Risk | Mitigation |
|------|-----------|
| `lsm.logging` updated in-place — `lsm.utils.logger` callsites missed | Comprehensive grep for `PlainTextLogger` / `LogVerbosity` / `create_plaintext_logger`; explicitly migrate `lsm/agents/log_formatter.py`; smoke test: `from lsm.utils.logger import PlainTextLogger` raises `ModuleNotFoundError` |
| `lsm.db` circular import cleanup breaks callers of re-exported `create_vectordb_provider` | Grep all `from lsm.db import create_vectordb_provider` before removing; update each callsite; run full test suite |
| HTMX SSE chat streaming UX correctness | Write SSE test helpers early; standardise `event: done` signal across all streaming endpoints; integration-test full query flow |
| marked.js streaming flash (raw markdown visible briefly) | Acceptable for a local single-user tool; document as known behaviour; can be refined post-v0.9.0 |
| Prompt injection via LLM markdown output | `mistune` configured with `escape=True`; additionally sanitize rendered HTML with `nh3.clean(...)`; add security tests |
| Dark mode flash-of-wrong-theme | Inline `<script>` in `<head>` (before CSS load) reads `localStorage` and sets `data-theme` synchronously; no flash |
| Web server port conflict on developer machine | `ServerConfig` makes port configurable; Uvicorn prints clear error on bind failure |
| `EventBufferHandler` blocking on slow subscriber | Callbacks wrapped in `try/except`; long-running Web SSE consumers run in asyncio task reading from a queue |
| TUI simplification breaks existing TUI tests | Run `tui_slow` and `tui_integration` tests after each screen removal step; fix before proceeding |
| Agent runtime invisible across TUI/Web processes | Shared `lsm_agent_runtime_state` SQLite table with heartbeat; stale detection cleans up crashed agents |
| `PlainTextLogger` callsites missed during migration | Full grep confirms scope; `import lsm.utils.logger` raises `ModuleNotFoundError` after deletion |
| Chat DB migration from flat files | Flat files remain untouched on disk as backup; migration runs once on startup if tables empty; idempotent |
| Concurrent config file write (Web UI + TUI) | Last-write-wins (existing `save_config_to_file` behaviour); acceptable for single-user tool; document |
| `python-multipart` not installed | FastAPI raises a clear error on first form POST; add to `pyproject.toml` explicitly |
| `mistune` not installed | `ImportError` on first page render; add to `pyproject.toml` explicitly |
| `nh3` not installed | `ImportError` on first markdown render with sanitisation; add to `pyproject.toml` explicitly |
| Ingest SSE stream disconnected mid-progress | Backend job continues regardless; `EventSource` reconnects automatically; progress resumes from queue |
| Large template/static directories not packaged correctly | Add `lsm/ui/web/templates/` and `lsm/ui/web/static/` to `pyproject.toml` `[tool.setuptools.package-data]` |
| Docs not found in installed package | Ship docs via packaged mirror (`lsm/ui/web/docs/user-guide`) plus prebuild sync script; `find_docs_root()` raises clear error if mirror missing |
| Packaged docs mirror drifts from top-level docs | Add CI/prebuild check that runs docs sync and fails on diff |
| Redaction misses expose sensitive data across sinks | Enforce logger-level mandatory `RedactingFilter`; add parity tests across console/file/event-buffer and maintain redaction pattern registry |
| Retry/edit/delete chat mutations race with new incoming messages | Enforce latest-only mutation preconditions server-side and reject stale operations with 409 + refresh hint |
| Assistant variant state drift (multiple active rows in one group) | Unique partial index on `(variant_group_id, is_active_variant)` + transactional updates |
| Provider response-chain IDs expire or become invalid | Treat local DB transcript as canonical; provider IDs are opportunistic optimization only |
| Shared compaction primitive causes behavior regressions between agents and web query | Add cross-caller parity tests and rollout behind one config toggle for first release |
| Unlimited assistant variants grow conversation DB unexpectedly | Add stats visibility in Admin screen (variant counts); allow future maintenance prune command if needed |
| Editing assistant text can desync displayed sources vs edited content | Mark edited assistant messages with `edited_at` badge and preserve original `sources_json` provenance metadata |
| LAN exposure increases attack surface | Keep default localhost bind; explicit `expose_to_lan` toggle; require LAN-mode access token for state-changing routes; document firewall requirement |
| Remote chain config->DB migration drops complex chain metadata | One-shot migration with dry-run report + JSON backup export before write |
| Per-conversation model override points to unavailable model | Validate against provider registry on save and fall back to service default with warning |
| Removing standalone notes flow surprises users who rely on old behavior | Add explicit export actions in chat UI, migration note in release docs, and compatibility message if old note commands are invoked |
| LAN-exposed state-changing endpoints are vulnerable to unauthorized browser/API writes | Require per-request access token in LAN mode for all non-GET API routes; reject missing/invalid tokens |
| LAN mode leaves read-only GET/SSE API endpoints unauthenticated in v0.9 | Document LAN mode as trusted-network-only and recommend firewall/VPN isolation when enabled |
| Export endpoints leak secrets unintentionally | Default `redact=true`; require explicit opt-out and show warning for raw export |
| Legacy `notes` config breaks loader/UI after removal | Keep backward-tolerant loader path for one cycle and migrate docs/tests to export flow |
| Remote chains stored in both config and DB diverge over time | Introduce migration marker; DB is sole writer after migration; stop serializing chains to config |
| Conversation search becomes slow on large datasets | Use SQLite FTS5 with index-backed queries; cap result count and paginate |
| Embedding inventory reports false "installed" status due cache-layout differences across platforms | Use conservative local-path checks, mark uncertain entries as `unknown`, and keep explicit `validate` action |
| Active fine-tuned model in DB and configured `global.embed_model` diverge and confuse users | Inventory UI shows both flags (`is_active_registry` and `is_configured_default`) and provides explicit guidance/action labels |
| Stale fine-tuned registry rows accumulate after model folder deletion/moves | Add explicit unregister endpoint (`DELETE /api/admin/finetune/models/{model_id}`) and surface action in inventory/list UI |
| Embedding inventory endpoint causes slow page load on large model caches | Avoid recursive deep scans; cap scan depth and cache inventory results with short TTL |
| finetune training with insufficient pairs | Pair count preview in Admin screen (before training) warns user; CLI help text includes minimum guidance |

---

## 9. Decisions Made

Confirmed decisions reflected throughout this document.

| Topic | Decision |
|-------|---------|
| Backend framework | **FastAPI** |
| Frontend approach | **HTMX + Jinja2** — no Node.js build toolchain |
| Streaming protocol | **SSE** — WebSockets deferred to a future version |
| Default `lsm` behaviour | Starts **web server**, prints address to stdout once bound |
| TUI subcommand | `lsm cli` |
| TUI ↔ web server co-running | Always **separate OS processes** — no in-process co-running |
| Agent runtime coordination | **Shared SQLite table** (`lsm_agent_runtime_state`) with heartbeat for cross-process visibility |
| Config file authority | Both UIs read/write the **same `config.json`** — no sync needed |
| Settings API wire format | `PUT /api/config` supports both HTMX form submissions and JSON clients |
| Chat migration trigger | **Both**: auto-run on first startup when needed, plus explicit flag (`lsm migrate --chats`) on the existing `migrate` subcommand |
| Logging module location | **`lsm/logging.py`** — stays in root; updated in-place with new handler classes; `lsm/utils/logger.py` deleted |
| Logger hierarchy root | **Named logger `"lsm"`** — file stays in root to match its hierarchy; moving to `lsm/utils/` was rejected as misleading |
| Log redaction model | **Always-on, logger-level redaction** before any sink emit; no sink-specific redaction logic |
| DB-backed log persistence | **Scrapped for now** (out of v0.9 scope and no post-v0.9 commitment in this plan) |
| Agent log verbosity model | Switch from `normal/verbose/debug` helper enum to standard logging levels |
| `configure_logging_from_args` | **Deleted** — replaced with direct `setup_logging()` calls at all callsites |
| `PlainTextLogger` / `create_plaintext_logger` | **Dead code** — zero callers anywhere; deleted with no migration |
| Embedding model warm-up | **Background preload** — starts after server is listening; routes return 503 if model not yet loaded |
| Path consolidation | **`lsm/utils/paths.py`** is the single module — `lsm/paths.py` deleted with no alias |
| `lsm.vectordb` rename | **Cancelled** — nesting it in `lsm.db` would worsen the circular dependency. `lsm.vectordb` stays as a peer top-level package |
| Why `lsm.vectordb` stays a peer package | Even without cycles, `lsm.db` remains foundational while `lsm.vectordb` is an application layer on top; keeping peer packages preserves this dependency boundary |
| `lsm.db` circular dependency fix | **Four targeted changes**: (a) remove re-export from `__init__`, (b) replace isinstance-guard with duck-type in `connection.py`, (c) remove `connection.py` indirection through `lsm.db` facade, (d) lazy imports in `migration.py` matching existing Chroma pattern |
| `lsm.finetune` location | **Stays at `lsm.finetune`** — no move |
| `lsm.ui.shell` | **Absorbed into `lsm.ui.tui`** — `shell/` package deleted entirely |
| TUI keybinding for CommandScreen | **`Ctrl+H`** |
| TUI query commands | **Removed entirely** — no queries in TUI; plain text shows a "use the web UI" message |
| TUI remote commands | `/remote list` retained; `/remote query` removed |
| Server status in TUI `/health` | **Yes** — polls `GET /api/health` with 2-second timeout |
| Ingest screen scope | **All** ingest-related commands (build + all flags, tag, wipe, db prune, db complete, cache clear) |
| Admin screen scope | Health, eval, migrate, cluster, graph, finetune (with pair-count preview), statistics, live log tail |
| Embedding model visibility | **Required inventory view** (configured + registry + catalog + local availability) in Admin UI/API and TUI `/embed-models` |
| Fine-tuned registry cleanup | **Supported** via explicit unregister/delete action; does not delete model files automatically |
| Chat history persistence | **SQLite DB** — long-term maintainable and required for future query pipeline inclusion |
| Chat message storage format | **Raw markdown** stored in DB; rendered server-side by `mistune` on load |
| "Infinite conversation" strategy | **Local transcript is canonical**; provider state (`previous_response_id` etc.) is optional acceleration only |
| Assistant retry behavior | **Keep sibling assistant variants without a hard cap**; exactly one variant is active for future context |
| Edit/delete scope in v0.9 | **Latest-message only** server-enforced mutations (edit-last-user, edit-last-assistant, delete-last-user, delete-last-assistant-group) |
| Delete latest assistant semantics | **Delete the entire latest assistant variant group** for that user turn |
| Chat branching | **Supported** — branch from selected message into a new conversation lineage |
| Chat archive semantics | **Dedicated archived view** plus search visibility; unarchive restores active list visibility |
| Chat deletion semantics | **Permanent hard delete** of conversation and messages |
| Chat search backend | **SQLite FTS5** across title, mode, message content, and citation metadata; search includes archived and active chats |
| Chat model selection | **Per-conversation provider/model override** in Query UI |
| Compaction ownership | **`lsm.providers` primitive** consumed by both agents and web query server |
| Port default | **127.0.0.1:8080** — configurable via `"server"` config object |
| Local network exposure | **Supported via `server.expose_to_lan`** (binds `0.0.0.0`); default remains localhost-only |
| LAN-mode state-change protection | **Required access token** for non-GET API routes when LAN exposure is enabled |
| Browser auto-open | **No** |
| Authentication (v0.9.0) | **No user-account auth**; localhost mode relies on loopback, LAN mode uses access-token write protection |
| Responsive/mobile support | **Required baseline support** for mobile and tablet layouts in web UI |
| Dark mode | **Supported** — default follows `prefers-color-scheme`; user can toggle; preference in `localStorage` |
| Markdown rendering (chat) | **`marked.js`** (client-side, vendored) for streaming; **`mistune`** (server-side) for stored messages |
| Markdown rendering (docs) | **`mistune`** (server-side) — same library as chat stored messages |
| HTML sanitisation | **`nh3`** post-processing sanitisation on rendered markdown before `| safe` template insertion (`bleach` is deprecated; `nh3` is the Rust-backed replacement) |
| Notes system | **Converged into chat exports** — standalone notes-save workflow removed |
| Legacy notes config handling | **Backward-tolerant read path** (warning) for one cycle, then full removal |
| Export capability | **Both conversation and single-message export** (`markdown` + `json`) |
| Export redaction default | **Enabled** (`redact=true`) with explicit opt-out for raw export |
| `Notes/` folder creation | **Removed from default global-folder bootstrap** in v0.9 |
| Docs in Web UI | **Yes** — Help screen serves `docs/user-guide/` files rendered as HTML via `/help/{slug}` |
| Docs bundling | **Packaged mirror** under `lsm/ui/web/docs/user-guide/`, synced from top-level docs prebuild and included via `pyproject.toml` package-data |
| Docs source of truth | Top-level `docs/user-guide/`; packaged mirror is generated/synced for runtime distribution |
| Remote chains persistence | **Move to DB-backed storage** with default seeding + JSON import/export |
| Remote chains rollout timing | **Ship in v0.9.0** (not staged to v0.9.x) |
| Remote chains authority after migration | **DB is source of truth**; config chains are import-only bootstrap input |
| Legacy chat markdown files after DB migration | **Retained**; user decides whether/when to clean up |
| Chat streaming markdown UX | **Raw text during stream, render on `event: done`** — acceptable for a local tool; incremental rendering deferred |
| Shims / backwards compat | **No import-path shims/aliases**; limited one-cycle config read-tolerance for legacy `notes` and remote-chain keys only |

---

## 10. Clarifications Resolved:

Integrated from `**User Feedback:**` blocks:
1. Chat migration supports both auto-run on first startup and explicit flag (`lsm migrate --chats`).
2. `lsm/utils/logger.py` migration uses standard logging levels.
3. Packaged docs mirror approach is accepted.
4. `PUT /api/config` keeps dual-mode (HTMX form + JSON client) with one shared write path.
5. Legacy markdown chat transcripts remain on disk; user decides cleanup timing.
6. DB-backed logs are scrapped for now; this plan does not include DB log persistence.
7. Log redaction is centralized inside `lsm/logging.py` and applies to all outputs before emit.
8. Assistant variant retention has no cap (single-user local system assumption).
9. Editing latest user removes old assistant variants and regenerates response.
10. Deleting latest assistant removes the entire assistant variant group for that turn.
11. Archive UX uses a dedicated archived view in addition to search visibility.
12. Chat search scope includes title, mode, message content, and citation metadata.
13. Server-side cache implications for variants/edits/deletes/branches are now integrated with explicit
    chain invalidation/re-anchor rules.
14. Full branch-to-new-chat support is included in schema, API, and UI design.
15. Per-conversation model selection is included in Query UI + API.
16. Editing the latest assistant response is included and fed back into future context.
17. Responsive/mobile web support is now part of v0.9 requirements.
18. Local-network server exposure is supported via server config (`expose_to_lan`).
19. Remote chains are researched as DB-backed with default preloads and JSON import/export.
20. Chat branching is designed as branch-from-any-message (full branch support).
21. Remote chains config->DB migration ships in v0.9.0 (not delayed to v0.9.x).
22. Standalone notes flow is considered redundant; chat/message export is the canonical replacement.
23. `configure_logging_from_args()` removed; callsites replaced with direct `setup_logging()` calls.
24. Chat migration uses `lsm migrate --chats` flag on existing `migrate` subcommand (not a separate command).
25. Agent runtime coordination uses shared SQLite table (`lsm_agent_runtime_state`) with heartbeat.
26. `bleach` replaced with `nh3` (bleach is deprecated since Jan 2023; nh3 is the Rust-backed replacement).
27. Embedding model preloaded as background task after server starts listening (non-blocking startup).
28. Plan now includes explicit embedding-model inventory visibility (Admin UI/API + TUI command) so
    configured, fine-tuned, catalog, and locally available models can be tracked in one place.
29. Fine-tuned model registry cleanup is explicit (`DELETE /api/admin/finetune/models/{model_id}`),
    and registry deletion does not implicitly delete model files from disk.
30. Web UI agent interaction prompts reuse the existing `InteractionRequest` contract and require an
    explicit `ack` route so browser rendering participates in the current two-phase timeout model.
31. Web UI live agent operations are keyed by `agent_id` (run identity), not `agent_name`, and the
    shared runtime-state table is run-centric for multi-run correctness.
32. Web UI does not expose `approve_session` in v0.9 because current approval caching is
    process-global rather than browser-session scoped.
33. Email/calendar support in the Web UI is delivered through Settings + specialized assistant
    launchers, not through separate standalone mail/calendar screens.
34. Browser-native OAuth routes are added for communication providers; normal Web UI agent/background
    flows use non-interactive tokens only and never spawn the temporary localhost callback server.
35. Agent interaction waits default to infinite duration, while fixed timeout behavior remains
    configurable for deployments that want automatic expiry.

## 11. Clarifications Required

No further clarifications are required for this research plan.

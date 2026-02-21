# Phase 1: Core Infrastructure & Configuration

**Why first:** Utils module and tiered model config are consumed by everything that follows. Logging changes and test harness shape all later agent work.

| Task | Description | Depends On |
| --- | --- | --- |
| 1.1 | Create `lsm.utils` module | None |
| 1.2 | Agent log format conversion | 1.1 |
| 1.3 | Tiered model configuration | None |
| 1.4 | Agent/tool test harness | None |

## 1.1: Create `lsm.utils` Module (COMPLETED)

- **Description:** Establish a shared utilities package for cross-cutting concerns currently scattered across modules.
- **Tasks:**
  - Create `lsm/utils/__init__.py` with public API.
  - Create `lsm/utils/logger.py` — centralized logger factory with configurable output (file, stream), format (plain text), and level (normal/verbose/debug).
  - Create `lsm/utils/paths.py` — common path resolution helpers (currently duplicated in config loader and agent harness).
  - Migrate existing utility code from `lsm/ingest/utils.py`, `lsm/agents/log_formatter.py`, and similar locations to use shared utilities where appropriate. Do NOT move ingest-specific code.
  - Commit changes to git using the COMMIT_MESSAGE format.
- **Files:**
  - `lsm/utils/__init__.py`
  - `lsm/utils/logger.py`
  - `lsm/utils/paths.py`
- **Success criteria:** Shared logger is importable from `lsm.utils`, existing tests pass, no duplication of path resolution logic.

## 1.2: Agent Log Format Conversion (COMPLETED)

- **Description:** Convert agent logs from JSON to plain text with configurable verbosity. Logs persist in agent workspace by default.
- **Tasks:**
  - Update `AgentLogEntry` serialization to emit plain text lines with timestamp, actor, and content.
  - Add verbosity levels: `normal` (actions and results only), `verbose` (+ tool arguments), `debug` (+ full LLM prompts/responses).
  - Ensure log files write to `<agents_folder>/<agent_name>/logs/` by default.
  - Update `log_formatter.py` and any JSON log consumers.
  - Commit changes to git using the COMMIT_MESSAGE format.
- **Files:**
  - `lsm/agents/log_formatter.py`
  - `lsm/agents/models.py`
  - `lsm/agents/harness.py`
- **Success criteria:** Agent runs produce plain text logs at configured verbosity, saved to workspace. Existing log-reading code (TUI log streaming) still works.

## 1.3: Tiered Model Configuration

- **Description:** Add quick/normal/complex model tiers so agents and tools can declare what LLM capability they need, with fallback resolution baked into config loading.
- **Tasks:**
  - Define tier schema in config: `llms.tiers: { quick: { provider, model }, normal: { ... }, complex: { ... } }`.
  - Each agent/tool declares its required tier (e.g., `tier = "quick"` for tagging, `tier = "complex"` for research synthesis).
  - During config loading, after `llms.services` are loaded, any service without an explicit model assignment inherits from its tier's default.
  - Update `resolve_service()` to incorporate tier fallback: explicit service config → tier default → "default" service.
  - Add validation: warn if a tier is referenced but not configured.
  - Write/update tests for tier schema validation, `resolve_service()` tier fallback, and backward compatibility with configs that omit tiers (TDD: write tests before implementation).
  - Update `example_config.json`, `config.json`, and relevant documentation.
  - Run the relevant test suite (`pytest tests/test_config/`) and verify all new and existing tests pass.
  - Commit changes to git using the COMMIT_MESSAGE format.
- **Files:**
  - `lsm/config/models/llm.py`
  - `lsm/config/loader.py`
  - `lsm/agents/base.py`
  - `lsm/agents/tools/base.py`
- **Success criteria:** Agents/tools can declare `tier = "quick"` and get the right model without explicit per-service config. Existing configs without tiers continue to work (tiers are optional).

## 1.4: Agent/Tool Test Harness

- **Description:** Design and implement a test harness for measuring agent and tool efficiency and detecting regressions across feature changes.
- **Tasks:**
  - Define benchmark task format: a reproducible scenario with inputs, expected outcomes, and scoring dimensions.
  - Define scoring metrics: token usage, tool call count, wall-clock time, output quality (manual or LLM-judged).
  - Define regression thresholds: acceptable variance per metric before flagging a regression.
  - Implement a minimal harness runner that can execute a benchmark task, record metrics, and compare against a baseline.
  - Add initial benchmark tasks for file operations (find, read, edit) as baseline for Phase 4 tooling comparisons.
  - Write tests for the harness runner itself: task loading, metric recording, baseline comparison, and regression detection (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/benchmarks/`) and verify all new tests pass.
  - Commit changes to git using the COMMIT_MESSAGE format.
- **Files:**
  - `tests/benchmarks/`
  - `tests/benchmarks/harness.py`
  - `tests/benchmarks/tasks/`
  - `.agents/docs/architecture/development/TESTING.md`
- **Success criteria:** Harness runner can execute benchmark tasks, persist results, and compare against baselines. Baseline metrics exist for current naive file operations.

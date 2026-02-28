# Phase 9: Evaluation Harness

**Status**: Pending

Implements the retrieval evaluation system. Per the evaluation-first principle (§7.2),
this must be in place before any retrieval feature can claim measurable improvement.
All subsequent retrieval phases (10, 11, 12, 14, 15) gate their validation on baselines
captured here.

Reference: [RESEARCH_PLAN.md §5.13](../RESEARCH_PLAN.md#513-evaluation-harness-lsm-eval-retrieval)

---

## 9.1: Evaluation Framework and CLI

**Description**: Create the evaluation harness with BEIR format support, metric computation,
and CLI commands.

**Tasks**:
- Create `lsm/eval/` package:
  - `lsm/eval/__init__.py`
  - `lsm/eval/harness.py`:
    - `EvalHarness.__init__(pipeline, dataset_path)`
    - `run(profile: str) -> EvalResult` — run all queries in dataset, collect metrics
    - `compare(result: EvalResult, baseline: EvalResult) -> ComparisonReport`
  - `lsm/eval/metrics.py`:
    - `recall_at_k(retrieved, relevant, k)` → float
    - `mrr(retrieved, relevant)` → float
    - `ndcg_at_k(retrieved, relevant, k)` → float
    - `diversity_at_k(retrieved, k)` → float
    - `latency_stats(timings)` → dict (mean, p50, p95, p99)
  - `lsm/eval/dataset.py`:
    - Load BEIR-format evaluation sets
    - Bundled synthetic dataset in `lsm/eval/data/`
  - `lsm/eval/baselines.py`:
    - Save/load named baselines to `lsm.db` or filesystem
    - `save_baseline(result, name)`, `load_baseline(name)`, `list_baselines()`
- Add CLI commands:
  - `lsm eval retrieval --profile <profile> [--compare <baseline>]`
  - `lsm eval save-baseline --name <name>`
  - `lsm eval list-baselines`
- Create bundled synthetic evaluation dataset (50+ queries with relevance judgments)

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/eval/__init__.py` — package init
- `lsm/eval/harness.py` — evaluation harness
- `lsm/eval/metrics.py` — metric functions
- `lsm/eval/dataset.py` — dataset loading
- `lsm/eval/baselines.py` — baseline management
- `lsm/eval/data/` — bundled synthetic dataset
- CLI entry point — eval commands
- `tests/test_eval/test_metrics.py`:
  - Test: recall@k computation
  - Test: MRR computation
  - Test: nDCG@k computation
  - Test: diversity computation
- `tests/test_eval/test_harness.py`:
  - Test: harness runs against synthetic dataset
  - Test: baseline save/load round-trip
  - Test: comparison report format

**Success criteria**: `lsm eval retrieval --profile dense_only` produces a metrics report.
Baselines can be saved and compared. Metrics are correct per standard definitions.

---

## 9.2: Phase 9 Code Review and Changelog

**Tasks**:
- Review metric implementations against standard definitions
- Review synthetic dataset quality and coverage
- Review CLI output format and usability
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CLI_USAGE.md` — document eval commands

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CLI_USAGE.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---

# Phase 4: Advanced Tooling

**Why fourth:** Builds on file graphing (Phase 2) to create the find/read/edit tools that agents (Phase 6) will use. This is the "tilth-inspired" structural awareness tooling.

**Depends on:** Phase 1.4 (test harness for benchmarking), Phase 2 (file graphing), Phase 3.3 (tool API standardization)

**Harness requirement:** Every new tool in this phase must be benchmarked against the naive baseline (captured in Phase 1.4) using the test harness. Benchmarks must demonstrate that the graph-aware tools reduce tool call count and/or token usage compared to naive find/read/edit workflows. If a tool does not outperform the baseline, it must be redesigned or the naive approach kept.

| Task | Description | Depends On |
|------|-------------|------------|
| 4.1 | Tool API design and registry updates | 3.3 |
| 4.2 | Line-hash editing engine | 2.4 |
| 4.3 | Find file and find section tools | 2.4 |
| 4.4 | Read/outline enhancements | 2.4 |
| 4.5 | Benchmark comparisons | 1.4, 4.2–4.4 |
| 4.6 | Tests and documentation | 4.1–4.5 |

## 4.1: Tool API Design and Registry Updates
- **Description:** Define new tool surfaces and align them with ToolRegistry expectations.
- **Tasks:**
  - Review existing file tools (`read_file`, `write_file`, `source_map`) for extension points.
  - Define schemas for `find_file`, `find_section`, and `edit_file` tools.
  - Update tool metadata (risk level, runner preference, network needs).
  - Write tests for tool schema registration, metadata correctness, and discoverability in the ToolRegistry (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/base.py`
  - `lsm/agents/tools/**`
- **Success criteria:** Tool definitions are registered and discoverable with consistent schemas.

## 4.2: Line-Hash Editing Engine
- **Description:** Implement a deterministic edit engine that uses line hashes to precisely identify replacement targets. Motivated by reducing the multi-call "glob → read → too big → grep → read again" pattern that wastes agent tokens. See [blog.can.ac/2026/02/12/the-harness-problem](https://blog.can.ac/2026/02/12/the-harness-problem/) and [tilth](https://github.com/jahala/tilth) for prior art.
- **Tasks:**
  - Define line-hash format: short hash per line (e.g., first 8 chars of SHA-256 of line content).
  - `edit_file` tool accepts `{ file, start_hash, end_hash, new_content }` to replace a line range.
  - On hash mismatch: return descriptive error including the actual hashes at the target lines, surrounding context, and suggestions for the correct range. This enables intelligent retry by the LLM.
  - On success: return the updated file graph (outline) so the agent has current structural awareness.
  - Add collision detection: if the same hash appears multiple times, require additional disambiguation (line number or surrounding context).
  - Write tests for line-hash computation, successful edits with matching hashes, hash mismatch error diagnostics, collision detection and disambiguation, and post-edit graph refresh (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_tools/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/edit_file.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** Edits apply only when hashes match. Failures include actionable diagnostics. Edit + re-read is a single tool call.

## 4.3: Find File and Find Section Tools
- **Description:** Provide fast file and section discovery using structural graphs, so agents get structural awareness in one call instead of 6.
- **Tasks:**
  - `find_file` tool: search by name pattern, content pattern, or both. Returns file paths with brief structural outlines.
  - `find_section` tool: given a file (or searched files), find sections by heading/function/class name. Returns the section's graph node with line range and line hashes.
  - Graph-aware filters: language, node type (function/class/heading), depth limit.
  - Output includes line hashes for immediate use with `edit_file`.
  - Write tests for `find_file` name/content pattern search, `find_section` heading/function/class lookup, graph-aware filter behavior, and line hash inclusion in output (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_tools/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/find_file.py`
  - `lsm/agents/tools/find_section.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** An agent can find a function and get its content + line hashes in a single tool call.

## 4.4: Read/Outline Enhancements
- **Description:** Extend read tools to provide structural outlines and section-targeted reads.
- **Tasks:**
  - `read_file` accepts optional `section` parameter (heading name, function name, or graph node ID) to return only that section.
  - `read_file` accepts optional `max_depth` to control outline depth.
  - `source_map` returns structural outlines derived from file graphs instead of flat content.
  - All read outputs include line hashes when `include_hashes=true`.
  - Write tests for section-targeted reads, `max_depth` outline filtering, `source_map` structural output, and `include_hashes` flag behavior (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_tools/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/read_file.py`
  - `lsm/agents/tools/source_map.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** Read tools can return section-only content and structural outlines.

## 4.5: Benchmark Comparisons
- **Description:** Use the test harness from 1.4 to validate that advanced tools outperform naive implementations.
- **Tasks:**
  - Define benchmark scenarios: find a function in a large codebase, edit a specific section, read an outline of a complex file.
  - Run each scenario with naive tools (current `read_file` + `write_file`) and record baseline metrics.
  - Run each scenario with advanced tools (line-hash edit, find_section, graph-aware read) and record metrics.
  - Compare: tool call count, total tokens consumed, wall-clock time, success rate.
  - Document results and flag any tools that do not demonstrate improvement.
  - Run the benchmark harness (`pytest tests/benchmarks/`) and verify all benchmark tasks complete without errors.
- **Files:**
  - `tests/benchmarks/tasks/`
  - `tests/benchmarks/results/`
- **Success criteria:** Advanced tools demonstrate measurable improvement over naive baselines in at least tool call count and token usage.

## 4.6: Tests and Documentation
- **Description:** Validate tool behavior and document new usage.
- **Tasks:**
  - Add tests for find/read/edit flows using fixtures.
  - Document tool schemas and examples in `docs/`.
  - Run the full test suite (`pytest tests/`) and verify all new and existing tests pass, including all tests added in tasks 4.1–4.5.
- **Files:**
  - `tests/test_tools/`
  - `docs/`
- **Success criteria:** Tooling works end-to-end with documented schemas and passing tests.

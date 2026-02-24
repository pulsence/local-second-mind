# Phase 4 Benchmark Results

Baseline: `tests/benchmarks/baselines/file_ops.json` (naive workflows)
Advanced: graph-aware tools (find_file/find_section/edit_file + structured read)

| Task | Baseline Tool Calls | Advanced Tool Calls | Baseline Tokens | Advanced Tokens | Improvement |
| --- | --- | --- | --- | --- | --- |
| file_find | 4 | 1 | 162 | 192 | Tool calls reduced (content search in one call) |
| file_read | 1 | 1 | 889 | 417 | Token usage reduced (section-only read) |
| file_edit | 3 | 2 | 769 | 482 | Tool calls + tokens reduced (find_section + edit_file) |

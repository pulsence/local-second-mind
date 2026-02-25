# lsm.agents.tools

Description: Tool registry, sandbox enforcement, runner abstraction (local/docker), and built-in tool implementations.
Folder Path: `lsm/agents/tools/`

## Modules

- [base.py](../../lsm/agents/tools/base.py): BaseTool abstract class and ToolRegistry
- [sandbox.py](../../lsm/agents/tools/sandbox.py): ToolSandbox permission enforcement and runner policy
- [runner.py](../../lsm/agents/tools/runner.py): Runner abstraction and LocalRunner implementation
- [docker_runner.py](../../lsm/agents/tools/docker_runner.py): DockerRunner implementation
- [wsl2_runner.py](../../lsm/agents/tools/wsl2_runner.py): WSL2Runner implementation
- [env_scrubber.py](../../lsm/agents/tools/env_scrubber.py): Environment variable scrubbing for runner isolation
- [spawn_agent.py](../../lsm/agents/tools/spawn_agent.py): Meta-system tool for spawning sub-agent runs
- [await_agent.py](../../lsm/agents/tools/await_agent.py): Meta-system tool for waiting on spawned sub-agent completion
- [collect_artifacts.py](../../lsm/agents/tools/collect_artifacts.py): Meta-system tool for collecting spawned sub-agent artifacts
- [ask_user.py](../../lsm/agents/tools/ask_user.py): Clarification tool for runtime user interaction
- [read_file.py](../../lsm/agents/tools/read_file.py): File reading tool; supports section-aware reads with outlines and optional line hashes via `get_file_graph()`
- [read_folder.py](../../lsm/agents/tools/read_folder.py): Folder listing tool
- [file_metadata.py](../../lsm/agents/tools/file_metadata.py): File metadata retrieval tool; `include_graph: true` adds full `FileGraph` output via `get_file_graph()`
- [find_file.py](../../lsm/agents/tools/find_file.py): Graph-aware file discovery by name/content pattern with structural outlines
- [find_section.py](../../lsm/agents/tools/find_section.py): Graph-aware section lookup with content + line hashes
- [hash_file.py](../../lsm/agents/tools/hash_file.py): File hashing tool
- [source_map.py](../../lsm/agents/tools/source_map.py): Source map aggregation tool; returns per-source outlines and propagates `node_id` fields into `node_ids` lists
- [edit_file.py](../../lsm/agents/tools/edit_file.py): Line-hash edit tool with post-edit outline refresh
- [bash.py](../../lsm/agents/tools/bash.py): Bash command execution tool with sandbox-enforced allow/deny lists
- [powershell.py](../../lsm/agents/tools/powershell.py): PowerShell command execution tool with sandbox-enforced allow/deny lists
- [write_file.py](../../lsm/agents/tools/write_file.py): File writing tool
- [append_file.py](../../lsm/agents/tools/append_file.py): File appending tool
- [create_folder.py](../../lsm/agents/tools/create_folder.py): Folder creation tool
- [memory_put.py](../../lsm/agents/tools/memory_put.py): Memory storage tool
- [memory_remove.py](../../lsm/agents/tools/memory_remove.py): Memory removal tool
- [memory_search.py](../../lsm/agents/tools/memory_search.py): Memory search tool
- [load_url.py](../../lsm/agents/tools/load_url.py): URL loading tool
- [query_llm.py](../../lsm/agents/tools/query_llm.py): LLM query tool
- [query_remote.py](../../lsm/agents/tools/query_remote.py): Remote source query tool
- [query_remote_chain.py](../../lsm/agents/tools/query_remote_chain.py): Remote source chain query tool
- [query_embeddings.py](../../lsm/agents/tools/query_embeddings.py): Embedding query tool
- [extract_snippets.py](../../lsm/agents/tools/extract_snippets.py): Snippet extraction tool
- [mcp_host.py](../../lsm/agents/tools/mcp_host.py): MCP host client; connects to external MCP servers, discovers tools, and registers them in ToolRegistry
- [similarity_search.py](../../lsm/agents/tools/similarity_search.py): Similarity search tool

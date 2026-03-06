from __future__ import annotations

import json
import shutil
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.tools import create_default_tool_registry
from lsm.agents.tools.sandbox import ToolSandbox

from tests.test_agents.execution_smoke_utils import (
    FakeHTTPResponse,
    FakeQueryLLMProvider,
    FakeRemoteChain,
    FakeRemoteProvider,
    build_pipeline,
    build_smoke_config,
    make_memory_store,
)


def test_default_registry_tools_execute_through_real_runtime(monkeypatch, tmp_path: Path) -> None:
    config = build_smoke_config(tmp_path)
    memory_store = make_memory_store(tmp_path)
    try:
        pipeline = build_pipeline(config)

        monkeypatch.setattr(
            "lsm.agents.tools.query_llm.create_provider",
            lambda cfg: FakeQueryLLMProvider(),
        )
        monkeypatch.setattr(
            "lsm.agents.tools.query_remote.create_remote_provider",
            lambda provider_type, provider_config: FakeRemoteProvider(),
        )
        monkeypatch.setattr(
            "lsm.agents.tools.query_remote_chain.RemoteProviderChain",
            FakeRemoteChain,
        )
        monkeypatch.setattr(
            "lsm.agents.tools.query_remote_chain.get_preconfigured_chain",
            lambda name: None,
        )
        monkeypatch.setattr(
            "lsm.agents.tools.load_url.requests.get",
            lambda url, timeout=10: FakeHTTPResponse(f"fetched:{url}"),
        )

        registry = create_default_tool_registry(
            config,
            collection=pipeline.db,
            embedder=pipeline.embedder,
            memory_store=memory_store,
            pipeline=pipeline,
        )
        sandbox = ToolSandbox(config.agents.sandbox)
        harness = AgentHarness(
            agent_config=config.agents,
            tool_registry=registry,
            llm_registry=config.llm,
            sandbox=sandbox,
            agent_name="meta",
            lsm_config=config,
        )

        workspace = tmp_path / "workspace"
        docs_dir = workspace / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        note_path = docs_dir / "note.md"

        executed: set[str] = set()

        def run_tool(name: str, args: dict):
            executed.add(name)
            tool = registry.lookup(name)
            return harness.sandbox.execute(tool, args)

        create_result = run_tool("create_folder", {"path": str(docs_dir)})
        assert "Created folder" in create_result

        write_result = run_tool(
            "write_file",
            {
                "path": str(note_path),
                "content": "# Sin\n\nOriginal paragraph.\n\n## Aquinas\n\nLine one.\nLine two.\n",
            },
        )
        assert "Wrote" in write_result

        append_result = run_tool(
            "append_file",
            {"path": str(note_path), "content": "\n## Notes\n\nAppended line.\n"},
        )
        assert "Appended" in append_result

        read_text = run_tool("read_file", {"path": str(note_path)})
        assert "Original paragraph." in read_text

        read_structured = json.loads(
            run_tool("read_file", {"path": str(note_path), "include_hashes": True})
        )
        assert read_structured["path"].endswith("note.md")
        assert read_structured["line_hashes"]

        folder_entries = json.loads(
            run_tool("read_folder", {"path": str(workspace), "recursive": True})
        )
        assert any(entry["name"] == "note.md" for entry in folder_entries)

        found_files = json.loads(
            run_tool(
                "find_file",
                {
                    "path": str(workspace),
                    "name_pattern": "note",
                    "content_pattern": "Aquinas",
                },
            )
        )
        assert found_files[0]["name"] == "note.md"

        sections = json.loads(
            run_tool(
                "find_section",
                {"path": str(note_path), "section": "Aquinas", "max_results": 1},
            )
        )
        assert sections[0]["node"]["name"] == "Aquinas"

        edit_result = json.loads(
            run_tool(
                "edit_file",
                {
                    "path": str(note_path),
                    "start_hash": sections[0]["start_hash"],
                    "end_hash": sections[0]["end_hash"],
                    "new_content": "## Aquinas\n\nRevised line.\n",
                },
            )
        )
        assert edit_result["status"] == "ok"
        assert "Revised line." in note_path.read_text(encoding="utf-8")

        metadata = json.loads(run_tool("file_metadata", {"paths": [str(note_path)]}))
        assert metadata[0]["ext"] == ".md"

        hash_payload = json.loads(run_tool("hash_file", {"path": str(note_path)}))
        assert hash_payload["path"].endswith("note.md")

        source_map = json.loads(
            run_tool(
                "source_map",
                {
                    "evidence": [
                        {"source_path": str(note_path), "snippet": "Aquinas excerpt", "score": 0.9}
                    ]
                },
            )
        )
        assert str(note_path) in source_map

        snippets = json.loads(
            run_tool(
                "extract_snippets",
                {
                    "query": "sin",
                    "paths": ["notes/aquinas.md"],
                    "max_snippets": 1,
                    "max_chars_per_snippet": 40,
                },
            )
        )
        assert snippets

        similar = json.loads(
            run_tool(
                "similarity_search",
                {"chunk_ids": ["c1", "c2", "c3"], "top_k": 3, "threshold": 0.5},
            )
        )
        assert similar

        query_context = json.loads(run_tool("query_context", {"query": "What is sin?", "k": 1}))
        assert query_context["candidate_count"] >= 1

        execute_context = json.loads(
            run_tool(
                "execute_context",
                {
                    "question": "What is sin?",
                    "context_block": "[S1] notes/aquinas.md\nSin is a voluntary act contrary to right reason.",
                },
            )
        )
        assert "Grounded synthesis" in execute_context["answer"]

        query_and_synthesize = json.loads(
            run_tool("query_and_synthesize", {"query": "What is sin?", "k": 1})
        )
        assert query_and_synthesize["candidates"]

        query_llm = json.loads(run_tool("query_llm", {"prompt": "Summarize this"}))
        assert "LLM tool response" in query_llm["answer"]

        remote_result = json.loads(
            run_tool("query_arxiv", {"input": {"query": "aquinas sin"}})
        )
        assert remote_result[0]["title"].startswith("Remote result")

        chain_result = json.loads(
            run_tool(
                "query_remote_chain",
                {"chain": "Research Digest", "input": {"query": "aquinas sin"}},
            )
        )
        assert chain_result[0]["title"].startswith("Chain result")

        loaded = run_tool("load_url", {"url": "https://example.com/test"})
        assert loaded == "fetched:https://example.com/test"

        memory_put = json.loads(
            run_tool(
                "memory_put",
                {
                    "key": "preferred_tone",
                    "value": {"tone": "concise"},
                    "type": "pinned",
                    "scope": "agent",
                    "rationale": "Smoke test",
                },
            )
        )
        assert memory_put["status"] == "pending"

        memory_search = json.loads(
            run_tool("memory_search", {"scope": "agent", "tags": [], "limit": 5})
        )
        assert memory_search == []

        memory_remove = json.loads(
            run_tool("memory_remove", {"memory_id": memory_put["memory_id"]})
        )
        assert memory_remove["status"] == "deleted"

        ask_user = run_tool("ask_user", {"prompt": "Continue?"})
        assert ask_user == "Continue with your best judgment."

        bash_output = run_tool("bash", {"command": "printf 'smoke-bash'"})
        assert bash_output == "smoke-bash"

        powershell_tool = registry.lookup("powershell")
        executed.add("powershell")
        if shutil.which("pwsh") or shutil.which("powershell"):
            powershell_output = harness.sandbox.execute(
                powershell_tool,
                {"command": "Write-Output 'smoke-powershell'"},
            )
            assert "smoke-powershell" in powershell_output
        else:
            try:
                harness.sandbox.execute(powershell_tool, {"command": "Write-Output 'smoke'"})
            except RuntimeError as exc:
                assert "PowerShell executable not found" in str(exc)
            else:
                raise AssertionError("powershell tool should fail when pwsh is unavailable")

        spawn_payload = json.loads(
            run_tool("spawn_agent", {"name": "assistant", "params": {"topic": "Tool smoke"}})
        )
        agent_id = spawn_payload["agent_id"]

        await_payload = json.loads(run_tool("await_agent", {"agent_id": agent_id, "timeout_seconds": 5}))
        assert await_payload["status"] == "completed"

        artifacts_payload = json.loads(run_tool("collect_artifacts", {"agent_id": agent_id}))
        assert artifacts_payload["artifacts"]

        assert executed == {tool.name for tool in registry.list_tools()}
    finally:
        memory_store.close()

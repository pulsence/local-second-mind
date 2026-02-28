"""Full integration coverage for ingest and query pipelines."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from time import perf_counter

import pytest

from lsm.config.models import (
    GlobalConfig,
    IngestConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LLMServiceConfig,
    LSMConfig,
    LocalSourcePolicy,
    ModeConfig,
    ModelKnowledgePolicy,
    QueryConfig,
    RemoteSourcePolicy,
    VectorDBConfig,
)
from lsm.ingest.api import run_ingest
from lsm.ingest.manifest import load_manifest
from lsm.query.api import query_sync
from lsm.query.retrieval import embed_text, retrieve_candidates
from lsm.query.session import SessionState
from lsm.vectordb import create_vectordb_provider
from tests.testing_config import TestConfig as RuntimeTestConfig


def _copy_fixture_documents(
    source_docs: Path,
    destination: Path,
    filenames: list[str],
) -> Path:
    """Copy selected synthetic fixture files into an isolated docs root."""
    destination.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        shutil.copy2(source_docs / filename, destination / filename)
    return destination


def _build_pipeline_config(
    *,
    docs_root: Path,
    tmp_path: Path,
    embed_model: str,
    provider_name: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    rerank_strategy: str = "none",
    no_rerank: bool = True,
    min_relevance: float = 0.0,
    chunk_size: int = 1400,
    chunk_overlap: int = 120,
    chunking_strategy: str = "structure",
    extensions: list[str] | None = None,
    vectordb_provider: str = "sqlite",
    vectordb_connection_string: str | None = None,
) -> LSMConfig:
    """Build an isolated config for full pipeline testing."""
    mode_name = "full_pipeline_mode"
    collection_name = f"full_pipeline_{tmp_path.name}"

    ingest = IngestConfig(
        roots=[docs_root],
        extensions=extensions or [".txt", ".md", ".html"],
        override_extensions=True,
        exclude_dirs=[],
        override_excludes=True,
        skip_errors=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
    )
    query = QueryConfig(
        mode=mode_name,
        rerank_strategy=rerank_strategy,
        no_rerank=no_rerank,
        k=8,
        retrieve_k=16,
        min_relevance=min_relevance,
    )
    llm = LLMRegistryConfig(
        providers=[
            LLMProviderConfig(
                provider_name=provider_name,
                api_key=api_key,
                base_url=base_url,
            )
        ],
        services={
            "query": LLMServiceConfig(
                provider=provider_name,
                model=model,
                temperature=0.1,
                max_tokens=500,
            ),
            "ranking": LLMServiceConfig(
                provider=provider_name,
                model=model,
                temperature=0.1,
                max_tokens=300,
            ),
        },
    )
    vectordb = VectorDBConfig(
        provider=vectordb_provider,
        path=tmp_path / "data",
        collection=collection_name,
        connection_string=vectordb_connection_string,
    )

    modes = {
        mode_name: ModeConfig(
            retrieval_profile="hybrid_rrf",
            synthesis_style="grounded",
            local_policy=LocalSourcePolicy(
                enabled=True,
                min_relevance=min_relevance,
                k=4,
            ),
            remote_policy=RemoteSourcePolicy(enabled=False),
            model_knowledge_policy=ModelKnowledgePolicy(enabled=False),
        )
    }

    return LSMConfig(
        ingest=ingest,
        query=query,
        llm=llm,
        vectordb=vectordb,
        modes=modes,
        global_settings=GlobalConfig(
            global_folder=tmp_path / "global",
            embed_model=embed_model,
            device="cpu",
            batch_size=16,
        ),
    )


def _resolve_live_llm_provider(
    test_config: RuntimeTestConfig,
) -> tuple[str, str, str | None, str | None]:
    """Select a configured live provider for end-to-end live pipeline checks."""
    if test_config.openai_api_key:
        return "openai", "gpt-5.2", test_config.openai_api_key, None
    if test_config.anthropic_api_key:
        return "anthropic", "claude-sonnet-4-5", test_config.anthropic_api_key, None
    if test_config.google_api_key:
        return "gemini", "gemini-2.5-flash", test_config.google_api_key, None
    if test_config.ollama_base_url:
        return "local", "llama3.1", None, test_config.ollama_base_url

    pytest.skip(
        "No live LLM credentials configured. "
        "Set one of LSM_TEST_OPENAI_API_KEY, LSM_TEST_ANTHROPIC_API_KEY, "
        "LSM_TEST_GOOGLE_API_KEY, or LSM_TEST_OLLAMA_BASE_URL."
    )


def _answer_looks_coherent(answer: str) -> bool:
    """Apply lightweight coherence checks to a synthesized answer."""
    words = [token for token in answer.replace("\n", " ").split(" ") if token.strip()]
    if len(words) < 12:
        return False

    lowered = answer.lower()
    return any(term in lowered for term in ("retrieval", "citation", "metadata", "chunk"))


@pytest.mark.integration
def test_end_to_end_ingest_and_retrieval_without_network(
    synthetic_data_root: Path,
    tmp_path: Path,
    real_embedder,
    test_config: RuntimeTestConfig,
) -> None:
    docs_root = _copy_fixture_documents(
        synthetic_data_root / "documents",
        tmp_path / "docs",
        ["philosophy_essay.txt", "research_paper.md", "technical_manual.html"],
    )
    config = _build_pipeline_config(
        docs_root=docs_root,
        tmp_path=tmp_path,
        embed_model=test_config.embed_model,
        provider_name="local",
        model="llama3.1",
        rerank_strategy="none",
        no_rerank=True,
        min_relevance=0.0,
    )

    ingest_result = run_ingest(config, force=True)
    assert ingest_result.total_files == 3
    assert ingest_result.completed_files == 3
    assert ingest_result.skipped_files == 0
    assert ingest_result.errors == []
    assert ingest_result.chunks_added > 0

    collection = create_vectordb_provider(config.vectordb)
    assert collection.count() == ingest_result.chunks_added
    manifest_data = load_manifest(connection=collection.connection)
    assert len(manifest_data) == 3
    assert all(entry.get("file_hash") for entry in manifest_data.values())

    query_embedding = embed_text(
        real_embedder,
        "How do citations and metadata improve trust in local retrieval systems?",
    )
    candidates = retrieve_candidates(collection, query_embedding, k=8)

    assert candidates
    assert any(
        ("citation" in candidate.text.lower()) or ("metadata" in candidate.text.lower())
        for candidate in candidates
    )
    for candidate in candidates:
        assert candidate.meta.get("source_path")
        assert candidate.meta.get("source_name")
        assert candidate.meta.get("chunk_index") is not None


@pytest.mark.live
def test_full_live_pipeline_with_real_llm_rerank_and_synthesis(
    synthetic_data_root: Path,
    tmp_path: Path,
    real_embedder,
    test_config: RuntimeTestConfig,
) -> None:
    provider_name, model, api_key, base_url = _resolve_live_llm_provider(test_config)
    docs_root = _copy_fixture_documents(
        synthetic_data_root / "documents",
        tmp_path / "docs",
        ["philosophy_essay.txt", "research_paper.md", "technical_manual.html"],
    )
    config = _build_pipeline_config(
        docs_root=docs_root,
        tmp_path=tmp_path,
        embed_model=test_config.embed_model,
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        rerank_strategy="llm",
        no_rerank=False,
        min_relevance=0.0,
    )

    ingest_result = run_ingest(config, force=True)
    assert ingest_result.chunks_added > 0

    collection = create_vectordb_provider(config.vectordb)
    state = SessionState(model=model)
    question = (
        "Explain what improves retrieval quality in this corpus and use "
        "inline citations like [S1]."
    )

    query_result = query_sync(
        question=question,
        config=config,
        state=state,
        embedder=real_embedder,
        collection=collection,
    )

    assert query_result.candidates
    assert query_result.answer.strip()
    assert "[S" in query_result.answer
    assert query_result.debug_info.get("synthesis_fallback") is not True
    assert _answer_looks_coherent(query_result.answer)
    assert query_result.remote_sources == []


@pytest.mark.live
@pytest.mark.live_vectordb
def test_full_pipeline_with_postgresql_store(
    synthetic_data_root: Path,
    tmp_path: Path,
    real_embedder,
    test_config: RuntimeTestConfig,
    live_postgres_connection_string: str,
) -> None:
    docs_root = _copy_fixture_documents(
        synthetic_data_root / "documents",
        tmp_path / "docs",
        ["philosophy_essay.txt", "research_paper.md", "technical_manual.html"],
    )
    config = _build_pipeline_config(
        docs_root=docs_root,
        tmp_path=tmp_path,
        embed_model=test_config.embed_model,
        provider_name="local",
        model="llama3.1",
        rerank_strategy="none",
        no_rerank=True,
        min_relevance=0.0,
        vectordb_provider="postgresql",
        vectordb_connection_string=live_postgres_connection_string,
    )

    collection = create_vectordb_provider(config.vectordb)
    if not collection.is_available():
        pytest.skip("PostgreSQL provider is unavailable in this environment")

    try:
        ingest_result = run_ingest(config, force=True)
        assert ingest_result.total_files == 3
        assert ingest_result.completed_files == 3
        assert ingest_result.skipped_files == 0
        assert ingest_result.errors == []
        assert ingest_result.chunks_added > 0

        assert collection.count() == ingest_result.chunks_added

        query_embedding = embed_text(
            real_embedder,
            "How do citations and metadata improve trust in local retrieval systems?",
        )
        candidates = retrieve_candidates(collection, query_embedding, k=8)

        assert candidates
        assert any(
            ("citation" in candidate.text.lower()) or ("metadata" in candidate.text.lower())
            for candidate in candidates
        )
        for candidate in candidates:
            assert candidate.meta.get("source_path")
            assert candidate.meta.get("source_name")
            assert candidate.meta.get("chunk_index") is not None
    finally:
        try:
            collection.delete_all()
        except Exception:
            pass


@pytest.mark.performance
@pytest.mark.skipif(not os.getenv("LSM_PERF_TEST"), reason="Set LSM_PERF_TEST=1 to run")
def test_performance_ingest_and_query_latency_over_100_chunks(
    synthetic_data_root: Path,
    tmp_path: Path,
    real_embedder,
    test_config: RuntimeTestConfig,
) -> None:
    docs_root = _copy_fixture_documents(
        synthetic_data_root / "documents",
        tmp_path / "docs",
        ["large_document.md"],
    )
    config = _build_pipeline_config(
        docs_root=docs_root,
        tmp_path=tmp_path,
        embed_model=test_config.embed_model,
        provider_name="local",
        model="llama3.1",
        rerank_strategy="none",
        no_rerank=True,
        min_relevance=0.0,
        chunk_size=1500,
        chunk_overlap=100,
        chunking_strategy="fixed",
        extensions=[".md"],
    )

    ingest_result = run_ingest(config, force=True)
    assert ingest_result.total_files == 1
    assert ingest_result.chunks_added >= 100

    collection = create_vectordb_provider(config.vectordb)
    assert collection.count() == ingest_result.chunks_added

    started_at = perf_counter()
    query_embedding = embed_text(
        real_embedder,
        "What controls are used during high-load ingest and retrieval runs?",
    )
    candidates = retrieve_candidates(collection, query_embedding, k=12)
    query_seconds = perf_counter() - started_at

    assert candidates
    max_seconds = float(os.getenv("LSM_PERF_QUERY_MAX_SECONDS", "8.0"))
    assert query_seconds <= max_seconds, (
        f"Expected end-to-end query embedding+retrieval <= {max_seconds:.2f}s, "
        f"got {query_seconds:.2f}s"
    )

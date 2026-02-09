from __future__ import annotations

from pathlib import Path

import pytest

from lsm.config.models import (
    ChainLink,
    IngestConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LLMServiceConfig,
    LSMConfig,
    QueryConfig,
    RemoteProviderChainConfig,
    RemoteProviderConfig,
    VectorDBConfig,
)
from lsm.remote.chain import RemoteProviderChain


def _base_config() -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(roots=[Path("/tmp")], manifest=Path("/tmp/manifest.json")),
        query=QueryConfig(mode="grounded"),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={"query": LLMServiceConfig(provider="openai", model="gpt-5.2")},
        ),
        vectordb=VectorDBConfig(persist_dir=Path("/tmp/.chroma"), collection="test"),
        remote_providers=[
            RemoteProviderConfig(name="openalex", type="openalex"),
            RemoteProviderConfig(name="crossref", type="crossref"),
        ],
        config_path=Path("/tmp/config.json"),
    )


def test_remote_provider_chain_executes_with_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    chain_config = RemoteProviderChainConfig(
        name="Research Digest",
        links=[
            ChainLink(source="openalex"),
            ChainLink(source="crossref", map=["doi:doi"]),
        ],
    )
    chain = RemoteProviderChain(config=config, chain_config=chain_config)

    class _OpenAlexProvider:
        def search_structured(self, _input_dict, max_results=5):
            return [{"doi": "10.1000/a"}, {"doi": "10.1000/b"}][:max_results]

    crossref_calls = []

    class _CrossrefProvider:
        def search_structured(self, input_dict, max_results=5):
            crossref_calls.append(input_dict)
            doi = input_dict.get("doi")
            return [{"doi": doi, "title": f"Resolved {doi}"}][:max_results]

    def _factory(provider_type, _cfg):
        if provider_type == "openalex":
            return _OpenAlexProvider()
        return _CrossrefProvider()

    monkeypatch.setattr("lsm.remote.chain.create_remote_provider", _factory)

    results = chain.execute({"query": "eucharist"}, max_results=5)
    assert len(results) == 2
    assert [item["doi"] for item in results] == ["10.1000/a", "10.1000/b"]
    assert crossref_calls == [{"doi": "10.1000/a"}, {"doi": "10.1000/b"}]


def test_remote_provider_chain_passes_through_without_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config()
    chain_config = RemoteProviderChainConfig(
        name="Passthrough",
        links=[ChainLink(source="openalex"), ChainLink(source="crossref")],
    )
    chain = RemoteProviderChain(config=config, chain_config=chain_config)

    class _ProviderA:
        def search_structured(self, _input_dict, max_results=5):
            return [{"query": "x", "title": "t"}][:max_results]

    calls = []

    class _ProviderB:
        def search_structured(self, input_dict, max_results=5):
            calls.append(input_dict)
            return [{"title": "done"}][:max_results]

    def _factory(provider_type, _cfg):
        if provider_type == "openalex":
            return _ProviderA()
        return _ProviderB()

    monkeypatch.setattr("lsm.remote.chain.create_remote_provider", _factory)
    results = chain.execute({"query": "seed"}, max_results=5)
    assert results == [{"title": "done"}]
    assert calls == [{"query": "x", "title": "t"}]


def test_remote_provider_chain_raises_for_missing_provider() -> None:
    config = _base_config()
    chain_config = RemoteProviderChainConfig(
        name="Bad",
        links=[ChainLink(source="missing")],
    )
    chain = RemoteProviderChain(config=config, chain_config=chain_config)

    with pytest.raises(ValueError, match="Remote provider not configured"):
        chain.execute({"query": "test"})


def test_remote_provider_chain_passes_provider_specific_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config()
    config.remote_providers = [
        RemoteProviderConfig(name="openalex", type="openalex", extra={"email": "you@example.com"}),
    ]
    chain_config = RemoteProviderChainConfig(name="Single", links=[ChainLink(source="openalex")])
    chain = RemoteProviderChain(config=config, chain_config=chain_config)

    captured_cfg = {}

    class _Provider:
        def search_structured(self, _input_dict, max_results=5):
            return [{"ok": True}][:max_results]

    def _factory(_provider_type, cfg):
        captured_cfg.update(cfg)
        return _Provider()

    monkeypatch.setattr("lsm.remote.chain.create_remote_provider", _factory)
    results = chain.execute({"query": "test"}, max_results=5)
    assert results == [{"ok": True}]
    assert captured_cfg["email"] == "you@example.com"

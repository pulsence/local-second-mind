from __future__ import annotations

from pathlib import Path

import pytest

from lsm.config.models import (
    ChainLink,
    GlobalConfig,
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
from lsm.remote.chains import build_chain


def _base_config() -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(roots=[Path("/tmp")], manifest=Path("/tmp/manifest.json")),
        query=QueryConfig(mode="grounded"),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={"query": LLMServiceConfig(provider="openai", model="gpt-5.2")},
        ),
        vectordb=VectorDBConfig(path=Path("/tmp/.chroma"), collection="test"),
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
    with pytest.raises(ValueError, match="Remote provider not configured"):
        RemoteProviderChain(config=config, chain_config=chain_config)


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


def test_remote_provider_chain_validates_mapping_fields() -> None:
    config = _base_config()
    chain_config = RemoteProviderChainConfig(
        name="BadMap",
        links=[
            ChainLink(source="openalex"),
            ChainLink(source="crossref", map=["unknown_field:doi"]),
        ],
    )

    with pytest.raises(ValueError, match="expects output field"):
        RemoteProviderChain(config=config, chain_config=chain_config)


def test_scholarly_chain_downloads_full_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config()
    config.global_settings = GlobalConfig(global_folder=tmp_path)
    config.remote_providers = [
        RemoteProviderConfig(name="openalex", type="openalex"),
        RemoteProviderConfig(name="crossref", type="crossref"),
        RemoteProviderConfig(name="unpaywall", type="unpaywall"),
        RemoteProviderConfig(name="core", type="core", api_key="test"),
    ]

    chain_config = RemoteProviderChainConfig(
        name="scholarly_discovery",
        links=[
            ChainLink(source="openalex"),
            ChainLink(source="crossref"),
            ChainLink(source="unpaywall", map=["doi:doi"]),
            ChainLink(source="core", map=["doi:doi"]),
        ],
    )

    class _OpenAlexProvider:
        def search_structured(self, _input_dict, max_results=5):
            return [{"doi": "10.1000/a"}][:max_results]

    class _CrossrefProvider:
        def search_structured(self, _input_dict, max_results=5):
            return [{"doi": "10.1000/a"}][:max_results]

    class _UnpaywallProvider:
        def search_structured(self, _input_dict, max_results=5):
            return [{"doi": "10.1000/a"}][:max_results]

    class _CoreProvider:
        def search_structured(self, _input_dict, max_results=5):
            return [
                {
                    "title": "Example",
                    "metadata": {
                        "pdf_url": "https://example.org/test.pdf",
                        "source_id": "10.1000/a",
                    },
                }
            ][:max_results]

    def _factory(provider_type, _cfg):
        if provider_type == "openalex":
            return _OpenAlexProvider()
        if provider_type == "crossref":
            return _CrossrefProvider()
        if provider_type == "unpaywall":
            return _UnpaywallProvider()
        return _CoreProvider()

    monkeypatch.setattr("lsm.remote.chain.create_remote_provider", _factory)

    response = type("Resp", (), {})()
    response.status_code = 200
    response.content = b"pdf-bytes"
    response.raise_for_status = lambda: None
    monkeypatch.setattr("lsm.remote.chains.requests.get", lambda *args, **kwargs: response)

    chain = build_chain(config=config, chain_config=chain_config)
    results = chain.execute({"query": "test"}, max_results=1)
    assert results
    metadata = results[0]["metadata"]
    assert metadata["downloaded_path"]
    assert Path(metadata["downloaded_path"]).exists()

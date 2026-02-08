from __future__ import annotations

from types import SimpleNamespace

from lsm.query.prefilter import extract_tags_from_prompt, prefilter_by_metadata


def test_extract_tags_from_prompt_basic() -> None:
    tags = extract_tags_from_prompt("Find papers about transformers and retrieval by author John Doe")
    assert "transformers" in tags
    assert "retrieval" in tags


def test_prefilter_by_metadata_uses_author_year_content_type_and_tags() -> None:
    where = prefilter_by_metadata(
        "Find theology documents by Jane Doe from 2024",
        available_metadata={
            "content_type": ["theology", "engineering"],
            "ai_tags": ["christology", "ethics"],
            "user_tags": ["doctrine"],
            "root_tags": ["theology"],
            "folder_tags": ["research"],
        },
    )
    assert where.get("author") is not None
    assert where.get("year") == "2024"
    assert where.get("content_type") == "theology"


def test_prefilter_by_metadata_matches_all_tag_fields() -> None:
    where = prefilter_by_metadata(
        "christology doctrine theology research",
        available_metadata={
            "ai_tags": ["christology"],
            "user_tags": ["doctrine"],
            "root_tags": ["theology"],
            "folder_tags": ["research"],
        },
    )

    assert where.get("ai_tags") == "christology"
    assert where.get("user_tags") == "doctrine"
    assert where.get("root_tags") == "theology"
    assert where.get("folder_tags") == "research"


def test_prefilter_by_metadata_uses_ai_decomposition_when_llm_config(monkeypatch) -> None:
    def _fake_decompose(query, method="deterministic", llm_config=None):
        assert method == "ai"
        assert llm_config is not None
        return SimpleNamespace(
            author="Jane Doe",
            keywords=["christology"],
            title="Christology",
            date_range=("2020", "2022"),
            doi=None,
            raw_query=query,
        )

    monkeypatch.setattr("lsm.query.prefilter.decompose_query", _fake_decompose)
    where = prefilter_by_metadata(
        "Find christology works by Jane Doe",
        available_metadata={
            "ai_tags": ["christology"],
            "content_type": ["theology"],
        },
        llm_config=object(),
    )
    assert where.get("author") == "Jane Doe"
    assert where.get("year") == "2020"
    assert where.get("title") == "Christology"
    assert where.get("ai_tags") == "christology"

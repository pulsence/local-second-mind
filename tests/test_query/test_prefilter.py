from __future__ import annotations

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

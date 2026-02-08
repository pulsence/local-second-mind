from __future__ import annotations

from lsm.query.decomposition import decompose_query, extract_fields_deterministic


def test_extract_fields_deterministic_doi_author_years() -> None:
    fields = extract_fields_deterministic(
        'Find papers by John Smith between 2021 and 2024 DOI 10.1000/xyz123 title: "Vector Search"'
    )
    assert fields.author is not None
    assert fields.doi == "10.1000/xyz123"
    assert fields.date_range == ("2021", "2024")
    assert fields.title is not None
    assert "vector" in fields.keywords


def test_decompose_query_ai_method_falls_back_cleanly() -> None:
    fields = decompose_query("papers about embeddings", method="ai", llm_config=object())
    assert fields.raw_query == "papers about embeddings"
    assert "embeddings" in fields.keywords

from __future__ import annotations

import json
from pathlib import Path


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_data"
DOCUMENTS_DIR = FIXTURE_ROOT / "documents"
CONFIGS_DIR = FIXTURE_ROOT / "configs"


REQUIRED_DOCUMENT_FILES = [
    DOCUMENTS_DIR / "philosophy_essay.txt",
    DOCUMENTS_DIR / "research_paper.md",
    DOCUMENTS_DIR / "technical_manual.html",
    DOCUMENTS_DIR / "short_note.txt",
    DOCUMENTS_DIR / "empty_with_whitespace.txt",
    DOCUMENTS_DIR / "unicode_content.txt",
    DOCUMENTS_DIR / "large_document.md",
    DOCUMENTS_DIR / "duplicate_content_1.txt",
    DOCUMENTS_DIR / "duplicate_content_2.txt",
    DOCUMENTS_DIR / "near_duplicate.txt",
    DOCUMENTS_DIR / "nested" / ".lsm_tags.json",
    DOCUMENTS_DIR / "nested" / "subfolder_a" / ".lsm_tags.json",
    DOCUMENTS_DIR / "nested" / "subfolder_a" / "notes_a1.md",
    DOCUMENTS_DIR / "nested" / "subfolder_a" / "notes_a2.txt",
    DOCUMENTS_DIR / "nested" / "subfolder_b" / "notes_b1.md",
]


REQUIRED_CONFIG_FILES = [
    CONFIGS_DIR / "test_config_openai.json",
    CONFIGS_DIR / "test_config_local.json",
    CONFIGS_DIR / "test_config_minimal.json",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_synthetic_fixture_files_exist() -> None:
    missing = [
        str(path.relative_to(FIXTURE_ROOT))
        for path in [*REQUIRED_DOCUMENT_FILES, *REQUIRED_CONFIG_FILES]
        if not path.exists()
    ]
    assert not missing, f"Missing synthetic fixture files: {missing}"


def test_document_fixture_files_are_readable() -> None:
    for path in REQUIRED_DOCUMENT_FILES:
        text = _read(path)
        assert isinstance(text, str)


def test_large_corpus_documents_have_expected_size() -> None:
    essay_words = len(_read(DOCUMENTS_DIR / "philosophy_essay.txt").split())
    paper_words = len(_read(DOCUMENTS_DIR / "research_paper.md").split())
    manual_words = len(_read(DOCUMENTS_DIR / "technical_manual.html").split())
    large_words = len(_read(DOCUMENTS_DIR / "large_document.md").split())

    assert essay_words >= 1800
    assert paper_words >= 2800
    assert manual_words >= 1300
    assert large_words >= 9000


def test_edge_case_document_content_shapes() -> None:
    short_note_lines = [
        line for line in _read(DOCUMENTS_DIR / "short_note.txt").splitlines() if line.strip()
    ]
    assert len(short_note_lines) == 2

    whitespace_text = _read(DOCUMENTS_DIR / "empty_with_whitespace.txt")
    assert whitespace_text.strip() == ""

    unicode_text = _read(DOCUMENTS_DIR / "unicode_content.txt")
    assert "Η γνώση" in unicode_text
    assert "知识管理" in unicode_text
    assert "إدارة المعرفة" in unicode_text


def test_duplicate_and_near_duplicate_fixtures() -> None:
    duplicate_1 = _read(DOCUMENTS_DIR / "duplicate_content_1.txt")
    duplicate_2 = _read(DOCUMENTS_DIR / "duplicate_content_2.txt")
    near_duplicate = _read(DOCUMENTS_DIR / "near_duplicate.txt")

    assert duplicate_1 == duplicate_2
    assert near_duplicate != duplicate_1
    assert duplicate_1.splitlines()[0] == near_duplicate.splitlines()[0]


def test_folder_tag_fixtures_are_valid_json() -> None:
    root_tags = json.loads(_read(DOCUMENTS_DIR / "nested" / ".lsm_tags.json"))
    sub_tags = json.loads(_read(DOCUMENTS_DIR / "nested" / "subfolder_a" / ".lsm_tags.json"))

    assert root_tags == {"tags": ["philosophy", "notes"]}
    assert sub_tags == {"tags": ["epistemology"]}


def test_synthetic_config_fixtures_are_valid_json_with_required_sections() -> None:
    required_sections = {"global", "ingest", "vectordb", "llms", "query"}
    for path in REQUIRED_CONFIG_FILES:
        data = json.loads(_read(path))
        assert required_sections.issubset(set(data.keys()))
        assert data["ingest"]["roots"]

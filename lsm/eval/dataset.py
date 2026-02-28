"""
BEIR-format dataset loading for retrieval evaluation.

A dataset consists of:
- queries: dict mapping query_id -> query text
- corpus: dict mapping doc_id -> document text
- qrels: dict mapping query_id -> {doc_id: relevance_score}

The bundled synthetic dataset lives in lsm/eval/data/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set

from lsm.logging import get_logger

logger = get_logger(__name__)

_BUNDLED_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class EvalDataset:
    """A retrieval evaluation dataset in BEIR format."""

    name: str
    queries: Dict[str, str]
    corpus: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_documents(self) -> int:
        return len(self.corpus)

    def relevant_for(self, query_id: str) -> Set[str]:
        """Get set of relevant document IDs for a query."""
        return {
            doc_id
            for doc_id, score in self.qrels.get(query_id, {}).items()
            if score > 0
        }


def load_dataset(path: Path | str) -> EvalDataset:
    """
    Load an evaluation dataset from a directory.

    Expects the directory to contain:
    - queries.json: {query_id: query_text, ...}
    - corpus.json: {doc_id: doc_text, ...}
    - qrels.json: {query_id: {doc_id: relevance, ...}, ...}

    Args:
        path: Path to the dataset directory.

    Returns:
        Loaded EvalDataset.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    queries_file = path / "queries.json"
    corpus_file = path / "corpus.json"
    qrels_file = path / "qrels.json"

    for f in (queries_file, corpus_file, qrels_file):
        if not f.exists():
            raise FileNotFoundError(f"Required dataset file not found: {f}")

    with open(queries_file, encoding="utf-8") as fh:
        queries = json.load(fh)
    with open(corpus_file, encoding="utf-8") as fh:
        corpus = json.load(fh)
    with open(qrels_file, encoding="utf-8") as fh:
        qrels = json.load(fh)

    logger.info(
        f"Loaded dataset '{path.name}': "
        f"{len(queries)} queries, {len(corpus)} documents, "
        f"{len(qrels)} qrels"
    )
    return EvalDataset(
        name=path.name,
        queries=queries,
        corpus=corpus,
        qrels=qrels,
    )


def load_bundled_dataset() -> EvalDataset:
    """Load the bundled synthetic evaluation dataset."""
    return load_dataset(_BUNDLED_DATA_DIR)

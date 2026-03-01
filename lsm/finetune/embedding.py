"""
Embedding model fine-tuning using heading-content pairs.

Extracts training data from the corpus (heading → section body pairs)
and fine-tunes a sentence-transformers model using contrastive learning.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lsm.db.tables import DEFAULT_TABLE_NAMES, TableNames
from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingPair:
    """A heading-content pair for contrastive fine-tuning."""

    anchor: str
    positive: str
    source_path: str


def extract_training_pairs(
    conn: Any,
    min_content_length: int = 50,
    max_pairs: Optional[int] = None,
    table_names: TableNames = DEFAULT_TABLE_NAMES,
) -> List[TrainingPair]:
    """Extract heading-content training pairs from the corpus.

    Uses chunks with headings as anchors and chunk text as positives.

    Args:
        conn: SQLite connection with chunk table.
        min_content_length: Minimum chunk text length to include.
        max_pairs: Optional limit on training pairs.

    Returns:
        List of TrainingPair objects.
    """
    tn = table_names
    sql = """
        SELECT heading, chunk_text, source_path
        FROM {chunks}
        WHERE is_current = 1
          AND node_type = 'chunk'
          AND heading IS NOT NULL
          AND heading != ''
          AND LENGTH(chunk_text) >= ?
        ORDER BY source_path, chunk_index
    """.format(chunks=tn.chunks)
    params = [min_content_length]
    if max_pairs:
        sql += " LIMIT ?"
        params.append(max_pairs)

    rows = conn.execute(sql, params).fetchall()

    pairs: List[TrainingPair] = []
    for row in rows:
        heading = row[0]
        text = row[1]
        source_path = row[2]
        if heading and text:
            pairs.append(
                TrainingPair(
                    anchor=heading.strip(),
                    positive=text.strip()[:500],
                    source_path=source_path or "",
                )
            )

    return pairs


def finetune_embedding_model(
    pairs: List[TrainingPair],
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_path: str = "./models/finetuned",
    epochs: int = 3,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
) -> Dict[str, Any]:
    """Fine-tune a sentence-transformers model on training pairs.

    Args:
        pairs: Training pairs (anchor=heading, positive=content).
        base_model: Base model to fine-tune.
        output_path: Directory to save fine-tuned model.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        warmup_ratio: Warmup fraction of total steps.

    Returns:
        Dict with model_id, output_path, dimension, num_pairs, epochs.
    """
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Fine-tuning requires sentence-transformers and torch. "
            "Install with: pip install sentence-transformers torch"
        ) from exc

    if not pairs:
        raise ValueError("No training pairs provided")

    logger.info(
        "Fine-tuning %s with %d pairs, %d epochs",
        base_model, len(pairs), epochs,
    )

    # Load base model
    model = SentenceTransformer(base_model)

    # Create training examples
    train_examples = [
        InputExample(texts=[p.anchor, p.positive])
        for p in pairs
    ]

    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    # Use MultipleNegativesRankingLoss for contrastive learning
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Fine-tune
    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
    )

    # Get dimension from model
    dimension = model.get_sentence_embedding_dimension()

    # Generate model ID
    model_hash = hashlib.sha1(
        f"{base_model}:{len(pairs)}:{epochs}".encode()
    ).hexdigest()[:12]
    model_id = f"finetuned-{model_hash}"

    return {
        "model_id": model_id,
        "base_model": base_model,
        "output_path": str(output_path),
        "dimension": dimension,
        "num_pairs": len(pairs),
        "epochs": epochs,
    }

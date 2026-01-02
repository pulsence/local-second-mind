from __future__ import annotations

import hashlib
from pathlib import Path
import chromadb
from chromadb.config import Settings

# -----------------------------
# Chroma persistence
# -----------------------------
def get_chroma_collection(persist_dir: Path, name: str):
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=name,
        # cosine is typical for sentence-transformers embeddings
        metadata={"hnsw:space": "cosine"},
    )

def make_chunk_id(source_path: str, file_hash: str, chunk_index: int) -> str:
    """
    Generate a unique chunk id per (source file path, file version, chunk index).

    This prevents collisions when two different files have identical content
    (same file_hash), and avoids DuplicateIDError in Chroma.
    """
    sp_norm = source_path.lower()
    sp_hash = hashlib.sha256(sp_norm.encode("utf-8")).hexdigest()
    return f"{sp_hash}:{file_hash}:{chunk_index}"

def chroma_write(collection, ids, documents, metadatas, embeddings):
    """
    Idempotent write with automatic sub-batching to respect Chroma max batch size.
    - Prefer upsert (overwrite existing IDs)
    - Fall back to add (+ delete-then-add recovery)
    """
    if not ids:
        return

    # Determine the maximum batch size allowed by Chroma.
    # Newer chromadb exposes collection._client.get_max_batch_size()
    # Some versions expose it on collection._client, others on collection._client._server.
    max_bs = None
    try:
        max_bs = collection._client.get_max_batch_size()
    except Exception:
        pass

    # Fallback: probe via private attribute if present
    if max_bs is None:
        max_bs = getattr(collection, "max_batch_size", None)

    # Conservative fallback if we cannot introspect
    if not isinstance(max_bs, int) or max_bs <= 0:
        max_bs = 4000

    # Write in sub-batches
    n = len(ids)
    for i in range(0, n, max_bs):
        j = min(i + max_bs, n)
        sub_ids = ids[i:j]
        sub_docs = documents[i:j]
        sub_metas = metadatas[i:j]
        sub_embs = embeddings[i:j]

        # Prefer upsert if available
        try:
            collection.upsert(
                ids=sub_ids,
                documents=sub_docs,
                metadatas=sub_metas,
                embeddings=sub_embs,
            )
            continue
        except AttributeError:
            pass

        try:
            collection.add(
                ids=sub_ids,
                documents=sub_docs,
                metadatas=sub_metas,
                embeddings=sub_embs,
            )
        except Exception as e:
            # Best-effort recovery for duplicates/conflicts
            try:
                collection.delete(ids=sub_ids)
                collection.add(
                    ids=sub_ids,
                    documents=sub_docs,
                    metadatas=sub_metas,
                    embeddings=sub_embs,
                )
            except Exception:
                raise e
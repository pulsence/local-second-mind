"""
AI-powered chunk tagging for knowledge base organization.

Provides on-demand tagging of chunks using LLM models.
Supports incremental tagging (only tags new chunks) and separation
of AI-generated tags from user-provided tags.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from chromadb.api.models.Collection import Collection

from lsm.config.models import LLMConfig
from lsm.providers import create_provider
from lsm.gui.shell.logging import get_logger
from lsm.vectordb.utils import require_chroma_collection

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Helper Functions for ChromaDB Tag Storage
# -----------------------------------------------------------------------------

def _serialize_tags(tags: List[str]) -> str:
    """
    Serialize tags list to JSON string for ChromaDB storage.

    ChromaDB only supports str, int, float, bool, and None in metadata.
    Lists must be stored as JSON strings.
    """
    return json.dumps(tags)


def _deserialize_tags(tags_json: Any) -> List[str]:
    """
    Deserialize tags from ChromaDB metadata.

    Handles both JSON strings (new format) and lists (if any legacy data exists).
    Returns empty list if tags are None or invalid.
    """
    if not tags_json:
        return []

    # If already a list (shouldn't happen with new code, but handle legacy data)
    if isinstance(tags_json, list):
        return tags_json

    # Parse JSON string
    if isinstance(tags_json, str):
        try:
            tags = json.loads(tags_json)
            return tags if isinstance(tags, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Invalid tags JSON: {tags_json!r}")
            return []

    return []


# -----------------------------------------------------------------------------
# Tag Generation
# -----------------------------------------------------------------------------

def generate_tags_for_chunk(
    text: str,
    llm_config: LLMConfig,
    num_tags: int = 3,
    existing_tags: Optional[List[str]] = None,
    max_retries: int = 1,
) -> List[str]:
    """
    Generate relevant tags for a text chunk using LLM.

    Args:
        text: Text content to tag
        llm_config: LLM configuration to use
        num_tags: Number of tags to generate (default: 3)
        existing_tags: Optional list of existing tags to consider
        max_retries: Maximum number of retry attempts (default: 1)

    Returns:
        List of generated tag strings

    Raises:
        Exception: If LLM call fails after all retries
    """
    # Validate LLM config
    llm_config.validate()

    # Create provider from config
    provider = create_provider(llm_config)

    # Try to generate tags with retries
    for attempt in range(max_retries + 1):
        try:
            tags = provider.generate_tags(
                text=text,
                num_tags=num_tags,
                existing_tags=existing_tags,
            )

            # Success if we got valid tags
            if tags:
                if attempt > 0:
                    logger.info(f"Successfully generated tags on retry attempt {attempt}")
                return tags
            else:
                # Empty result, try again
                if attempt < max_retries:
                    logger.warning(f"No tags generated on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    logger.warning(f"No tags generated after {max_retries + 1} attempts")
                    return []

        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Tag generation failed on attempt {attempt + 1}, retrying: {e}")
                continue
            else:
                logger.error(f"Failed to generate tags after {max_retries + 1} attempts: {e}")
                raise

    return []


# -----------------------------------------------------------------------------
# Incremental Tagging
# -----------------------------------------------------------------------------

def get_untagged_chunks(
    collection: Collection,
    batch_size: int = 100,
    processed_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Get chunks that haven't been AI-tagged yet.

    Args:
        collection: ChromaDB collection
        batch_size: Number of untagged chunks to retrieve
        processed_ids: Set of chunk IDs already processed (to skip)

    Returns:
        List of metadata dictionaries for untagged chunks
    """
    # ChromaDB doesn't support querying for None/null values directly,
    # so we get all chunks and filter manually.
    if processed_ids is None:
        processed_ids = set()

    collection = require_chroma_collection(collection, "get_untagged_chunks")
    try:
        logger.debug("Fetching chunks and filtering for untagged ones")

        # Fetch a large batch and filter for untagged chunks
        # We need to fetch more than batch_size since some will be tagged
        fetch_limit = max(batch_size * 10, 1000)  # Fetch enough to find untagged ones

        results = collection.get(
            limit=fetch_limit,
            include=["metadatas", "documents"],
        )

        if not results or not results.get("metadatas"):
            return []

        untagged = []
        for i, meta in enumerate(results["metadatas"]):
            chunk_id = results["ids"][i]

            # Skip if already processed
            if chunk_id in processed_ids:
                continue

            # Check if ai_tags is missing or empty
            if "ai_tags" not in meta or not meta.get("ai_tags"):
                chunk_data = {
                    "id": chunk_id,
                    "metadata": meta,
                    "text": results["documents"][i] if results.get("documents") else "",
                }
                untagged.append(chunk_data)

                # Stop if we have enough
                if len(untagged) >= batch_size:
                    break

        logger.debug(f"Found {len(untagged)} untagged chunks out of {len(results['ids'])} fetched")
        return untagged

    except Exception as e:
        logger.error(f"Error fetching chunks: {e}")
        return []


def tag_chunks(
    collection: Collection,
    llm_config: LLMConfig,
    num_tags: int = 3,
    batch_size: int = 100,
    max_chunks: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Tag untagged chunks in a collection using AI.

    Only tags chunks that haven't been AI-tagged before (incremental).
    Updates chunks with ai_tags and ai_tagged_at metadata fields.

    Args:
        collection: ChromaDB collection
        llm_config: LLM configuration for tagging
        num_tags: Number of tags to generate per chunk (default: 3)
        batch_size: Number of chunks to process per batch (default: 100)
        max_chunks: Maximum number of chunks to tag (None = no limit)
        dry_run: If True, don't actually update the database

    Returns:
        Tuple of (chunks_tagged, chunks_failed)

    Example:
        >>> config = LSMConfig.load("config.json")
        >>> collection = create_vectordb_provider(config.vectordb).get_collection()
        >>> tagging_config = config.llm.get_tagging_config()
        >>> tagged, failed = tag_chunks(collection, tagging_config, num_tags=3)
        >>> print(f"Tagged {tagged} chunks, {failed} failed")
    """
    collection = require_chroma_collection(collection, "tag_chunks")
    logger.info("Starting AI chunk tagging...")
    logger.info(f"  Model: {llm_config.model}")
    logger.info(f"  Tags per chunk: {num_tags}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max chunks: {max_chunks or 'unlimited'}")
    logger.info(f"  Dry run: {dry_run}")

    tagged_count = 0
    failed_count = 0
    total_processed = 0
    processed_ids: set = set()  # Track processed chunk IDs

    # Get existing tags for context (sample from collection)
    existing_tags: List[str] = []
    try:
        sample_results = collection.get(limit=100, include=["metadatas"])
        if sample_results and sample_results.get("metadatas"):
            for meta in sample_results["metadatas"]:
                if "ai_tags" in meta and meta["ai_tags"]:
                    # Deserialize tags from JSON
                    tags = _deserialize_tags(meta["ai_tags"])
                    existing_tags.extend(tags)
        existing_tags = list(set(existing_tags))  # Deduplicate
        logger.info(f"Found {len(existing_tags)} existing unique tags for context")
    except Exception as e:
        logger.warning(f"Could not fetch existing tags: {e}")

    # Process in batches
    while True:
        # Check if we've hit the max
        if max_chunks and total_processed >= max_chunks:
            logger.info(f"Reached max_chunks limit ({max_chunks})")
            break

        # Get next batch of untagged chunks
        remaining = batch_size
        if max_chunks:
            remaining = min(batch_size, max_chunks - total_processed)

        untagged = get_untagged_chunks(collection, batch_size=remaining, processed_ids=processed_ids)

        if not untagged:
            logger.info("No more untagged chunks found")
            break

        logger.info(f"Processing batch of {len(untagged)} untagged chunks...")

        # Tag each chunk
        for chunk_data in untagged:
            # Check if we've hit the max before processing
            if max_chunks and total_processed >= max_chunks:
                logger.info(f"Reached max_chunks limit ({max_chunks})")
                break

            chunk_id = chunk_data["id"]
            text = chunk_data["text"]
            metadata = chunk_data["metadata"]

            # Get source info for logging
            source_path = metadata.get("source_path", "unknown")
            chunk_index = metadata.get("chunk_index", 0)

            try:
                # Generate tags
                tags = generate_tags_for_chunk(
                    text=text,
                    llm_config=llm_config,
                    num_tags=num_tags,
                    existing_tags=existing_tags if existing_tags else None,
                )

                if tags:
                    # Log with source info and tags BEFORE serialization - show filename only for readability
                    filename = Path(source_path).name if source_path != "unknown" else source_path
                    tags_str = ", ".join(tags)
                    logger.info(f"  ✓ Chunk #{chunk_index} from '{filename}': {tags_str}")

                    # Update metadata - serialize tags as JSON string for ChromaDB
                    metadata["ai_tags"] = _serialize_tags(tags)
                    metadata["ai_tagged_at"] = datetime.now().isoformat()

                    # Add to existing tags pool
                    existing_tags.extend(tags)
                    existing_tags = list(set(existing_tags))  # Deduplicate

                    # Update in database
                    if not dry_run:
                        collection.update(
                            ids=[chunk_id],
                            metadatas=[metadata],
                        )

                    tagged_count += 1
                else:
                    logger.warning(f"No tags generated for chunk #{chunk_index} from {source_path}")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Failed to tag chunk #{chunk_index} from {source_path}: {e}")
                failed_count += 1

            # Mark as processed
            processed_ids.add(chunk_id)
            total_processed += 1

            # Progress update
            if total_processed % 10 == 0:
                logger.info(f"Progress: {total_processed} chunks processed ({tagged_count} tagged, {failed_count} failed)")

    logger.info("AI tagging completed")
    logger.info(f"  Total processed: {total_processed}")
    logger.info(f"  Successfully tagged: {tagged_count}")
    logger.info(f"  Failed: {failed_count}")

    return tagged_count, failed_count


# -----------------------------------------------------------------------------
# Tag Management
# -----------------------------------------------------------------------------

def add_user_tags(
    collection: Collection,
    chunk_id: str,
    tags: List[str],
) -> None:
    """
    Add user-provided tags to a chunk.

    User tags are stored separately from AI tags in the 'user_tags' field.

    Args:
        collection: ChromaDB collection
        chunk_id: ID of chunk to tag
        tags: List of tags to add

    Raises:
        ValueError: If chunk doesn't exist
    """
    collection = require_chroma_collection(collection, "add_user_tags")
    # Get current metadata
    results = collection.get(ids=[chunk_id], include=["metadatas"])

    if not results or not results.get("metadatas"):
        raise ValueError(f"Chunk not found: {chunk_id}")

    metadata = results["metadatas"][0]

    # Get existing user tags - deserialize from JSON
    existing_tags = _deserialize_tags(metadata.get("user_tags"))

    # Add new tags (avoid duplicates)
    updated_tags = list(set(existing_tags + [t.lower().strip() for t in tags]))

    # Update metadata - serialize as JSON string for ChromaDB
    metadata["user_tags"] = _serialize_tags(updated_tags)
    metadata["user_tagged_at"] = datetime.now().isoformat()

    # Save
    collection.update(ids=[chunk_id], metadatas=[metadata])

    logger.info(f"Added user tags to {chunk_id}: {tags}")


def remove_user_tags(
    collection: Collection,
    chunk_id: str,
    tags: List[str],
) -> None:
    """
    Remove user-provided tags from a chunk.

    Args:
        collection: ChromaDB collection
        chunk_id: ID of chunk
        tags: List of tags to remove

    Raises:
        ValueError: If chunk doesn't exist
    """
    collection = require_chroma_collection(collection, "remove_user_tags")
    # Get current metadata
    results = collection.get(ids=[chunk_id], include=["metadatas"])

    if not results or not results.get("metadatas"):
        raise ValueError(f"Chunk not found: {chunk_id}")

    metadata = results["metadatas"][0]

    # Get existing user tags - deserialize from JSON
    existing_tags = _deserialize_tags(metadata.get("user_tags"))

    # Remove specified tags
    tags_to_remove = {t.lower().strip() for t in tags}
    updated_tags = [t for t in existing_tags if t not in tags_to_remove]

    # Update metadata - serialize as JSON string for ChromaDB
    metadata["user_tags"] = _serialize_tags(updated_tags)
    metadata["user_tagged_at"] = datetime.now().isoformat()

    # Save
    collection.update(ids=[chunk_id], metadatas=[metadata])

    logger.info(f"Removed user tags from {chunk_id}: {tags}")


def get_all_tags(collection: Collection) -> Dict[str, List[str]]:
    """
    Get all unique tags in the collection.

    Returns:
        Dictionary with 'ai_tags' and 'user_tags' lists
    """
    collection = require_chroma_collection(collection, "get_all_tags")
    ai_tags: set[str] = set()
    user_tags: set[str] = set()

    # Sample chunks to get tags (limit to avoid memory issues)
    results = collection.get(limit=10000, include=["metadatas"])

    if results and results.get("metadatas"):
        for meta in results["metadatas"]:
            # Collect AI tags - deserialize from JSON
            if "ai_tags" in meta and meta["ai_tags"]:
                tags = _deserialize_tags(meta["ai_tags"])
                ai_tags.update(tags)

            # Collect user tags - deserialize from JSON
            if "user_tags" in meta and meta["user_tags"]:
                tags = _deserialize_tags(meta["user_tags"])
                user_tags.update(tags)

    return {
        "ai_tags": sorted(list(ai_tags)),
        "user_tags": sorted(list(user_tags)),
    }

"""
AI-powered chunk tagging for knowledge base organization.

Provides on-demand tagging of chunks using LLM models.
Supports incremental tagging (only tags new chunks) and separation
of AI-generated tags from user-provided tags.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from chromadb.api.models.Collection import Collection
from openai import OpenAI

from lsm.config.models import LLMConfig
from lsm.cli.logging import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Tag Generation
# -----------------------------------------------------------------------------

def generate_tags_for_chunk(
    text: str,
    llm_config: LLMConfig,
    num_tags: int = 3,
    existing_tags: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate relevant tags for a text chunk using LLM.

    Args:
        text: Text content to tag
        llm_config: LLM configuration to use
        num_tags: Number of tags to generate (default: 3)
        existing_tags: Optional list of existing tags to consider

    Returns:
        List of generated tag strings

    Raises:
        Exception: If LLM call fails
    """
    # Validate LLM config
    llm_config.validate()

    # Create OpenAI client
    client = OpenAI(api_key=llm_config.api_key)

    # Build prompt
    existing_context = ""
    if existing_tags:
        existing_context = f"\n\nExisting tags in this knowledge base: {', '.join(existing_tags[:20])}"

    prompt = f"""Analyze the following text and generate {num_tags} relevant tags.

Guidelines:
- Tags should be concise (1-3 words)
- Tags should be specific to the content
- Tags should help with organization and retrieval
- Use lowercase
- Separate multi-word tags with hyphens (e.g., "machine-learning")
{existing_context}

Text:
{text[:2000]}

Respond with ONLY a JSON array of tags, like: ["tag1", "tag2", "tag3"]
"""

    try:
        response = client.chat.completions.create(
            model=llm_config.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise, relevant tags for text content."},
                {"role": "user", "content": prompt}
            ],
            temperature=llm_config.temperature,
            max_tokens=min(llm_config.max_tokens, 200),  # Tags don't need many tokens
        )

        # Extract tags from response
        content = response.choices[0].message.content.strip()

        # Try to parse as JSON
        try:
            tags = json.loads(content)
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                # Normalize tags
                tags = [t.lower().strip() for t in tags if t.strip()]
                return tags[:num_tags]
        except json.JSONDecodeError:
            # If not JSON, try to extract tags from comma-separated list
            logger.warning(f"LLM response not JSON, attempting to parse: {content}")
            tags = [t.strip().lower() for t in content.split(",")]
            tags = [t for t in tags if t and not t.startswith("[") and not t.startswith("]")]
            return tags[:num_tags]

        logger.error(f"Could not parse tags from LLM response: {content}")
        return []

    except Exception as e:
        logger.error(f"Failed to generate tags: {e}")
        raise


# -----------------------------------------------------------------------------
# Incremental Tagging
# -----------------------------------------------------------------------------

def get_untagged_chunks(
    collection: Collection,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Get chunks that haven't been AI-tagged yet.

    Args:
        collection: ChromaDB collection
        batch_size: Number of chunks to retrieve per batch

    Returns:
        List of metadata dictionaries for untagged chunks
    """
    # Query for chunks without ai_tags field or with null ai_tags
    try:
        results = collection.get(
            where={
                "$or": [
                    {"ai_tags": {"$eq": None}},
                    {"ai_tags": {"$size": 0}},
                ]
            },
            limit=batch_size,
            include=["metadatas", "documents"],
        )

        if not results or not results.get("metadatas"):
            return []

        # Combine metadata with documents and IDs
        untagged = []
        for i, meta in enumerate(results["metadatas"]):
            chunk_data = {
                "id": results["ids"][i],
                "metadata": meta,
                "text": results["documents"][i] if results.get("documents") else "",
            }
            untagged.append(chunk_data)

        return untagged

    except Exception as e:
        logger.error(f"Error querying untagged chunks: {e}")
        # Fallback: get all chunks and filter manually
        logger.info("Falling back to manual filtering of untagged chunks")

        results = collection.get(
            limit=batch_size,
            include=["metadatas", "documents"],
        )

        if not results or not results.get("metadatas"):
            return []

        untagged = []
        for i, meta in enumerate(results["metadatas"]):
            # Check if ai_tags is missing or empty
            if "ai_tags" not in meta or not meta.get("ai_tags"):
                chunk_data = {
                    "id": results["ids"][i],
                    "metadata": meta,
                    "text": results["documents"][i] if results.get("documents") else "",
                }
                untagged.append(chunk_data)

        return untagged


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
        >>> collection = get_chroma_collection(config.persist_dir, config.collection)
        >>> tagging_config = config.llm.get_tagging_config()
        >>> tagged, failed = tag_chunks(collection, tagging_config, num_tags=3)
        >>> print(f"Tagged {tagged} chunks, {failed} failed")
    """
    logger.info("Starting AI chunk tagging...")
    logger.info(f"  Model: {llm_config.model}")
    logger.info(f"  Tags per chunk: {num_tags}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max chunks: {max_chunks or 'unlimited'}")
    logger.info(f"  Dry run: {dry_run}")

    tagged_count = 0
    failed_count = 0
    total_processed = 0

    # Get existing tags for context (sample from collection)
    existing_tags: List[str] = []
    try:
        sample_results = collection.get(limit=100, include=["metadatas"])
        if sample_results and sample_results.get("metadatas"):
            for meta in sample_results["metadatas"]:
                if "ai_tags" in meta and meta["ai_tags"]:
                    existing_tags.extend(meta["ai_tags"])
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

        untagged = get_untagged_chunks(collection, batch_size=remaining)

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

            try:
                # Generate tags
                tags = generate_tags_for_chunk(
                    text=text,
                    llm_config=llm_config,
                    num_tags=num_tags,
                    existing_tags=existing_tags if existing_tags else None,
                )

                if tags:
                    # Update metadata
                    metadata["ai_tags"] = tags
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
                    logger.debug(f"Tagged chunk {chunk_id}: {tags}")
                else:
                    logger.warning(f"No tags generated for chunk {chunk_id}")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Failed to tag chunk {chunk_id}: {e}")
                failed_count += 1

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
    # Get current metadata
    results = collection.get(ids=[chunk_id], include=["metadatas"])

    if not results or not results.get("metadatas"):
        raise ValueError(f"Chunk not found: {chunk_id}")

    metadata = results["metadatas"][0]

    # Get existing user tags
    existing_tags = metadata.get("user_tags", [])
    if not isinstance(existing_tags, list):
        existing_tags = []

    # Add new tags (avoid duplicates)
    updated_tags = list(set(existing_tags + [t.lower().strip() for t in tags]))

    # Update metadata
    metadata["user_tags"] = updated_tags
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
    # Get current metadata
    results = collection.get(ids=[chunk_id], include=["metadatas"])

    if not results or not results.get("metadatas"):
        raise ValueError(f"Chunk not found: {chunk_id}")

    metadata = results["metadatas"][0]

    # Get existing user tags
    existing_tags = metadata.get("user_tags", [])
    if not isinstance(existing_tags, list):
        existing_tags = []

    # Remove specified tags
    tags_to_remove = {t.lower().strip() for t in tags}
    updated_tags = [t for t in existing_tags if t not in tags_to_remove]

    # Update metadata
    metadata["user_tags"] = updated_tags
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
    ai_tags: set[str] = set()
    user_tags: set[str] = set()

    # Sample chunks to get tags (limit to avoid memory issues)
    results = collection.get(limit=10000, include=["metadatas"])

    if results and results.get("metadatas"):
        for meta in results["metadatas"]:
            # Collect AI tags
            if "ai_tags" in meta and meta["ai_tags"]:
                ai_tags.update(meta["ai_tags"])

            # Collect user tags
            if "user_tags" in meta and meta["user_tags"]:
                user_tags.update(meta["user_tags"])

    return {
        "ai_tags": sorted(list(ai_tags)),
        "user_tags": sorted(list(user_tags)),
    }

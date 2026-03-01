"""
Graph builder for ingest-time knowledge graph construction.

Extracts nodes and edges from FileGraph heading hierarchy and
document-internal links (wikilinks, markdown links, DOI citations).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lsm.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DBGraphNode:
    """Node destined for lsm_graph_nodes table."""

    node_id: str
    node_type: str
    label: str
    source_path: str
    heading_path: Optional[str] = None


@dataclass
class DBGraphEdge:
    """Edge destined for lsm_graph_edges table."""

    src_id: str
    dst_id: str
    edge_type: str
    weight: float = 1.0


# -- Regex patterns for link extraction --
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_DOI_RE = re.compile(r"\b(?:doi:|https?://doi\.org/)(10\.\d{4,}/\S+)", re.IGNORECASE)


def _stable_id(source_path: str, *parts: str) -> str:
    """Generate a deterministic node ID."""
    raw = "|".join([source_path] + list(parts))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_graph_from_file_graph(
    file_graph,
    source_path: str,
    raw_text: str,
) -> Tuple[List[DBGraphNode], List[DBGraphEdge]]:
    """Build graph nodes and edges from a FileGraph and document text.

    Args:
        file_graph: FileGraph instance (from lsm.utils.file_graph).
        source_path: Canonical source path for the file.
        raw_text: Raw document text for link extraction.

    Returns:
        Tuple of (nodes, edges).
    """
    nodes: List[DBGraphNode] = []
    edges: List[DBGraphEdge] = []

    # 1. File-level root node
    file_node_id = _stable_id(source_path, "file")
    nodes.append(
        DBGraphNode(
            node_id=file_node_id,
            node_type="file",
            label=source_path.rsplit("/", 1)[-1] if "/" in source_path else source_path,
            source_path=source_path,
        )
    )

    # 2. Build heading hierarchy nodes from FileGraph
    if file_graph is not None:
        node_map = file_graph.node_map() if hasattr(file_graph, "node_map") else {}
        fg_nodes = file_graph.nodes if hasattr(file_graph, "nodes") else ()

        for fg_node in fg_nodes:
            heading_path_str = _build_heading_path(fg_node, node_map)
            db_node_id = _stable_id(source_path, "heading", fg_node.id)
            nodes.append(
                DBGraphNode(
                    node_id=db_node_id,
                    node_type=fg_node.node_type,
                    label=fg_node.name,
                    source_path=source_path,
                    heading_path=heading_path_str,
                )
            )

            # "contains" edge: parent → child
            if fg_node.parent_id is not None:
                parent_db_id = _stable_id(source_path, "heading", fg_node.parent_id)
                edges.append(
                    DBGraphEdge(
                        src_id=parent_db_id,
                        dst_id=db_node_id,
                        edge_type="contains",
                    )
                )
            else:
                # Root heading → connect to file node
                edges.append(
                    DBGraphEdge(
                        src_id=file_node_id,
                        dst_id=db_node_id,
                        edge_type="contains",
                    )
                )

    # 3. Extract links from text
    link_nodes, link_edges = _extract_links(raw_text, source_path, file_node_id)
    nodes.extend(link_nodes)
    edges.extend(link_edges)

    return nodes, edges


def _build_heading_path(fg_node, node_map: Dict[str, Any]) -> str:
    """Build a slash-separated heading path from a FileGraph node."""
    parts = []
    current = fg_node
    while current is not None:
        parts.append(current.name)
        current = node_map.get(current.parent_id) if current.parent_id else None
    parts.reverse()
    return " / ".join(parts)


def _extract_links(
    text: str,
    source_path: str,
    file_node_id: str,
) -> Tuple[List[DBGraphNode], List[DBGraphEdge]]:
    """Extract wikilinks, markdown links, and DOI references."""
    nodes: List[DBGraphNode] = []
    edges: List[DBGraphEdge] = []
    seen_targets: set = set()

    # Wikilinks: [[target]] or [[target|display]]
    for match in _WIKILINK_RE.finditer(text):
        target = match.group(1).strip()
        if not target or target in seen_targets:
            continue
        seen_targets.add(target)
        target_id = _stable_id(target, "link_target")
        nodes.append(
            DBGraphNode(
                node_id=target_id,
                node_type="link_target",
                label=target,
                source_path=target,
            )
        )
        edges.append(
            DBGraphEdge(
                src_id=file_node_id,
                dst_id=target_id,
                edge_type="references",
            )
        )

    # Markdown links: [text](path) — only internal (no http)
    for match in _MD_LINK_RE.finditer(text):
        href = match.group(2).strip()
        if href.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if href in seen_targets:
            continue
        seen_targets.add(href)
        target_id = _stable_id(href, "link_target")
        nodes.append(
            DBGraphNode(
                node_id=target_id,
                node_type="link_target",
                label=href,
                source_path=href,
            )
        )
        edges.append(
            DBGraphEdge(
                src_id=file_node_id,
                dst_id=target_id,
                edge_type="references",
            )
        )

    # DOI references
    for match in _DOI_RE.finditer(text):
        doi = match.group(1).strip()
        if doi in seen_targets:
            continue
        seen_targets.add(doi)
        doi_id = _stable_id(doi, "doi")
        nodes.append(
            DBGraphNode(
                node_id=doi_id,
                node_type="doi",
                label=doi,
                source_path="",
            )
        )
        edges.append(
            DBGraphEdge(
                src_id=file_node_id,
                dst_id=doi_id,
                edge_type="cites",
            )
        )

    return nodes, edges

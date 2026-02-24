from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GRAPH_NODE_VERSION = 1


@dataclass(frozen=True)
class GraphNode:
    """Unified node schema for file graphs."""

    id: str
    node_type: str
    name: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    depth: int
    parent_id: Optional[str]
    children: Tuple[str, ...]
    metadata: Dict[str, Any]
    line_hash: str

    def validate(self) -> None:
        errors: List[str] = []
        if not self.node_type:
            errors.append("node_type must be non-empty")
        if not self.name:
            errors.append("name must be non-empty")
        if self.start_line < 1:
            errors.append("start_line must be >= 1")
        if self.end_line < self.start_line:
            errors.append("end_line must be >= start_line")
        if self.start_char < 0:
            errors.append("start_char must be >= 0")
        if self.end_char < self.start_char:
            errors.append("end_char must be >= start_char")
        if self.depth < 0:
            errors.append("depth must be >= 0")
        if not isinstance(self.children, tuple):
            errors.append("children must be a tuple")
        if not isinstance(self.metadata, dict):
            errors.append("metadata must be a dict")
        if not isinstance(self.line_hash, str) or not self.line_hash:
            errors.append("line_hash must be a non-empty string")
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "name": self.name,
            "span": {
                "start_line": self.start_line,
                "end_line": self.end_line,
                "start_char": self.start_char,
                "end_char": self.end_char,
            },
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "metadata": dict(self.metadata),
            "line_hash": self.line_hash,
            "version": GRAPH_NODE_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphNode":
        span = payload.get("span") or {}
        return cls(
            id=str(payload.get("id", "")),
            node_type=str(payload.get("node_type", "")),
            name=str(payload.get("name", "")),
            start_line=int(span.get("start_line", 0) or 0),
            end_line=int(span.get("end_line", 0) or 0),
            start_char=int(span.get("start_char", 0) or 0),
            end_char=int(span.get("end_char", 0) or 0),
            depth=int(payload.get("depth", 0) or 0),
            parent_id=payload.get("parent_id"),
            children=tuple(payload.get("children") or ()),
            metadata=dict(payload.get("metadata") or {}),
            line_hash=str(payload.get("line_hash", "")),
        )


@dataclass(frozen=True)
class FileGraph:
    path: str
    content_hash: str
    nodes: Tuple[GraphNode, ...]
    root_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "nodes": [node.to_dict() for node in self.nodes],
            "root_ids": list(self.root_ids),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FileGraph":
        return cls(
            path=str(payload.get("path", "")),
            content_hash=str(payload.get("content_hash", "")),
            nodes=tuple(GraphNode.from_dict(item) for item in payload.get("nodes", [])),
            root_ids=tuple(payload.get("root_ids", [])),
        )

    def node_map(self) -> Dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}

    def with_path(self, path: Path) -> "FileGraph":
        resolved = str(Path(path).resolve())
        if resolved == self.path:
            return self
        return FileGraph(
            path=resolved,
            content_hash=self.content_hash,
            nodes=self.nodes,
            root_ids=self.root_ids,
        )


def stable_node_id(
    content_hash: str,
    node_type: str,
    name: str,
    start_line: int,
    end_line: int,
    start_char: int,
    end_char: int,
    depth: int,
    parent_id: Optional[str],
) -> str:
    payload = "|".join(
        [
            content_hash,
            node_type,
            name,
            str(start_line),
            str(end_line),
            str(start_char),
            str(end_char),
            str(depth),
            parent_id or "",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def order_nodes(nodes: Iterable[GraphNode]) -> List[GraphNode]:
    return sorted(
        nodes,
        key=lambda node: (
            node.start_line,
            node.start_char,
            node.end_line,
            node.end_char,
            node.depth,
            node.node_type,
            node.name,
            node.id,
        ),
    )


def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_line_hash(lines: Sequence[str]) -> str:
    normalized = "\n".join(lines).rstrip("\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _decode_bytes(content: bytes) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="ignore")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _line_offsets(text: str) -> Tuple[List[str], List[int]]:
    lines = text.split("\n")
    offsets: List[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1  # account for newline
    return lines, offsets


def _line_span_hash(lines: Sequence[str], start_line: int, end_line: int) -> str:
    if not lines:
        return compute_line_hash([])
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return compute_line_hash(lines[start_idx:end_idx])


def _build_basic_graph(path: Path, text: str, content_hash: str) -> FileGraph:
    lines, _ = _line_offsets(text)
    total_lines = max(len(lines), 1)
    end_char = len(text)
    line_hash = _line_span_hash(lines, 1, total_lines)
    root_id = stable_node_id(
        content_hash,
        "document",
        path.name,
        1,
        total_lines,
        0,
        end_char,
        0,
        None,
    )
    root = GraphNode(
        id=root_id,
        node_type="document",
        name=path.name,
        start_line=1,
        end_line=total_lines,
        start_char=0,
        end_char=end_char,
        depth=0,
        parent_id=None,
        children=(),
        metadata={"path": str(path)},
        line_hash=line_hash,
    )
    root.validate()
    return FileGraph(
        path=str(path.resolve()),
        content_hash=content_hash,
        nodes=(root,),
        root_ids=(root_id,),
    )


_GRAPH_CACHE: Dict[str, FileGraph] = {}


def get_file_graph(path: Path | str) -> FileGraph:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    content_bytes = file_path.read_bytes()
    content_hash = compute_content_hash(content_bytes)
    cached = _GRAPH_CACHE.get(content_hash)
    if cached is not None:
        return cached.with_path(file_path)

    text = _decode_bytes(content_bytes)
    graph = _build_basic_graph(file_path, text, content_hash)
    _GRAPH_CACHE[content_hash] = graph
    return graph

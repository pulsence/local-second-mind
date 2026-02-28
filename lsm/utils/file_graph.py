from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from lsm.utils.text_processing import extract_docx_text, is_list_block, split_paragraphs


GRAPH_NODE_VERSION = 1
LINE_HASH_CHARS = 8

CODE_EXTENSIONS = {
    ".py",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
}

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".docx",
}

HTML_EXTENSIONS = {
    ".html",
    ".htm",
}


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


def compute_line_hashes(
    lines: Sequence[str],
    *,
    size: int = LINE_HASH_CHARS,
) -> List[str]:
    hashes: List[str] = []
    for line in lines:
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        hashes.append(digest[:size] if size else digest)
    return hashes


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


@dataclass
class _NodeDraft:
    node_type: str
    name: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


def _apply_section_spans(
    drafts: List[_NodeDraft],
    heading_indices: List[Tuple[int, int]],
    end_limit: Optional[int] = None,
) -> None:
    if end_limit is None:
        end_limit = len(drafts) - 1
    for pos, (draft_idx, level) in enumerate(heading_indices):
        end_idx = end_limit
        for next_idx, next_level in heading_indices[pos + 1 :]:
            if next_level <= level:
                end_idx = next_idx - 1
                break
        if end_idx < draft_idx:
            end_idx = draft_idx
        tail = drafts[end_idx]
        draft = drafts[draft_idx]
        if tail.end_line > draft.end_line:
            draft.end_line = tail.end_line
        if tail.end_char > draft.end_char:
            draft.end_char = tail.end_char


def _assemble_graph(
    path: Path,
    text: str,
    content_hash: str,
    drafts: Sequence[_NodeDraft],
    parent_by_node_id: Optional[Dict[str, Optional[str]]] = None,
) -> FileGraph:
    lines, _ = _line_offsets(text)
    total_lines = max(len(lines), 1)
    end_char = len(text)
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
        line_hash=_line_span_hash(lines, 1, total_lines),
    )

    base_nodes: List[GraphNode] = []
    for draft in drafts:
        node_id = stable_node_id(
            content_hash,
            draft.node_type,
            draft.name,
            draft.start_line,
            draft.end_line,
            draft.start_char,
            draft.end_char,
            0,
            None,
        )
        node = GraphNode(
            id=node_id,
            node_type=draft.node_type,
            name=draft.name,
            start_line=draft.start_line,
            end_line=draft.end_line,
            start_char=draft.start_char,
            end_char=draft.end_char,
            depth=0,
            parent_id=None,
            children=(),
            metadata=draft.metadata,
            line_hash=_line_span_hash(lines, draft.start_line, draft.end_line),
        )
        base_nodes.append(node)

    parent_map: Dict[str, str] = {}
    if parent_by_node_id is None:
        for node in base_nodes:
            parent_candidate: Optional[GraphNode] = None
            for other in base_nodes:
                if other.id == node.id:
                    continue
                if other.start_line <= node.start_line and other.end_line >= node.end_line:
                    if parent_candidate is None:
                        parent_candidate = other
                        continue
                    span_len = other.end_line - other.start_line
                    parent_span = parent_candidate.end_line - parent_candidate.start_line
                    if span_len < parent_span:
                        parent_candidate = other
            parent_map[node.id] = parent_candidate.id if parent_candidate else root_id
    else:
        node_ids = {node.id for node in base_nodes}
        for node in base_nodes:
            candidate = parent_by_node_id.get(node.id)
            if candidate is None or candidate == node.id:
                parent_map[node.id] = root_id
            elif candidate != root_id and candidate not in node_ids:
                parent_map[node.id] = root_id
            else:
                parent_map[node.id] = candidate

    def _depth_for(node_id: str) -> int:
        depth = 0
        current = node_id
        visited = set()
        while True:
            if current in visited:
                return depth
            visited.add(current)
            parent_id = parent_map.get(current)
            if not parent_id:
                return depth
            depth += 1
            if parent_id == root_id:
                return depth
            current = parent_id

    children_map: Dict[str, List[str]] = {root_id: []}
    for node in base_nodes:
        parent_id = parent_map[node.id]
        children_map.setdefault(parent_id, []).append(node.id)

    node_lookup = {node.id: node for node in base_nodes}
    ordered_nodes: List[GraphNode] = []
    for node in base_nodes:
        depth = _depth_for(node.id)
        parent_id = parent_map[node.id]
        children_ids = children_map.get(node.id, [])
        children_ids_sorted = sorted(
            children_ids,
            key=lambda cid: (
                node_lookup[cid].start_line,
                node_lookup[cid].start_char,
                node_lookup[cid].end_line,
                node_lookup[cid].end_char,
            ),
        )
        ordered_nodes.append(
            GraphNode(
                id=node.id,
                node_type=node.node_type,
                name=node.name,
                start_line=node.start_line,
                end_line=node.end_line,
                start_char=node.start_char,
                end_char=node.end_char,
                depth=depth,
                parent_id=parent_id,
                children=tuple(children_ids_sorted),
                metadata=node.metadata,
                line_hash=node.line_hash,
            )
        )

    root_children = children_map.get(root_id, [])
    root_children_sorted = sorted(
        root_children,
        key=lambda cid: (
            node_lookup[cid].start_line,
            node_lookup[cid].start_char,
            node_lookup[cid].end_line,
            node_lookup[cid].end_char,
        ),
    )
    root = GraphNode(
        id=root.id,
        node_type=root.node_type,
        name=root.name,
        start_line=root.start_line,
        end_line=root.end_line,
        start_char=root.start_char,
        end_char=root.end_char,
        depth=root.depth,
        parent_id=root.parent_id,
        children=tuple(root_children_sorted),
        metadata=root.metadata,
        line_hash=root.line_hash,
    )

    root.validate()
    for node in ordered_nodes:
        node.validate()

    return FileGraph(
        path=str(path.resolve()),
        content_hash=content_hash,
        nodes=tuple([root] + order_nodes(ordered_nodes)),
        root_ids=(root_id,),
    )


_PY_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][\w]*)\s*\(")
_PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][\w]*)")
_PY_IMPORT_RE = re.compile(r"^\s*(?:import\s+[\w\.]+|from\s+[\w\.]+\s+import\b)")
_PY_BLOCK_RE = re.compile(r"^\s*(if|elif|else|for|while|try|except|with)\b")

_JS_FUNC_RE = re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][\w]*)\s*\(")
_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w]*)")
_JS_IMPORT_RE = re.compile(r"^\s*import\b")
_JS_BLOCK_RE = re.compile(r"^\s*(if|for|while|switch|try|catch)\b")

_GENERIC_FUNC_RE = re.compile(r"^\s*(?:def|function)\s+([A-Za-z_][\w]*)")
_GENERIC_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][\w]*)")


def _leading_indent(line: str) -> int:
    expanded = line.replace("\t", "    ")
    return len(expanded) - len(expanded.lstrip(" "))


def _python_block_end(lines: Sequence[str], start_idx: int, indent: int) -> int:
    last_content = start_idx
    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        if _leading_indent(line) <= indent:
            return last_content
        last_content = idx
    return last_content


def _brace_block_end(lines: Sequence[str], start_idx: int) -> int:
    depth = 0
    started = False
    last_content = start_idx
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        depth += line.count("{")
        if "{" in line:
            started = True
        depth -= line.count("}")
        if line.strip():
            last_content = idx
        if started and depth <= 0:
            return last_content
    return last_content


def _language_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".py", ".pyw"}:
        return "python"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    return "generic"


def _try_tree_sitter_nodes(text: str, language: str) -> Optional[List[_NodeDraft]]:
    try:
        from tree_sitter import Parser
        from tree_sitter_languages import get_language
    except Exception:
        return None

    try:
        ts_language = get_language(language)
    except Exception:
        return None

    parser = Parser()
    parser.set_language(ts_language)
    tree = parser.parse(text.encode("utf-8"))

    if language == "python":
        mapping = {
            "function_definition": "function",
            "class_definition": "class",
            "import_statement": "import",
            "import_from_statement": "import",
        }
    elif language in {"javascript", "typescript"}:
        mapping = {
            "function_declaration": "function",
            "class_declaration": "class",
            "import_statement": "import",
        }
    else:
        mapping = {}

    if not mapping:
        return None

    nodes: List[_NodeDraft] = []
    cursor = tree.walk()
    stack = [cursor.node]
    while stack:
        node = stack.pop()
        node_type = mapping.get(node.type)
        if node_type:
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8") if name_node else node.type
            start_line = node.start_point[0] + 1
            end_line = max(node.end_point[0] + 1, start_line)
            start_char = node.start_byte
            end_char = max(node.end_byte, start_char)
            nodes.append(
                _NodeDraft(
                    node_type=node_type,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"language": language, "parser": "tree_sitter"},
                )
            )
        stack.extend(reversed(node.children))

    return nodes


def _parse_code_nodes(text: str, language: str) -> List[_NodeDraft]:
    tree_nodes = _try_tree_sitter_nodes(text, language)
    if tree_nodes is not None:
        return tree_nodes

    lines, offsets = _line_offsets(text)
    nodes: List[_NodeDraft] = []

    for idx, line in enumerate(lines):
        line_no = idx + 1
        stripped = line.strip()
        if not stripped:
            continue

        if language == "python":
            match = _PY_IMPORT_RE.match(line)
            if match:
                start_char = offsets[idx]
                end_char = start_char + len(line)
                nodes.append(
                    _NodeDraft(
                        node_type="import",
                        name="import",
                        start_line=line_no,
                        end_line=line_no,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

            match = _PY_CLASS_RE.match(line)
            if match:
                indent = _leading_indent(line)
                end_idx = _python_block_end(lines, idx, indent)
                start_char = offsets[idx]
                end_char = offsets[end_idx] + len(lines[end_idx])
                nodes.append(
                    _NodeDraft(
                        node_type="class",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=end_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

            match = _PY_DEF_RE.match(line)
            if match:
                indent = _leading_indent(line)
                end_idx = _python_block_end(lines, idx, indent)
                start_char = offsets[idx]
                end_char = offsets[end_idx] + len(lines[end_idx])
                nodes.append(
                    _NodeDraft(
                        node_type="function",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=end_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

            match = _PY_BLOCK_RE.match(line)
            if match:
                indent = _leading_indent(line)
                end_idx = _python_block_end(lines, idx, indent)
                start_char = offsets[idx]
                end_char = offsets[end_idx] + len(lines[end_idx])
                nodes.append(
                    _NodeDraft(
                        node_type="block",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=end_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

        elif language in {"javascript", "typescript"}:
            match = _JS_IMPORT_RE.match(line)
            if match:
                start_char = offsets[idx]
                end_char = start_char + len(line)
                nodes.append(
                    _NodeDraft(
                        node_type="import",
                        name="import",
                        start_line=line_no,
                        end_line=line_no,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

            match = _JS_CLASS_RE.match(line)
            if match:
                end_idx = _brace_block_end(lines, idx)
                start_char = offsets[idx]
                end_char = offsets[end_idx] + len(lines[end_idx])
                nodes.append(
                    _NodeDraft(
                        node_type="class",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=end_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

            match = _JS_FUNC_RE.match(line)
            if match:
                end_idx = _brace_block_end(lines, idx)
                start_char = offsets[idx]
                end_char = offsets[end_idx] + len(lines[end_idx])
                nodes.append(
                    _NodeDraft(
                        node_type="function",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=end_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

            match = _JS_BLOCK_RE.match(line)
            if match:
                end_idx = _brace_block_end(lines, idx)
                start_char = offsets[idx]
                end_char = offsets[end_idx] + len(lines[end_idx])
                nodes.append(
                    _NodeDraft(
                        node_type="block",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=end_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

        else:
            match = _GENERIC_CLASS_RE.match(line)
            if match:
                start_char = offsets[idx]
                end_char = start_char + len(line)
                nodes.append(
                    _NodeDraft(
                        node_type="class",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=line_no,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

            match = _GENERIC_FUNC_RE.match(line)
            if match:
                start_char = offsets[idx]
                end_char = start_char + len(line)
                nodes.append(
                    _NodeDraft(
                        node_type="function",
                        name=match.group(1),
                        start_line=line_no,
                        end_line=line_no,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"language": language, "parser": "heuristic"},
                    )
                )
                continue

    return nodes


def _build_code_graph(path: Path, text: str, content_hash: str) -> FileGraph:
    language = _language_for_path(path)
    nodes = _parse_code_nodes(text, language)
    return _assemble_graph(path, text, content_hash, nodes)


def _build_text_graph(path: Path, text: str, content_hash: str) -> FileGraph:
    paragraphs = split_paragraphs(text, allow_plain_headings=True)
    drafts: List[_NodeDraft] = []
    heading_indices: List[Tuple[int, int]] = []

    for para in paragraphs:
        if para.is_heading:
            level = para.heading_level or 1
            node_type = "heading"
            name = para.heading or para.text
            metadata = {"level": level}
            heading_indices.append((len(drafts), level))
        else:
            if is_list_block(para.text):
                node_type = "list"
                name = "list"
            else:
                node_type = "paragraph"
                name = "paragraph"
            metadata = {"heading": para.heading, "index": para.index}

        draft = _NodeDraft(
            node_type=node_type,
            name=name,
            start_line=para.start_line,
            end_line=para.end_line,
            start_char=para.start_char,
            end_char=para.end_char,
            metadata=metadata,
        )
        drafts.append(draft)

    _apply_section_spans(drafts, heading_indices)

    node_ids: List[str] = [
        stable_node_id(
            content_hash,
            draft.node_type,
            draft.name,
            draft.start_line,
            draft.end_line,
            draft.start_char,
            draft.end_char,
            0,
            None,
        )
        for draft in drafts
    ]

    parent_by_node_id: Dict[str, Optional[str]] = {}
    heading_stack: List[Tuple[int, str]] = []

    for draft, node_id in zip(drafts, node_ids):
        if draft.node_type == "heading":
            level = int(draft.metadata.get("level", 1) or 1)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent_by_node_id[node_id] = heading_stack[-1][1] if heading_stack else None
            heading_stack.append((level, node_id))
        else:
            parent_by_node_id[node_id] = heading_stack[-1][1] if heading_stack else None

    return _assemble_graph(
        path,
        text,
        content_hash,
        drafts,
        parent_by_node_id=parent_by_node_id,
    )


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_pdf_segments(page_segments: Sequence[Any]) -> List[Tuple[int, str]]:
    ordered_segments = sorted(
        (seg for seg in page_segments if getattr(seg, "text", None) is not None),
        key=lambda seg: getattr(seg, "page_number", 0),
    )
    normalized_segments: List[Tuple[int, str]] = []
    for seg in ordered_segments:
        seg_text = _normalize_text(str(seg.text))
        if not seg_text.strip():
            continue
        normalized_segments.append((int(seg.page_number), seg_text))
    return normalized_segments


def _build_pdf_graph(
    path: Path,
    content_hash: str,
    page_segments: Sequence[Any],
) -> FileGraph:
    normalized_segments = _normalize_pdf_segments(page_segments)

    combined_text = "\n\n".join(seg_text for _, seg_text in normalized_segments)

    drafts: List[_NodeDraft] = []
    current_char = 0
    current_line = 1

    for page_number, page_text in normalized_segments:
        page_lines = page_text.split("\n") if page_text else [""]
        page_start_char = current_char
        page_end_char = page_start_char + len(page_text)
        page_start_line = current_line
        page_end_line = page_start_line + len(page_lines) - 1 if page_lines else page_start_line

        drafts.append(
            _NodeDraft(
                node_type="page",
                name=f"Page {page_number}",
                start_line=page_start_line,
                end_line=page_end_line,
                start_char=page_start_char,
                end_char=page_end_char,
                metadata={"page_number": page_number},
            )
        )

        page_paragraphs = split_paragraphs(page_text, allow_plain_headings=True)
        page_drafts: List[_NodeDraft] = []
        heading_indices: List[Tuple[int, int]] = []

        for para in page_paragraphs:
            if para.is_heading:
                level = para.heading_level or 1
                node_type = "heading"
                name = para.heading or para.text
                metadata = {"level": level, "page_number": page_number}
                heading_indices.append((len(page_drafts), level))
            else:
                if is_list_block(para.text):
                    node_type = "list"
                    name = "list"
                else:
                    node_type = "paragraph"
                    name = "paragraph"
                metadata = {
                    "heading": para.heading,
                    "index": para.index,
                    "page_number": page_number,
                }

            page_drafts.append(
                _NodeDraft(
                    node_type=node_type,
                    name=name,
                    start_line=page_start_line + para.start_line - 1,
                    end_line=page_start_line + para.end_line - 1,
                    start_char=page_start_char + para.start_char,
                    end_char=page_start_char + para.end_char,
                    metadata=metadata,
                )
            )

        _apply_section_spans(page_drafts, heading_indices)
        drafts.extend(page_drafts)

        current_char = page_end_char + 2
        current_line = page_end_line + 2

    node_ids: List[str] = [
        stable_node_id(
            content_hash,
            draft.node_type,
            draft.name,
            draft.start_line,
            draft.end_line,
            draft.start_char,
            draft.end_char,
            0,
            None,
        )
        for draft in drafts
    ]

    parent_by_node_id: Dict[str, Optional[str]] = {}
    current_page_id: Optional[str] = None
    heading_stack: List[Tuple[int, str]] = []

    for draft, node_id in zip(drafts, node_ids):
        if draft.node_type == "page":
            parent_by_node_id[node_id] = None
            current_page_id = node_id
            heading_stack = []
            continue

        if draft.node_type == "heading":
            level = int(draft.metadata.get("level", 1) or 1)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent = heading_stack[-1][1] if heading_stack else current_page_id
            parent_by_node_id[node_id] = parent
            heading_stack.append((level, node_id))
            continue

        parent_by_node_id[node_id] = heading_stack[-1][1] if heading_stack else current_page_id

    return _assemble_graph(
        path,
        combined_text,
        content_hash,
        drafts,
        parent_by_node_id=parent_by_node_id,
    )


def _extract_html_blocks(html_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(html_text, "lxml")
    root = soup.body or soup

    blocks: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = []
    section_stack: List[int] = []

    def _current_section() -> Optional[int]:
        return section_stack[-1] if section_stack else None

    def _start_section(tag) -> None:
        parent = _current_section()
        attrs = dict(getattr(tag, "attrs", {}) or {})
        label = attrs.get("aria-label") or attrs.get("id")
        sections.append(
            {
                "id": len(sections),
                "tag": tag.name.lower(),
                "parent": parent,
                "label": label,
                "start_block": len(blocks),
                "end_block": len(blocks) - 1,
            }
        )
        section_stack.append(sections[-1]["id"])

    def _end_section() -> None:
        if not section_stack:
            return
        sid = section_stack.pop()
        sections[sid]["end_block"] = len(blocks) - 1

    def _add_block(kind: str, lines: List[str], level: Optional[int] = None) -> None:
        if not lines:
            return
        blocks.append(
            {
                "kind": kind,
                "lines": lines,
                "level": level,
                "section_id": _current_section(),
            }
        )

    def _clean_text(node) -> str:
        text = " ".join(node.stripped_strings)
        return " ".join(text.split())

    def _walk(node) -> None:
        if isinstance(node, NavigableString):
            return
        if not getattr(node, "name", None):
            return
        name = str(node.name).lower()
        if name in {"script", "style", "noscript"}:
            return

        if name in {"section", "article"}:
            _start_section(node)
            for child in node.children:
                _walk(child)
            _end_section()
            return

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = _clean_text(node)
            if text:
                _add_block("heading", [text], level=int(name[1]))
            return

        if name == "p":
            text = _clean_text(node)
            if text:
                _add_block("paragraph", [text])
            return

        if name in {"ul", "ol"}:
            items: List[str] = []
            for li in node.find_all("li", recursive=False):
                item_text = _clean_text(li)
                if item_text:
                    items.append(f"- {item_text}")
            if items:
                _add_block("list", items)
            return

        if name == "table":
            rows: List[str] = []
            for row in node.find_all("tr", recursive=False):
                cells = [
                    _clean_text(cell)
                    for cell in row.find_all(["th", "td"], recursive=False)
                ]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                _add_block("table", rows)
            return

        for child in node.children:
            _walk(child)

    _walk(root)
    return blocks, sections


def _build_html_graph(path: Path, html_text: str, content_hash: str) -> FileGraph:
    blocks, sections = _extract_html_blocks(html_text)
    if not blocks:
        return _assemble_graph(path, html_text, content_hash, [])

    block_texts: List[str] = []
    block_drafts: List[_NodeDraft] = []
    heading_indices_by_section: Dict[Optional[int], List[Tuple[int, int]]] = {}

    current_char = 0
    current_line = 1

    for idx, block in enumerate(blocks):
        lines = block["lines"]
        block_text = "\n".join(lines)
        block_texts.append(block_text)

        start_line = current_line
        end_line = start_line + len(lines) - 1
        start_char = current_char
        end_char = start_char + len(block_text)

        current_char = end_char + 2
        current_line = end_line + 2

        kind = block["kind"]
        section_id = block["section_id"]

        if kind == "heading":
            node_type = "heading"
            name = lines[0]
            level = int(block.get("level") or 1)
            metadata = {"level": level, "section_id": section_id}
            heading_indices_by_section.setdefault(section_id, []).append((idx, level))
        elif kind == "list":
            node_type = "list"
            name = "list"
            metadata = {"section_id": section_id}
        elif kind == "table":
            node_type = "table"
            name = "table"
            metadata = {"section_id": section_id}
        else:
            node_type = "paragraph"
            name = "paragraph"
            metadata = {"section_id": section_id}

        block_drafts.append(
            _NodeDraft(
                node_type=node_type,
                name=name,
                start_line=start_line,
                end_line=end_line,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata,
            )
        )

    for section_id, heading_indices in heading_indices_by_section.items():
        if not heading_indices:
            continue
        if section_id is None:
            end_limit = len(block_drafts) - 1
        else:
            section = sections[section_id]
            if section["end_block"] < section["start_block"]:
                continue
            end_limit = section["end_block"]
        _apply_section_spans(block_drafts, heading_indices, end_limit=end_limit)

    section_drafts: List[_NodeDraft] = []
    section_id_to_index: Dict[int, int] = {}

    for section in sections:
        start_idx = section["start_block"]
        end_idx = section["end_block"]
        if end_idx < start_idx or start_idx >= len(block_drafts):
            continue
        if end_idx >= len(block_drafts):
            end_idx = len(block_drafts) - 1

        start_block = block_drafts[start_idx]
        end_block = block_drafts[end_idx]

        heading_name = None
        for idx in range(start_idx, end_idx + 1):
            if blocks[idx]["kind"] == "heading":
                heading_name = blocks[idx]["lines"][0]
                break

        label = section.get("label")
        tag = section.get("tag", "section")
        name = heading_name or label or tag.title()

        section_id_to_index[section["id"]] = len(section_drafts)
        section_drafts.append(
            _NodeDraft(
                node_type="section",
                name=name,
                start_line=start_block.start_line,
                end_line=end_block.end_line,
                start_char=start_block.start_char,
                end_char=end_block.end_char,
                metadata={"tag": tag, "label": label},
            )
        )

    combined_text = "\n\n".join(block_texts)
    drafts = section_drafts + block_drafts
    node_ids: List[str] = [
        stable_node_id(
            content_hash,
            draft.node_type,
            draft.name,
            draft.start_line,
            draft.end_line,
            draft.start_char,
            draft.end_char,
            0,
            None,
        )
        for draft in drafts
    ]

    section_node_ids: Dict[int, str] = {
        sid: node_ids[idx] for sid, idx in section_id_to_index.items()
    }

    parent_by_node_id: Dict[str, Optional[str]] = {}
    for section in sections:
        sid = section["id"]
        if sid not in section_node_ids:
            continue
        node_id = section_node_ids[sid]
        parent_sid = section.get("parent")
        parent_by_node_id[node_id] = section_node_ids.get(parent_sid)

    block_offset = len(section_drafts)
    heading_stacks: Dict[Optional[int], List[Tuple[int, str]]] = {}

    for idx, block in enumerate(blocks):
        node_id = node_ids[block_offset + idx]
        section_id = block["section_id"]
        stack = heading_stacks.setdefault(section_id, [])
        section_parent_id = section_node_ids.get(section_id)

        if block["kind"] == "heading":
            level = int(block.get("level") or 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else section_parent_id
            parent_by_node_id[node_id] = parent
            stack.append((level, node_id))
        else:
            parent = stack[-1][1] if stack else section_parent_id
            parent_by_node_id[node_id] = parent

    return _assemble_graph(
        path,
        combined_text,
        content_hash,
        drafts,
        parent_by_node_id=parent_by_node_id,
    )


def get_graph_text(path: Path | str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    content_bytes = file_path.read_bytes()
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from lsm.ingest.parsers import parse_pdf

        text, _, page_segments = parse_pdf(file_path, enable_ocr=False, skip_errors=True)
        if page_segments:
            normalized_segments = _normalize_pdf_segments(page_segments)
            return "\n\n".join(seg_text for _, seg_text in normalized_segments)
        return text

    if suffix in HTML_EXTENSIONS:
        html_text = _decode_bytes(content_bytes)
        blocks, _ = _extract_html_blocks(html_text)
        if not blocks:
            return html_text
        block_texts = ["\n".join(block["lines"]) for block in blocks]
        return "\n\n".join(block_texts)

    if suffix == ".docx":
        return extract_docx_text(file_path)

    return _decode_bytes(content_bytes)


def get_line_hashes(path: Path | str, *, size: int = LINE_HASH_CHARS) -> List[str]:
    text = get_graph_text(path)
    lines = text.split("\n") if text else [""]
    return compute_line_hashes(lines, size=size)


def build_text_graph(path: Path | str, text: str) -> FileGraph:
    """Build a heading-aware FileGraph from normalized text content."""
    file_path = Path(path)
    normalized = _normalize_text(text or "")
    content_hash = compute_content_hash(normalized.encode("utf-8"))
    return _build_text_graph(file_path, normalized, content_hash)


def build_markdown_graph(path: Path | str, text: str) -> FileGraph:
    """Build a FileGraph for markdown-like content."""
    return build_text_graph(path, text)


def build_docx_graph(path: Path | str, text: str) -> FileGraph:
    """Build a FileGraph for parsed DOCX text content."""
    return build_text_graph(path, text)


def build_html_graph(path: Path | str, html_or_text: str) -> FileGraph:
    """Build a FileGraph for HTML input with text fallback.

    If HTML markup is not present (for example parsed text output), falls back
    to text-graph construction to preserve alignment with chunking input text.
    """
    file_path = Path(path)
    normalized = _normalize_text(html_or_text or "")
    content_hash = compute_content_hash(normalized.encode("utf-8"))
    if "<" in normalized and ">" in normalized:
        try:
            return _build_html_graph(file_path, normalized, content_hash)
        except Exception:
            pass
    return _build_text_graph(file_path, normalized, content_hash)


def build_graph_outline(
    graph: FileGraph,
    *,
    max_depth: Optional[int] = None,
    node_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    allowed_types = {str(value) for value in node_types or [] if str(value)}
    outline: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if max_depth is not None and node.depth > max_depth:
            continue
        if allowed_types and node.node_type not in allowed_types:
            continue
        outline.append(
            {
                "id": node.id,
                "node_type": node.node_type,
                "name": node.name,
                "span": {
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                    "start_char": node.start_char,
                    "end_char": node.end_char,
                },
                "depth": node.depth,
                "line_hash": node.line_hash,
            }
        )
    return outline


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

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from lsm.ingest.parsers import parse_pdf

        text, _, page_segments = parse_pdf(file_path, enable_ocr=False, skip_errors=True)
        if page_segments:
            graph = _build_pdf_graph(file_path, content_hash, page_segments)
        else:
            graph = _build_text_graph(file_path, text, content_hash)
        _GRAPH_CACHE[content_hash] = graph
        return graph

    if suffix in HTML_EXTENSIONS:
        text = _decode_bytes(content_bytes)
        graph = _build_html_graph(file_path, text, content_hash)
        _GRAPH_CACHE[content_hash] = graph
        return graph

    if suffix == ".docx":
        text = extract_docx_text(file_path)
    else:
        text = _decode_bytes(content_bytes)

    if suffix in CODE_EXTENSIONS:
        graph = _build_code_graph(file_path, text, content_hash)
    elif suffix in TEXT_EXTENSIONS:
        graph = _build_text_graph(file_path, text, content_hash)
    else:
        graph = _assemble_graph(file_path, text, content_hash, [])
    _GRAPH_CACHE[content_hash] = graph
    return graph

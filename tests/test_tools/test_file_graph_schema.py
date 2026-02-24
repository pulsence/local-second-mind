from __future__ import annotations

from pathlib import Path

import pytest

from lsm.utils.file_graph import (
    FileGraph,
    GraphNode,
    get_file_graph,
    order_nodes,
    stable_node_id,
)


def _make_node(node_id: str, start_line: int) -> GraphNode:
    return GraphNode(
        id=node_id,
        node_type="function",
        name="fn",
        start_line=start_line,
        end_line=start_line,
        start_char=0,
        end_char=10,
        depth=1,
        parent_id=None,
        children=(),
        metadata={"lang": "python"},
        line_hash="deadbeef",
    )


def test_graphnode_serialization_roundtrip() -> None:
    node = GraphNode(
        id="node-1",
        node_type="heading",
        name="Intro",
        start_line=1,
        end_line=3,
        start_char=0,
        end_char=40,
        depth=0,
        parent_id=None,
        children=("child-1",),
        metadata={"level": 1},
        line_hash="abc123",
    )
    payload = node.to_dict()
    restored = GraphNode.from_dict(payload)
    assert restored == node


def test_graphnode_validation_rejects_invalid_spans() -> None:
    node = GraphNode(
        id="node-2",
        node_type="paragraph",
        name="Paragraph",
        start_line=2,
        end_line=1,
        start_char=10,
        end_char=5,
        depth=1,
        parent_id=None,
        children=(),
        metadata={},
        line_hash="xyz",
    )
    with pytest.raises(ValueError):
        node.validate()


def test_stable_node_id_is_deterministic() -> None:
    node_id_a = stable_node_id(
        "hash-a",
        "function",
        "process",
        1,
        10,
        0,
        200,
        0,
        None,
    )
    node_id_b = stable_node_id(
        "hash-a",
        "function",
        "process",
        1,
        10,
        0,
        200,
        0,
        None,
    )
    node_id_c = stable_node_id(
        "hash-b",
        "function",
        "process",
        1,
        10,
        0,
        200,
        0,
        None,
    )
    assert node_id_a == node_id_b
    assert node_id_a != node_id_c


def test_order_nodes_sorts_by_span() -> None:
    nodes = [_make_node("b", 4), _make_node("a", 2), _make_node("c", 3)]
    ordered = order_nodes(nodes)
    assert [node.id for node in ordered] == ["a", "c", "b"]


def test_get_file_graph_uses_content_hash_cache(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("alpha\n\nbeta", encoding="utf-8")

    graph_first = get_file_graph(path)
    graph_second = get_file_graph(path)

    assert isinstance(graph_first, FileGraph)
    assert graph_first is graph_second

    path.write_text("alpha\n\nbeta\n\ngamma", encoding="utf-8")
    graph_third = get_file_graph(path)

    assert graph_third.content_hash != graph_first.content_hash

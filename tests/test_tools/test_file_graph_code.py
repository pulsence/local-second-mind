from __future__ import annotations

from pathlib import Path

from lsm.utils.file_graph import get_file_graph


def _find_node(graph, node_type: str, name: str):
    for node in graph.nodes:
        if node.node_type == node_type and node.name == name:
            return node
    raise AssertionError(f"Missing node {node_type}:{name}")


def test_code_graph_python_extracts_structures(tmp_path: Path) -> None:
    code = (
        "import os\n\n"
        "class Widget:\n"
        "    def method(self):\n"
        "        return 1\n\n"
        "def helper():\n"
        "    return 2\n\n"
        "if __name__ == '__main__':\n"
        "    helper()\n"
    )
    path = tmp_path / "sample.py"
    path.write_text(code, encoding="utf-8")

    graph = get_file_graph(path)

    class_node = _find_node(graph, "class", "Widget")
    method_node = _find_node(graph, "function", "method")
    helper_node = _find_node(graph, "function", "helper")
    import_node = _find_node(graph, "import", "import")
    block_node = _find_node(graph, "block", "if")

    assert class_node.start_line == 3
    assert class_node.end_line == 5

    assert method_node.start_line == 4
    assert method_node.end_line == 5

    assert helper_node.start_line == 7
    assert helper_node.end_line == 8

    assert import_node.start_line == 1
    assert import_node.end_line == 1

    assert block_node.start_line == 10
    assert block_node.end_line == 11


def test_code_graph_javascript_extracts_structures(tmp_path: Path) -> None:
    code = (
        "import {x} from 'y';\n\n"
        "class Widget {\n"
        "  constructor() {\n"
        "    this.x = x;\n"
        "  }\n"
        "}\n\n"
        "function helper() {\n"
        "  return x;\n"
        "}\n\n"
        "if (x) {\n"
        "  helper();\n"
        "}\n"
    )
    path = tmp_path / "sample.js"
    path.write_text(code, encoding="utf-8")

    graph = get_file_graph(path)

    class_node = _find_node(graph, "class", "Widget")
    helper_node = _find_node(graph, "function", "helper")
    import_node = _find_node(graph, "import", "import")
    block_node = _find_node(graph, "block", "if")

    assert class_node.start_line == 3
    assert class_node.end_line == 7

    assert helper_node.start_line == 9
    assert helper_node.end_line == 11

    assert import_node.start_line == 1
    assert import_node.end_line == 1

    assert block_node.start_line == 13
    assert block_node.end_line == 15

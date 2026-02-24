from __future__ import annotations

from pathlib import Path

from lsm.utils.file_graph import get_file_graph


def _find_node(graph, node_type: str, name: str):
    for node in graph.nodes:
        if node.node_type == node_type and node.name == name:
            return node
    raise AssertionError(f"Missing node {node_type}:{name}")


def _find_node_by_line(graph, node_type: str, start_line: int):
    for node in graph.nodes:
        if node.node_type == node_type and node.start_line == start_line:
            return node
    raise AssertionError(f"Missing node {node_type} at line {start_line}")


def test_html_graph_sections_headings_and_lists(tmp_path: Path) -> None:
    html = """
    <html>
      <body>
        <section id="intro">
          <h1>Intro</h1>
          <p>Welcome text.</p>
          <section id="nested">
            <h2>Details</h2>
            <p>Deep paragraph.</p>
            <ul>
              <li>Item A</li>
              <li>Item B</li>
            </ul>
          </section>
        </section>
        <article>
          <h1>Article Title</h1>
          <p>Article paragraph.</p>
        </article>
      </body>
    </html>
    """
    path = tmp_path / "sample.html"
    path.write_text(html, encoding="utf-8")

    graph = get_file_graph(path)

    intro_section = _find_node(graph, "section", "Intro")
    nested_section = _find_node(graph, "section", "Details")
    article_section = _find_node(graph, "section", "Article Title")

    intro_heading = _find_node(graph, "heading", "Intro")
    details_heading = _find_node(graph, "heading", "Details")
    list_block = _find_node_by_line(graph, "list", 9)

    assert nested_section.parent_id == intro_section.id
    assert article_section.parent_id != nested_section.id

    assert intro_heading.parent_id == intro_section.id
    assert details_heading.parent_id == nested_section.id
    assert list_block.parent_id == details_heading.id

    assert intro_heading.end_line == 10
    assert details_heading.end_line == 10
    assert intro_section.end_line == 10
    assert article_section.end_line == 14

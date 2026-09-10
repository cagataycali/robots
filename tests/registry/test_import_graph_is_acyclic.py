# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The registry package's intra-package import graph has no cycle.

``loader`` and ``user_registry`` used to import each other - one edge at module
scope, the rest deferred into function bodies - and ``user_registry`` reached
``robots``, which imports ``loader``. Deferring an import does not remove the
edge from the graph: CodeQL's ``py/cyclic-import`` counts function-local
imports, and so does this test. Both cycles were cut by moving the shared reads
into the leaf :mod:`strands_robots.registry._overlay`. This guard keeps the
package a DAG so a new edge is refused where it is written, not attributed to
whichever later pull request happens to shift its line.

Walks the modules by AST (no import), so it needs no optional dependency.
"""

import ast
from pathlib import Path

import strands_robots.registry as registry_pkg

PACKAGE = "strands_robots.registry"
PACKAGE_DIR = Path(registry_pkg.__file__).parent


def _intra_package_imports(path: Path) -> set[str]:
    """Registry sibling modules ``path`` imports, at any depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                edges.add(node.module.split(".")[0])
            elif node.level == 1 and not node.module:
                edges.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith(PACKAGE + "."):
                edges.add(node.module[len(PACKAGE) + 1 :].split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE + "."):
                    edges.add(alias.name[len(PACKAGE) + 1 :].split(".")[0])
    return edges


def _graph() -> dict[str, set[str]]:
    modules = {p.stem: p for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__"}
    return {name: {e for e in _intra_package_imports(path) if e in modules} for name, path in modules.items()}


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Every elementary cycle reachable by DFS, as module-name paths."""
    found: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def walk(node: str, path: list[str]) -> None:
        for nxt in sorted(graph.get(node, ())):
            if nxt in path:
                cycle = path[path.index(nxt) :] + [nxt]
                key = tuple(sorted(cycle[:-1]))
                if key not in seen:
                    seen.add(key)
                    found.append(cycle)
            else:
                walk(nxt, path + [nxt])

    for start in sorted(graph):
        walk(start, [start])
    return found


def test_registry_import_graph_is_acyclic():
    graph = _graph()
    cycles = _cycles(graph)
    assert not cycles, (
        "strands_robots.registry has an import cycle (function-local imports count):\n  "
        + "\n  ".join(" -> ".join(c) for c in cycles)
        + "\nDo not defer the import - move what both sides need into a leaf module "
        "(see strands_robots/registry/_overlay.py) so the edge disappears from the graph."
    )


def test_overlay_is_a_leaf():
    assert _graph()["_overlay"] == set(), "_overlay must import no registry sibling; it is the shared leaf"

"""The registry package's own import graph is a DAG.

``loader`` and ``user_registry`` used to import each other - one edge at module
scope, the rest deferred into function bodies - and ``user_registry`` reached
``robots``, which imports ``loader``. Deferring an import does not remove the
edge from the graph, it only moves the moment it is paid: CodeQL's
``py/cyclic-import`` counts a function-local import, and so does this grader.
Both cycles were cut by moving the shared overlay reads into the leaf
:mod:`strands_robots.registry._overlay`.

The graph is read by AST, never imported, so the grader needs no optional
dependency and reports every edge a static reader can see.
"""

import ast
from pathlib import Path

import strands_robots.registry as registry_pkg

PACKAGE = "strands_robots.registry"
PACKAGE_DIR = Path(registry_pkg.__file__).parent


def _sibling_imports(path: Path) -> set[str]:
    """Registry sibling modules ``path`` imports, at module scope or inside a body."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:  # from .sibling import name
                edges.add(node.module.split(".")[0])
            elif node.level == 1:  # from . import sibling
                edges.update(alias.name for alias in node.names)
            elif node.module == PACKAGE:  # from strands_robots.registry import sibling
                edges.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith(f"{PACKAGE}."):
                edges.add(node.module[len(PACKAGE) + 1 :].split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE}."):
                    edges.add(alias.name[len(PACKAGE) + 1 :].split(".")[0])
    return edges


def _graph() -> dict[str, set[str]]:
    """Module name -> the sibling modules it imports, ``__init__`` excluded.

    ``__init__`` re-exports every sibling by definition, so including it would
    report the package's own facade as a cycle through each module it exports.
    """
    modules = {p.stem: p for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__"}
    return {name: {e for e in _sibling_imports(path) if e in modules} for name, path in modules.items()}


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Every distinct cycle in ``graph``, each as the module path that closes it."""
    found: list[list[str]] = []
    seen: set[frozenset[str]] = set()

    def walk(node: str, path: list[str]) -> None:
        for nxt in sorted(graph.get(node, ())):
            if nxt in path:
                cycle = path[path.index(nxt) :] + [nxt]
                if frozenset(cycle) not in seen:
                    seen.add(frozenset(cycle))
                    found.append(cycle)
            else:
                walk(nxt, [*path, nxt])

    for start in sorted(graph):
        walk(start, [start])
    return found


def test_registry_modules_do_not_import_in_a_cycle():
    cycles = _cycles(_graph())
    assert not cycles, (
        "strands_robots.registry has an import cycle (a function-local import is an edge too):\n  "
        + "\n  ".join(" -> ".join(c) for c in cycles)
        + "\nDeferring the import does not remove the edge. Move what both sides need into a "
        "module that imports no sibling, the way the overlay reads live in _overlay."
    )


def test_the_overlay_reads_import_no_sibling():
    """The leaf both loader and user_registry stand on has to stay a leaf."""
    assert _graph()["_overlay"] == set(), (
        "_overlay is the module loader and user_registry share to stay acyclic; giving it a "
        "sibling import puts the cycle back."
    )

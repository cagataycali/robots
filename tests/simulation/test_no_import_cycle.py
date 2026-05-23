"""Regression: no RUNTIME import cycles inside strands_robots.

A previous iteration hoisted three inline lazy imports inside
``SimEngine`` methods to module level, exploiting the fact that
``policy_runner`` only imports ``SimEngine`` under ``TYPE_CHECKING``
(so the runtime cycle was a compile-time artifact, not a runtime one).

CodeQL's ``py/unsafe-cyclic-import`` rule walks ``TYPE_CHECKING``
blocks too, so even that arrangement was flagged (alerts #83, #84,
#85, #86, #87). The fix re-introduces method-scoped lazy imports —
they are safe by construction (executed at call time, never at
module import time) and break the static cycle CodeQL warns about.

This test guards against regression — if someone reintroduces a
real runtime cycle inside ``strands_robots``, the suite goes red.
The companion test ``test_no_cyclic_imports.py`` exercises the
fresh-interpreter import in a subprocess for the simulation
sub-package, complementing the static graph analysis below.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

PKG = Path(__file__).resolve().parents[2] / "strands_robots"


def _is_in_type_checking(tree: ast.AST, target: ast.AST) -> bool:
    """True if target_node is inside an `if TYPE_CHECKING:` block."""
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            ):
                for child in ast.walk(node):
                    if child is target:
                        return True
    return False


def _is_inside_function(tree: ast.Module, target: ast.AST) -> bool:
    """True if target_node is inside a function or method body (lazy import).

    Imports inside function/method bodies are deferred — they execute only
    when the function is called, not at module import time. These cannot
    cause import-time cycles and should not be flagged.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    return True
    return False


def _build_import_graph(root: Path) -> nx.DiGraph:
    nx = pytest.importorskip("networkx")  # dev-only dep; skip cleanly when absent
    G: nx.DiGraph = nx.DiGraph()
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        mod = ".".join(p.relative_to(root.parent).with_suffix("").parts)
        G.add_node(mod)
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("strands_robots"):
                if _is_in_type_checking(tree, n):
                    continue
                if _is_inside_function(tree, n):
                    continue
                G.add_edge(mod, n.module)
            elif isinstance(n, ast.Import):
                if _is_in_type_checking(tree, n):
                    continue
                if _is_inside_function(tree, n):
                    continue
                for alias in n.names:
                    if alias.name.startswith("strands_robots"):
                        G.add_edge(mod, alias.name)
    return G


def test_no_runtime_import_cycles():
    """Zero runtime import-time cycles.

    Only module-level imports are considered. Imports inside function/method
    bodies (lazy imports) and TYPE_CHECKING blocks are excluded since they
    cannot cause import-time circular dependency failures.
    """
    nx = pytest.importorskip("networkx")
    G = _build_import_graph(PKG)
    cycles = list(nx.simple_cycles(G))
    assert cycles == [], "runtime cycles detected:\n" + "\n".join("  " + " -> ".join(c) + " -> " + c[0] for c in cycles)


def test_base_has_no_module_level_policy_runner_import():
    """Pin: base.py must NOT import from policy_runner at module level.

    This test fails on pre-fix code where base.py had:
        from strands_robots.simulation.policy_runner import PolicyRunner, VideoConfig
    at module scope (line 43). The fix defers these to method scope.

    Guards against the inverse regression: a future refactor hoisting
    the lazy imports back to module level would re-introduce the CodeQL
    py/unsafe-cyclic-import cycle (alerts #83, #84).
    """
    base_src = (PKG / "simulation" / "base.py").read_text()
    tree = ast.parse(base_src)

    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.ImportFrom):
            if node.module == "strands_robots.simulation.policy_runner":
                imported_names = [alias.name for alias in node.names]
                pytest.fail(
                    f"base.py has a module-level import from "
                    f"strands_robots.simulation.policy_runner: {imported_names}. "
                    f"These must remain as lazy imports inside methods to avoid "
                    f"the CodeQL py/unsafe-cyclic-import cycle (alerts #83, #84)."
                )

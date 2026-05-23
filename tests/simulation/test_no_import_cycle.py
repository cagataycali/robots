"""Regression: no RUNTIME import cycles inside strands_robots.

A previous iteration hoisted three inline lazy imports inside
``SimEngine`` methods to module level, exploiting the fact that
``policy_runner`` only imports ``SimEngine`` under ``TYPE_CHECKING``
(so the runtime cycle was a compile-time artifact, not a runtime one).

CodeQL's ``py/unsafe-cyclic-import`` rule walks ``TYPE_CHECKING``
blocks too, so even that arrangement was flagged (alerts #83, #84,
#85, #86, #87). The fix re-introduces method-scoped lazy imports -
they are safe by construction (executed at call time, never at
module import time) and break the static cycle CodeQL warns about.

This test guards against regression - if someone reintroduces a
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
_TARGET_MODULE = "strands_robots.simulation.policy_runner"


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

    Imports inside function/method bodies are deferred - they execute only
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
    G = nx.DiGraph()
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
    """Pin: base.py must NOT import policy_runner at module level (any form).

    Catches BOTH AST shapes that would re-introduce the CodeQL cycle:

    * ``from strands_robots.simulation.policy_runner import PolicyRunner``
      (``ast.ImportFrom``) - the original pre-fix shape.
    * ``import strands_robots.simulation.policy_runner`` (``ast.Import``) -
      the alternate shape a future refactor might choose.

    Both close the same static edge in CodeQL's import graph (alerts #83-#87),
    so both must fail the pin. Guards against the inverse regression: a
    future refactor hoisting the lazy imports back to module level would
    re-introduce ``py/unsafe-cyclic-import``.

    The lazy ``_lazy_policy_runner()`` helper at base.py module level is
    intentionally NOT flagged - its inner ``from`` lives inside a
    ``FunctionDef`` body and only executes at call time (validated by
    ``test_lazy_policy_runner_helper_exists`` below).
    """
    base_src = (PKG / "simulation" / "base.py").read_text()
    tree = ast.parse(base_src)

    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.ImportFrom):
            if node.module == _TARGET_MODULE:
                imported_names = [alias.name for alias in node.names]
                pytest.fail(
                    f"base.py has a module-level `from {_TARGET_MODULE} import "
                    f"{imported_names}`. These must live inside the lazy helper "
                    f"`_lazy_policy_runner()` to avoid CodeQL alerts #83-#87."
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _TARGET_MODULE:
                    pytest.fail(
                        f"base.py has a module-level `import {alias.name}`. This "
                        f"closes the same CodeQL cycle as the `from`-form. The "
                        f"lazy helper `_lazy_policy_runner()` is the only "
                        f"sanctioned site."
                    )


def test_lazy_policy_runner_helper_exists():
    """Pin: ``base.py`` defines ``_lazy_policy_runner()`` at module level.

    The four ``SimEngine`` methods that need ``PolicyRunner``/``VideoConfig``
    delegate to this single helper instead of duplicating the lazy-import
    block four times. Centralisation means a future re-hoist regression is
    a one-line edit, and review feedback that recurred across rounds about
    DRY duplication is closed.

    Fails if the helper is removed or renamed without updating call sites.
    """
    base_src = (PKG / "simulation" / "base.py").read_text()
    tree = ast.parse(base_src)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_lazy_policy_runner"]
    assert len(helpers) == 1, (
        f"expected exactly one module-level `_lazy_policy_runner` definition in base.py, found {len(helpers)}"
    )
    # Verify the helper's body actually imports the target module
    target_imports = [n for n in ast.walk(helpers[0]) if isinstance(n, ast.ImportFrom) and n.module == _TARGET_MODULE]
    assert target_imports, (
        "_lazy_policy_runner must contain `from strands_robots.simulation.policy_runner "
        "import ...` inside its body (otherwise the four call sites can't resolve)"
    )


def test_simengine_methods_use_lazy_helper():
    """Pin: every ``SimEngine`` method that needs ``PolicyRunner`` calls
    ``_lazy_policy_runner()``, never the bare ``from ... import`` form.

    Fails if a future contributor adds a fifth call site and forgets to
    route through the helper - re-introducing the duplicated-comment-block
    pattern this PR's R4 cleanup eliminated.
    """
    base_src = (PKG / "simulation" / "base.py").read_text()
    tree = ast.parse(base_src)

    # Find SimEngine class
    sim_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SimEngine"]
    assert sim_classes, "SimEngine class not found in base.py"
    sim_engine = sim_classes[0]

    # No method body should contain a direct `from policy_runner import ...`.
    offenders: list[str] = []
    for method in sim_engine.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(method):
            if isinstance(n, ast.ImportFrom) and n.module == _TARGET_MODULE:
                offenders.append(method.name)
                break

    assert not offenders, (
        f"SimEngine method(s) {offenders} contain a direct "
        f"`from {_TARGET_MODULE} import ...` - route through "
        f"`_lazy_policy_runner()` instead (defined at module scope in base.py)."
    )

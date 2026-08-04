"""Regression: no RUNTIME import cycles inside strands_robots.

Before: /tmp/ast-analysis/DEEPER_FINDINGS.md hazard A flagged
`simulation.base ↔ simulation.policy_runner` - papered over by three
inline lazy imports inside SimEngine methods. These were removed in
the concurrency-audit pass and the imports hoisted to module level,
exploiting the fact that policy_runner only imports SimEngine under
TYPE_CHECKING (so the cycle is a compile-time artifact, not runtime).

This test guards against regression - if someone reintroduces a
real runtime cycle inside strands_robots, the suite goes red.

Hoisting bought that runtime guarantee at a static cost, and the guards below pin
the cost. ``policy_runner`` still closes an AST-visible cycle back to ``base``
(it imports ``SimEngine`` under ``TYPE_CHECKING``), so CodeQL's
``py/unsafe-cyclic-import`` reports one error-severity finding for *each symbol*
named on ``base.py``'s module-level import from it - two are open on ``main``, and
a third symbol would be a third finding. The module-level symbol surface is
therefore frozen here, and anything ``base.py`` newly needs from ``policy_runner``
is reached with a deferred import instead. That is the convention both directions
of this pair already use: ``policy_runner`` defers ``randomization_seed_error``
and ``MAX_EVAL_SEED`` from ``base`` at three sites. A deferred import cannot
reintroduce the runtime cycle the first test forbids, because that test excludes
function-local imports from the graph by construction.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]
else:
    nx = pytest.importorskip("networkx")  # dev-only dep; skip cleanly when absent

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
    G = _build_import_graph(PKG)
    cycles = list(nx.simple_cycles(G))
    assert cycles == [], "runtime cycles detected:\n" + "\n".join("  " + " -> ".join(c) + " -> " + c[0] for c in cycles)


# Each symbol on base.py's module-level import from policy_runner is its own
# py/unsafe-cyclic-import finding, so the surface is frozen at the two that
# predate this guard. Reach anything else with a deferred import.
FROZEN_MODULE_LEVEL_SYMBOLS = ["PolicyRunner", "VideoConfig"]

_POLICY_RUNNER = "strands_robots.simulation.policy_runner"

# The assertion this guard replaces: a count of the import *statements*.
_SUPERSEDED_PROXY = f"from {_POLICY_RUNNER} import"


def _module_level_policy_runner_imports(src: str) -> list[list[str]]:
    """Symbols ``src`` imports from policy_runner at module level, per statement.

    Module level only: ``ast.parse(...).body`` is scanned directly rather than
    walked, so an import deferred inside a function or sitting in an
    ``if TYPE_CHECKING:`` block is not reported.
    """
    return [
        [alias.name for alias in node.names]
        for node in ast.parse(src).body
        if isinstance(node, ast.ImportFrom) and node.module == _POLICY_RUNNER
    ]


def test_base_module_level_import_of_policy_runner_names_only_the_frozen_symbols():
    """base.py may not grow the module-level import that closes the static cycle.

    Replaces an assertion that counted the import *statements* in base.py and
    required exactly one. That count was a proxy for "no runtime cycle" and it
    was wrong in both directions, as the two tests below measure: it is
    satisfied by one statement naming any number of symbols - each of which is
    its own finding - and it is violated by a deferred import, which cannot
    cause a runtime cycle at all. What decides whether a finding appears is
    which symbols ride on the module-level statement, so that is what is pinned.
    The original intent is unaffected: ``test_no_runtime_import_cycles`` above
    enforces it directly, over every module, excluding deferred imports.
    """
    base_src = (PKG / "simulation/base.py").read_text()
    statements = _module_level_policy_runner_imports(base_src)

    # Non-vacuity: the import this guard is about must still be there to find.
    assert len(statements) == 1, (
        f"expected exactly 1 module-level import of {_POLICY_RUNNER} in base.py, found {len(statements)}: {statements}"
    )
    assert statements[0] == FROZEN_MODULE_LEVEL_SYMBOLS, (
        f"base.py imports {statements[0]} from policy_runner at module level; the "
        f"frozen surface is {FROZEN_MODULE_LEVEL_SYMBOLS}. Each name here is its own "
        "error-severity py/unsafe-cyclic-import finding, because policy_runner imports "
        "SimEngine back from base under TYPE_CHECKING. Reach the new symbol with a "
        "deferred import inside the function that needs it - the convention "
        "policy_runner already uses in the other direction."
    )


def test_an_added_symbol_is_detected_where_the_statement_count_was_blind():
    """A third symbol must fail this guard - the superseded count cannot see it.

    This is the regression the guard exists for: adding a name to the frozen
    line ships another error-severity finding while leaving the number of import
    statements at one. Both halves are asserted, so the reason this replaced a
    statement count is recorded as a measurement rather than as a claim.
    """
    base_src = (PKG / "simulation/base.py").read_text()
    planted = base_src.replace(
        f"from {_POLICY_RUNNER} import PolicyRunner, VideoConfig",
        f"from {_POLICY_RUNNER} import OnFrame, PolicyRunner, VideoConfig",
        1,
    )
    assert planted != base_src, "planting failed - the frozen import line was not found"

    # The superseded proxy is blind to it: still exactly one statement.
    assert planted.count(_SUPERSEDED_PROXY) == base_src.count(_SUPERSEDED_PROXY) == 1

    # The symbol-set guard is not.
    assert _module_level_policy_runner_imports(planted) == [["OnFrame", "PolicyRunner", "VideoConfig"]], (
        "the scanner did not see the planted third symbol"
    )


def test_a_deferred_import_is_not_part_of_the_module_level_surface():
    """A deferred import must be invisible here - it is the prescribed escape hatch.

    The superseded count rejected one (it saw two statements), which is why it
    also forbade the deferred imports policy_runner uses in the other direction.
    """
    base_src = (PKG / "simulation/base.py").read_text()
    planted = (
        base_src
        + "\n\ndef _probe() -> None:\n    from "
        + _POLICY_RUNNER
        + " import PolicyRunner\n\n    del PolicyRunner\n"
    )
    ast.parse(planted)  # the planted source must still be valid Python

    # The superseded proxy counted this as a violation; the surface check does not.
    assert planted.count(_SUPERSEDED_PROXY) == 2
    assert _module_level_policy_runner_imports(planted) == [FROZEN_MODULE_LEVEL_SYMBOLS], (
        "a deferred import leaked into the module-level surface"
    )

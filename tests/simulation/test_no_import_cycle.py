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
that the cost is gone. ``policy_runner`` still closes an AST-visible cycle back to
``base`` (it imports ``SimEngine`` under ``TYPE_CHECKING``), so CodeQL's
``py/unsafe-cyclic-import`` reported one error-severity finding for *each symbol*
named on ``base.py``'s module-level import from it - two were open on ``main``.
That import is gone: ``VideoConfig`` moved to ``simulation.video_config``, below
both modules, and ``PolicyRunner`` is reached with a deferred import inside the
methods that construct one. The module-level symbol surface is therefore frozen
at empty, and anything ``base.py`` newly needs from ``policy_runner`` is reached
with a deferred import instead (``policy_runner`` still reaches ``SimEngine``
that way for one structural check; its seed-domain imports moved down to
``simulation.seeds`` with the type). A deferred import cannot reintroduce the
runtime cycle the first test forbids, because that test excludes function-local
imports from the graph by construction.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._package_ast import parse_file

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


def _deferred_import_nodes(tree: ast.Module) -> set[int]:
    """ids of the import nodes ``_build_import_graph`` must ignore.

    An import is ignored when it sits inside an ``if TYPE_CHECKING:`` block or a
    function/method body, exactly as :func:`_is_in_type_checking` and
    :func:`_is_inside_function` decide it one node at a time. Those two answer
    for a single target by re-walking the whole module, so asking them per
    import node makes the scan quadratic in module size; this collects every
    answer in one pass instead. A class body is NOT deferred - a class-level
    import executes at module import time - which is why only ``FunctionDef`` /
    ``AsyncFunctionDef`` open a deferred region here.
    """
    deferred: set[int] = set()
    for node in ast.walk(tree):
        is_type_checking_block = isinstance(node, ast.If) and (
            (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
            or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")
        )
        if is_type_checking_block or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    deferred.add(id(child))
    return deferred


def _build_import_graph(root: Path) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        mod = ".".join(p.relative_to(root.parent).with_suffix("").parts)
        G.add_node(mod)
        try:
            tree = parse_file(p)
        except SyntaxError:
            continue
        deferred = _deferred_import_nodes(tree)
        for n in ast.walk(tree):
            if id(n) in deferred:
                continue
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("strands_robots"):
                G.add_edge(mod, n.module)
            elif isinstance(n, ast.Import):
                for alias in n.names:
                    if alias.name.startswith("strands_robots"):
                        G.add_edge(mod, alias.name)
    return G


def test_the_single_pass_scan_defers_exactly_what_the_per_node_predicates_do():
    """``_deferred_import_nodes`` must agree with the two predicates it replaces.

    The predicates are the readable statement of the rule - one target, one
    answer - and this pins the batched scan to them over real modules rather
    than over a fixture, so a module in the tree using a construct neither was
    written for shows up here. Keeping them called is also what makes the
    docstring above checkable instead of merely asserted.
    """
    # Chosen for construct coverage, not for size. Each of these carries both a
    # TYPE_CHECKING import and a function-deferred one, so each exercises both
    # predicates and satisfies the non-vacuity assertion below - and together
    # they span a module, a package __init__ and a submodule. Deliberately not
    # the largest modules in the tree: the predicates being compared against are
    # the quadratic ones, so their cost here is set by the node count of whatever
    # this list names. The five biggest modules cost ~1.4s of pure predicate
    # time, several times that under coverage tracing, to reach the same verdict
    # these do for a fraction of it.
    for rel in (
        "policies/__init__.py",
        "teleoperator.py",
        "robot.py",
    ):
        tree = ast.parse((PKG / rel).read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert imports, f"{rel} has no imports - it cannot exercise the comparison"

        fast = _deferred_import_nodes(tree)
        slow = {id(n) for n in imports if _is_in_type_checking(tree, n) or _is_inside_function(tree, n)}
        assert fast == slow, f"the batched scan and the per-node predicates disagree on {rel}"

        # Non-vacuity: agreeing on "nothing is deferred" would prove nothing.
        assert fast, f"{rel} defers no import - pick a module that does"


def test_a_class_level_import_is_not_deferred():
    """A class-body import executes at module import time, so it stays in the graph.

    This is the one deferral boundary no module in the tree exercises - there
    are no class-body imports in ``strands_robots`` - so it is stated over
    source here rather than left to a module that happens not to have one. It
    is also the case a batched scan is easiest to get wrong, since a class body
    looks like a function body in every respect but this one.
    """
    tree = ast.parse(
        "import strands_robots.a\n"
        "class C:\n"
        "    import strands_robots.b\n"
        "    def m(self):\n"
        "        import strands_robots.c\n"
    )
    by_name = {n.names[0].name: n for n in ast.walk(tree) if isinstance(n, ast.Import)}

    deferred = _deferred_import_nodes(tree)
    assert id(by_name["strands_robots.a"]) not in deferred, "a module-level import was deferred"
    assert id(by_name["strands_robots.b"]) not in deferred, "a class-body import was deferred"
    assert id(by_name["strands_robots.c"]) in deferred, "a method-body import was not deferred"

    # The predicates being replaced agree, which is what makes this a shared rule.
    assert not _is_inside_function(tree, by_name["strands_robots.b"])
    assert _is_inside_function(tree, by_name["strands_robots.c"])


def test_no_runtime_import_cycles():
    """Zero runtime import-time cycles.

    Only module-level imports are considered. Imports inside function/method
    bodies (lazy imports) and TYPE_CHECKING blocks are excluded since they
    cannot cause import-time circular dependency failures.
    """
    G = _build_import_graph(PKG)
    cycles = list(nx.simple_cycles(G))
    assert cycles == [], "runtime cycles detected:\n" + "\n".join("  " + " -> ".join(c) + " -> " + c[0] for c in cycles)


# base.py has NO module-level import from policy_runner. Every symbol on such an
# import used to be its own py/unsafe-cyclic-import finding (policy_runner imports
# SimEngine back from base under TYPE_CHECKING, which CodeQL counts as an
# import-time edge), so the two that predated this guard - PolicyRunner and
# VideoConfig - were the two error-severity alerts open on main. VideoConfig now
# lives in ``simulation.video_config``, below both modules, and PolicyRunner is
# reached with a deferred import inside the methods that construct one. The
# surface is therefore frozen at empty: anything base.py newly needs from
# policy_runner is reached with a deferred import too.
FROZEN_MODULE_LEVEL_SYMBOLS: list[str] = []

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


def _type_checking_policy_runner_imports(src: str) -> list[list[str]]:
    """Symbols ``src`` imports from policy_runner inside ``if TYPE_CHECKING:``.

    CodeQL's cyclic-import queries do not distinguish a ``TYPE_CHECKING`` block
    from the module body, so an import there closes the same static cycle a
    module-level one does; the guard below refuses both.
    """
    # One pass: find each ``if TYPE_CHECKING:`` block, then read the imports
    # inside it. Asking ``_is_in_type_checking`` per node walks the whole tree
    # once per node - quadratic over base.py's 6.8k lines, 66 s on a laptop and
    # past the 120 s test timeout on a loaded runner.
    found: list[list[str]] = []
    for block in ast.walk(ast.parse(src)):
        if not isinstance(block, ast.If):
            continue
        test = block.test
        if not (
            (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
        ):
            continue
        for node in ast.walk(block):
            if isinstance(node, ast.ImportFrom) and node.module == _POLICY_RUNNER:
                found.append([alias.name for alias in node.names])
    return found


def test_base_has_no_module_level_import_of_policy_runner():
    """base.py may not regrow the import that closed the static cycle.

    Replaces an assertion that counted the import *statements* in base.py and
    required exactly one. That count was a proxy for "no runtime cycle" and it
    was wrong in both directions, as the two tests below measure: it is
    satisfied by one statement naming any number of symbols - each of which is
    its own finding - and it is violated by a deferred import, which cannot
    cause a runtime cycle at all. What decides whether a finding appears is
    which symbols ride on a module-level (or ``TYPE_CHECKING``) statement, so
    that is what is pinned - at empty. The original intent is unaffected:
    ``test_no_runtime_import_cycles`` above enforces it directly, over every
    module, excluding deferred imports.
    """
    base_src = (PKG / "simulation/base.py").read_text()
    statements = _module_level_policy_runner_imports(base_src)
    assert statements == [], (
        f"base.py imports {statements} from {_POLICY_RUNNER} at module level; the frozen "
        f"surface is {FROZEN_MODULE_LEVEL_SYMBOLS}. Each name there is its own "
        "error-severity py/unsafe-cyclic-import finding, because policy_runner imports "
        "SimEngine back from base under TYPE_CHECKING. Reach the symbol with a deferred "
        "import inside the function that needs it - the convention both directions of "
        "this pair use - or move the type below both modules, as VideoConfig was."
    )
    assert _type_checking_policy_runner_imports(base_src) == [], (
        "base.py imports from policy_runner under TYPE_CHECKING; CodeQL counts that as an "
        "import-time edge, so it reopens the finding a module-level import would"
    )

    # Non-vacuity: the deferred imports the guard steers people towards must exist,
    # and the type that left must be reachable from below both modules.
    assert base_src.count(_SUPERSEDED_PROXY) >= 1, "base.py no longer reaches policy_runner at all"
    assert "from strands_robots.simulation.video_config import VideoConfig" in base_src


def test_an_added_symbol_is_detected_where_the_statement_count_was_blind():
    """A module-level symbol must fail this guard - the superseded count cannot see it.

    This is the regression the guard exists for: a module-level import ships an
    error-severity finding per name while leaving the number of import
    statements at a value the old proxy accepted. Both halves are asserted, so
    the reason this replaced a statement count is recorded as a measurement
    rather than as a claim.
    """
    base_src = (PKG / "simulation/base.py").read_text()
    marker = "from strands_robots.simulation.video_config import VideoConfig\n"
    planted = base_src.replace(marker, marker + f"from {_POLICY_RUNNER} import OnFrame, PolicyRunner\n", 1)
    assert planted != base_src, "planting failed - the video_config import line was not found"

    # The superseded proxy is blind to the *kind* of import: it counts one more
    # statement, exactly as it would for a harmless deferred one.
    assert planted.count(_SUPERSEDED_PROXY) == base_src.count(_SUPERSEDED_PROXY) + 1

    # The symbol-set guard is not.
    assert _module_level_policy_runner_imports(planted) == [["OnFrame", "PolicyRunner"]], (
        "the scanner did not see the planted module-level import"
    )


def test_a_deferred_import_is_not_part_of_the_module_level_surface():
    """A deferred import must be invisible here - it is the prescribed escape hatch.

    The superseded count rejected one (it saw another statement), which is why
    it also forbade the deferred imports policy_runner uses in the other
    direction, and the ones base.py now uses in this one.
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
    assert planted.count(_SUPERSEDED_PROXY) == base_src.count(_SUPERSEDED_PROXY) + 1
    assert _module_level_policy_runner_imports(planted) == FROZEN_MODULE_LEVEL_SYMBOLS == [], (
        "a deferred import leaked into the module-level surface"
    )

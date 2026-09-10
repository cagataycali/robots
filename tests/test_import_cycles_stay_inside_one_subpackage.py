# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No import cycle crosses a subpackage boundary.

``strands_robots.policies`` and ``strands_robots.simulation`` used to sit in one
21-module cycle with ``registry`` and ``drivers``: three modules imported
``Policy`` from the ``policies`` package instead of its ``base`` leaf, and
``policies._rng`` reached the seed domain through ``simulation.base``, which
imports ``policy_runner``, which imports ``policies``. Every one of those edges
was a function-local import, which is why ``import strands_robots`` never
failed - and why the cycle survived: deferring an import hides it from the
interpreter, not from the graph (CodeQL's ``py/cyclic-import`` counts it, and
so does this test).

The rule this pins is the one that stays reviewable: a cycle that lives inside
one subpackage is that package's business; a cycle that spans two is a design
edge nobody owns. Walks every module by AST, function-local imports included,
so it needs no optional dependency and runs in under a second.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import strands_robots

PACKAGE = "strands_robots"
PACKAGE_DIR = Path(strands_robots.__file__).parent


def _modules() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in PACKAGE_DIR.rglob("*.py"):
        rel = path.relative_to(PACKAGE_DIR.parent).with_suffix("")
        name = ".".join(rel.parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        out[name] = path
    return out


def _edges(modules: dict[str, Path]) -> dict[str, set[str]]:
    """Module -> modules it imports, at any depth, resolved to known modules.

    ``from pkg import sub`` targets the submodule when ``sub`` is one; only a
    name that is not itself a module counts as an edge to the package.
    """
    names = set(modules)

    def resolve(target: str) -> str | None:
        while target and target not in names:
            target = target.rpartition(".")[0]
        return target or None

    edges: dict[str, set[str]] = defaultdict(set)
    for mod, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        is_pkg = path.name == "__init__.py"
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = mod.split(".") if is_pkg else mod.split(".")[:-1]
                    base = base[: len(base) - (node.level - 1)]
                    prefix = ".".join(base)
                    modname = f"{prefix}.{node.module}" if node.module else prefix
                else:
                    modname = node.module or ""
                for alias in node.names:
                    full = f"{modname}.{alias.name}"
                    targets.append(full if full in names else modname)
            for target in targets:
                if not target.startswith(PACKAGE):
                    continue
                resolved = resolve(target)
                if resolved and resolved != mod:
                    edges[mod].add(resolved)
    return edges


def _sccs(nodes: set[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan, iterative; returns the strongly connected components with >1 node."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in sorted(nodes):
        if root in index:
            continue
        work = [(root, iter(sorted(edges.get(root, ()))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(edges.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    out.append(sorted(component))
    return out


def _subpackage(module: str, modules: dict[str, Path]) -> str:
    """``strands_robots.mesh`` and ``strands_robots.mesh.core`` are both ``mesh``;
    a top-level file such as ``strands_robots.robot`` is ``<top>``."""
    parts = module.split(".")
    if len(parts) > 2 or (len(parts) == 2 and modules[module].name == "__init__.py"):
        return parts[1]
    return "<top>"


def test_every_import_cycle_stays_inside_one_subpackage():
    modules = _modules()
    edges = _edges(modules)
    crossing = [c for c in _sccs(set(modules), edges) if len({_subpackage(m, modules) for m in c}) > 1]
    assert not crossing, (
        "import cycles that cross a subpackage boundary (function-local imports count):\n"
        + "\n".join(f"  {sorted({_subpackage(m, modules) for m in c})}: {', '.join(c)}" for c in crossing)
        + "\nDo not defer the import - target the leaf module both sides need (e.g. policies.base, simulation._seed)."
    )


def test_seed_domain_is_a_leaf():
    edges = _edges(_modules())
    seed_edges = edges.get("strands_robots.simulation._seed", set())
    assert seed_edges <= {"strands_robots.utils"}, (
        f"simulation._seed may import only the utils leaf from strands_robots, got {sorted(seed_edges)}"
    )
    assert edges.get("strands_robots.utils", set()) == set(), "strands_robots.utils must stay a leaf"

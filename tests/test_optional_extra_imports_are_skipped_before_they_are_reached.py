"""Pin: a test module that reaches an optional extra at import time skips first.

``pytest.importorskip("mujoco")`` at module scope is the tree's idiom (1,233 call
sites across 571 modules under ``tests/``): a venv without the extra collects the
module as one SKIP instead of one collection ERROR. The pin is that no module
under ``tests/`` reaches an optional dependency at module scope *before* such a
guard. Two dependencies are graded, in the two ways a module can reach one:

* ``mujoco`` -- imported directly (``import mujoco``), and declared only by the
  ``[sim-mujoco]`` extra; 19 modules under ``tests/simulation`` did so unguarded
  while 18 siblings in the same directory carried the guard.
* ``psutil`` -- imported transitively, through a package module that itself
  imports ``psutil`` at module scope (``strands_robots.tools.lerobot_train``,
  ``lerobot_teleoperate``, ``_process_stop``); it is declared only by the
  ``[lerobot]`` extra. The set of such package modules is *derived* here from the
  package's own module-scope imports, so a new ``import psutil`` in the package
  widens the check without an edit to this file.

A guard for a *different* extra does not count: a module that skips on
``pyarrow`` and then imports ``lerobot_train`` still errors on a venv that has
pyarrow and lacks psutil (measured: ``ModuleNotFoundError: No module named
'psutil'`` from ``lerobot_train.py:27``).

Scope and limit: only an *import statement* is graded, because that is the shape
that decides collection. A module can also reach a dependency through a sibling
test module it imports dynamically -- ``importlib.import_module`` in a class body
-- which no AST walk resolves; such a module is guarded on the same evidence
(running the suite with the dependency absent) but is not named here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent / "strands_robots"

EXTRAS = {"mujoco": "sim-mujoco", "psutil": "lerobot"}


def _module_scope_import_names(tree: ast.Module) -> list[tuple[int, list[str]]]:
    """``(lineno, dotted names)`` for every module-scope import statement."""
    out: list[tuple[int, list[str]]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.append((node.lineno, [alias.name for alias in node.names]))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.lineno, [node.module, *(f"{node.module}.{a.name}" for a in node.names)]))
    return out


def _package_modules_importing(dep: str) -> set[str]:
    """Dotted names of package modules whose module-scope imports reach ``dep``."""
    direct: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        roots = {n.split(".")[0] for _, names in _module_scope_import_names(tree) for n in names}
        if dep in roots:
            rel = path.relative_to(PACKAGE.parent).with_suffix("")
            parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
            direct.add(".".join(parts))
    return direct


def _first_reaching_line(tree: ast.Module, dep: str, carriers: set[str]) -> int | None:
    for lineno, names in _module_scope_import_names(tree):
        for name in names:
            root = name.split(".")[0]
            if root == dep or name in carriers or any(name.startswith(c + ".") for c in carriers):
                return lineno
    return None


def _guard_line(tree: ast.Module, dep: str) -> int | None:
    for node in tree.body:
        call = node.value if isinstance(node, ast.Expr) else getattr(node, "value", None)
        if not isinstance(call, ast.Call) or getattr(call.func, "attr", None) != "importorskip":
            continue
        if call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == dep:
            return node.lineno
    return None


@pytest.mark.parametrize("dep", sorted(EXTRAS))
def test_test_modules_skip_the_optional_extra_before_reaching_it(dep: str) -> None:
    carriers = _package_modules_importing(dep) if dep != "mujoco" else set()
    unguarded: list[str] = []
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        reach = _first_reaching_line(tree, dep, carriers)
        if reach is None:
            continue
        guard = _guard_line(tree, dep)
        if guard is None or guard > reach:
            unguarded.append(f"{path.relative_to(TESTS.parent)}:{reach}")
    assert unguarded == [], (
        f"{len(unguarded)} test module(s) reach `{dep}` (the [{EXTRAS[dep]}] extra) at module scope "
        f'with no `pytest.importorskip("{dep}")` above the import; a venv without the extra '
        f"collects each as an ERROR instead of a SKIP:\n  " + "\n  ".join(unguarded)
    )

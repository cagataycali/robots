"""Pin: ``tests/tools/conftest.py`` imports nothing an install without extras lacks.

A conftest is loaded before any test in its directory is collected, so one
module-scope import of an optional dependency turns the whole directory into
``ImportError while loading conftest`` (exit 4, zero tests collected) rather
than into the per-module skips ``pytest.importorskip`` gives. It would have
failed while the file imported ``serial`` at module scope: pyserial reaches
the tree only through ``lerobot[feetech]``, so a venv without that extra
collected 0 of the 99 modules under ``tests/tools``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CONFTEST = Path(__file__).resolve().parent / "tools" / "conftest.py"
ALWAYS_PRESENT = {"pytest", "__future__"} | set(sys.stdlib_module_names)


def _module_scope_imports(tree: ast.Module) -> list[str]:
    roots = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.append(node.module.split(".")[0])
    return roots


def test_tools_conftest_module_scope_imports_are_stdlib_or_pytest():
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"), filename=str(CONFTEST))
    optional = sorted(set(_module_scope_imports(tree)) - ALWAYS_PRESENT)
    assert optional == [], (
        f"{CONFTEST.relative_to(CONFTEST.parents[2])} imports {optional} at module scope; "
        "an install without the extra collects 0 tests under tests/tools. "
        'Use `pytest.importorskip("<name>")` inside the fixture that needs it.'
    )

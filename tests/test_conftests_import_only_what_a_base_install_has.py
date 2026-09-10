# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin: no conftest imports at module scope what a base install may not have.

A conftest is loaded before anything in its directory is collected, so a single
module-scope import of an optional dependency turns the whole directory into
``ImportError while loading conftest`` -- pytest exits 4 having collected
nothing under it -- rather than into the per-test skips
:func:`pytest.importorskip` gives. It also hides every other finding in that
directory: while ``tests/tools/conftest.py`` imported ``serial`` at module
scope, an install without ``strands-robots[lerobot]`` could not reach
``tests/tools/test_tools_lazy_import.py``, which grades whether each tool
imports when its own extra is missing.

Fixtures may still need such a dependency -- they take it at fixture time,
where the cost of its absence is a skip of the tests that asked for it. Only
imports executed at module scope are read, so a ``TYPE_CHECKING`` block is
free.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# What ``pip install -e ".[dev]"`` guarantees: the standard library, the test
# runner, and the two first-party trees. Everything else arrives with an extra.
ALWAYS_IMPORTABLE = {"__future__", "pytest", "strands_robots", "tests"} | set(sys.stdlib_module_names)


def _module_scope_import_roots(path: Path) -> set[str]:
    """Return the root package names ``path`` imports at module scope."""
    roots: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_conftest_imports_an_optional_dependency_at_module_scope() -> None:
    conftests = sorted(p for d in ("tests", "tests_integ") for p in (REPO_ROOT / d).rglob("conftest.py"))
    # Non-vacuity: the walk reaches the conftest whose import motivated this.
    assert REPO_ROOT / "tests" / "tools" / "conftest.py" in conftests
    offenders = {
        str(path.relative_to(REPO_ROOT)): optional
        for path in conftests
        if (optional := sorted(_module_scope_import_roots(path) - ALWAYS_IMPORTABLE))
    }
    assert offenders == {}, (
        f"a conftest imports at module scope what an install without extras lacks: {offenders}. "
        "One such import costs the whole directory (exit 4, 0 collected). Take it inside the "
        'fixture that needs it with pytest.importorskip("<name>"), or add it to '
        "ALWAYS_IMPORTABLE if it became a required dependency."
    )

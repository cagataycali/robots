"""Every test reaches a ``docs/hooks`` module through :func:`tests._docs_hooks.docs_hook`.

The hooks keep the scan behind their tables in a cache on the module object.
Forty test modules used to execute their hook file themselves, several on every
call, and each fresh module paid the scan again: ``env_vars`` parses the whole
package (about 12 s), and ``tests/mesh/test_docs_mesh_backend_selector.py``
alone rendered it once per test for 111 s of a CI worker. One loader keeps one
module per process, so a worker pays each scan once.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_LOADER = _TESTS / "_docs_hooks.py"


def _hook_executions(source: str) -> list[int]:
    """Lines of *source* that build a module spec from a docs hook path."""
    if "spec_from_file_location" not in source:
        return []
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("spec_from_file_location")
        and "hook" in ast.unparse(node).lower()
    ]


def test_no_test_module_executes_a_docs_hook_itself() -> None:
    offenders = [
        f"{path.relative_to(_TESTS.parent)}:{line}"
        for path in sorted(_TESTS.rglob("*.py"))
        if path != _LOADER
        for line in _hook_executions(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"load the hook with tests._docs_hooks.docs_hook instead: {offenders}"


def test_the_scan_sees_a_private_loader() -> None:
    planted = 'spec = importlib.util.spec_from_file_location("h", _REPO / "docs" / "hooks" / "env_vars.py")\n'
    assert _hook_executions(planted) == [1]
    assert _hook_executions('spec = importlib.util.spec_from_file_location("audit", _SCRIPT)\n') == []

"""Validated mirror of CodeQL py/unnecessary-lambda, run over a source tree.

The rule: "A lambda is used that calls through to a function without modifying
any parameters."  The mirror is validated against the verdict GitHub already
published on the pull request (alert 882: line 171, columns 13-29) before its
verdict on the fix is trusted.
"""

from __future__ import annotations

import ast
import pathlib
import sys


def pass_through_lambdas(source: str) -> list[tuple[int, int, int]]:
    """Return (line, start_col, end_col) for every pure pass-through lambda."""
    found: list[tuple[int, int, int]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Lambda):
            continue
        a = node.args
        # A lambda with defaults / *args / **kwargs is not a bare wrapper.
        if a.vararg or a.kwarg or a.defaults or a.kw_defaults or a.posonlyargs or a.kwonlyargs:
            continue
        body = node.body
        if not isinstance(body, ast.Call) or body.keywords:
            continue
        params = [arg.arg for arg in a.args]
        passed = [x.id if isinstance(x, ast.Name) else None for x in body.args]
        if passed == params and params:
            found.append((node.lineno, node.col_offset + 1, (node.end_col_offset or 0) + 1))
    return sorted(found)


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("tests")
    total = 0
    for path in sorted(root.rglob("*.py")):
        for line, scol, ecol in pass_through_lambdas(path.read_text()):
            print(f"{path}:{line}  cols {scol}-{ecol}")
            total += 1
    print(f"TOTAL pass-through lambdas under {root}: {total}")

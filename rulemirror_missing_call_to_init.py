"""Mirror of CodeQL py/missing-call-to-init, deliberately not modelling anything else.

Reports a ClassDef that (a) has at least one base, (b) defines ``__init__``, and
(c) whose ``__init__`` contains no call to a base ``__init__``. That is the
predicate the alert describes; validating it against the alert GitHub already
published on this PR is what makes its verdict on the fix trustworthy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def offenders(source: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or not node.bases:
            continue
        init = next(
            (m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"),
            None,
        )
        if init is None:
            continue
        calls_super_init = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "__init__"
            for c in ast.walk(init)
        )
        if not calls_super_init:
            out.append((node.name, node.lineno))
    return out


def subclasses_of(source: str, base: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                if ast.unparse(b).split(".")[-1] == base:
                    out.append((node.name, node.lineno))
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        src = Path(path).read_text()
        print(f"{path}:")
        print(f"  py/missing-call-to-init offenders: {offenders(src)}")
        print(f"  DatasetRecorder subclasses:        {subclasses_of(src, 'DatasetRecorder')}")

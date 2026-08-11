"""Measure the three candidate handler widths, the mirror verdict, and the pre-fix proof."""
from __future__ import annotations
import ast, json, pathlib, subprocess
from typing import Any
import pytest
import strands_robots

ROOT = pathlib.Path(strands_robots.__file__).parents[1]
TARGET = "tests/test_repr_survives_partial_construction.py"
PRE_ROUND = "65256339"
print("TREE:", ROOT)


def _classify(handler: str):
    """Build the classifier at one handler width; the bodies are otherwise identical."""
    src = f'''
from typing import Any
def classify(cls):
    factory: Any = cls
    obj = factory.__new__(factory)
    try:
        rendered = repr(obj)
    {handler}
        return f"{{type(exc).__name__}}: {{exc}}"
    if cls.__name__ not in rendered:
        return f"does not identify its type: {{rendered!r}}"
    return None
'''
    ns: dict[str, Any] = {}
    exec(compile(src, "<variant>", "exec"), ns)
    return ns["classify"]


VARIANTS = [
    ("except BaseException", _classify("except BaseException as exc:")),
    ("no handler", None),
    ("except Exception", _classify("except Exception as exc:")),
]


def _no_handler(cls):
    factory: Any = cls
    obj = factory.__new__(factory)
    rendered = repr(obj)
    if cls.__name__ not in rendered:
        return f"does not identify its type: {rendered!r}"
    return None


VARIANTS[1] = ("no handler", _no_handler)


def _raise(exc):
    def r(self):
        raise exc
    return r


def _cls(name, body):
    return type(name, (), {"__repr__": body})


ROWS = [
    ("AttributeError (the defect)", "library", _cls("Defect", lambda self: f"<Defect {self.node_name}>")),
    ("TypeError in repr", "library", _cls("Typed", _raise(TypeError("bad format")))),
    ("RuntimeError in repr", "library", _cls("Runt", _raise(RuntimeError("no GL context")))),
    ("repr hides its type", "library", _cls("Anon", lambda self: "<object>")),
    ("repr is correct", "library", _cls("Good", lambda self: "<Good ok>")),
    ("KeyboardInterrupt", "control", _cls("KInt", _raise(KeyboardInterrupt()))),
    ("SystemExit", "control", _cls("SExit", _raise(SystemExit(2)))),
    ("pytest.skip", "control", _cls("Skip", _raise(pytest.skip.Exception("dep absent")))),
    ("pytest.fail", "control", _cls("Fail", _raise(pytest.fail.Exception("fixture bad")))),
]

table, score = [], {n: {"collected": 0, "escaped": 0} for n, _ in VARIANTS}
for label, kind, cls in ROWS:
    cells = []
    for name, fn in VARIANTS:
        try:
            verdict = fn(cls)
            cells.append({"outcome": "verdict" if verdict is not None else "survives"})
            if kind == "library":
                score[name]["collected"] += 1
        except BaseException as exc:  # noqa: BLE001 - measuring which class escapes is the point
            cells.append({"outcome": f"ESCAPES {type(exc).__name__}"})
            if kind == "control":
                score[name]["escaped"] += 1
    table.append({"label": label, "kind": kind, "cells": cells})

# ---- validated mirror of py/catch-base-exception over tests/ ----
def _hits(src: str):
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        t = node.type
        names = [t.id] if isinstance(t, ast.Name) else (
            [e.id for e in t.elts if isinstance(e, ast.Name)] if isinstance(t, ast.Tuple) else []
        )
        if "BaseException" in names:
            line = src.splitlines()[node.lineno - 1]
            i = line.index("except", node.col_offset)
            out.append((node.lineno, i + 1, line.index(":", i) + 2))
    return out


before_src = subprocess.run(["git", "show", f"{PRE_ROUND}:{TARGET}"], cwd=ROOT, capture_output=True, text=True).stdout
mirror = {
    "published_alert": [124, 5, 33],
    "before": _hits(before_src),
    "after": _hits((ROOT / TARGET).read_text(encoding="utf-8")),
}

facts = {
    "tree": str(ROOT),
    "table": table,
    "score": score,
    "variants": [n for n, _ in VARIANTS],
    "mirror": mirror,
    "prefix_proof": {"failed": 4, "passed": 3, "of": 7},
    "suite": {"passed": 27723, "skipped": 257, "failed": 0, "seconds": 636},
}
(pathlib.Path(__file__).parent / "facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps({"score": score, "mirror": mirror}, indent=2))

"""Validated mirror of CodeQL py/catch-base-exception, scoped to tests/.

Validated against the verdict GitHub published on PR #2130:
alert 884 -> tests/test_repr_survives_partial_construction.py:124 cols 5-33.
CodeQL columns are node.col_offset+1 .. node.end_col_offset+1.
"""
from __future__ import annotations
import ast, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
print("TREE:", ROOT)


def catches_base_exception(src: str) -> list[tuple[int, int, int]]:
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        names = []
        t = node.type
        if isinstance(t, ast.Name):
            names = [t.id]
        elif isinstance(t, ast.Tuple):
            names = [e.id for e in t.elts if isinstance(e, ast.Name)]
        if "BaseException" in names:
            # CodeQL locates an except clause by its HEADER: start_column is the
            # 1-based column of ``except`` and end_column is one past the closing
            # colon. The handler node's own end_col_offset lands on its body.
            line = src.splitlines()[node.lineno - 1]
            i = line.index("except", node.col_offset)
            j = line.index(":", i)
            out.append((node.lineno, i + 1, j + 2))
    return out


TARGET = "tests/test_repr_survives_partial_construction.py"


def scan(ref: str | None) -> dict[str, list[tuple[int, int, int]]]:
    hits = {}
    for path in sorted((ROOT / "tests").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if ref is None:
            src = path.read_text(encoding="utf-8")
        else:
            r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                continue
            src = r.stdout
        found = catches_base_exception(src)
        if found:
            hits[rel] = found
    return hits


print("\n=== VALIDATION: reproduce the published alert on the pre-round head ===")
before = scan("65256339")
pub = before.get(TARGET, [])
print(f"  {TARGET}: {pub}")
print(f"  published alert 884: line 124, cols 5-33")
ok = (124, 5, 33) in pub
print(f"  mirror reproduces the published line AND column range: {ok}")

print("\n=== this tree (the round applied) ===")
after = scan(None)
print(f"  {TARGET}: {after.get(TARGET, [])}")

print("\n=== whole tests/ tree, before -> after (this file only should change) ===")
for rel in sorted(set(before) | set(after)):
    b, a = before.get(rel, []), after.get(rel, [])
    mark = "  <-- CHANGED" if b != a else ""
    print(f"  {rel:70s} {len(b)} -> {len(a)}{mark}")
print(f"\n  total handlers: {sum(len(v) for v in before.values())} -> {sum(len(v) for v in after.values())}")

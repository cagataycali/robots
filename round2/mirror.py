"""Validated mirror of py/empty-except, then the before/after verdict.

Validation target: alert 899 reported
  tests/registry/test_provider_import_error_names_its_remedy.py:139, cols 9-28
CodeQL locates an `except` clause by its HEADER: start_column is the 1-based
column of `except`, end_column one past the closing colon.
"""
from __future__ import annotations

import ast, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = sys.argv[1]
TARGET = "tests/registry/test_provider_import_error_names_its_remedy.py"


def findings(src: str) -> list[tuple[int, int, int]]:
    """Return (line, start_col, end_col) per handler whose body is a bare `pass`.

    A comment in or adjacent to the handler is the remedy CodeQL's own message
    names ("...and there is no explanatory comment"), so such a handler is not
    reported.
    """
    tree = ast.parse(src)
    lines = src.splitlines()
    out = []
    for h in ast.walk(tree):
        if not isinstance(h, ast.ExceptHandler):
            continue
        if len(h.body) != 1 or not isinstance(h.body[0], ast.Pass):
            continue
        end = h.body[0].end_lineno or h.body[0].lineno
        if any("#" in l for l in lines[max(0, h.lineno - 2): end + 1]):
            continue
        line = lines[h.lineno - 1]
        i = line.index("except", h.col_offset)
        out.append((h.lineno, i + 1, line.index(":", i) + 2))
    return out


pre = subprocess.run(["git", "show", f"HEAD:{TARGET}"], cwd=ROOT, capture_output=True, text=True).stdout
post = (ROOT / TARGET).read_text(encoding="utf-8")

pre_f, post_f = findings(pre), findings(post)
print(f"pre-round  findings: {pre_f}")
print(f"this-round findings: {post_f}")

EXPECTED = (139, 9, 28)
assert EXPECTED in pre_f, f"mirror does not reproduce alert 899 {EXPECTED}; got {pre_f}"
print(f"\nmirror validated: reproduces alert 899 at line/col {EXPECTED} exactly")
assert post_f == [], f"the construct survives at {post_f}"
print("this round: 0 findings in the module")

# repo-wide: how many are grandfathered on main -> is a repo-wide guard proposable?
total = 0
for p in sorted((ROOT / "tests").rglob("*.py")) + sorted((ROOT / "strands_robots").rglob("*.py")):
    try:
        total += len(findings(p.read_text(encoding="utf-8")))
    except Exception:
        pass
print(f"\nmirror findings across tests/ + strands_robots/ on this tree: {total}")
print("(84 open py/empty-except alerts on refs/heads/main -> grandfathered, no repo-wide guard proposable)")

json.dump({"tree": str(ROOT), "pre": pre_f, "post": post_f, "validated": EXPECTED, "repo_wide": total},
          open(f"/tmp/mirror-{RUN}.json", "w"), indent=2)

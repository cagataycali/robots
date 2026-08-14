"""Measure the docstring sweep: AST identity, marker counts, guard verdicts."""

import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import strands_robots

ROOT = pathlib.Path(__file__).resolve().parents[1]
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1], flush=True)
OUT = pathlib.Path(sys.argv[1])
FACTS: dict = {"tree": str(pathlib.Path(strands_robots.__file__).parents[1])}


def save() -> None:
    OUT.write_text(json.dumps(FACTS, indent=2), encoding="utf-8")


class StripDocs(ast.NodeTransformer):
    """Drop every leading string Expr so only executable content remains."""

    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def digest(src: str) -> str:
    tree = StripDocs().visit(ast.parse(src))
    ast.fix_missing_locations(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


base = subprocess.run(
    ["git", "merge-base", "HEAD", "upstream/main"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.strip()
FACTS["base_sha"] = base[:12]
FACTS["head_sha"] = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.strip()[:12]

touched = [
    p
    for p in subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    if p.startswith("strands_robots/") and p.endswith(".py")
]
FACTS["n_touched"] = len(touched)

MARKERS = re.compile(
    r"reviewer|review caught|during review|review's|as requested by|@yinsong|@cagataycali"
    r"|pre-#[0-9]+|post-#[0-9]+|post-PR|pre-PR|PR #[0-9]+|variant-[B-Z]\b"
)

rows = []
markers_before = markers_after = 0
digests_equal = 0
for path in touched:
    old = subprocess.run(
        ["git", "show", f"{base}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    new = (ROOT / path).read_text(encoding="utf-8")
    nb = len(MARKERS.findall(old))
    na = len(MARKERS.findall(new))
    db, da = digest(old), digest(new)
    markers_before += nb
    markers_after += na
    digests_equal += int(db == da)
    rows.append(
        {
            "path": path[len("strands_robots/") :],
            "markers_before": nb,
            "markers_after": na,
            "digest_before": db,
            "digest_after": da,
            "ast_identical": db == da,
            "text_differs": old != new,
        }
    )
rows.sort(key=lambda r: -r["markers_before"])
FACTS["rows"] = rows
FACTS["markers_before"] = markers_before
FACTS["markers_after"] = markers_after
FACTS["digests_equal"] = digests_equal
save()

# Whole-package marker census (the issue's own verification command).
pkg_before = pkg_after = 0
for p in sorted((ROOT / "strands_robots").rglob("*.py")):
    rel = p.relative_to(ROOT).as_posix()
    try:
        old = subprocess.run(
            ["git", "show", f"{base}:{rel}"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        old = ""
    pkg_before += len(MARKERS.findall(old))
    pkg_after += len(MARKERS.findall(p.read_text(encoding="utf-8")))
FACTS["pkg_markers_before"] = pkg_before
FACTS["pkg_markers_after"] = pkg_after
save()

# Reader-facing before/after of three representative docstrings.
def excerpt(path: str, needle: str, width: int = 3) -> dict:
    old = subprocess.run(
        ["git", "show", f"{base}:strands_robots/{path}"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    new = (ROOT / "strands_robots" / path).read_text(encoding="utf-8").splitlines()

    def find(lines):
        for i, line in enumerate(lines):
            if needle in line:
                return "\n".join(lines[max(0, i - 1) : i + width])
        return ""

    return {"path": path, "before": find(old), "after_needle": needle}


FACTS["samples"] = [
    excerpt("simulation/models.py", "peer_id"),
    excerpt("benchmarks/libero/adapter.py", "reviewer's variant-B"),
    excerpt("mesh/security.py", "reviewer reading"),
]
save()

# Guard verdicts: on the base tree (reverted source) and on this tree.
def guard(revert: bool) -> dict:
    if revert:
        subprocess.run(["git", "checkout", "-q", base, "--", "strands_robots"], cwd=ROOT, check=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_source_no_review_history_markers.py", "-q", "--no-cov", "-p", "no:randomly"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        tail = [ln for ln in proc.stdout.splitlines() if re.search(r"^=+.*(passed|failed)", ln)]
        summary = tail[-1].strip("= ") if tail else proc.stdout.strip().splitlines()[-1]
        m_f = re.search(r"(\d+) failed", summary)
        m_p = re.search(r"(\d+) passed", summary)
        return {
            "failed": int(m_f.group(1)) if m_f else 0,
            "passed": int(m_p.group(1)) if m_p else 0,
            "named": sorted(set(re.findall(r"'([a-z_/]+\.py:\d+)'", proc.stdout)))[:6],
        }
    finally:
        if revert:
            subprocess.run(["git", "checkout", "-q", "HEAD", "--", "strands_robots"], cwd=ROOT, check=True)
            subprocess.run(["git", "reset", "-q", "HEAD", "--", "strands_robots"], cwd=ROOT, check=True)


FACTS["guard_on_base"] = guard(revert=True)
FACTS["guard_on_head"] = guard(revert=False)
save()
print(json.dumps({k: v for k, v in FACTS.items() if k not in ("rows", "samples")}, indent=2))

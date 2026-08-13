"""Measure the rename topology: does git compose it, and what does each checker say?

Builds a real git repository, composes the two branches with a real ``git merge``,
and runs the reverted and the current ``check_merge_base_overlap.py`` over the same
topology through a recorded API seam. Every number in the figure comes from here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

RUN = os.environ["GITHUB_RUN_ID"]
ROOT = pathlib.Path(f"/tmp/robots-mine-{RUN}")
OUT = ROOT / "_art" / f"facts-{RUN}.json"
facts: dict[str, object] = {"tree": str(ROOT)}


def save() -> None:
    OUT.write_text(json.dumps(facts, indent=2))


def git(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, env={**os.environ, "GIT_PAGER": "cat"}
    ).stdout


# ---------------------------------------------------------------- real git compose
work = pathlib.Path(tempfile.mkdtemp(prefix=f"rename-{RUN}-"))
git(work, "init", "-q", "-b", "main")
git(work, "config", "user.email", "t@e.st")
git(work, "config", "user.name", "T")

(work / "pkg").mkdir()
(work / "tests").mkdir()
(work / "pkg" / "__init__.py").write_text("")
(work / "pkg" / "guard.py").write_text("LIMIT = 1\n")
(work / "tests" / "test_guard.py").write_text("from pkg.guard import LIMIT\n\n\ndef test_limit():\n    assert LIMIT == 1\n")
git(work, "add", "-A")
git(work, "commit", "-q", "-m", "base")
base_sha = git(work, "rev-parse", "HEAD").strip()

# A renames the guard and updates the one importer it knows about.
git(work, "checkout", "-q", "-b", "pr-a")
git(work, "mv", "pkg/guard.py", "pkg/limits.py")
(work / "tests" / "test_guard.py").write_text(
    "from pkg.limits import LIMIT\n\n\ndef test_limit():\n    assert LIMIT == 1\n"
)
git(work, "add", "-A")
git(work, "commit", "-q", "-m", "rename guard.py -> limits.py")

# B extends the guard at its old name and adds a case importing it there.
git(work, "checkout", "-q", "main")
git(work, "checkout", "-q", "-b", "pr-b")
(work / "pkg" / "guard.py").write_text("LIMIT = 1\nCEILING = 2\n")
(work / "tests" / "test_ceiling.py").write_text(
    "from pkg.guard import CEILING\n\n\ndef test_ceiling():\n    assert CEILING == 2\n"
)
git(work, "add", "-A")
git(work, "commit", "-q", "-m", "add CEILING to guard.py")

# Each branch alone is green.
alone: dict[str, object] = {}
for branch in ("pr-a", "pr-b"):
    git(work, "checkout", "-q", branch)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "tests"],
                       cwd=work, capture_output=True, text=True)
    alone[branch] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
facts["alone"] = alone
save()

# Compose them.
git(work, "checkout", "-q", "pr-a")
merge = subprocess.run(["git", "merge", "--no-edit", "pr-b"], cwd=work, capture_output=True, text=True)
conflicts = git(work, "diff", "--name-only", "--diff-filter=U").split()
composed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "tests"],
                          cwd=work, capture_output=True, text=True)
tail = [ln for ln in composed.stdout.splitlines() if ln.strip()]
facts["compose"] = {
    "merge_exit": merge.returncode,
    "merge_stdout": merge.stdout.strip().splitlines()[-1] if merge.stdout.strip() else "",
    "conflicts": conflicts,
    "guard_py_exists": (work / "pkg" / "guard.py").exists(),
    "limits_py_exists": (work / "pkg" / "limits.py").exists(),
    "pytest_tail": tail[-1] if tail else "?",
    "error": next((ln.strip() for ln in composed.stdout.splitlines() if "ModuleNotFoundError" in ln), ""),
}
save()

# ------------------------------------------------------- both checkers, one topology
SHARED = "pkg/guard.py"
RENAMED = "pkg/limits.py"


def load(label: str, source: str) -> object:
    path = pathlib.Path(tempfile.mkdtemp(prefix=f"chk-{label}-")) / "chk.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"chk_{label}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"chk_{label}"] = mod          # dataclasses resolves __module__ via sys.modules
    spec.loader.exec_module(mod)
    return mod


branch_src = (ROOT / "scripts/check_merge_base_overlap.py").read_text()
main_src = subprocess.run(
    ["git", "show", f"{git(ROOT, 'rev-parse', 'HEAD').strip()}:scripts/check_merge_base_overlap.py"],
    cwd=ROOT, capture_output=True, text=True).stdout

PULLS = [
    {"number": 10, "draft": False, "head": {"sha": "headA"}, "html_url": "u10"},
    {"number": 20, "draft": False, "head": {"sha": "headB"}, "html_url": "u20"},
]
# A renames guard.py and touches the importer; B edits guard.py and adds a case.
COMPARES = {
    "main...headA": {"merge_base_commit": {"sha": base_sha}, "behind_by": 0, "files": [
        {"filename": RENAMED, "previous_filename": SHARED, "status": "renamed"},
        {"filename": "tests/test_guard.py"},
    ]},
    "headA...main": {"merge_base_commit": {"sha": base_sha}, "behind_by": 0, "files": []},
    "main...headB": {"merge_base_commit": {"sha": base_sha}, "behind_by": 0, "files": [
        {"filename": SHARED}, {"filename": "tests/test_ceiling.py"},
    ]},
    "headB...main": {"merge_base_commit": {"sha": base_sha}, "behind_by": 0, "files": []},
}


def fake_get(url: str, token: str) -> object:
    if "/pulls?" in url:
        return PULLS if url.endswith("page=1") else []
    if "/files?" in url:
        number = int(url.split("/pulls/", 1)[1].split("/", 1)[0])
        page = int(url.rsplit("page=", 1)[1])
        head = "headA" if number == 10 else "headB"
        return COMPARES[f"main...{head}"]["files"] if page == 1 else []
    return COMPARES[url.split("/compare/", 1)[1]]


verdicts: dict[str, object] = {}
for label, source in (("main", main_src), ("branch", branch_src)):
    mod = load(label, source)
    mod._get = fake_get
    import contextlib, io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # main() returns the exit code; only the __main__ block wraps it in sys.exit.
        try:
            code = int(mod.main(["--all-open", "--github-repo", "owner/name", "--token", "t"]) or 0)
        except SystemExit as exc:
            code = int(exc.code or 0)
    report = buf.getvalue()
    verdicts[label] = {
        "exit": code,
        "pairs_reported": "#10 + #20" in report,
        "names_shared_path": SHARED in report,
        "report": report,
    }
    print(f"{label}: exit={code} pair_reported={'#10 + #20' in report}")

facts["verdicts"] = verdicts
save()

# ------------------------------------------------------------------ live open queue
facts["live"] = {
    "open_prs": 7,
    "pairs": 21,
    "renames_in_open_set": 0,
    "max_changed_files": 10,
    "requests_before": 15,
    "requests_after": 22,
    "merged_precedent": "#2057: tests/simulation/test_args_docstring_completeness.py -> tests/test_args_docstring_completeness.py",
}
facts["mutations"] = [
    ("M1  entry_paths reads only filename (the defect)", 3, 0),
    ("M2  entry_paths reads only previous_filename", 5, 6),
    ("M3  head side back on the capped compare list", 2, 0),
    ("M4  pull_request_paths reads page 1 only", 2, 0),
    ("M5  base side drops the cap guard", 0, 1),
]
save()

assert facts["compose"]["merge_exit"] == 0, "the merge must succeed for the point to hold"
assert facts["compose"]["conflicts"] == [], "git must compose this with no conflict"
assert facts["compose"]["guard_py_exists"] is False
assert facts["compose"]["limits_py_exists"] is True
assert "ModuleNotFoundError" in facts["compose"]["error"], facts["compose"]
assert verdicts["main"]["exit"] == 0 and verdicts["main"]["pairs_reported"] is False
assert verdicts["branch"]["exit"] == 1 and verdicts["branch"]["pairs_reported"] is True
assert verdicts["branch"]["names_shared_path"] is True
print("\nall capture assertions hold ->", OUT)

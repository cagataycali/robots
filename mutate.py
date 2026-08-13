"""Mutation table: 8 plausible regressions x 2 arms (new sweep tests / pre-existing 30)."""
import ast, pathlib, re, shutil, subprocess, sys

ROOT = pathlib.Path(sys.argv[1])
SCRIPT = ROOT / "scripts" / "check_merge_base_overlap.py"
TESTS = ROOT / "tests" / "test_merge_base_overlap.py"
SAVE = pathlib.Path("/tmp/mutsave-%s.py" % sys.argv[2])
shutil.copy(SCRIPT, SAVE)
src0 = SCRIPT.read_text(encoding="utf-8")

NEW = [n.name for n in ast.walk(ast.parse(TESTS.read_text(encoding="utf-8")))
       if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
       and ("sweep" in n.name or "open_pull" in n.name or "prose_only" in n.name
            or "stale_base" in n.name or "truncated" in n.name or "draft" in n.name
            or "disjoint" in n.name or "unreadable" in n.name or "all_open" in n.name)]
K_NEW = " or ".join(NEW)

MUTATIONS = [
    ("M1 pairwise mode never compares a pair", "pair_overlaps",
     "    for left, right in itertools.combinations(ordered, 2):",
     "    for left, right in []:"),
    ("M2 a pair finding does not set the exit status", "_run_sweep",
     "    return 1 if any(row[2] for row in pairs) or any(row[2] for row in stale) else 0",
     "    return 1 if any(row[2] for row in stale) else 0"),
    ("M3 stale-base mode skips every pull request", "stale_base_overlaps",
     "        if row.landed_since is None or row.behind_by == 0:",
     "        if True:"),
    ("M4 drafts are swept too", "resolve_open_pull_requests",
     '            if not isinstance(row, dict) or row.get("draft"):',
     "            if not isinstance(row, dict):"),
    ("M5 a truncated path set is read as complete", "compare_paths",
     "    if len(entries) >= _COMPARE_FILE_CAP:",
     "    if False:"),
    ("M6 an unreadable base side drops the whole pull request", "collect_open_pull_requests",
     '            landed_since = None\n            unevaluated.append((number, f"stale-base mode only - base-side path set unreadable: {error}"))',
     '            unevaluated.append((number, f"dropped: {error}"))\n            continue'),
    ("M7 the sweep carries its own rule: prose blocks", "pair_overlaps",
     "        blocking, prose = partition_overlap(overlapping_paths(left.edits, right.edits))",
     "        blocking, prose = (overlapping_paths(left.edits, right.edits), ())"),
    ("M8 --all-open with --head is honoured, not refused", "main",
     "    if args.all_open and args.head is not None:",
     "    if False:"),
]

def run(k: str) -> str:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-cov", "-p", "no:randomly", "--tb=no", "-k", k],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    summary = [l for l in out.splitlines() if re.match(r"^=+.*(passed|failed|error)", l)]
    tail = summary[-1] if summary else out.splitlines()[-1] if out.splitlines() else "?"
    f = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", tail))
    p = int((re.search(r"(\d+) passed", tail) or ["", "0"])[1])
    return f"{f} failed / {p} passed"

def fn_range(src: str, name: str) -> tuple[int, int]:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.lineno, node.end_lineno or node.lineno
    raise AssertionError(f"no function {name}")

print(f"{'mutation':<52} {'new sweep tests':<22} {'pre-existing 30':<20}")
print("-" * 96)
try:
    print(f"{'(unmutated control)':<52} {run(K_NEW):<22} {run('not (' + K_NEW + ')'):<20}")
    for label, fname, old, new in MUTATIONS:
        lo, hi = fn_range(src0, fname)
        lines = src0.splitlines(keepends=True)
        region = "".join(lines[lo - 1:hi])
        in_fn, in_file = region.count(old), src0.count(old)
        assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname} (want 1)"
        mutated = src0.replace(region, region.replace(old, new, 1), 1)
        assert mutated != src0, f"{label}: no-op mutation"
        SCRIPT.write_text(mutated, encoding="utf-8")
        print(f"{label:<52} {run(K_NEW):<22} {run('not (' + K_NEW + ')'):<20}  [in_fn={in_fn} in_file={in_file}]")
        SCRIPT.write_text(src0, encoding="utf-8")
finally:
    shutil.copy(SAVE, SCRIPT)
assert SCRIPT.read_text(encoding="utf-8") == src0, "restore failed"
print("\nrestored byte-identically")

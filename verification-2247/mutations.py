import ast, json, pathlib, re, subprocess, sys

SRC = pathlib.Path("scripts/check_merge_base_overlap.py")
TEST = "tests/test_merge_base_overlap.py"
orig = SRC.read_text()
tree = ast.parse(orig)
fns = {n.name: (n.lineno, n.end_lineno) for n in ast.walk(tree)
       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

MUTS = [
    ("M1  paths_from_entries drops previous_filename (the defect)",
     "paths_from_entries",
     '        for key in ("filename", "previous_filename")\n',
     '        for key in ("filename",)\n'),
    ("M2  compare_fork_point enforces the compare cap again (old behaviour)",
     "compare_fork_point",
     "    _, merge_base_sha, behind_by = _compare_payload(repo, base, head, token)\n",
     "    _, merge_base_sha, behind_by = compare_paths(repo, base, head, token)\n"),
    ("M3  pull_request_paths returns after page 1",
     "pull_request_paths",
     "        if len(rows) < _PULL_FILE_PAGE:\n",
     "        if True:\n"),
]

def failing_names(paths):
    p = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no", "-rf"],
                       capture_output=True, text=True)
    return sorted({ln.split("::")[-1].split(" ")[0]
                   for ln in p.stdout.splitlines() if ln.startswith("FAILED")})

rows = []
try:
    base_fail = failing_names([TEST])
    assert base_fail == [], f"unmutated control is not clean: {base_fail}"
    print("control: clean\n")
    for label, fn, old, new in MUTS:
        lo, hi = fns[fn]
        region = "".join(orig.splitlines(keepends=True)[lo - 1 : hi])
        in_fn, in_file = region.count(old), orig.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} in_file={in_file}"
        SRC.write_text(orig.replace(region, region.replace(old, new, 1), 1))
        names = failing_names([TEST])
        rows.append({"label": label, "in_fn": in_fn, "in_file": in_file, "failing": names})
        print(f"{label}\n    anchor in_fn={in_fn} in_file={in_file}\n    fails ({len(names)}): {names}")
        SRC.write_text(orig)
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig
    print("\nsource restored byte-identically")

pathlib.Path(f"/tmp/mutrows-{__import__('os').environ['GITHUB_RUN_ID']}.json").write_text(json.dumps(rows, indent=2))

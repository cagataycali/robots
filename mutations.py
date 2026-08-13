import ast, json, os, pathlib, re, subprocess, sys
MINE = pathlib.Path(sys.argv[1]); RUN = sys.argv[2]
SRC = MINE / "strands_robots/simulation/mujoco/simulation.py"
TESTFILE = "tests/simulation/test_finalizer_reports_only_real_cleanup_failures.py"
NEW = ("TestTheFinalizerCompletesAtInterpreterShutdown or "
       "TestNoFinalizerReachableTeardownNeedsTheImportSystem")
orig = SRC.read_text()
tree = ast.parse(orig)
fn = next(f for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == "MuJoCoSimEngine"
          for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "cleanup")
lo, hi = fn.lineno - 1, fn.end_lineno
lines = orig.splitlines(keepends=True)
region = "".join(lines[lo:hi])

MUTATIONS = [
    ("M1 revert: local import back in cleanup()",
     "        try:\n            self._shutdown_ros_bridge()\n",
     "        import contextlib as _cl2\n\n        try:\n            self._shutdown_ros_bridge()\n"),
    ("M2 same, using the from-form",
     "        try:\n            self._shutdown_ros_bridge()\n",
     "        from contextlib import suppress as _sup\n\n        try:\n            self._shutdown_ros_bridge()\n"),
    ("M3 move the import into destroy() instead",
     "    def destroy(self) -> dict[str, Any]:\n", None),  # handled specially
]
# region uses contextlib.suppress; build the try/except form of the first step so M1/M2 anchor
FIRST = "        with contextlib.suppress(Exception):\n            self._shutdown_ros_bridge()\n"
assert region.count(FIRST) == 1, region.count(FIRST)

def run(label, mutated_full):
    SRC.write_text(mutated_full)
    res = {}
    for arm, kexpr in (("new", NEW), ("pre_existing", f"not ({NEW})")):
        r = subprocess.run([sys.executable, "-m", "pytest", TESTFILE, "-q", "--no-cov",
                            "-p", "no:randomly", "-k", kexpr, "--tb=no"],
                           cwd=MINE, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "MUJOCO_GL": "egl"})
        m = re.search(r"(\d+) failed", r.stdout); p = re.search(r"(\d+) passed", r.stdout)
        res[arm] = {"failed": int(m.group(1)) if m else 0, "passed": int(p.group(1)) if p else 0}
    return res

out = {}
try:
    out["M0 unmutated control"] = run("M0", orig)
    # M1: re-introduce a function-local stdlib import at the top of cleanup's body
    m1 = orig.replace(FIRST, "        import contextlib as _cl2\n\n" + FIRST.replace("contextlib.", "_cl2."), 1)
    assert m1 != orig and "import contextlib as _cl2" in m1
    out["M1 function-local stdlib import returns to cleanup()"] = run("M1", m1)
    # M2: the from-form of the same regression
    m2 = orig.replace(FIRST,
                      "        from contextlib import suppress as _sup\n\n"
                      "        with _sup(Exception):\n            self._shutdown_ros_bridge()\n", 1)
    assert m2 != orig and "from contextlib import suppress" in m2
    out["M2 same regression written as a from-import"] = run("M2", m2)
    # M3: put the import in destroy() -- the other finalizer-reachable method
    danchor = "    def destroy(self) -> dict[str, Any]:\n"
    assert orig.count(danchor) == 1, orig.count(danchor)
    dbody = ast.parse(orig)
    dfn = next(f for c in ast.walk(dbody) if isinstance(c, ast.ClassDef) and c.name == "MuJoCoSimEngine"
               for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "destroy")
    dl = orig.splitlines(keepends=True)
    ins = dfn.body[0].end_lineno  # after the docstring
    m3 = "".join(dl[:ins]) + "        import json as _j  # regression\n\n" + "".join(dl[ins:])
    assert "import json as _j" in m3
    out["M3 stdlib import lands in destroy() instead"] = run("M3", m3)
    # M4: drop the docstring Note that tells a reader why the import is at module scope
    note = "        Note:\n            Every name this method needs is bound at module scope."
    assert orig.count(note) == 1
    m4 = orig.replace(note, "        Note:\n            Placeholder.", 1)
    out["M4 delete the reason from the docstring"] = run("M4", m4)
finally:
    SRC.write_text(orig)
assert SRC.read_text() == orig, "RESTORE FAILED"
pathlib.Path(f"/tmp/mut-{RUN}.json").write_text(json.dumps(out, indent=2))
print(f"cleanup() region lines {fn.lineno}-{fn.end_lineno}; anchor in_fn={region.count(FIRST)} in_file={orig.count(FIRST)}")
for k, v in out.items():
    print(f"{k:52s} new={v['new']['failed']}F/{v['new']['passed']}P  pre_existing={v['pre_existing']['failed']}F/{v['pre_existing']['passed']}P")
print("restored byte-identical")

import ast, os, pathlib, re, subprocess, sys

SRC = pathlib.Path("tests/simulation/mujoco/_gl_probe.py")
TST = "tests/simulation/mujoco/test_gl_probe.py"
orig = SRC.read_text(encoding="utf-8")
BASE = subprocess.run(["git", "merge-base", "HEAD", "upstream/main"], capture_output=True, text=True).stdout.strip()


def tests_in(src):
    return [n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


base_tst = subprocess.run(["git", "show", f"{BASE}:{TST}"], capture_output=True, text=True).stdout
OLD = tests_in(base_tst)
ALL = tests_in(pathlib.Path(TST).read_text(encoding="utf-8"))
NEW = [t for t in ALL if t not in OLD]
assert len(OLD) == 3 and len(NEW) == 6, (OLD, NEW)
print(f"pre-existing arm ({len(OLD)}): {OLD}")
print(f"new arm ({len(NEW)}): {NEW}\n")

MUTS = [
    ("M1 drop the early-out (restore the original defect)", "_probe_gl_once",
     "    if _HARDWARE_PROBE_RESULT is not None:\n        return _HARDWARE_PROBE_RESULT\n", ""),
    ("M2 latch only on success (retry a graceful failure)", "_probe_gl_once",
     "    _HARDWARE_PROBE_RESULT = False\n    try:\n        import mujoco as mj\n",
     "    try:\n        import mujoco as mj\n"),
    ("M3 the force-skip poisons the latch", "gl_available",
     '    if os.environ.get("ROBOT_TEST_MUJOCO") == "0":\n        return False\n',
     '    if os.environ.get("ROBOT_TEST_MUJOCO") == "0":\n'
     "        global _HARDWARE_PROBE_RESULT\n        _HARDWARE_PROBE_RESULT = False\n        return False\n"),
    ("M4 probe before honouring the force-skip", "gl_available",
     '    if os.environ.get("ROBOT_TEST_MUJOCO") == "0":\n        return False\n    return _probe_gl_once()\n',
     '    answer = _probe_gl_once()\n    if os.environ.get("ROBOT_TEST_MUJOCO") == "0":\n'
     "        return False\n    return answer\n"),
    ("M5 drop the global so the latch never persists", "_probe_gl_once",
     "    global _HARDWARE_PROBE_RESULT\n", ""),
]


def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(name)


def run(names):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TST, "-q", "--no-cov", "-p", "no:randomly", "-k", " or ".join(names)],
        capture_output=True, text=True, timeout=400, env={**os.environ, "MUJOCO_GL": "egl"},
    )
    out = r.stdout + r.stderr
    n = lambda w: int((re.search(rf"(\d+) {w}", out) or [0, 0])[1])  # noqa: E731
    bad = n("failed") + n("error") + n("errors")
    return bad, n("passed")


print(f"{'mutation':<50} {'new arm':>18} {'pre-existing':>18}")
print("-" * 90)
bn, bp = run(NEW)
on, op = run(OLD)
print(f"{'(unmutated control)':<50} {f'{bn} bad / {bp} pass':>18} {f'{on} bad / {op} pass':>18}")
caught_new = caught_old = 0
try:
    for label, fname, old, new in MUTS:
        lo, hi = fn_range(orig, fname)
        region = "".join(orig.splitlines(keepends=True)[lo - 1:hi])
        in_fn, in_file = region.count(old), orig.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} in_file={in_file}"
        SRC.write_text(orig.replace(old, new, 1), encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") != orig
        mn, _ = run(NEW)
        mo, _ = run(OLD)
        caught_new += mn > 0
        caught_old += mo > 0
        print(f"{label:<50} {f'{mn} bad':>18} {f'{mo} bad':>18}   [in_fn={in_fn} in_file={in_file}]")
finally:
    SRC.write_text(orig, encoding="utf-8")
    assert SRC.read_text(encoding="utf-8") == orig, "RESTORE FAILED"
print(f"\ncaught: new arm {caught_new}/{len(MUTS)}   pre-existing arm {caught_old}/{len(MUTS)}")
print("restore: byte-identical")

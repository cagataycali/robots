import ast, pathlib, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/mujoco/simulation.py")
TEST = "tests/simulation/test_recording_rate_matches_control_frequency.py"
NEW_K = "AsyncEntryPointRefusesBeforeItReportsStarted or MultiRobotEntryPointRefusesTheSameDisagreement"
original = SRC.read_text()

def fn_range(src, name):
    tree = ast.parse(src)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == name:
                    return fn.lineno, fn.end_lineno
    raise SystemExit(f"{name} not found")

def apply(method, style):
    """Return mutated source. style='delete' drops the guard; 'discard' keeps the call."""
    guard = (
        f'        if err := self._validate_recording_rate(control_frequency, "{method}"):\n'
        "            return err\n"
    )
    lo, hi = fn_range(original, method)
    lines = original.splitlines(keepends=True)
    region = "".join(lines[lo - 1 : hi])
    in_fn, in_file = region.count(guard), original.count(guard)
    assert in_fn == 1, (method, in_fn)
    print(f"    anchor for {method}: in_fn={in_fn} in_file={in_file}")
    if style == "delete":
        repl = ""
    else:
        repl = f'        self._validate_recording_rate(control_frequency, "{method}")\n'
    new_region = region.replace(guard, repl, 1)
    out = "".join(lines[: lo - 1]) + new_region + "".join(lines[hi:])
    ast.parse(out)
    return out

def run(k):
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "--no-cov", "-p", "no:randomly", "-k", k],
        capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"},
    )
    tail = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l]
    return tail[-1].strip() if tail else "(no summary)"

MUTATIONS = [
    ("start_policy", "delete", "drop the guard entirely"),
    ("start_policy", "discard", "keep the call, discard the refusal"),
    ("run_multi_policy", "delete", "drop the guard entirely"),
    ("run_multi_policy", "discard", "keep the call, discard the refusal"),
]
rows = []
try:
    for method, style, label in MUTATIONS:
        print(f"\n== {method}: {label} ==")
        SRC.write_text(apply(method, style))
        new = run(NEW_K)
        old = run(f"not ({NEW_K})")
        rows.append((method, label, new, old))
        print(f"    new tests      : {new}")
        print(f"    existing tests : {old}")
finally:
    SRC.write_text(original)
    assert SRC.read_text() == original
    print("\nsource restored byte-identically")

print("\n=== MUTATION TABLE ===")
for m, lab, new, old in rows:
    print(f"{m:17s} | {lab:36s} | new: {new:26s} | existing: {old}")

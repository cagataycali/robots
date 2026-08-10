import ast, pathlib, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/base.py")
NEW = ["tests/simulation/test_eval_facade_preflight_refusals.py"]
STRUCT = ["tests/simulation/test_recording_rate_matches_control_frequency.py"]
ORIG = SRC.read_text()

CASES = [
    ("evaluate_benchmark", 'if err := self._validate_recording_rate(control_frequency, "evaluate_benchmark"):',
     '        self._validate_recording_rate(control_frequency, "evaluate_benchmark")\n'),
    ("eval_policy", 'if err := self._validate_recording_rate(control_frequency, "eval_policy"):',
     '        self._validate_recording_rate(control_frequency, "eval_policy")\n'),
]

def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(name)

def run(files):
    p = subprocess.run([sys.executable, "-m", "pytest", *files, "-q", "--no-cov", "-p", "no:randomly",
                        "--timeout=180"], capture_output=True, text=True)
    return sum(1 for l in p.stdout.splitlines() if l.startswith("FAILED")), p.stdout

rows = []
try:
    for fname, anchor, repl in CASES:
        lo, hi = fn_range(ORIG, fname)
        lines = ORIG.splitlines(keepends=True)
        hits = [i for i in range(lo - 1, hi) if lines[i].strip() == anchor]
        assert len(hits) == 1, hits
        i = hits[0]
        assert lines[i + 1].strip() == "return err"
        SRC.write_text("".join(lines[:i] + [repl] + lines[i + 2:]))
        ast.parse(SRC.read_text())
        f_new, _ = run(NEW)
        f_struct, out = run(STRUCT)
        rows.append((fname, f_new, f_struct))
        SRC.write_text(ORIG)
finally:
    SRC.write_text(ORIG)
assert SRC.read_text() == ORIG

print("Guard CALL kept, refusal DISCARDED (structural test still sees the call):")
print("| mutation | new tests failing | structural-parity module failing |")
print("|---|---|---|")
for fname, a, b in rows:
    print(f"| `{fname}` calls the rate guard but discards its refusal | {a} | {b} |")
assert all(r[1] > 0 for r in rows), "new tests missed it"
assert all(r[2] == 0 for r in rows), "structural module unexpectedly caught it"
print("\nsource restored byte-identical:", SRC.read_text() == ORIG)

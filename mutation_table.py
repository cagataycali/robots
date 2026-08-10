import ast, os, pathlib, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/mujoco/simulation.py")
ORIG = SRC.read_text()
F = ["tests/simulation/test_policy_config_mapping_validation.py"]
NEW = "test_start_policy_reports_a_policy_kwargs_error_instead_of_a_false_started"
GUARD = ('        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "start_policy"):\n'
         '            return err\n')
MUTS = [
    ("guard DELETED entirely", ""),
    ("call kept, `return err` DISCARDED",
     '        self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "start_policy")\n'),
]

def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(name)

def run(k):
    r = subprocess.run([sys.executable, "-m", "pytest", *F, "-q", "--no-cov", "-p", "no:randomly", "-k", k, "--tb=no"],
                       capture_output=True, text=True,
                       env={**os.environ, "MUJOCO_GL": "egl", "HF_HUB_OFFLINE": "1"})
    t = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l]
    return t[-1].strip().strip("=").strip() if t else "??"

lo, hi = fn_range(ORIG, "start_policy")
lines = ORIG.splitlines(keepends=True)
region = "".join(lines[lo - 1 : hi])
print(f"anchor: in start_policy={region.count(GUARD)}  in file={ORIG.count(GUARD)}")
assert region.count(GUARD) == 1
print(f"\n{'mutation':38s} {'the new case':22s} the 10 pre-existing cases")
print("-" * 90)
try:
    for label, repl in MUTS:
        mutated = "".join(lines[: lo - 1]) + region.replace(GUARD, repl, 1) + "".join(lines[hi:])
        assert mutated != ORIG
        ast.parse(mutated)
        SRC.write_text(mutated)
        print(f"{label:38s} {run(NEW):22s} {run('not ' + NEW)}")
finally:
    SRC.write_text(ORIG)
    assert SRC.read_text() == ORIG, "RESTORE FAILED"
    print("\nsource restored byte-identically")

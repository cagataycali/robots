"""Mutation table: 5 plausible regressions x 2 arms (new module / pre-existing)."""
import ast, pathlib, subprocess, sys

SRC = pathlib.Path("strands_robots/device_connect/__init__.py")
ORIG = SRC.read_text()
NEW = ["tests/test_device_connect_sync_bringup_outcomes.py"]
OLD = [
    "tests/test_device_connect_hardening.py", "tests/test_device_connect_drivers.py",
    "tests/test_device_connect_all_robots.py", "tests/test_device_connect_heartbeat.py",
    "tests/test_device_connect_public_member_docstrings.py", "tests/test_robot_factory.py",
]

TIMEOUT_ARM = '''    if not started:
        raise TimeoutError(
            f"init_device_connect_sync: the Device Connect runtime did not come up "
            f"within {_INIT_TIMEOUT_S:g}s. The bring-up is still running on its "
            f"background thread; check that the messaging URL / broker is reachable."
        )
'''
ERR_ARM = '''    if error_holder[0] is not None:
        raise error_holder[0]
'''

MUTS = [
    ("M1 timeout arm deleted (returns None again)", TIMEOUT_ARM, ""),
    ("M2 timeout checked before the recorded error", ERR_ARM + TIMEOUT_ARM, TIMEOUT_ARM + ERR_ARM),
    ("M3 budget re-hardcoded at the wait site",
     "started = ready.wait(timeout=_INIT_TIMEOUT_S)", "started = ready.wait(timeout=30.0)"),
    ("M4 recorded exception re-wrapped, not re-raised",
     "        raise error_holder[0]\n", "        raise RuntimeError(str(error_holder[0]))\n"),
    ("M5 budget omitted from the refusal message",
     'f"within {_INIT_TIMEOUT_S:g}s. The bring-up is still running on its "',
     'f"eventually. The bring-up is still running on its "'),
]

# anchor scoping evidence
fn_lo, fn_hi = 0, 0
tree = ast.parse(ORIG)
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.name == "init_device_connect_sync":
        fn_lo, fn_hi = n.lineno, n.end_lineno
region = "\n".join(ORIG.splitlines()[fn_lo - 1 : fn_hi]) + "\n"
for label, old, _new in MUTS:
    print(f"  anchor {label[:2]}: in_fn={region.count(old)} in_file={ORIG.count(old)}")
    assert region.count(old) == 1, label
print()

def run(paths):
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"], capture_output=True, text=True)
    import re
    f = re.search(r"(\d+) failed", r.stdout)
    p = re.search(r"(\d+) passed", r.stdout)
    return int(f.group(1)) if f else 0, int(p.group(1)) if p else 0

try:
    print(f"{'mutation':46s} {'new module':>14s} {'pre-existing':>14s}")
    for label, old, new in MUTS:
        assert ORIG.count(old) == 1
        SRC.write_text(ORIG.replace(old, new, 1))
        nf, np_ = run(NEW)
        of, op = run(OLD)
        print(f"  {label:44s} {nf:3d} failed{'':4s} {of:3d} failed  (of {op} passed)")
    SRC.write_text(ORIG)
    nf, np_ = run(NEW)
    of, op = run(OLD)
    print(f"  {'(unmutated control)':44s} {nf:3d} failed{'':4s} {of:3d} failed  ({np_}/{op} passed)")
finally:
    SRC.write_text(ORIG)
    assert SRC.read_text() == ORIG
    print("\nrestored byte-identical")

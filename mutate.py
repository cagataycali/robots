import ast, pathlib, re, subprocess, sys

SRC = pathlib.Path("strands_robots/tools/pose_tool.py")
ORIG = SRC.read_text()
NEW_K = "TestTheUnknownMotorDeferralHolds or TestTheComputedTargetDeferralHolds"
FILES = ["tests/tools/test_pose_tool.py", "tests/tools/test_pose_tool_target_domain.py",
         "tests/tools/test_pose_tool_interpolation_options.py", "tests/tools/test_pose_tool_emergency_stop.py"]

def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise SystemExit(f"no fn {name}")

def scoped_replace(src, fn, old, new):
    lo, hi = fn_range(src, fn)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    print(f"      anchor in_fn={in_fn} in_file={in_file}")
    assert in_fn == 1, (fn, in_fn, in_file)
    lines[lo - 1:hi] = [region.replace(old, new)]
    out = "".join(lines)
    assert out != src
    ast.parse(out)
    return out

MUTS = [
    ("M1 remove the unknown-motor deferral", "_joint_delta_error",
     "    config = _DEFAULT_MOTOR_CONFIGS.get(name)\n    if config is None:\n        return None\n",
     "    config = _DEFAULT_MOTOR_CONFIGS[name]\n"),
    ("M2 the read tolerates an unconfigured motor", "read_motor_position",
     '            logger.error(f"Failed to read motor {motor_name}: {e}")\n\n        return None\n',
     '            logger.error(f"Failed to read motor {motor_name}: {e}")\n\n        return 0.0\n'),
    ("M3 incremental_move tolerates a missing reading", "incremental_move",
     "        if current_pos is None:\n            return False\n",
     "        if current_pos is None:\n            current_pos = 0.0\n"),
    ("M4 bound the delta by the endpoints, not the travel", "_joint_delta_error",
     "    if abs(delta) > span:\n", "    if not low <= delta <= high:\n"),
    ("M5 drop the clamp the computed target relies on", "degrees_to_position",
     "        degrees = max(min_deg, min(max_deg, degrees))\n", "        degrees = float(degrees)\n"),
]

def run(k):
    p = subprocess.run([sys.executable, "-m", "pytest", *FILES, "-q", "-p", "no:randomly",
                        "--no-cov", "-k", k], capture_output=True, text=True)
    f = int((re.search(r"(\d+) failed", p.stdout) or [0, 0])[1] or 0)
    ps = int((re.search(r"(\d+) passed", p.stdout) or [0, 0])[1] or 0)
    e = 1 if "error" in p.stdout.lower() and "errors" in p.stdout.lower() else 0
    return f, ps, e

print("=== unmutated control ===")
print("  new-class arm:      failed=%d passed=%d" % run(NEW_K)[:2])
print("  pre-existing arm:   failed=%d passed=%d" % run(f"not ({NEW_K})")[:2])

rows = []
try:
    for label, fn, old, new in MUTS:
        print(f"\n=== {label} ===")
        SRC.write_text(scoped_replace(ORIG, fn, old, new))
        nf, np_, _ = run(NEW_K)
        of, op, _ = run(f"not ({NEW_K})")
        rows.append((label, nf, of))
        print(f"      new classes: failed={nf} passed={np_}   |   pre-existing: failed={of} passed={op}")
finally:
    SRC.write_text(ORIG)
    assert SRC.read_text() == ORIG, "RESTORE FAILED"
    print("\nrestored byte-identically:", SRC.read_text() == ORIG)

print("\n=== TABLE ===")
for label, nf, of in rows:
    print(f"  {label:<52} new={'CAUGHT' if nf else 'missed'}({nf})  pre-existing={'caught' if of else 'BLIND'}({of})")
import json
json.dump([{"label": l, "new_failed": n, "old_failed": o} for l, n, o in rows],
          open("/tmp/art-mutations.json", "w"), indent=2)
print("dumped /tmp/art-mutations.json")
print(f"\ncaught by the new classes: {sum(1 for _,n,_ in rows if n)}/{len(rows)}")
print(f"invisible to the pre-existing cases: {sum(1 for _,_,o in rows if not o)}/{len(rows)}")

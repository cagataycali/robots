import ast, pathlib, re, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "strands_robots/simulation/mujoco/motion_primitives.py"
NEW = "tests/simulation/mujoco/test_primitive_resolution_refusals.py"
OLD = [
    "tests/simulation/mujoco/test_motion_primitives.py",
    "tests/simulation/mujoco/test_primitive_teardown_abort.py",
    "tests/simulation/mujoco/test_primitive_robot_auto_resolution.py",
    "tests/simulation/mujoco/test_motion_primitive_numeric_domains.py",
    "tests/simulation/mujoco/test_set_gripper_setpoint_range_sources.py",
]

# (label, enclosing function, old, new)
MUTATIONS = [
    ("M1 move_to: proceed with no EE frame", "move_to",
     "            if frame is None:\n", "            if False:  # MUTANT\n"),
    ("M2 move_to: proceed with no actuators", "move_to",
     "            if not jact:\n", "            if False:  # MUTANT\n"),
    ("M3 move_to: proceed when all actuators are grippers", "move_to",
     "            if not arm_jact:\n", "            if False:  # MUTANT\n"),
    ("M4 set_gripper: proceed with no gripper actuator", "set_gripper",
     "            if not gripper_acts:\n", "            if False:  # MUTANT\n"),
    ("M5 rotate_wrist: proceed with no actuators", "rotate_wrist",
     "            if not jact:\n", "            if False:  # MUTANT\n"),
    ("M6 rotate_wrist: distal fallback reaches past the gripper classification", "rotate_wrist",
     "            non_gripper = [j for j in candidates if jact[j] not in grip_acts]\n",
     "            non_gripper = list(candidates)  # MUTANT\n"),
    ("M7 _short_name: hand back MuJoCo's None instead of an empty name", "_short_name",
     '        return name or ""\n', "        return name  # MUTANT\n"),
]

original = SRC.read_text()
lines = original.splitlines(keepends=True)
tree = ast.parse(original)


def fn_range(name):
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            if best is None or n.lineno > best.lineno:
                best = n
    return best.lineno, best.end_lineno


def run(paths, extra=()):
    p = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly", "--tb=no", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=1200,
        env={**__import__("os").environ, "MUJOCO_GL": "egl"},
    )
    out = p.stdout + p.stderr
    f = re.search(r"(\d+) failed", out)
    ps = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0), (int(ps.group(1)) if ps else 0)

print("=== unmutated control ===")
print("   new :", run([NEW]))
print("   old :", run(OLD))
rows = []
try:
    for label, fname, old, new in MUTATIONS:
        lo, hi = fn_range(fname)
        region = "".join(lines[lo - 1:hi])
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname} (file: {in_file})"
        head = "".join(lines[: lo - 1])
        tail = "".join(lines[hi:])
        mutated = head + region.replace(old, new, 1) + tail
        assert mutated != original
        SRC.write_text(mutated)
        nf, np_ = run([NEW])
        of, op = run(OLD)
        rows.append((label, fname, in_fn, in_file, nf, of))
        print(f"{label}\n    in_fn={in_fn} in_file={in_file}  NEW: {nf} failed/{np_} passed   OLD: {of} failed/{op} passed")
finally:
    SRC.write_text(original)
    assert SRC.read_text() == original, "RESTORE FAILED"
    print("\nrestored byte-identically:", SRC.read_text() == original)

print("\n===== MUTATION TABLE =====")
print(f"{'mutation':<74} {'new':>5} {'pre-existing':>13}")
for label, fname, in_fn, in_file, nf, of in rows:
    print(f"{label:<74} {nf:>5} {of:>13}")
print(f"\ncaught by the new module: {sum(1 for r in rows if r[4] > 0)}/{len(rows)}")
print(f"caught by pre-existing  : {sum(1 for r in rows if r[5] > 0)}/{len(rows)}")

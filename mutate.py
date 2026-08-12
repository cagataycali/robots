"""7-regression mutation table for the motion-primitive result envelopes.

Run from a repo checkout: ``PYTHONPATH=. python3 mutate.py``.

Each mutation is scoped to the enclosing function by AST line range - four of
the seven anchors appear twice in the file, so the scoping is load-bearing and
the in_fn / in_file counts are printed as the justification. The source is
restored byte-identically in a ``finally``.

Two arms per mutation: the cases this change adds, and the pre-existing suite
over the same primitives.
"""

import ast, json, pathlib, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/motion_primitives_base.py")
ORIGINAL = SRC.read_text()

NEW_K = ("TestMoveToResultEnvelope or TestRotateWristResultEnvelope or "
         "TestSetGripperResultEnvelope or TestRegistryLookupSeamDefault or "
         "test_servo_timeout_reports_the_residual_and_a_budget_that_fixes_it")
OLD_K = f"not ({NEW_K})"

BASE = "tests/simulation/test_motion_primitives_base.py"
MJ = "tests/simulation/mujoco/test_motion_primitives.py"
SIBLINGS = [
    "tests/simulation/mujoco/test_motion_primitive_numeric_domains.py",
    "tests/simulation/mujoco/test_move_to_body_frame_end_effector.py",
    "tests/simulation/mujoco/test_primitive_teardown_abort.py",
    "tests/simulation/mujoco/test_primitive_robot_auto_resolution.py",
    "tests/simulation/mujoco/test_set_gripper_setpoint_range_sources.py",
    "tests/simulation/mujoco/test_gripper_action_key_equivalence.py",
]


def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.body[0].lineno, n.end_lineno
    raise AssertionError(f"no function {name}")


def apply(src, fn, old, new):
    lo, hi = fn_range(src, fn)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"{fn}: anchor appears {in_fn}x in function (file {in_file}x)"
    print(f"    anchor in_fn={in_fn} in_file={in_file}")
    lines[lo - 1:hi] = [region.replace(old, new, 1)]
    out = "".join(lines)
    assert out != src
    ast.parse(out)
    return out


MUTATIONS = [
    ("M1 move_to: not-reached half regressed to success",
     "_move_to_result", "        if reached:\n", "        if True:\n"),
    ("M2 move_to: refusal drops the json payload",
     "_move_to_result", "            payload,\n        )\n", "        )\n"),
    ("M3 move_to: the two residuals collapse into one",
     "_move_to_result", '"position_error_m": position_error,', '"position_error_m": ik_residual,'),
    ("M4 rotate_wrist: not-reached half regressed to success",
     "_rotate_wrist_result", "        if reached:\n", "        if True:\n"),
    ("M5 rotate_wrist: refusal drops the json payload",
     "_rotate_wrist_result", "            payload,\n        )\n", "        )\n"),
    ("M6 default registry seam stops resolving",
     "_get_registry_robot", "        return get_robot(data_config)\n", "        return None\n"),
    ("M7 set_gripper: payload drops setpoint_sources",
     "_set_gripper_result", '"setpoint_sources": setpoint_sources,\n', ""),
]


def run(files, k):
    import os, re
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "--no-header", "--no-cov",
         "-p", "no:randomly", "-k", k],
        capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"},
    )
    f = re.search(r"(\d+) failed", r.stdout)
    p = re.search(r"(\d+) passed", r.stdout)
    return (int(f.group(1)) if f else 0, int(p.group(1)) if p else 0)


rows = []
try:
    print("=== CONTROL (unmutated) ===")
    ctl_new = run([BASE, MJ], NEW_K)
    ctl_old = run([BASE, MJ, *SIBLINGS], OLD_K)
    print(f"  new arm: {ctl_new[0]} failed / {ctl_new[1]} passed")
    print(f"  old arm: {ctl_old[0]} failed / {ctl_old[1]} passed")
    for label, fn, old, new in MUTATIONS:
        print(f"\n=== {label}")
        SRC.write_text(apply(ORIGINAL, fn, old, new))
        n = run([BASE, MJ], NEW_K)
        o = run([BASE, MJ, *SIBLINGS], OLD_K)
        print(f"    new cases: {n[0]} failed / {n[1]} passed   |   pre-existing: {o[0]} failed / {o[1]} passed")
        rows.append(dict(label=label, new_failed=n[0], old_failed=o[0]))
finally:
    SRC.write_text(ORIGINAL)
    assert SRC.read_text() == ORIGINAL, "restore failed"
    print("\nrestored byte-identically: True")

caught = sum(1 for r in rows if r["new_failed"] > 0)
blind = sum(1 for r in rows if r["old_failed"] == 0)
print(f"\nSUMMARY: caught by the new cases {caught}/{len(rows)};"
      f" invisible to the pre-existing suite {blind}/{len(rows)}")
print(json.dumps(rows, indent=1))

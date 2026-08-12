import ast, pathlib, subprocess, sys

SIM = pathlib.Path("strands_robots/simulation/mujoco/simulation.py")
ORIG = SIM.read_text()
SAVE = pathlib.Path("/tmp/sim-save-" + __import__("os").environ["GITHUB_RUN_ID"] + ".py")
SAVE.write_text(ORIG)

# AST line range of the hook
tree = ast.parse(ORIG)
fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_maybe_install_wbc_torque_control":
        fn = node
assert fn is not None
lines = ORIG.splitlines(keepends=True)
lo, hi = fn.lineno - 1, fn.end_lineno
region = "".join(lines[lo:hi])
print(f"hook lines {fn.lineno}-{fn.end_lineno}\n")

MUTS = [
    ("M1 drop the ImportError tolerance",
     """        try:
            from strands_robots.policies.wbc import (
                WBCPolicy,
                install_wbc_torque_control,
                wbc_uses_position_servo,
            )
        except ImportError:
            return None
""",
     """        from strands_robots.policies.wbc import (
            WBCPolicy,
            install_wbc_torque_control,
            wbc_uses_position_servo,
        )
"""),
    ("M2 drop the no-compiled-world guard",
     "        if world is None or world._model is None:\n            return None\n", ""),
    ("M3 drop the manual-install-wins guard",
     '        if isinstance(backend_state, dict) and backend_state.get("action_controller") is not None:\n'
     "            return None  # a manually-installed controller wins\n", ""),
    ("M4 drop the position-servo guard",
     "        if not wbc_uses_position_servo(self, policy, robot_name):\n            return None\n", ""),
    ("M5 drop the non-WBC-policy guard",
     "        if not isinstance(policy, WBCPolicy):\n            return None\n", ""),
    ("M7 revert the cleanup to gains only",
     """        return _cleanup""",
     """        return controller.uninstall"""),
    ("M6 invert the position-servo guard",
     "        if not wbc_uses_position_servo(self, policy, robot_name):\n",
     "        if wbc_uses_position_servo(self, policy, robot_name):\n"),
]

NEW = ("test_skips_when_the_wbc_extra_is_absent or test_skips_without_a_world or "
       "test_skips_when_the_world_has_no_compiled_model or "
       "test_skips_when_the_driven_actuators_are_already_torque_motors or "
       "test_skips_when_no_wbc_joint_resolves_in_the_scene or "
       "test_the_hook_declines_in_exactly_the_five_ways_this_module_drives or "
       "test_cleanup_unregisters_so_a_second_rollout_gets_the_shim or "
       "test_cleanup_leaves_a_controller_it_did_not_install")

def run(k: str) -> tuple[int, int]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/policies/wbc", "-q", "--no-cov",
         "-p", "no:randomly", "-k", k],
        capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"},
    )
    out = p.stdout
    import re
    f = re.search(r"(\d+) failed", out)
    ps = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0, int(ps.group(1)) if ps else 0)

print(f"{'mutation':40s} {'new cases':>18s} {'490 pre-existing':>20s}")
print("-" * 82)
b0 = run(NEW); b1 = run(f"not ({NEW})")
print(f"{'(unmutated control)':40s} {str(b0):>18s} {str(b1):>20s}")
rows = []
try:
    for label, old, new in MUTS:
        in_fn, in_file = region.count(old), ORIG.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} in_file={in_file}"
        mutated = ORIG.replace(old, new)
        assert mutated != ORIG
        SIM.write_text(mutated)
        a = run(NEW); b = run(f"not ({NEW})")
        rows.append((label, in_fn, in_file, a, b))
        print(f"{label:40s} {str(a):>18s} {str(b):>20s}   [in_fn={in_fn} in_file={in_file}]")
        SIM.write_text(ORIG)
finally:
    SIM.write_text(ORIG)

assert SIM.read_text() == SAVE.read_text(), "restore must be byte-identical"
print("\nrestored byte-identically:", SIM.read_text() == ORIG)
caught_new = sum(1 for _l, _a, _b, a, _bb in rows if a[0] > 0)
caught_old = sum(1 for _l, _a, _b, _a2, b in rows if b[0] > 0)
print(f"caught by the new cases: {caught_new} of {len(rows)}")
print(f"caught by the 490 pre-existing: {caught_old} of {len(rows)}")
import json, os
json.dump({"rows": [[l, a, b] for l, _f, _fi, a, b in rows], "control": [b0, b1]},
          open(f"/tmp/mut-{os.environ['GITHUB_RUN_ID']}.json", "w"))

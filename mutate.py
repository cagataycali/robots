import ast, json, pathlib, subprocess, sys
ROOT = pathlib.Path(".").resolve()
SRC = ROOT / "strands_robots/simulation/mujoco/motion_primitives.py"
NEW = "tests/simulation/mujoco/test_move_to_body_frame_end_effector.py"
OLD = ["tests/simulation/mujoco/test_motion_primitives.py",
       "tests/simulation/mujoco/test_motion_primitive_numeric_domains.py",
       "tests/simulation/mujoco/test_primitive_robot_auto_resolution.py",
       "tests/simulation/mujoco/test_primitive_teardown_abort.py",
       "tests/simulation/mujoco/test_set_gripper_setpoint_range_sources.py"]
original = SRC.read_text()

def fn_range(src, name):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"{name} not found")

MUTS = [
  ("M1 body pos reads the inertial frame (xipos)", "_frame_world_pose",
   "return np.array(data.xpos[bid], dtype=np.float64), np.array(data.xquat[bid], dtype=np.float64)",
   "return np.array(data.xipos[bid], dtype=np.float64), np.array(data.xquat[bid], dtype=np.float64)"),
  ("M2 body quat reports identity", "_frame_world_pose",
   "return np.array(data.xpos[bid], dtype=np.float64), np.array(data.xquat[bid], dtype=np.float64)",
   "return np.array(data.xpos[bid], dtype=np.float64), np.zeros(4, dtype=np.float64)"),
  ("M3 readback assumes a site frame", "move_to",
   "ee_pos, ee_quat = self._frame_world_pose(model, data, frame_name, frame_type)",
   'ee_pos, ee_quat = self._frame_world_pose(model, data, frame_name, "site")'),
  ("M4 IK bridge built on a hardcoded site frame", "move_to",
   "                    frame_type,\n",
   '                    "site",\n'),
  ("M5 payload reports a hardcoded frame_type", "move_to",
   '"frame_type": frame_type,\n            }\n        if reached:',
   '"frame_type": "site",\n            }\n        if reached:'),
]

def run(paths):
    p = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-header",
                        "-p", "no:randomly", "--no-cov", "--timeout=180"],
                       capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"})
    out = p.stdout
    import re
    f = re.search(r"(\d+) failed", out); ps = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0, int(ps.group(1)) if ps else 0)

rows = []
try:
    for label, fname, old, new in MUTS:
        lo, hi = fn_range(original, fname)
        region = "".join(original.splitlines(keepends=True)[lo-1:hi])
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname} (file: {in_file})"
        mutated = original.replace(old, new)
        assert mutated != original
        SRC.write_text(mutated)
        newf, newp = run([NEW]); oldf, oldp = run(OLD)
        rows.append((label, f"in_fn={in_fn}/in_file={in_file}", newf, newp, oldf, oldp))
        print(f"{label}\n    anchor in_fn={in_fn} in_file={in_file}\n    NEW: {newf} failed / {newp} passed"
              f"\n    PRE-EXISTING: {oldf} failed / {oldp} passed", flush=True)
finally:
    SRC.write_text(original)
    assert SRC.read_text() == original
    print("\nRESTORED byte-identical:", SRC.read_text() == original)
json.dump(rows, open("/tmp/mut.json","w"))
print("\nBLIND to pre-existing:", sum(1 for r in rows if r[4] == 0), "of", len(rows))

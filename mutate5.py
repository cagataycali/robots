import ast, json, os, pathlib, re, subprocess, sys
SRC = pathlib.Path("strands_robots/simulation/mujoco/motion_primitives.py")
NEW = "tests/simulation/mujoco/test_move_to_body_frame_end_effector.py"
OLD = ["tests/simulation/mujoco/test_motion_primitives.py",
       "tests/simulation/mujoco/test_motion_primitive_numeric_domains.py",
       "tests/simulation/mujoco/test_primitive_robot_auto_resolution.py",
       "tests/simulation/mujoco/test_primitive_teardown_abort.py",
       "tests/simulation/mujoco/test_set_gripper_setpoint_range_sources.py"]
original = SRC.read_text()
def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name: return n.lineno, n.end_lineno
    raise AssertionError(name)
old = '\n            "frame_type": frame_type,\n        }'
new = '\n            "frame_type": "site",\n        }'
lo, hi = fn_range(original, "move_to")
region = "".join(original.splitlines(keepends=True)[lo-1:hi])
in_fn, in_file = region.count(old), original.count(old)
print(f"M5 anchor in_fn={in_fn} in_file={in_file}")
assert in_fn == 1 and in_file == 1
def run(paths):
    p = subprocess.run([sys.executable,"-m","pytest",*paths,"-q","--no-header","-p","no:randomly","--no-cov","--timeout=180"],
                       capture_output=True, text=True, env={**os.environ,"MUJOCO_GL":"egl"})
    f=re.search(r"(\d+) failed",p.stdout); ps=re.search(r"(\d+) passed",p.stdout)
    return (int(f.group(1)) if f else 0, int(ps.group(1)) if ps else 0)
try:
    SRC.write_text(original.replace(old,new))
    nf,np_ = run([NEW]); of,op = run(OLD)
    print(f"M5 payload reports a hardcoded frame_type\n    NEW: {nf} failed / {np_} passed\n    PRE-EXISTING: {of} failed / {op} passed")
    rows=json.load(open("/tmp/mut.json"))
    rows.append(("M5 payload reports a hardcoded frame_type", f"in_fn={in_fn}/in_file={in_file}", nf, np_, of, op))
    json.dump(rows, open("/tmp/mut.json","w"))
finally:
    SRC.write_text(original)
    print("RESTORED byte-identical:", SRC.read_text()==original)
rows=json.load(open("/tmp/mut.json"))
print("\nBLIND to pre-existing:", sum(1 for r in rows if r[4]==0), "of", len(rows))

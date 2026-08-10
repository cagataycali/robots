import ast, os, pathlib, subprocess, sys

run = os.environ["GITHUB_RUN_ID"]
MINE = pathlib.Path(f"/tmp/robots-mine-{run}")
os.chdir(MINE)
NEW = "tests/simulation/mujoco/test_add_camera_pose_rule_has_one_owner.py"
EXISTING = [
    "tests/simulation/mujoco/test_add_object_camera_vector_validation.py",
    "tests/simulation/mujoco/test_vector_params_read_by_membership.py",
    "tests/simulation/test_pose_vector_domain_across_backends.py",
    "tests/simulation/test_camera_pixel_count_domain.py",
    "tests/simulation/mujoco/test_agenttool_contract.py",
]
SIM = pathlib.Path("strands_robots/simulation/mujoco/simulation.py")
UTILS = pathlib.Path("strands_robots/utils.py")


def fn_range(path, name):
    src = path.read_text()
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    return src, fn.lineno, fn.end_lineno


def scoped_replace(path, fnname, old, new):
    """Replace `old` -> `new`, asserting exactly one hit INSIDE fnname."""
    src, lo, hi = fn_range(path, fnname)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"{path}::{fnname}: anchor in_fn={in_fn} in_file={in_file}"
    print(f"    anchor in_fn={in_fn} in_file={in_file}  ({path.name}::{fnname})")
    region2 = region.replace(old, new, 1)
    out = "".join(lines[:lo - 1]) + region2 + "".join(lines[hi:])
    ast.parse(out)
    path.write_text(out)


def run_pytest(paths):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
        capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"},
    )
    tail = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l or "error" in l]
    return (tail[-1] if tail else "??"), r.returncode


MUTATIONS = [
    ("M1 re-add a second application of the pose rule to add_camera", SIM, "add_camera",
     '        target, _terr = coerce_pose_vector("add_camera", "target", target, 3)\n',
     '        target, _terr = coerce_pose_vector("add_camera", "target", target, 3)\n'
     '        if (_x := pose_vector_error("add_camera", "target", target, 3)) is not None:\n'
     '            return {"status": "error", "content": [{"text": _x}]}\n'),
    ("M2 coerce_pose_vector stops propagating the read's refusal", UTILS, "coerce_pose_vector",
     "    if err is not None:\n        return None, err\n", "    if False:\n        return None, err\n"),
    ("M3 delete the degenerate-orientation refusal below the deleted loop", SIM, "add_camera",
     "        if all(abs(pos[i] - tgt[i]) < 1e-9 for i in range(3)):\n",
     "        if False:\n"),
]

saved = {p: p.read_text() for p in (SIM, UTILS)}
extra_import = '    positive_finite_number_error,\n'
print("### MUTATION TABLE  (new module | pre-existing suite)\n")
try:
    for label, path, fnname, old, new in MUTATIONS:
        print(label)
        if "M1" in label:  # the mutation needs the import the fix removed
            s = path.read_text()
            assert s.count(extra_import) == 1
            path.write_text(s.replace(extra_import, "    pose_vector_error,\n" + extra_import, 1))
        scoped_replace(path, fnname, old, new)
        a, rca = run_pytest([NEW])
        b, rcb = run_pytest(EXISTING)
        print(f"    new module      : {a}   -> {'CAUGHT' if rca else 'missed'}")
        print(f"    pre-existing    : {b}   -> {'CAUGHT' if rcb else 'MISSED'}")
        for p, t in saved.items():
            p.write_text(t)
        print()
    # M3 against the WHOLE mujoco backend suite, to size the blast radius
    print("M3 again, against the entire tests/simulation/mujoco suite")
    scoped_replace(SIM, "add_camera",
                   "        if all(abs(pos[i] - tgt[i]) < 1e-9 for i in range(3)):\n",
                   "        if False:\n")
    c, rcc = run_pytest(["tests/simulation/mujoco"])
    print(f"    tests/simulation/mujoco : {c}   -> {'CAUGHT' if rcc else 'MISSED'}")
finally:
    for p, t in saved.items():
        p.write_text(t)
    ok = all(p.read_text() == t for p, t in saved.items())
    print(f"\nrestore byte-identical: {ok}")
    assert ok

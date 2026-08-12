"""Mutation table: do the new tests catch plausible regressions the existing suite misses?"""
import ast, json, os, pathlib, re, shutil, subprocess, sys

RUN = os.environ["GITHUB_RUN_ID"]
SRC = pathlib.Path("strands_robots/simulation/isaac/simulation.py")
NEW = "tests/simulation/isaac/test_camera_readback_pixel_domain.py"
SAVE = pathlib.Path(f"/tmp/mut-save-{RUN}.py")
shutil.copy(SRC, SAVE)

def fn_range(src, name):
    for fn in ast.walk(ast.parse(src)):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == name:
            return fn.lineno, fn.end_lineno
    raise AssertionError(name)

MUTATIONS = [
    ("M1 get_frame: delete the pixel-floor guard", "get_frame",
     '                if arg is not None:\n'
     '                    if (dim_err := positive_count_error(arg, arg_name, "get_frame")) is not None:\n'
     '                        raise ValueError(dim_err)\n', ''),
    ("M2 get_frame: keep the call, drop the raise", "get_frame",
     '                        raise ValueError(dim_err)\n', '                        pass\n'),
    ("M3 get_camera_params: delete the guard", "get_camera_params",
     '                if arg is not None:\n'
     '                    if (dim_err := positive_count_error(arg, arg_name, "get_camera_params")) is not None:\n'
     '                        raise ValueError(dim_err)\n', ''),
    ("M4 get_camera_params: keep the call, drop the raise", "get_camera_params",
     '                        raise ValueError(dim_err)\n', '                        pass\n'),
    ("M5 get_frame: reword the refusal locally", "get_frame",
     '                        raise ValueError(dim_err)\n',
     '                        raise ValueError(f"bad size: {arg!r}")\n'),
    ("M6 get_frame: widen the floor to the integral-real domain", "get_frame",
     'positive_count_error(arg, arg_name, "get_frame")',
     'positive_whole_number_error(arg, arg_name, "get_frame")'),
    ("M7 get_camera_params: drop the prim->GL basis correction", "get_camera_params",
     "T[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz) @ prim_to_gl",
     "T[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz)"),
    ("M8 get_frame: silently drop the depth buffer", "get_frame",
     "depth_arr = None if depth is None else np.asarray(depth, dtype=np.float32)",
     "depth_arr = None"),
]

def run(paths, extra=None):
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"]
    if extra:
        cmd += extra
    env = dict(os.environ, MUJOCO_GL="egl")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    f = re.search(r"(\d+) failed", out)
    p = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0, int(p.group(1)) if p else 0)

# arm B = the pre-existing isaac suite, this module deselected
base_new = run([NEW])
base_old = run(["tests/simulation/isaac"], ["--ignore", NEW])
print(f"UNMUTATED   new={base_new}  pre-existing-isaac={base_old}\n")

rows = []
for label, fname, old, new in MUTATIONS:
    src = SAVE.read_text()
    lo, hi = fn_range(src, fname)
    region = "".join(src.splitlines(keepends=True)[lo - 1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname} (file: {in_file})"
    print(f"{label}\n    anchor in_fn={in_fn} in_file={in_file}")
    lines = src.splitlines(keepends=True)
    mutated_region = region.replace(old, new, 1)
    out = "".join(lines[:lo - 1]) + mutated_region + "".join(lines[hi:])
    assert out != src
    if "positive_whole_number_error" in new:
        out = out.replace("from strands_robots.utils import (", "from strands_robots.utils import (\n    positive_whole_number_error,", 1)
    ast.parse(out)
    SRC.write_text(out)
    try:
        nf, npv = run([NEW])
        of, opv = run(["tests/simulation/isaac"], ["--ignore", NEW])
    finally:
        shutil.copy(SAVE, SRC)
    assert SRC.read_text() == SAVE.read_text(), "restore failed"
    rows.append((label, nf, of))
    print(f"    new-tests: {nf} failed   pre-existing: {of} failed"
          f"   -> {'CAUGHT' if nf else 'MISSED'} / {'caught' if of else 'BLIND'}\n")

print("=== SUMMARY ===")
print(f"{'mutation':56s} | new tests | pre-existing isaac suite")
for label, nf, of in rows:
    print(f"{label:56s} | {nf:>5} fail | {of:>5} fail {'' if of else '  <- BLIND'}")
print(f"\ncaught by new: {sum(1 for _, nf, _ in rows if nf)}/{len(rows)}   "
      f"caught by pre-existing: {sum(1 for _, _, of in rows if of)}/{len(rows)}")
json.dump([{"label": l, "new_failed": nf, "old_failed": of} for l, nf, of in rows],
          open(f"/tmp/mut-{RUN}.json", "w"), indent=2)

import ast, json, pathlib, shutil, subprocess, sys, tempfile

SRC = pathlib.Path("strands_robots/simulation/base.py")
FILES = ["tests/simulation/test_get_world_point_math.py", "tests/simulation/mujoco/test_get_world_point.py"]
NEW = [
    "test_base_facade_without_camera_params_is_a_structured_error",
    "test_the_missing_path_stubs_reproduce_the_base_defaults",
    "test_a_failed_camera_params_read_reports_its_reason",
    "test_the_two_backend_reads_are_distinguishable",
    "test_orthographic_free_camera_renders_but_reports_unreadable_intrinsics",
    "test_a_perspective_free_camera_on_the_same_scene_still_grounds",
]
K_NEW = " or ".join(NEW)
K_OLD = f"not ({K_NEW})"

# (label, enclosing function, old, new)
MUTATIONS = [
    ("M1 NIE arm no longer catches", "get_world_point",
     '            except NotImplementedError:\n'
     '                return _err(\n'
     '                    "get_world_point is unavailable: this backend has no camera-params path (get_camera_params is not implemented)."\n',
     '            except ZeroDivisionError:\n'
     '                return _err(\n'
     '                    "get_world_point is unavailable: this backend has no camera-params path (get_camera_params is not implemented)."\n'),
    ("M2 NIE report drops the method name", "get_world_point",
     '                    "get_world_point is unavailable: this backend has no camera-params path (get_camera_params is not implemented)."\n',
     '                    "get_world_point is unavailable: this backend has no camera-params path."\n'),
    ("M3 handled tuple narrowed to KeyError", "get_world_point",
     '            except (KeyError, ValueError, RuntimeError, TypeError) as e:\n'
     '                return _err(f"get_world_point failed to read camera parameters: {e}")\n',
     '            except KeyError as e:\n'
     '                return _err(f"get_world_point failed to read camera parameters: {e}")\n'),
    ("M4 params report drops the reason", "get_world_point",
     '                return _err(f"get_world_point failed to read camera parameters: {e}")\n',
     '                return _err("get_world_point failed to read camera parameters.")\n'),
    ("M5 params report copies the frame wording", "get_world_point",
     '                return _err(f"get_world_point failed to read camera parameters: {e}")\n',
     '                return _err(f"get_world_point failed to render camera frame: {e}")\n'),
    ("M6 base default stops naming the method", "get_camera_params",
     '        raise NotImplementedError("get_camera_params not implemented by this backend")\n',
     '        raise NotImplementedError("not implemented")\n'),
]

original = SRC.read_text()
backup = pathlib.Path(tempfile.mkdtemp()) / "base.py"
backup.write_text(original)
lines = original.splitlines(keepends=True)
tree = ast.parse(original)
ranges = {}
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef):
        ranges.setdefault(n.name, []).append((n.lineno, n.end_lineno))


def run(extra):
    cmd = [sys.executable, "-m", "pytest", *FILES, "-q", "-p", "no:randomly", "--no-header", "--no-cov", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"}).stdout
    import re
    f = re.search(r"(\d+) failed", out)
    return int(f.group(1)) if f else 0


rows = []
try:
    for label, fn, old, new in MUTATIONS:
        (lo, hi) = ranges[fn][0]
        region = "".join(lines[lo - 1:hi])
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: anchor in_fn={in_fn} (in_file={in_file})"
        mutated = region.replace(old, new, 1)
        SRC.write_text("".join(lines[:lo - 1]) + mutated + "".join(lines[hi:]))
        ast.parse(SRC.read_text())
        new_fail, old_fail = run(["-k", K_NEW]), run(["-k", K_OLD])
        rows.append((label, in_fn, in_file, new_fail, old_fail))
        print(f"  {label:42s} in_fn={in_fn} in_file={in_file}  new={new_fail:2d} failed | pre-existing={old_fail:2d} failed")
        SRC.write_text(original)
finally:
    shutil.copy(backup, SRC)
    assert SRC.read_text() == original, "RESTORE FAILED"

print("\n(unmutated control)")
print(f"  new cases: {run(['-k', K_NEW])} failed   pre-existing: {run(['-k', K_OLD])} failed")
caught_new = sum(1 for r in rows if r[3] > 0)
blind_old = sum(1 for r in rows if r[4] == 0)
print(f"\ncaught by the new cases: {caught_new}/{len(rows)}   invisible to the pre-existing: {blind_old}/{len(rows)}")
pathlib.Path(f"/tmp/mut-{__import__('os').environ['GITHUB_RUN_ID']}.json").write_text(json.dumps(rows, indent=2))
print("source restored byte-identical:", SRC.read_text() == original)

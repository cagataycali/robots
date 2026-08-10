"""Mutate the auto-resolve contract 4 ways; each must be caught by the new tests."""
import ast, pathlib, subprocess, sys

NEW = "tests/simulation/mujoco/test_motion_primitives.py tests/simulation/mujoco/test_primitive_teardown_abort.py tests/simulation/mujoco/test_set_gripper_setpoint_range_sources.py tests/simulation/mujoco/test_gripper_action_key_equivalence.py"
PRIM = pathlib.Path("strands_robots/simulation/mujoco/motion_primitives.py")
BASE = pathlib.Path("strands_robots/simulation/base.py")

def fn_range(path, name):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise SystemExit(f"no {name} in {path}")

def scoped_replace(path, fn, old, new):
    """Replace `old` with `new` inside fn only; assert exactly one hit there."""
    src = path.read_text()
    lo, hi = fn_range(path, fn)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"anchor appears {in_fn}x inside {fn} (in file: {in_file})"
    print(f"    anchor: in_fn={in_fn} in_file={in_file}")
    head, tail = "".join(lines[:lo - 1]), "".join(lines[hi:])
    out = head + region.replace(old, new, 1) + tail
    ast.parse(out)
    path.write_text(out)

MUTATIONS = [
    ("M1 the sole-robot resolution is removed entirely", PRIM, "_primitive_resolve_robot",
     """        if robot_name is None:
            try:
                robot_name = self._resolve_single_robot(None)
            except ValueError as e:
                return None, _err(str(e))
""", ""),
    ("M2 the ValueError is no longer converted to an envelope", PRIM, "_primitive_resolve_robot",
     """            except ValueError as e:
                return None, _err(str(e))
""", """            except ValueError:
                raise
"""),
    ("M3 an ambiguous scene silently resolves the first robot", BASE, "_resolve_single_robot",
     '        raise ValueError(f"Multiple robots registered; specify robot_name. Available: {names}")\n',
     "        return names[0]\n"),
    ("M4 the refusal no longer carries the reason", PRIM, "_primitive_resolve_robot",
     "                return None, _err(str(e))\n",
     '                return None, _err("motion primitive failed")\n'),
]

originals = {p: p.read_text() for p in (PRIM, BASE)}
rows = []
try:
    for label, path, fn, old, new in MUTATIONS:
        print(f"\n=== {label} ===")
        scoped_replace(path, fn, old, new)
        r = subprocess.run([sys.executable, "-m", "pytest", *NEW.split(), "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
                           capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"})
        tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
        verdict = tail[-1] if tail else "(no summary)"
        print(f"    {verdict}")
        rows.append((label, verdict))
        path.write_text(originals[path])
finally:
    for p, s in originals.items():
        p.write_text(s)

assert PRIM.read_text() == originals[PRIM] and BASE.read_text() == originals[BASE]
print("\nsources restored byte-identically")
print("\n=== MUTATION TABLE ===")
for label, verdict in rows:
    print(f"{label:60s} -> {verdict}")

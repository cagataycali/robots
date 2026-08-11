import ast
import pathlib
import subprocess
import sys

MINE = pathlib.Path.cwd()
GLBLOCK = f"/tmp/glblock-{__import__('os').environ['GITHUB_RUN_ID']}"

# (label, production file, target function, old, new, the retained GL-free test)
MUTATIONS = [
    (
        "M1 a non-string name resolves to a real entity instead of 'not found'",
        "strands_robots/simulation/mujoco/backend.py",
        "mj_name_to_id",
        "    if not isinstance(name, str):\n        return -1\n",
        "    if not isinstance(name, str):\n        return 0\n",
        "tests/simulation/mujoco/test_entity_name_lookup_type_safety.py::TestTheSessionSurvives",
    ),
    (
        "M2 a registered name stops resolving",
        "strands_robots/simulation/models.py",
        "registered",
        "    return isinstance(name, str) and name in registry\n",
        "    return False\n",
        "tests/simulation/test_unhashable_entity_name_is_reported.py::test_a_registered_name_is_unaffected",
    ),
    (
        "M3 the per-camera dimensions are dropped on the way to add_camera",
        "strands_robots/benchmarks/libero/adapter.py",
        "_install_libero_cameras",
        "                result = add_camera(name=cam_name, **cam_kwargs)\n",
        '                result = add_camera(\n'
        '                    name=cam_name, **{k: v for k, v in cam_kwargs.items() if k not in ("width", "height")}\n'
        "                )\n",
        "tests/benchmarks/libero/test_libero_camera_config_domain.py::TestOnTheRealMuJoCoBackend::test_a_usable_config_installs",
    ),
]


def fn_range(src: str, name: str) -> tuple[int, int]:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.lineno, node.end_lineno or node.lineno
    raise AssertionError(f"function {name} not found")


def run(target: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-cov", "-p", "no:randomly", "-p", "noglhost"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": GLBLOCK, "MUJOCO_GL": "egl"},
        cwd=MINE,
    )
    for line in reversed(proc.stdout.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip().strip("=").strip()
    return "no summary"


print("=== mutation check: with the production behaviour reverted, do the RETAINED")
print("=== GL-free assertions still fail?  (all runs on the emulated GL-free host)\n")
rows = []
for label, relpath, fname, old, new, target in MUTATIONS:
    path = pathlib.Path(relpath)
    src = path.read_text(encoding="utf-8")
    lo, hi = fn_range(src, fname)
    lines = src.splitlines(keepends=True)
    inside = "".join(lines[lo - 1 : hi])
    in_fn, in_file = inside.count(old), src.count(old)
    assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname} (file: {in_file})"
    print(f"  anchor scoping: in {fname}()={in_fn}  in whole file={in_file}")
    mutated = "".join(lines[: lo - 1]) + inside.replace(old, new, 1) + "".join(lines[hi:])
    ast.parse(mutated)
    try:
        path.write_text(mutated, encoding="utf-8")
        summary = run(target)
    finally:
        path.write_text(src, encoding="utf-8")
        assert path.read_text(encoding="utf-8") == src, f"{label}: restore failed"
    rows.append((label, target.split("::")[-1], summary))
    print(f"  {label}\n      -> {summary}\n")

print("=== control: the same retained tests on unmutated source ===")
for _, _, _, _, _, target in MUTATIONS:
    print(f"  {target.split('::')[-1]}: {run(target)}")

"""Reproduce scenarios A and B from the issue against the pristine tree."""
import ast, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path.cwd()
GUARD = "tests/test_mujoco_render_assertions_are_gl_gated.py"
IN_SCOPE = ROOT / "tests/simulation/mujoco/test_entity_name_lookup_type_safety.py"

# Scenario B payload: a SECOND requires_gl-gated render-success assertion, appended
# to a module ALREADY in EXPECTED_IN_SCOPE.  Exactly what the guard's own remedy
# text instructs a contributor to write.
SPLIT = '''

    @requires_gl
    def test_planted_second_gated_render(self, sim) -> None:
        assert sim.render(camera_name="default")["status"] == "success"
'''

# Scenario A payload: a NEW module, correctly gated.
NEW_MODULE = '''"""Planted: a new in-scope module, correctly gated."""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402


@requires_gl
def test_planted(sim) -> None:
    assert sim.render(camera_name="default")["status"] == "success"
'''


def run() -> str:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", GUARD, "-q", "--no-cov", "-p", "no:randomly"],
        capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"},
    )
    tail = [l for l in p.stdout.splitlines() if l.strip()]
    summary = [l for l in tail if "passed" in l or "failed" in l]
    asserts = [l.strip() for l in p.stdout.splitlines() if l.strip().startswith("E       assert")]
    return (summary[-1] if summary else "?") + "\n      " + "\n      ".join(asserts)


print("=== baseline ===")
print(run())

# --- Scenario B: split an assertion inside an in-scope module -----------------
src = IN_SCOPE.read_text(encoding="utf-8")
assert "test_planted_second_gated_render" not in src
# append inside the class that already carries the gated case
lines = src.splitlines(keepends=True)
tree = ast.parse(src)
target = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and any(
        isinstance(d, ast.Name) and d.id == "requires_gl" for d in node.decorator_list
    ):
        target = node
assert target is not None, "no requires_gl-decorated function found"
end = target.end_lineno
out = "".join(lines[:end]) + SPLIT + "".join(lines[end:])
assert ast.parse(out)
IN_SCOPE.write_text(out, encoding="utf-8")
try:
    print("\n=== scenario B: second gated assertion in an in-scope module ===")
    print(run())
finally:
    IN_SCOPE.write_text(src, encoding="utf-8")
assert IN_SCOPE.read_text(encoding="utf-8") == src

# --- Scenario A: a new in-scope module, correctly gated ----------------------
planted = ROOT / "tests/simulation/mujoco/test_zz_planted_new_module.py"
assert not planted.exists()
planted.write_text(NEW_MODULE, encoding="utf-8")
try:
    print("\n=== scenario A: a NEW module, correctly gated ===")
    print(run())
finally:
    planted.unlink()
assert not planted.exists()
print("\n=== tree restored ===")
print(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout or "(clean)")

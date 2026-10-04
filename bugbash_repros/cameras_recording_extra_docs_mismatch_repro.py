"""
Repro: docs/learn/data/record.md:45 claims
    "start_cameras_recording() ... runs under [sim-mujoco] alone"

But Isaac has the method too (strands_robots/simulation/isaac/simulation.py:8461),
whose own docstring explicitly says filenames match MuJoCo "so cross-backend
tooling finds Isaac videos the same way". Newton/mjlab correctly lack it.

A user who installs `strands-robots[sim-isaac]` reads record.md, sees the verb
scoped to mujoco, and either (a) installs sim-mujoco unnecessarily or
(b) assumes Isaac can't record MP4 and switches backends.

This repro uses pure AST inspection -- no Isaac runtime required -- to show
both halves of the mismatch exist in the shipped tree.
"""
from __future__ import annotations

import ast
import pathlib
import sys


def find_method(path: str, name: str) -> int | None:
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.lineno
    return None


def doc_line_match(path: str, needle: str) -> tuple[int, str] | None:
    for i, line in enumerate(pathlib.Path(path).read_text().splitlines(), start=1):
        if needle in line:
            return i, line
    return None


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]

    isaac_line = find_method(str(root / "strands_robots/simulation/isaac/simulation.py"), "start_cameras_recording")
    mj_line = find_method(str(root / "strands_robots/simulation/mujoco/rendering.py"), "start_cameras_recording")

    # Docs claim
    docs_hit = doc_line_match(str(root / "docs/learn/data/record.md"), "runs under `[sim-mujoco]` alone")

    print("CODE SIDE:")
    print(f"  mujoco  start_cameras_recording:  rendering.py:{mj_line}")
    print(f"  isaac   start_cameras_recording:  simulation.py:{isaac_line}")
    print()
    print("DOCS SIDE:")
    if docs_hit:
        d_lineno, d_text = docs_hit
        print(f"  record.md:{d_lineno}:  {d_text.strip()}")

    # Expected: Isaac should be either (a) listed in the extras clause, or
    # (b) scoped out explicitly. Neither holds -> defect reproduces.
    assert isaac_line is not None, "Isaac must expose start_cameras_recording"
    assert mj_line is not None,    "MuJoCo must expose start_cameras_recording"
    assert docs_hit is not None,   "Docs must contain the misleading claim"

    print()
    print("VERDICT: docs claim scopes verb to [sim-mujoco] alone, but Isaac ships it too.")
    print("         Scope the claim ('`[sim-mujoco]` or `[sim-isaac]`') or add it to the extras list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

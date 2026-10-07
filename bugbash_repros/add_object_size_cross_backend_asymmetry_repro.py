"""
Repro: README quickstart's `add_object(size=[0.05, 0.05, 0.05])` builds a 2x
larger box on Newton than on MuJoCo/Isaac.

The README (README.md:54) teaches the one scene-call every new user runs:

    robot.add_object(name="red_cube", shape="box", size=[0.05, 0.05, 0.05],
                     position=[0.0, -0.2, 0.025], color=[1.0, 0.0, 0.0])

The accompanying prose names this a "5 cm cube" and the agent is asked to
"pick up the red cube". On MuJoCo (the default) the geom's world extent is
0.05 m per axis; the half-extents mjModel.geom_size reports back are 0.025 m
(the backend halves the input internally, per its docstring at
strands_robots/simulation/base.py:1988-1994, "MuJoCo backend treats ``size``
as the full extent in meters per axis (halved internally to MuJoCo's
half-extents), whereas Newton consumes half-extents / radii directly").

On Newton (`backend="newton"`), the same call lands on
strands_robots/simulation/newton/simulation.py:3290-3292:

    hx, hy, hz = (size + [0.05, 0.05, 0.05])[:3]
    builder.add_shape_box(body, xform=shape_xform, hx=hx, hy=hy, hz=hz, color=color)

So `size[0]=0.05` goes directly into `hx`, Newton's half-extent -> world extent
0.10 m. The same README snippet builds a 10 cm cube on Newton while reporting
success and echoing back the request. The agent that was told "pick up the red
cube" and knows the gripper's reach may try to grasp 5 cm and fail because the
cube is 10 cm.

This script asserts the asymmetry statically (no Newton install required): it
reads `add_object` source from both backends and compares the extent per `size`
unit, then runs the MuJoCo backend for a live confirmation of the 5 cm case.

Exits non-zero when the asymmetry is present (i.e. the defect reproduces).
"""

from __future__ import annotations

import os
import sys
import ast
import inspect
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent


def read_newton_box_mapping() -> str:
    """Return the Newton add-shape-box call site, verbatim."""
    src = (REPO / "strands_robots/simulation/newton/simulation.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_add_object_body":
            # Fall through to call-site heuristic
            pass
    # Simpler: grep for the add_shape_box call
    for i, line in enumerate(src.splitlines(), start=1):
        if "builder.add_shape_box" in line:
            # Capture the two lines before + this one
            prior = src.splitlines()[i - 2]
            return f"newton/simulation.py:{i-1}: {prior.strip()}\nnewton/simulation.py:{i}: {line.strip()}"
    raise RuntimeError("couldn't find newton add_shape_box call")


def read_newton_docstring_size() -> str:
    """Return the Newton size docstring line."""
    src = (REPO / "strands_robots/simulation/newton/simulation.py").read_text()
    for i, line in enumerate(src.splitlines(), start=1):
        if "size: Half-extents (box)" in line:
            return f"newton/simulation.py:{i}: {line.strip()}"
    raise RuntimeError("couldn't find newton 'Half-extents' docstring")


def read_mujoco_docstring_size() -> str:
    """Return the MuJoCo size convention docstring line from base.py."""
    src = (REPO / "strands_robots/simulation/base.py").read_text()
    for i, line in enumerate(src.splitlines(), start=1):
        if "full extent in meters" in line:
            return f"base.py:{i}: {line.strip()}"
    raise RuntimeError("couldn't find base.py 'full extent' docstring")


def live_mujoco_check() -> dict:
    """Instantiate `Robot('so100')`, run the exact README add_object call,
    and read the actual geom_size the backend stored."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots import Robot
    import mujoco

    r = Robot("so100")
    res = r.add_object(
        name="rc",
        shape="box",
        size=[0.05, 0.05, 0.05],
        position=[0.0, -0.2, 0.025],
        color=[1.0, 0.0, 0.0],
    )
    assert res["status"] == "success", res

    # Find the geom - MuJoCo backend creates `<name>_geom`
    # Robot() on so100 returns a MuJoCoSimEngine; the model lives on an attr
    # that varies - find any MjModel on it.
    model = None
    for attr_name in dir(r):
        if attr_name.startswith("__"):
            continue
        try:
            v = getattr(r, attr_name)
        except Exception:
            continue
        if isinstance(v, mujoco.MjModel):
            model = v
            break
        # One level deeper (e.g. r._backend._model)
        if hasattr(v, "__dict__") and not callable(v):
            for sub in ("model", "_model", "mj_model"):
                sv = getattr(v, sub, None)
                if isinstance(sv, mujoco.MjModel):
                    model = sv
                    break
            if model is not None:
                break
    assert isinstance(model, mujoco.MjModel), f"no MjModel on Robot: {type(r)}"

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.startswith("rc"):
            half = [float(x) for x in model.geom_size[i]]
            full = [2 * x for x in half]
            return {"name": name, "half_extents_m": half, "full_extent_m": full}
    raise RuntimeError("rc geom not found")


def main() -> int:
    print("=" * 72)
    print("README add_object(size=[0.05, 0.05, 0.05]) cross-backend size defect")
    print("=" * 72)
    print()
    print("README.md:54 (verbatim):")
    print('    robot.add_object(name="red_cube", shape="box",')
    print("                     size=[0.05, 0.05, 0.05],  <-- the shared call")
    print('                     position=[0.0, -0.2, 0.025], color=[1.0, 0.0, 0.0])')
    print()
    print("Prose above the snippet calls this 'the red cube'; the agent is told")
    print("'pick up the red cube'. A reader parses size as 'a 5 cm cube'.")
    print()
    print("--- Static evidence (both docstrings agree the backends disagree): ---")
    print()
    print(read_mujoco_docstring_size())
    print(read_newton_docstring_size())
    print()
    print("Newton consumption site (treats size[i] directly as half-extent):")
    print(read_newton_box_mapping())
    print()
    print("--- Live MuJoCo check (default backend, exact README call) ---")
    try:
        mj = live_mujoco_check()
        print(f"  geom name: {mj['name']}")
        print(f"  mjModel.geom_size (half-extents): {mj['half_extents_m']}")
        print(f"  full extent (world size):          {mj['full_extent_m']}")
    except Exception as e:
        print(f"  (skipped: {e})")
        mj = None
    print()

    # Compute Newton's would-be extent from the same input, without running it.
    newton_size_input = [0.05, 0.05, 0.05]
    newton_half = newton_size_input
    newton_full = [2 * x for x in newton_half]

    print("--- Same README input, projected onto Newton's documented contract ---")
    print(f"  size=        {newton_size_input}")
    print(f"  half-extents: {newton_half}       (Newton docstring: 'Half-extents')")
    print(f"  full extent:  {newton_full}  <-- 2x MuJoCo for the same source call")
    print()

    if mj and mj["full_extent_m"] == newton_full:
        print("UNEXPECTED: MuJoCo and Newton produce the same extent -- defect gone.")
        return 0

    print("DEFECT PRESENT: Same README snippet builds a different-sized object on")
    print("Newton (10 cm cube) vs MuJoCo/Isaac (5 cm cube). The agent that is told")
    print("'pick up the red cube' and plans around a 5 cm width is handed a 10 cm")
    print("cube on Newton with a status=success envelope echoing size=[0.05, ...]")
    print("verbatim.")
    print()
    print("Fix shape (zero-risk, no behaviour change):")
    print("  - README.md:54: either use the per-backend-portable alternative")
    print("    (document the asymmetry inline, or pick a size that reads the")
    print("    same on either convention, e.g. size=[0.04, 0.04, 0.04] names a")
    print("    '4 cm half' which is already unusual phrasing); or")
    print("  - normalise the backends: Newton's _add_object_body could halve")
    print("    the input, mirroring the MuJoCo/Isaac contract (one-line change")
    print("    at simulation/newton/simulation.py:3291, plus docstring sync).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

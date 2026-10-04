"""Minimal repro: ``add_object(color=...)`` promises ``0..1`` but accepts anything.

Harness: cagataycali/robots-harness#NEW
Upstream: strands-labs/robots

Context
-------
The MuJoCo/Newton/Isaac ``add_object`` docstrings all state:

    color: ``[r, g, b]`` or ``[r, g, b, a]`` in 0..1 (default mid-grey).

``coerce_rgba`` in ``strands_robots/utils.py:1610`` enforces 7 clauses on
``color``: non-None, is-sequence, finite, numeric, no-bool, no-NaN/inf,
count in {3,4}. It does **NOT** enforce the ``0..1`` range the three
backends all document. The sibling ``_validate_mass`` guard
(``base.py:2763``) enforces its documented ``(0, inf)`` range for an
analogous "docstring says X" / "silent-wrong past X" risk.

Three unusable colours - a 0..255 scale (the single most common RGB
convention outside OpenGL), negatives, and 1e9 - all return
``status=success`` and land in ``model.geom_rgba`` verbatim. MuJoCo's
renderer clamps on draw but the stored contract is already violated:
``get_scene_description``, dataset recording's ``randomize_colors`` lookup,
and ``set_geom_properties`` echoes all see the raw out-of-range row.

The README's headline snippet teaches ``color=[1.0, 0.0, 0.0]`` - the
exact format a new user flips to ``color=[255, 0, 0]`` when they transcribe
from a web-RGB design doc, which is why this hits the quickstart.

Expected behaviour
------------------
Either refuse the out-of-range colour with a message that cites the
documented ``0..1`` interval (symmetric with ``_validate_mass`` /
``finite_number_error``), **or** change the three docstrings so they say
what the code actually does.

Reproduces: 2026-10-04 (v0.5.3 bugbash fire #46)
Upstream HEAD: 609fdff
"""

from __future__ import annotations

import os
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from strands_robots import Robot


FAIL = False


def record(label: str, result: dict[str, Any], expected_status: str) -> None:
    global FAIL
    status = result.get("status")
    ok = status == expected_status
    FAIL = FAIL or not ok
    marker = "PASS" if ok else "FAIL"
    text = result.get("content", [{}])[0].get("text", "")
    print(f"[{marker}] {label}: status={status!r}  text={text[:120]!r}")


def _rgba_of(robot: Robot, geom_name: str) -> list[float] | None:
    model = robot.mj_model
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return None
    return list(model.geom_rgba[gid])


def main() -> int:
    print("--- v0.5.3 add_object color 0..1 contract (docstring vs code) ---\n")
    robot = Robot("so100")

    print("Documented contract: color must be in 0..1 "
          "(mujoco/simulation.py:4880, newton/simulation.py:856, isaac/simulation.py:4034)\n")

    print("1. [255, 0, 0] - the 0..255 scale a new user coming from web RGB would write:")
    r = robot.add_object(name="scale_255", shape="box", size=[0.03] * 3,
                        position=[0.0, -0.2, 0.025], color=[255, 0, 0])
    record("add_object color=[255,0,0]", r, expected_status="error")
    print(f"       stored rgba: {_rgba_of(robot, 'scale_255_geom')}")

    print("\n2. [-1.0, -1.0, 0.0] - a user who negated the wrong axis:")
    r = robot.add_object(name="negative", shape="box", size=[0.03] * 3,
                        position=[0.1, -0.2, 0.025], color=[-1.0, -1.0, 0.0])
    record("add_object color=[-1,-1,0]", r, expected_status="error")
    print(f"       stored rgba: {_rgba_of(robot, 'negative_geom')}")

    print("\n3. [1e9, 1e9, 1e9] - a scalar that is finite yet absurd:")
    r = robot.add_object(name="huge", shape="box", size=[0.03] * 3,
                        position=[0.2, -0.2, 0.025], color=[1e9, 1e9, 1e9])
    record("add_object color=[1e9,1e9,1e9]", r, expected_status="error")
    print(f"       stored rgba: {_rgba_of(robot, 'huge_geom')}")

    print("\n4. [1.0, 0.0, 0.0] - the README's own example, control case:")
    r = robot.add_object(name="good", shape="box", size=[0.03] * 3,
                        position=[0.3, -0.2, 0.025], color=[1.0, 0.0, 0.0])
    record("add_object color=[1,0,0]", r, expected_status="success")
    print(f"       stored rgba: {_rgba_of(robot, 'good_geom')}")

    print("\nSibling parity probe: _validate_mass does enforce its documented range")
    r = robot.add_object(name="bad_mass", shape="box", size=[0.03] * 3,
                        position=[0.4, -0.2, 0.025], color=[1.0, 0.0, 0.0], mass=-1.0)
    record("add_object mass=-1.0", r, expected_status="error")

    print()
    if FAIL:
        print("REPRO: color range contract claimed in docstring is not enforced.")
        print("coerce_rgba() in strands_robots/utils.py:1610 refuses NaN/inf/bool/wrong count")
        print("but accepts any finite number including 255, -1.0, and 1e9.")
        print("The three storage sites (newton: dict, mujoco: geom_rgba, isaac: prim)")
        print("all keep the out-of-range row.")
        return 1
    print("All refusals surfaced correctly - defect fixed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

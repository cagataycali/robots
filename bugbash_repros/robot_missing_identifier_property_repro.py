"""
Robot("so100") returns a MuJoCoSimEngine with NO stable, public property that
tells the user which robot they got.

End-user path: a user follows README.md's hero snippet verbatim

    from strands_robots import Robot
    robot = Robot("so100")           # ← user typed "so100"

and then asks "what did I just build?" The three most natural Python
reflexes all fail or mislead:

  1. `robot.name`           → AttributeError "no attribute 'name'"
  2. `robot.robot_name`     → AttributeError "no attribute 'robot_name'"
  3. `robot.tool_name`      → returns "so100_sim" (NOT "so100")

The only public, name-shaped data attribute on the engine is
`tool_name` (property) and its backing field `tool_name_str`. Both
carry the Strands-agent tool name, which the sim-path mutates to
"<name>_sim". The user-typed identifier (and the string every public
method of the engine expects back) is only recoverable via
`robot.list_robots()[0]` or the leading-underscore `_world.robots`.

Yet the identifier the user typed, "so100", is the string the sim's own
public API requires:

    robot.robot_joint_names("so100")      # OK
    robot.robot_joint_names("so100_sim")  # 404 — "sim" is not a registered robot
    robot.list_robots()                   # ['so100']  ← source of truth

The only way to recover the user's input is `robot.list_robots()[0]` or
`robot._world.robots`, a leading-underscore private attribute, or
`robot.tool_name_str.removesuffix("_sim")` — a string-munge pattern the
docs never mention.

HardwareRobot is symmetric for `tool_name` (there `tool_name == name`), so
the ergonomic trap is sim-specific; the "Did you mean: tool_name?" hint
Python produces becomes actively wrong for the sim path.

Impact on the README-quickstart user:

* An LLM-driven agent that attempts `robot.robot_name` to decide which
  checkpoint to pull burns a round-trip on the AttributeError, follows the
  "tool_name" hint, binds "so100_sim" to a downstream API that expected
  "so100", and the second call fails too — this time with a 404 that
  blames the user for mis-spelling a robot name.
* A human reader of the README has no documented way to answer "what's
  the canonical name of this robot?" without reading simulation/base.py.

A ≤ 15-LOC fix adds a backend-shared `robot_name` property that returns
the canonical identifier (`"so100"`), and a `__repr__` that names the
engine kind, the robot name, and the mode so `repr(robot)` is useful.

This repro is end-user path only: it imports `strands_robots.Robot` and
reads public attributes. No simulation.base internals.
"""

from __future__ import annotations

import os
import sys
import traceback

# README quickstart uses default MUJOCO_GL; the EGL hint is for headless hosts.
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> int:
    from strands_robots import Robot

    print("# README quickstart: robot = Robot('so100')")
    robot = Robot("so100")

    # ------------------------------------------------------------------ #
    # Claim 1: `robot.name` is not defined.
    # ------------------------------------------------------------------ #
    print("\n[1] robot.name")
    try:
        _ = robot.name  # type: ignore[attr-defined]
        print("    OK (unexpected)")
    except AttributeError as e:
        print(f"    AttributeError: {e}")

    # ------------------------------------------------------------------ #
    # Claim 2: `robot.robot_name` is not defined.
    # ------------------------------------------------------------------ #
    print("\n[2] robot.robot_name")
    try:
        _ = robot.robot_name  # type: ignore[attr-defined]
        print("    OK (unexpected)")
    except AttributeError as e:
        print(f"    AttributeError: {e}")

    # The attribute surface a user discovers via dir() offers no "give me
    # the robot's identifier" property. `tool_name` is the only public
    # name-shaped attribute, and it is the Strands tool name — which the
    # sim path suffixes with '_sim' (step [3] confirms the asymmetry).
    name_shaped_public = sorted(
        a for a in dir(robot)
        if ("name" in a) and not a.startswith("_") and not callable(getattr(robot, a, None))
    )
    print(f"    name-shaped public (data) attrs: {name_shaped_public}")

    # ------------------------------------------------------------------ #
    # Claim 3: `robot.tool_name` returns "so100_sim" — NOT the string
    # the user typed, and NOT the string the sim's public API expects.
    # ------------------------------------------------------------------ #
    print("\n[3] robot.tool_name vs list_robots()")
    tool_name = robot.tool_name
    registered = robot.list_robots()
    print(f"    robot.tool_name       = {tool_name!r}")
    print(f"    robot.list_robots()   = {registered!r}")
    print(f"    user typed            = 'so100'")
    assert tool_name == "so100_sim", (
        "pin: sim adds '_sim' suffix; a fix MUST keep tool_name stable "
        "and add a separate robot_name property"
    )
    assert registered == ["so100"], (
        "pin: list_robots() returns the user-typed identifier"
    )
    assert tool_name != registered[0], (
        "the asymmetry this defect is about: tool_name != list_robots()[0]"
    )

    # ------------------------------------------------------------------ #
    # Claim 4: the asymmetry breaks a copy-paste from the sim's own
    # public API. `robot_joint_names(name)` accepts "so100", NOT
    # "so100_sim", so a user who follows Python's "Did you mean:
    # tool_name" hint hits a second 404.
    # ------------------------------------------------------------------ #
    print("\n[4] robot_joint_names(robot.tool_name) — the natural wrong call")
    try:
        labels = robot.robot_joint_names(robot.tool_name)
        print(f"    OK: {labels}")
    except Exception as e:
        print(f"    {type(e).__name__}: {e}")
        # Does the refusal cite 'list_robots()' or point the user at the
        # right source? If not, the second 404 compounds the first.

    # Verify the right form works.
    labels = robot.robot_joint_names("so100")
    print(f"    robot_joint_names('so100') = {labels}")

    # ------------------------------------------------------------------ #
    # Claim 5: `__repr__` tells the user nothing.
    # ------------------------------------------------------------------ #
    print("\n[5] repr(robot)")
    r = repr(robot)
    print(f"    {r}")
    assert "so100" not in r, "pin: repr() carries no robot name today"

    try:
        robot.cleanup()
    except Exception:  # noqa: BLE001 -- cleanup is best-effort for the repro
        pass

    # ------------------------------------------------------------------ #
    # Pin the fix contract. These are the invariants a 15-LOC patch
    # should flip from `False` to `True`.
    # ------------------------------------------------------------------ #
    print("\n# After-fix invariants (currently FAIL, should PASS):")
    robot = Robot("so100")
    checks = {
        "robot.robot_name == 'so100'": getattr(robot, "robot_name", None) == "so100",
        "'so100' in repr(robot)": "so100" in repr(robot),
        "robot.tool_name still == 'so100_sim'": robot.tool_name == "so100_sim",
        "robot.list_robots() still == ['so100']": robot.list_robots() == ["so100"],
    }
    all_pass = all(checks.values())
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}: {k}")
    try:
        robot.cleanup()
    except Exception:
        pass

    # Exit 1 while the bug stands; 0 after the fix lands.
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)

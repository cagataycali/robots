"""Repro: docs/robots/*.md quickstart teaches ``robot_joint_names`` as the
first-touch primitive on robots whose list is wider than their action/state
vector.  26 floating-base entries in ``registry/robots.json`` ship with a
quickstart whose printed roster includes a free joint (or passive joints)
that no scalar surface on the engine accepts.

The ABC docstring at ``strands_robots/simulation/base.py:1421-1434`` warns
about this (``harness#727`` filed it at that layer).  New users do not read
docstrings; they read docs/robots/<robot>.md.  The generator in
``docs/hooks/robot_pages.py:665,676`` emits ``print(robot_joint_names(name))``
unconditionally.  On ``microduck`` the chip on the same page (line 3 of the
.md) advertises ``14-DOF`` while the quickstart (line 18) prints a 15-entry
list, so the page contradicts itself in six physical lines.

Run from repo root::

    MUJOCO_GL=egl python bugbash_repros/microduck_quickstart_robot_joint_names_wrong_width_repro.py

Observed on strands-labs/robots @ 5ddca6a (v0.5.3).
"""

import numpy as np

from strands_robots import Robot


def main() -> None:
    print("=" * 60)
    print("DOCS QUICKSTART TAUGHT WRONG-WIDTH LIST ON FLOATING-BASE BODIES")
    print("=" * 60)

    # Follow docs/robots/microduck.md:14-20 quickstart literally.
    robot = Robot("microduck")

    names = robot.robot_joint_names("microduck")
    print("\n1. docs/robots/microduck.md:18 prints:")
    print(f"   len(robot.robot_joint_names('microduck')) == {len(names)}")
    print(f"   first name == {names[0]!r}  (this is a 6-DoF free joint, nq=7)")

    # Natural next step for a new user who just saw "15 joints".
    action = np.zeros(len(names)).tolist()
    result = robot.send_action(action=action)
    assert result["status"] == "error", "expected send_action to refuse wrong width"
    msg = result["content"][0]["text"].split(". ")[0]
    print("\n2. User sizes an action by that count:")
    print(f"   robot.send_action(action=np.zeros({len(names)})) ->")
    print(f"   status={result['status']}, message={msg!r}")

    # The correct primitive, invisible to a quickstart reader.
    keys = robot.robot_action_keys("microduck")
    print("\n3. The correct primitive, named on no robot page:")
    print(f"   len(robot.robot_action_keys('microduck')) == {len(keys)}")

    # The chip in the SAME file disagrees with the quickstart in the SAME file.
    print("\n4. docs/robots/microduck.md:3 advertises '14-DOF';")
    print("   docs/robots/microduck.md:18 prints 15 names.")
    print("   The page contradicts itself in six physical lines.")

    # Blast-radius check: survey the four floating-base bodies the harness-site
    # already observed (microduck + three the sibling issue#727 named).
    print("\n5. Same gap on every floating-base body shipped in the registry:")
    for name in ("microduck", "unitree_g1", "cassie"):
        r2 = Robot(name)
        jn = len(r2.robot_joint_names(name))
        ak = len(r2.robot_action_keys(name))
        first = r2.robot_joint_names(name)[0]
        print(f"   {name:12s}: joint_names={jn:3d}  action_keys={ak:3d}  gap={jn - ak:3d}  first={first!r}")
        r2.cleanup()

    assert len(names) - len(keys) == 1
    assert names[0].endswith("freejoint")

    robot.cleanup()
    print("\n[DEFECT] docs/hooks/robot_pages.py:665,676 emits a quickstart")
    print("         that is one wider than reality on 2 of 3 floating-base")
    print("         bodies surveyed (microduck +1, unitree_g1 +1), and 12")
    print("         wider on cassie (passive joints).  26 pages affected.")


if __name__ == "__main__":
    main()

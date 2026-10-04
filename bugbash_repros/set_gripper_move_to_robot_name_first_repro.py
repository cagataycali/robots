"""Minimal repro: set_gripper() and move_to() put `robot_name` BEFORE the payload.

Siblings on the same Robot handle keep payload first:
  send_action(action, robot_name=None, ...)              # payload first ✓
  set_joint_positions(positions, robot_name=None, ...)   # payload first ✓

But two motion primitives invert it:
  set_gripper(robot_name=None, state=None, steps=12)     # robot_name first ✗
  move_to(robot_name=None, position=None, ...)           # robot_name first ✗

So the natural positional calls that the module's own docstrings use —
    set_gripper("close") -> move_to([x, y, z])
  (strands_robots/simulation/mujoco/motion_primitives.py:501
   strands_robots/simulation/isaac/motion_primitives.py:780)
— silently misbind: the payload is eaten by `robot_name`, the real parameter
is left as None, and `set_gripper` then lies about what the user passed:

    "set_gripper: 'state' must be \"open\" or \"close\", got None."

The user did not pass None; they passed "close". The error points at the
wrong word.

Run:
    MUJOCO_GL=egl python bugbash_repros/set_gripper_move_to_robot_name_first_repro.py

Expected (if the signature matched its siblings or the error knew about the
misbind): either the call succeeds, or the refusal names *the robot_name
positional eating the payload*, so the user flips to the correct keyword
form instead of chasing a phantom None.

Actual on v0.5.3-dev (commit bb61ecb):
    set_gripper("open")              → refusal says got None  (LIE)
    set_gripper("close")             → refusal says got None  (LIE)
    set_gripper(0.5)                 → refusal says got None  (LIE, numeric too)
    move_to([0.2, 0.0, 0.1])         → refusal says "requires 'position'"
                                       (truthful but no "robot_name ate your
                                        list" hint — still a silent misbind)

Upstream file:line:
  strands_robots/simulation/mujoco/motion_primitives.py:1130  (set_gripper)
  strands_robots/simulation/mujoco/motion_primitives.py:470   (move_to)
  strands_robots/simulation/isaac/motion_primitives.py:1188   (set_gripper)
  strands_robots/simulation/isaac/motion_primitives.py:717    (move_to)
  strands_robots/simulation/motion_primitives_base.py:240     (lying refusal)

For comparison (payload-first siblings on the SAME Robot):
  strands_robots/simulation/base.py (send_action)
  strands_robots/simulation/physics.py (set_joint_positions)
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")

    from strands_robots import Robot

    r = Robot("so101", mesh=False)

    def _show(label: str, out: dict) -> None:
        status = out.get("status")
        text = ""
        content = out.get("content", [])
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text", "")
        print(f"  {label:<42} status={status!r:<10} text={text!r}")

    print("# set_gripper — docstring says `set_gripper(\"close\")`, user tries exactly that")
    _show("set_gripper('open')  (positional)",  r.set_gripper("open"))
    _show("set_gripper('close') (positional)",  r.set_gripper("close"))
    _show("set_gripper(0.5)     (positional)",  r.set_gripper(0.5))

    print("\n# move_to — natural positional usage from any IK tutorial")
    _show("move_to([0.2, 0.0, 0.1]) (positional)", r.move_to([0.2, 0.0, 0.1]))

    print("\n# What actually works (keyword form) — payload second, robot_name omitted")
    _show("set_gripper(state='open')               ", r.set_gripper(state="open"))
    _show("move_to(position=[0.2, 0.0, 0.1])       ", r.move_to(position=[0.2, 0.0, 0.1]))

    print("\n# Sibling siblings keep payload FIRST (so positional *is* idiomatic there)")
    _show("send_action({'gripper': 0.5}) (positional)", r.send_action({"gripper": 0.5}))
    _show("set_joint_positions([0]*6)  (positional)",   r.set_joint_positions([0.0] * 6))

    print(
        "\nDefect summary: set_gripper + move_to put `robot_name` first, so the only\n"
        "two positional primitives a user ever reaches for from a tutorial misbind\n"
        "their payload. set_gripper then reports `got None` — a lie about what the\n"
        "user typed. The module's own docstrings use the broken positional form\n"
        "(mujoco/motion_primitives.py:501, isaac/motion_primitives.py:780)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

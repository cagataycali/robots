"""Repro for silent out-of-range clamp on send_action (status=success lies).

Target: so101 sim quickstart (docs/robots/so101.md, docs/start/first-robot.md).

What the docs say:
  docs/robots/so101.md:40
    > Gripper `6`: the low end closes, the high end opens.
  (no numeric range is published anywhere on a quickstart page)

What send_action promises (strands_robots/simulation/base.py:2492 `send_action`):
    "A batch is applied whole or not at all: when any action key cannot be
     resolved, nothing is written... status is 'error' when n_substeps is
     outside its domain."

What send_action actually does when the VALUE (not the key) is out of range:
    rendering.py:1231 `_warn_ctrl_clamp` docstring (verbatim):
      "the commanded trajectory is silently NOT reproduced for that actuator
       while the call still reports success."

So the signal path is:
  stderr WARNING once per (prefix,key) → but the public API return is
  {status: "success", content: [...]}. A user who follows the docs and tries
  gripper=-1.0 (plausible normalized convention) sees "success" and never
  learns the arm did not reach the target.

Run:
    cd <repo-root>
    MUJOCO_GL=egl python bugbash_repros/so101_silent_clamp_repro.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

TOL = 0.05  # radians


def ran_the_command(commanded: float, measured: float) -> bool:
    return abs(commanded - measured) <= TOL


def one(robot: Robot, key: str, value: float, joint_idx: str) -> None:
    robot.reset()
    envelope = robot.send_action({key: value}, n_substeps=200)
    state = robot.get_robot_state()["content"][1]["json"]["state"]
    measured = state[joint_idx]["position"]
    status = envelope.get("status")
    text = envelope.get("content", [{}])[0].get("text", "")
    reproduced = ran_the_command(value, measured)
    print(
        f"send_action({key!r}={value:+.2f}) -> status={status!r} "
        f"measured={measured:+.4f}  reproduced_within_{TOL}_rad={reproduced}"
    )
    print(f"    text: {text}")
    if status == "success" and not reproduced:
        print(
            f"    ^^^ DEFECT: envelope claims success, measurement disagrees "
            f"(stderr emitted a one-shot warning that the Python caller never sees)."
        )


def main() -> int:
    robot = Robot("so101")

    print("=== so101 silent clamp repro ===\n")

    # 1. Gripper negative — the docs say "low end closes" but publish no numeric
    #    low, so a user trying -1.0 is reasonable; it is in fact 5.7x the
    #    actual joint lower limit of ~-0.1745 rad.
    one(robot, "gripper", -1.0, "6")
    print()

    # 2. Gripper above upper joint limit (~1.745 rad).
    one(robot, "gripper", 2.0, "6")
    print()

    # 3. shoulder_pan an order of magnitude past the actuator's joint range
    #    (~[-1.92, 1.92] rad).
    one(robot, "shoulder_pan", 10.0, "1")
    print()

    # 4. Control case — a value well within the ctrlrange: should be reproduced
    #    AND envelope should remain success. This is the one true success
    #    today; the defect is only that the three above also return success.
    one(robot, "shoulder_pan", 0.5, "1")

    robot.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())

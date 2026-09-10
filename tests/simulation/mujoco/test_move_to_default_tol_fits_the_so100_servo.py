"""``move_to``'s default tolerance is one the bundled so100 servos can settle to.

Measured (deepdive D-058): the agent's first motion on ``Robot("so100")`` -
end-effector +0.10 m straight up, a reachable point - returned ``did not reach
... within tol=0.01 m after max_steps=200 (residual 0.0105 m; IK residual was
0.0085 m)``; the retry with ``tol=0.015`` "reached ... in 1 steps" because the
arm was already there. One wasted LLM turn per motion for a 0.5 mm miss. The
default is now 0.015 m, where those servos settle inside the step budget.
"""

from __future__ import annotations

import importlib.util
import inspect

import pytest

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")


def test_default_tol_is_0_015() -> None:
    from strands_robots.simulation.mujoco.motion_primitives import MotionPrimitivesMixin

    assert inspect.signature(MotionPrimitivesMixin.move_to).parameters["tol"].default == 0.015


@requires_mujoco
def test_so100_reaches_ten_cm_up_with_the_default_tol() -> None:
    from strands_robots import Robot

    sim = Robot("so100")
    try:
        ee = sim.get_body_state(body_name="so100/Wrist_Pitch_Roll")["content"][1]["json"]["position"]
        target = [ee[0], ee[1], ee[2] + 0.10]
        res = sim.move_to(robot_name="so100", position=target)
        assert res["status"] == "success", res["content"][0]["text"]
    finally:
        sim.destroy()

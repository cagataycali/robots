"""Regression tests: ``zero_dynamics`` clears what its result claims.

Two defects, both of which reported ``status="success"`` with text asserting the
buffers were cleared:

1. **``qacc`` was repopulated before the method returned.** The trailing
   ``mj_forward`` runs the full forward-dynamics pipeline, which recomputes
   ``data.qacc`` from the current state - undoing the zeroing while the success
   text still said ``qacc``. Measured 49.14 rad/s^2 on a Panda after a 100-step
   run. That is not cosmetic: ``inverse_dynamics`` reads ``data.qacc`` as the
   *desired* acceleration (see its docstring), so the documented
   "teleport -> zero_dynamics -> inverse_dynamics" recipe solved for a bogus
   target.

2. **The robot-scoped path skipped the floating base.** ``robot.joint_ids`` holds
   only articulated joints; a floating base's ``<freejoint>`` is resolved
   separately. So ``zero_dynamics(robot_name=...)`` left the base's 6 DOFs at
   their pre-teleport velocity - exactly the DOFs whose discontinuity causes the
   "QACC nan" divergence this method exists to prevent. A Go2 with ``qvel=7``
   everywhere kept ``[7,7,7]`` linear and ``[7,7,7]`` angular while the result
   claimed "12 DOFs ... zeroed" (of 18).
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def arm():
    s = Simulation(tool_name="zero_dynamics_arm", mesh=False)
    s.create_world()
    s.add_robot(name="panda")
    yield s
    s.destroy()


@pytest.fixture
def quadruped():
    s = Simulation(tool_name="zero_dynamics_go2", mesh=False)
    s.create_world()
    if s.add_robot(name="go2", data_config="unitree_go2")["status"] != "success":
        s.destroy()
        pytest.skip("unitree_go2 asset unavailable")
    yield s
    s.destroy()


def test_qacc_is_actually_zero_after_the_call(arm) -> None:
    """The buffer the result names must really be cleared when it returns."""
    model, data = arm.mj_model, arm.mj_data
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.abs(data.qacc).max() > 1.0, "fixture must build up a real acceleration"

    result = arm.zero_dynamics()
    assert result["status"] == "success"
    assert "qacc" in result["content"][0]["text"]

    data = arm.mj_data
    assert np.abs(data.qvel).max() == pytest.approx(0.0)
    assert np.abs(data.qacc).max() == pytest.approx(0.0)  # pre-fix: 49.14
    assert np.abs(data.qacc_warmstart).max() == pytest.approx(0.0)


def test_scoped_call_zeroes_the_floating_base(quadruped) -> None:
    """A robot-scoped call must include the base freejoint's 6 DOFs."""
    model, data = quadruped.mj_model, quadruped.mj_data
    data.qvel[:] = 7.0
    mujoco.mj_forward(model, data)

    result = quadruped.zero_dynamics(robot_name="go2")
    assert result["status"] == "success"

    data = quadruped.mj_data
    still_moving = [i for i in range(model.nv) if abs(float(data.qvel[i])) > 1e-9]
    # Pre-fix: DOFs [0..5] (the base) were left at 7.0.
    assert still_moving == [], f"DOFs still moving: {still_moving}"


def test_scoped_call_reports_the_dof_count_it_zeroed(quadruped) -> None:
    """The reported scope must match the DOFs actually cleared."""
    model = quadruped.mj_model
    data = quadruped.mj_data
    data.qvel[:] = 3.0
    mujoco.mj_forward(model, data)

    text = quadruped.zero_dynamics(robot_name="go2")["content"][0]["text"]
    # A Go2 is a floating base plus 12 leg joints: 6 + 12 == nv.
    assert f"{model.nv} DOFs" in text, text


def test_unscoped_call_zeroes_every_dof(quadruped) -> None:
    model, data = quadruped.mj_model, quadruped.mj_data
    data.qvel[:] = 5.0
    mujoco.mj_forward(model, data)

    quadruped.zero_dynamics()
    data = quadruped.mj_data
    assert np.abs(data.qvel).max() == pytest.approx(0.0)
    assert np.abs(data.qacc).max() == pytest.approx(0.0)


def test_unknown_robot_still_errors(arm) -> None:
    assert arm.zero_dynamics(robot_name="nope")["status"] == "error"

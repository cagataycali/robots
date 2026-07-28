"""Regression tests: a teleport that keeps its momentum says so.

``set_joint_positions`` writes ``qpos`` and runs ``mj_forward``. It does NOT clear
``qvel`` - deliberately, since a caller may be setting a pose and a velocity
together - so a teleport made from a MOVING state carries the old momentum into
the new pose and the very next step integrates away from it.

The result text said only ``"Set 7/7 joint positions, FK updated"``, so the
leftover motion was invisible. Measured on a Panda teleported mid-swing
(commanded joint2 to -0.4 rad):

    teleport state              qvel after    drift after 1 step   after 10
    from rest                     0.185 rad/s     0.503 mrad         27.8
    from moving, no zero_dynamics 2.213 rad/s     4.469 mrad         46.6
    from moving, + zero_dynamics  0.000 rad/s     0.059 mrad          3.3

76x more first-step drift than the same teleport after ``zero_dynamics``, and by
10 steps the requested pose was visibly gone - while the call reported success.

``zero_dynamics`` documents this hazard ("writing qpos directly leaves qvel/qacc
holding pre-teleport values"), but nothing on the teleport primitive itself
pointed there. The behaviour is unchanged (silently zeroing would break the
documented "writes to qpos" contract); the leftover velocity is now reported with
the remedy named.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.physics import _TELEPORT_QVEL_NOTE_THRESHOLD  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_POSE = {f"joint{i + 1}": v for i, v in enumerate([0.3, -0.4, 0.2, -1.8, 0.1, 1.2, 0.5])}


@pytest.fixture
def sim():
    s = Simulation(tool_name="teleport_qvel_note", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    s.step(n_steps=100)
    yield s
    s.destroy()


def _set_moving(sim) -> float:
    """Drive the arm so it carries real velocity; return the peak |qvel|."""
    sim.send_action(action={"joint2": -1.2})
    sim.step(n_steps=40)
    peak = float(np.abs(sim.mj_data.qvel).max())
    assert peak > _TELEPORT_QVEL_NOTE_THRESHOLD, f"premise: the arm must be moving, got {peak}"
    return peak


def test_a_teleport_from_a_moving_state_is_flagged(sim) -> None:
    """The core defect: retained momentum reported as plain success."""
    _set_moving(sim)
    result = sim.set_joint_positions(robot_name="a", positions=_POSE)
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "qvel was left untouched" in text
    assert "zero_dynamics" in text, "the remedy must be named"
    assert "rad/s" in text


def test_the_note_reports_the_actual_residual(sim) -> None:
    """Not a generic warning - the number a caller can act on."""
    peak = _set_moving(sim)
    text = sim.set_joint_positions(robot_name="a", positions=_POSE)["content"][0]["text"]
    reported = float(text.split("carries up to ")[1].split(" rad/s")[0])
    assert reported == pytest.approx(peak, rel=0.05)


def test_a_teleport_from_rest_is_not_flagged(sim) -> None:
    """Guard against crying wolf: a settling arm shows a few tenths of a rad/s."""
    result = sim.set_joint_positions(robot_name="a", positions=_POSE)
    assert result["status"] == "success"
    assert "qvel was left untouched" not in result["content"][0]["text"]


def test_zero_dynamics_silences_the_note(sim) -> None:
    """The suggested remedy must actually work."""
    _set_moving(sim)
    assert sim.zero_dynamics(robot_name="a")["status"] == "success"
    result = sim.set_joint_positions(robot_name="a", positions=_POSE)
    assert "qvel was left untouched" not in result["content"][0]["text"]


def test_the_remedy_removes_the_measured_drift(sim) -> None:
    """The behavioural claim behind the note, measured both ways.

    Same teleport, same target joint: with the leftover velocity the first step
    integrates far off the requested pose; after zero_dynamics it barely moves.
    """

    def drift_after(zero_first: bool) -> float:
        s = Simulation(tool_name="teleport_drift", mesh=False)
        s.create_world()
        assert s.add_robot(name="a", data_config="panda")["status"] == "success"
        s.step(n_steps=100)
        try:
            s.send_action(action={"joint2": -1.2})
            s.step(n_steps=40)
            s.set_joint_positions(robot_name="a", positions=_POSE)
            if zero_first:
                s.zero_dynamics(robot_name="a")
            model, data = s.mj_model, s.mj_data
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "a/joint2")
            adr = int(model.jnt_qposadr[jid])
            mujoco.mj_step(model, data)
            return abs(float(data.qpos[adr]) - _POSE["joint2"])
        finally:
            s.destroy()

    kept = drift_after(False)
    zeroed = drift_after(True)
    assert kept > zeroed * 5, f"expected the retained velocity to dominate: kept={kept:.6f} zeroed={zeroed:.6f}"


def test_qvel_is_still_left_untouched(sim) -> None:
    """The fix is a report, NOT a behaviour change - the contract is unchanged.

    A caller setting a pose and a velocity together must keep working, so the
    teleport must not silently zero what it was handed.
    """
    peak = _set_moving(sim)
    sim.set_joint_positions(robot_name="a", positions=_POSE)
    assert float(np.abs(sim.mj_data.qvel).max()) == pytest.approx(peak, rel=1e-6)


def test_setting_a_velocity_after_a_pose_still_works(sim) -> None:
    """The documented alternative remedy: choose the velocity explicitly."""
    _set_moving(sim)
    assert sim.set_joint_positions(robot_name="a", positions=_POSE)["status"] == "success"
    assert sim.set_joint_velocities(robot_name="a", velocities={"joint2": 0.0})["status"] == "success"
    model, data = sim.mj_model, sim.mj_data
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "a/joint2")
    assert float(data.qvel[int(model.jnt_dofadr[jid])]) == pytest.approx(0.0)


def test_the_threshold_sits_above_settling_noise() -> None:
    """A resting arm under gravity showed 0.185 rad/s; that is not a warning."""
    assert 0.2 < _TELEPORT_QVEL_NOTE_THRESHOLD < 2.0


def test_the_docstring_states_that_velocity_is_not_cleared(sim) -> None:
    """The absence of this is what made the momentum invisible."""
    doc = type(sim).set_joint_positions.__doc__ or ""
    assert "VELOCITY IS NOT CLEARED" in doc
    assert "zero_dynamics" in doc

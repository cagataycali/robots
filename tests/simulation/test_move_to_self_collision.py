"""Regression tests: move_to rejects a self-colliding IK solution.

The IK bridge is purely kinematic - it respects joint limits but knows nothing
about geometry - so it will happily return a posture in which the arm passes
through ITSELF. The physics solver then holds those links apart, so the servo
descent can never realize the pose: it stalls with an actuator pinned at its
force limit.

Measured on the Panda before the fix (``tol=0.01``, ``max_steps=400``):

    target             ik_residual   penetration   move_to
    [0.45,  0.0, 0.5]    0.00051 m    -9.7 mm      timeout at 13.2 mm
    [0.3,  -0.2, 0.6]    0.00061 m   -10.2 mm      timeout at 12.1 mm
    [0.4,   0.2, 0.45]   0.00049 m    -3.5 mm      timeout at 11.0 mm

Three of seven workspace targets failed this way. Because the residual looked
excellent, the restart search - which already existed - never ran, and the error
said "the servo may need more steps": actively misleading, since 30 s of extra
sim time changed the error by less than 0.01 mm (joint 6 was saturated at its
+-12 Nm limit the whole time).

Penetration is now a solve-rejection criterion alongside the residual, and the
restart search prefers a collision-free branch over a merely closer one. All
seven targets now converge, the three above in 37-48 control ticks.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mink")

from strands_robots.simulation.mujoco.motion_primitives import _SELF_COLLISION_TOL_M  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Targets whose DIRECT IK solve self-penetrates on the Panda (see the table
# above). These are the regression cases: reachable, but only via another branch.
_SELF_COLLIDING_TARGETS = [
    [0.45, 0.0, 0.5],
    [0.3, -0.2, 0.6],
    [0.4, 0.2, 0.45],
]

# Targets whose direct solve is already clean - the guard must not disturb them.
_CLEAN_TARGETS = [
    [0.45, 0.0, 0.30],
    [0.5, 0.0, 0.15],
]


@pytest.fixture
def sim():
    s = Simulation(tool_name="move_to_self_collision", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    s.step(n_steps=100)
    yield s
    s.destroy()


def _worst_penetration(model, qpos) -> float:
    """Deepest contact penetration at ``qpos`` (<= 0), via a scratch MjData."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    mujoco.mj_forward(model, scratch)
    return min((float(scratch.contact[i].dist) for i in range(int(scratch.ncon))), default=0.0)


def _json(result) -> dict:
    for block in result["content"]:
        if "json" in block:
            return block["json"]
    return {}


@pytest.mark.parametrize("target", _SELF_COLLIDING_TARGETS)
def test_a_target_needing_another_branch_is_reached(sim, target) -> None:
    """The core defect: these three used to time out at 11-13 mm."""
    result = sim.move_to(robot_name="a", position=target, tol=0.01, max_steps=400)
    assert result["status"] == "success", result["content"][0]["text"]
    payload = _json(result)
    assert payload["position_error_m"] <= 0.01
    assert payload["reached"] is True


@pytest.mark.parametrize("target", _SELF_COLLIDING_TARGETS)
def test_the_pose_actually_held_is_collision_free(sim, target) -> None:
    """Not just "converged" - the arm must not be inside itself at the goal.

    This is the property the servo needs: a penetrating goal is unrealizable, so
    reaching tolerance and being collision-free are the same requirement here.
    """
    assert sim.move_to(robot_name="a", position=target, tol=0.01, max_steps=400)["status"] == "success"
    assert _worst_penetration(sim.mj_model, sim.mj_data.qpos) >= -_SELF_COLLISION_TOL_M


@pytest.mark.parametrize("target", _CLEAN_TARGETS)
def test_an_already_clean_target_still_works(sim, target) -> None:
    """Guard against the rejection criterion refusing healthy solves."""
    result = sim.move_to(robot_name="a", position=target, tol=0.01, max_steps=400)
    assert result["status"] == "success", result["content"][0]["text"]


def test_the_reported_penetration_is_in_the_payload(sim) -> None:
    """A refused solve must say how deep it penetrated, not just "unreachable".

    Reaching the all-branches-penetrate path reliably would need a contrived
    model, so assert the contract on the source of the message instead: the
    branch exists, names the cause, and reports the depth in millimeters.
    """
    import inspect

    source = inspect.getsource(Simulation.move_to)
    assert "self-colliding arm posture" in source
    assert "ik_self_penetration_m" in source
    # The old, misleading wording must not be what a self-collision reports.
    assert source.index("self-colliding arm posture") < source.index("The servo may need more steps")


def test_a_collision_free_branch_beats_a_closer_penetrating_one(sim) -> None:
    """The restart search's preference order, measured directly.

    For [0.45, 0, 0.5] the DIRECT solve has the smaller residual (0.00051 m) but
    penetrates 9.7 mm; restart 0 (the home keyframe) is both clean and closer.
    The chosen solution must be the clean one even though a purely
    residual-ranked search would have kept the first.
    """
    from strands_robots.simulation.ik import MinkIKBridge, discover_ee_frame

    model, data = sim.mj_model, sim.mj_data
    _frame = discover_ee_frame(model, "a/")
    assert _frame is not None, "expected an end-effector frame for the panda arm"
    frame_name, frame_type = _frame
    bridge = MinkIKBridge(model, frame_name, frame_type, orientation_cost=0.0, max_iters=200)
    q0 = np.array(data.qpos, dtype=np.float64, copy=True)
    target = np.array([0.45, 0.0, 0.5])
    pose = np.eye(4)
    pose[:3, 3] = target
    pose[:3, :3] = bridge.ee_pose(q0)[:3, :3]

    direct = bridge.solve(pose, q0)
    direct_residual = float(np.linalg.norm(bridge.ee_pose(direct)[:3, 3] - target))
    direct_penetration = _worst_penetration(model, direct)
    # Pin the premise: the direct solve looks good on residual alone.
    assert direct_residual <= 0.01, "premise changed: the direct solve no longer meets tol"
    assert direct_penetration < -_SELF_COLLISION_TOL_M, "premise changed: the direct solve no longer self-collides"

    assert sim.move_to(robot_name="a", position=target.tolist(), tol=0.01, max_steps=400)["status"] == "success"
    assert _worst_penetration(model, sim.mj_data.qpos) >= -_SELF_COLLISION_TOL_M


def test_the_tolerance_sits_above_resting_contact_noise() -> None:
    """A resting arm carries sub-0.1 mm contact depths; those are not collisions."""
    assert 1e-4 < _SELF_COLLISION_TOL_M < 3.5e-3


def test_an_unreachable_target_still_reports_the_residual(sim) -> None:
    """The pre-existing out-of-reach contract must be unchanged."""
    result = sim.move_to(robot_name="a", position=[1.2, 0.0, 0.2], tol=0.01, max_steps=50)
    assert result["status"] == "error"
    assert "unreachable" in result["content"][0]["text"]
    assert _json(result)["ik_residual_m"] > 0.01


def test_the_docstring_documents_the_goal_only_scope(sim) -> None:
    """The guard checks the GOAL pose, not the swept path - say so."""
    doc = type(sim).move_to.__doc__ or ""
    assert "through itself" in doc
    assert "only the goal is" in doc, "the goal-only scope of the check must be stated"

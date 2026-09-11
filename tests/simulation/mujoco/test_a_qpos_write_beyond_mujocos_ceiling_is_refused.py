"""A caller value too large to be a joint coordinate is refused before the write.

``mj_step`` runs ``mj_checkPos`` before it integrates: a ``qpos`` entry past
``mjMAXVAL`` makes MuJoCo declare the simulation unstable and reset the ENTIRE
state - every joint of every robot and every object - reporting that only as a
``WARNING`` on stderr. Measured on the bundled g1: ``set_joint_positions`` on
the floating base, ``move_object`` and ``add_object`` each returned ``success``
for ``1e11``, and the next ``step`` rewound ``data.time`` 0.402 -> 0.002, put a
settled cube back at its 0.30 spawn and dropped a 0.796 rad shoulder to 0.

A joint's declared range already bounds a LIMITED joint, so the hole was the
joint that declares none (a floating base, a continuous hinge) and the freejoint
pose of a dynamic object, whose quaternion is not renormalized on write either.
``set_joint_velocities`` has held ``qvel`` to this same ceiling since D-052;
these are the ``qpos`` surfaces that had not caught up.

The ceiling is MuJoCo's own, so ``mjMAXVAL`` exactly is still writable - the
guard refuses what ``mj_checkPos`` refuses and nothing more. A static body is
welded with no freejoint, owns no ``qpos`` entry and is stable however far away
it sits, so it is not held to the ceiling at all.
"""

from __future__ import annotations

import importlib.util

import pytest

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")

# 'hinge' declares a range (limited); 'spin' declares none, so autolimits leaves
# it unlimited - the joint kind that had nothing bounding its magnitude.
_ARM = """
<mujoco model="turret">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.1"/>
      <body name="turret" pos="0 0 0.2">
        <geom type="box" size="0.02 0.02 0.02"/>
        <joint name="spin" type="hinge" axis="0 0 1" damping="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="hinge_act" joint="hinge" kp="50"/></actuator>
</mujoco>
"""

_HUGE = 1e11  # finite, and past mjMAXVAL (1e10)


@pytest.fixture
def sim(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "turret.xml"
    path.write_text(_ARM)
    s = Simulation()
    s.create_world(timestep=0.002)
    assert s.add_robot("arm", urdf_path=str(path))["status"] == "success"
    # A pose the servo holds and a settled object: both are lost if the world
    # is reset, so either one witnesses the harm through a public surface.
    assert s.set_joint_positions(robot_name="arm", positions={"hinge": 0.5})["status"] == "success"
    assert (
        s.add_object(name="cube", shape="box", position=[0.3, 0.0, 0.4], size=[0.02] * 3, mass=0.05)["status"]
        == "success"
    )
    s.step(n_steps=1)
    yield s
    s.destroy()


def _ceiling() -> float:
    import mujoco

    return float(mujoco.mjMAXVAL)


def _hinge_position(sim) -> float:
    return float(sim.get_robot_state("arm")["content"][1]["json"]["state"]["hinge"]["position"])


def _world_survived(sim) -> bool:
    """True when a step later the arm still holds the pose it was left in.

    ``mj_resetData`` would put ``hinge`` back at 0; the servo only creeps toward
    its 0.5 setpoint, so a pose above 0.3 means no reset happened.
    """
    sim.step(n_steps=1)
    return _hinge_position(sim) > 0.3


# Every MuJoCo surface that writes a caller-supplied value into qpos, and the
# component of each that reaches it. move_object's quaternion is listed because
# a freejoint's wxyz slice is qpos too and nothing renormalizes it on write.
def _refused_calls():
    return {
        "set_joint_positions/unlimited joint": lambda s, v: s.set_joint_positions(
            robot_name="arm", positions={"spin": v}
        ),
        "move_object/position": lambda s, v: s.move_object("cube", position=[v, 0.0, 0.4]),
        "move_object/orientation": lambda s, v: s.move_object("cube", orientation=[v, 0.0, 0.0, 0.0]),
        "add_object/position": lambda s, v: s.add_object(
            name="far", shape="box", position=[v, 0.0, 0.4], size=[0.02] * 3, mass=0.05
        ),
    }


@requires_mujoco
@pytest.mark.parametrize("surface", sorted(_refused_calls()))
def test_a_value_mujoco_would_reset_the_world_on_is_refused(sim, surface) -> None:
    res = _refused_calls()[surface](sim, _HUGE)

    assert res["status"] == "error"
    text = res["content"][0]["text"]
    assert "nothing written" in text
    assert f"{_HUGE:.4g}" in text, "the refusal names the offending value"
    assert _world_survived(sim), "the scene the caller built outlived the refused write"


@requires_mujoco
@pytest.mark.parametrize("surface", sorted(_refused_calls()))
def test_the_refusal_names_the_consequence_and_mujocos_own_ceiling(sim, surface) -> None:
    text = _refused_calls()[surface](sim, _HUGE)["content"][0]["text"]

    # One shared refusal, so the accepted domain cannot drift between surfaces.
    assert "reset every joint and object to its initial state" in text
    assert f"mjMAXVAL={_ceiling():.0e}" in text


@requires_mujoco
@pytest.mark.parametrize("surface", sorted(_refused_calls()))
def test_the_ceiling_itself_is_still_writable(sim, surface) -> None:
    """The guard refuses what ``mj_checkPos`` refuses - and not the boundary it allows."""
    res = _refused_calls()[surface](sim, _ceiling())

    assert res["status"] == "success"
    assert _world_survived(sim)


@requires_mujoco
def test_an_ordinary_pose_is_still_applied(sim) -> None:
    assert sim.set_joint_positions(robot_name="arm", positions={"spin": 0.25})["status"] == "success"
    assert sim.move_object("cube", position=[0.35, 0.0, 0.4])["status"] == "success"
    assert (
        sim.add_object(name="near", shape="box", position=[0.1, 0.0, 0.4], size=[0.02] * 3, mass=0.05)["status"]
        == "success"
    )
    assert _world_survived(sim)


@requires_mujoco
def test_a_non_unit_quaternion_is_still_accepted(sim) -> None:
    """Magnitude is not the contract for an orientation - only the ceiling is.

    ``[0, 2, 0, 0]`` and ``[0, 1, 0, 0]`` are the same half turn, and
    ``coerce_orientation_quaternion`` accepts any usable direction, so holding
    the components to the ceiling must not narrow that to unit quaternions.
    """
    assert sim.move_object("cube", orientation=[0.0, 2.0, 0.0, 0.0])["status"] == "success"
    assert _world_survived(sim)


@requires_mujoco
def test_a_static_body_owns_no_qpos_entry_so_it_is_not_held_to_the_ceiling(sim) -> None:
    """A welded body has no freejoint to overflow, and a far-away one is stable."""
    assert (
        sim.add_object(name="wall", shape="box", position=[_HUGE, 0.0, 0.4], size=[0.02] * 3, is_static=True)["status"]
        == "success"
    )
    assert sim.move_object("wall", position=[2 * _HUGE, 0.0, 0.4])["status"] == "success"
    assert _world_survived(sim)


@requires_mujoco
def test_a_limited_joint_keeps_the_message_naming_its_own_range(sim) -> None:
    """A range is the more specific finding, so it is reported ahead of the ceiling."""
    text = sim.set_joint_positions(robot_name="arm", positions={"hinge": _HUGE})["content"][0]["text"]

    assert "outside [-1.57, 1.57]" in text
    assert "mjMAXVAL" not in text


@requires_mujoco
@pytest.mark.parametrize("surface", sorted(_refused_calls()))
def test_a_boolean_is_answered_by_the_surfaces_own_domain_first(sim, surface) -> None:
    """The ceiling check never sees a value its caller's domain has not accepted.

    ``float(True)`` is ``1.0``, so a boolean would sail under the ceiling and be
    written as 1 radian or 1 meter. Each surface refuses it by name before the
    ceiling is consulted - which is what lets the ceiling check coerce with a
    bare ``float()``.
    """
    text = _refused_calls()[surface](sim, True)["content"][0]["text"]

    assert "bool" in text
    assert "mjMAXVAL" not in text, "the boolean is answered by its own domain, not by the ceiling"

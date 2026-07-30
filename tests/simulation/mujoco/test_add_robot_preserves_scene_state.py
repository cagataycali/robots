"""``add_robot`` gives the new robot a clean state without rewinding the scene.

:meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.add_robot` documents
composition as state-preserving -- "this method merges the robot's bodies,
actuators, assets, and sensors into the existing scene XML. This preserves
previously-created world state (gravity, objects, cameras, other robots)" -- and
it also owes the robot it just added a defined starting configuration, because
the recompile that merged it in leaves its new joints wherever the compiler put
them.

Those two obligations were met with a single ``mj_resetData`` over the whole
world, so the second one cancelled the first: adding a robot returned every
other robot to its reference pose and released its actuator setpoints, teleported
settled objects back to their declared spawn, and rewound the clock -- all
reported as a successful add. These tests pin both halves, so neither can be
restored by weakening the other: the scene is untouched, AND the new robot starts
from the reference configuration.
"""

import pathlib
import re

import pytest

from strands_robots.simulation import Simulation

# Two-link arm with position servos. Damped and limited so a commanded setpoint
# settles instead of oscillating, and declared in radians (MuJoCo's compiler
# defaults to degrees, which would cap these joints at ~2 degrees of travel).
_ARM_XML = """<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
      <body name="fore" pos="0.25 0 0">
        <joint name="lift" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="60"/>
    <position name="lift_act" joint="lift" kp="60"/>
  </actuator>
  <keyframe>
    <key name="stow" qpos="1.2 -1.1"/>
  </keyframe>
</mujoco>"""

# Floating base: exercises the free-joint slice widths (7 qpos / 6 qvel) of the
# scoped reset, which a hinge-only arm never reaches.
_BASE_XML = """<mujoco model="probe_base">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chassis" pos="0.4 0 0.35">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.08 0.05"/>
    </body>
  </worldbody>
</mujoco>"""

_PARKED = {"pan": 0.9, "lift": -0.7}


@pytest.fixture
def models(tmp_path: pathlib.Path) -> dict[str, str]:
    """Write the inline models to disk so ``add_robot(urdf_path=...)`` can load them."""
    arm = tmp_path / "arm.xml"
    arm.write_text(_ARM_XML)
    base = tmp_path / "base.xml"
    base.write_text(_BASE_XML)
    return {"arm": str(arm), "base": str(base)}


@pytest.fixture
def sim():
    """A MuJoCo world with no robots yet."""
    engine = Simulation(backend="mujoco", tool_name="scene_state_sim", mesh=False)
    assert engine.create_world()["status"] == "success"
    try:
        yield engine
    finally:
        engine.cleanup()


def _json(result: dict) -> dict:
    """The structured block of a tool result envelope (it is not always first)."""
    return next(c["json"] for c in result["content"] if "json" in c)


def _reported_clock(engine) -> tuple[float, int]:
    """Elapsed sim time and step count as ``get_state`` reports them."""
    text = engine.get_state()["content"][0]["text"]
    m = re.search(r"t=([0-9.]+)s \(step (\d+)\)", text)
    assert m is not None, text
    return float(m.group(1)), int(m.group(2))


def _joints(engine, robot_name: str) -> dict[str, float]:
    """The robot's joint positions, read through the public observation surface."""
    obs = engine.get_observation(robot_name=robot_name)
    return {k: float(v) for k, v in obs.items() if not k.endswith(".vel") and not hasattr(v, "shape")}


def _park(engine, robot_name: str) -> dict[str, float]:
    """Command ``robot_name`` to a distinctive pose and let it settle there."""
    keys = engine.robot_action_keys(robot_name)
    assert keys == ["pan_act", "lift_act"], keys
    assert engine.send_action(dict(zip(keys, _PARKED.values())), robot_name=robot_name)["status"] == "success"
    engine.step(500)
    parked = _joints(engine, robot_name)
    # Non-vacuity: a reset to the reference configuration has to be detectable.
    assert min(abs(v) for v in parked.values()) > 0.1, parked
    return parked


def test_a_parked_arm_holds_its_pose_when_another_robot_is_added(sim, models):
    """The arm is where it was parked, and is still being held there afterwards.

    The pose surviving the call is only half the contract: the actuator setpoints
    holding it have to survive too, or the arm reads correctly for one instant and
    then collapses under gravity over the following steps.
    """
    assert sim.add_robot(name="a", urdf_path=models["arm"])["status"] == "success"
    parked = _park(sim, "a")

    assert sim.add_robot(name="b", urdf_path=models["arm"])["status"] == "success"

    assert _joints(sim, "a") == pytest.approx(parked, abs=1e-6)
    sim.step(800)
    assert _joints(sim, "a") == pytest.approx(parked, abs=1e-3)


def test_a_settled_object_stays_where_it_settled_when_a_robot_is_added(sim, models):
    """A free body that has fallen to rest is not teleported back to its spawn."""
    assert (
        sim.add_object(name="crate", shape="box", size=[0.08, 0.08, 0.08], position=[0.7, 0.0, 0.5])["status"]
        == "success"
    )
    sim.step(600)
    settled = [float(v) for v in _json(sim.get_body_state(body_name="crate"))["position"]]
    # Non-vacuity: it has to have moved from its spawn for the check to mean anything.
    assert abs(settled[2] - 0.5) > 0.1, settled

    assert sim.add_robot(name="a", urdf_path=models["arm"])["status"] == "success"

    after = [float(v) for v in _json(sim.get_body_state(body_name="crate"))["position"]]
    assert after == pytest.approx(settled, abs=1e-6)


def test_the_added_robot_starts_at_the_reference_configuration(sim, models):
    """The other half: the new robot is at the reference pose and unpowered.

    Without this, "stop resetting the world" would pass by leaving the new robot
    at whatever configuration the recompile happened to produce.
    """
    assert sim.add_robot(name="a", urdf_path=models["arm"])["status"] == "success"
    _park(sim, "a")

    assert sim.add_robot(name="b", urdf_path=models["arm"])["status"] == "success"

    assert _joints(sim, "b") == pytest.approx({"pan": 0.0, "lift": 0.0}, abs=1e-9)
    # Unpowered, not merely momentarily at zero: with no setpoint the arm only
    # sags away from the reference pose under gravity.
    sim.step(400)
    assert _joints(sim, "b")["lift"] != pytest.approx(0.0, abs=1e-3)


def test_a_free_base_robot_is_added_at_its_reference_pose(sim, models):
    """A floating base is placed at its declared spawn, not at the world origin.

    A free joint spends 7 ``qpos`` entries against 6 ``qvel`` entries, so a reset
    that reused one width for both would read past the joint.
    """
    assert sim.add_robot(name="a", urdf_path=models["arm"])["status"] == "success"
    parked = _park(sim, "a")

    assert sim.add_robot(name="rover", urdf_path=models["base"])["status"] == "success"

    pose = _json(sim.get_body_state(body_name="rover/chassis"))
    assert [float(v) for v in pose["position"]] == pytest.approx([0.4, 0.0, 0.35], abs=1e-6)
    assert [float(v) for v in pose["linear_velocity"]] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert _joints(sim, "a") == pytest.approx(parked, abs=1e-6)


def test_the_clock_is_not_rewound_when_a_robot_is_added(sim, models):
    """Elapsed simulation time keeps counting the steps that actually ran.

    Rewinding it while the bodies keep their state reports a scene that never
    existed, and every recorded timestamp downstream is taken from this clock.
    """
    assert sim.add_robot(name="a", urdf_path=models["arm"])["status"] == "success"
    sim.step(300)
    before = _reported_clock(sim)
    assert before == (pytest.approx(0.6), 300)

    assert sim.add_robot(name="b", urdf_path=models["arm"])["status"] == "success"

    assert _reported_clock(sim) == before


def test_a_keyframe_spawned_arm_is_not_snapped_back_to_its_home_pose(sim, models):
    """A robot that has moved since its keyframe spawn stays where it moved to.

    The world-wide reset used to be followed by a re-apply of every robot's
    captured keyframe pose, which was the only reason a keyframe spawn survived
    the call at all -- and it overwrote wherever that robot had since been driven.
    """
    assert sim.add_robot(name="a", urdf_path=models["arm"], keyframe="stow")["status"] == "success"
    assert _joints(sim, "a") == pytest.approx({"pan": 1.2, "lift": -1.1}, abs=1e-6)
    parked = _park(sim, "a")
    assert parked["pan"] != pytest.approx(1.2, abs=1e-2)

    assert sim.add_robot(name="b", urdf_path=models["arm"])["status"] == "success"

    assert _joints(sim, "a") == pytest.approx(parked, abs=1e-6)

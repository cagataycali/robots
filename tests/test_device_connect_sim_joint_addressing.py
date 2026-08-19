"""The sim driver's 10Hz joint publisher reports each joint's own position.

``SimulationDeviceDriver._publishState`` emits an ``observationUpdate`` whose
``joints`` field is documented as ``{joint name: position (radians)}``. Getting
that mapping right needs three facts about MuJoCo state that the simulation's
own ``get_observation`` surface already owns:

* a joint's position lives at its qpos ADDRESS (``model.jnt_qposadr[jid]``),
  which equals the joint id only while every joint is single-DoF. One floating
  base ahead of a chain occupies seven qpos slots for one joint id, shifting
  every later joint by six.
* a free joint has no scalar position - its qpos is ``[xyz, quat]`` - so it is
  excluded from the per-joint state rather than reported as one number.
* the read has to be serialised against a concurrent physics step.

The publisher used to index ``data.qpos`` by joint id directly, so on a
floating-base robot it published the base pose under leg-joint names. These
tests pin the published mapping against the observation surface, on a model
where the two addressings differ, so a second in-driver reader cannot drift
from the one that answers this question.
"""

import asyncio
import pathlib
import unittest
from typing import Any

import pytest

pytest.importorskip("device_connect_edge")
pytest.importorskip("mujoco")

from strands_robots import create_simulation  # noqa: E402
from strands_robots.device_connect.sim_driver import SimulationDeviceDriver  # noqa: E402

# A floating base ahead of two hinges: joint ids are (0, 1, 2) while the qpos
# addresses are (0, 7, 8). The base is authored away from the origin and both
# hinges are driven to distinct angles, so every candidate reading is a
# different number and a mis-addressed publish names which slot it came from.
_FREE_BASE_ARM = """<mujoco model="free_base_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="base" pos="0 0.25 0.5">
      <freejoint name="floating_base"/>
      <geom type="box" size="0.08 0.08 0.04" mass="1"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# The same two hinges with no floating base: joint ids and qpos addresses
# coincide, which is the shape the publisher was already correct for.
_FIXED_BASE_ARM = """<mujoco model="fixed_base_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="base" pos="0 0.25 0.5">
      <geom type="box" size="0.08 0.08 0.04" mass="1"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_ANGLES = {"j1": 0.3, "j2": -0.6}


def _ok(result: dict[str, Any], what: str) -> dict[str, Any]:
    """Return ``result`` after refusing a non-success tool envelope.

    Raises rather than asserting so the scene is still built under ``python -O``.
    """
    if result.get("status") != "success":
        raise AssertionError(f"{what} failed: {result}")
    return result


def _sim_with(tmp_path: pathlib.Path, xml: str, name: str = "arm") -> Any:
    """Build a one-robot MuJoCo world from ``xml`` with both hinges driven."""
    model_path = tmp_path / f"{name}.xml"
    model_path.write_text(xml, encoding="utf-8")
    # Typed loose: ``create_simulation`` is declared to return the backend-
    # agnostic engine, while this fixture drives MuJoCo-only surfaces.
    sim: Any = create_simulation(backend="mujoco", tool_name="joint_addressing")
    _ok(sim.create_world(), "create_world")
    _ok(sim.add_robot(name=name, urdf_path=str(model_path)), "add_robot")
    _ok(sim.set_joint_positions(dict(_ANGLES), robot_name=name), "set_joint_positions")
    return sim


def _publish(sim: Any) -> dict[str, dict[str, float]]:
    """Run one publish cycle and return the emitted joints per robot."""
    for robot in sim._world.robots.values():
        robot.policy_running = True
        robot.policy_steps = 1
        robot.policy_instruction = "hold"

    captured: dict[str, dict[str, float]] = {}

    async def record_observation(**kwargs: Any) -> None:
        captured[kwargs["robot_name"]] = dict(kwargs["joints"])

    async def record_state(**kwargs: Any) -> None:
        return None

    driver: Any = SimulationDeviceDriver(sim)
    driver.stateUpdate = record_state
    driver.observationUpdate = record_observation
    asyncio.run(driver._publishState())
    return captured


class TestTheJointPublisherAddressesEachJointsOwnState:
    """The published position of a joint is that joint's position."""

    def test_a_floating_base_shifts_no_published_joint_off_its_own_slot(self, tmp_path):
        """Indexing qpos by joint id reads the base pose for a shifted joint.

        With a free joint at id 0, ``j1`` sits at qpos address 7 and ``j2`` at
        8, while their joint ids are 1 and 2 - the base's y and z slots. A
        publisher that indexes by id therefore reports the base's lateral
        offset and HEIGHT as two joint angles, in metres, under names whose
        documented unit is radians.
        """
        sim = _sim_with(tmp_path, _FREE_BASE_ARM)
        model = sim._world._model
        robot = sim._world.robots["arm"]
        addresses = [int(model.jnt_qposadr[jid]) for jid in robot.joint_ids]
        assert addresses != list(robot.joint_ids), "premise: this model must shift the joints"

        joints = _publish(sim)["arm"]
        assert joints == pytest.approx(_ANGLES), (
            f"published {joints} for a robot whose joints are {_ANGLES}; "
            f"joint ids {list(robot.joint_ids)} address qpos slots {addresses}"
        )

    def test_the_published_joints_are_the_ones_the_observation_surface_reports(self, tmp_path):
        """The publisher and ``get_observation`` answer the same question.

        Two readers of one robot's joint state must not disagree, whichever
        addressing each happens to use.
        """
        sim = _sim_with(tmp_path, _FREE_BASE_ARM)
        observation = sim.get_observation("arm", skip_images=True)
        expected = {
            name: float(observation[name])
            for name in sim._world.robots["arm"].joint_names
            if isinstance(observation.get(name), (int, float))
        }
        assert expected, "premise: the observation surface must report some joints"

        joints = _publish(sim)["arm"]
        assert joints == pytest.approx(expected)

    def test_a_free_joint_is_not_published_as_a_scalar_position(self, tmp_path):
        """A free joint's qpos is ``[xyz, quat]``, so it has no scalar position.

        Publishing one anyway reports a single component of the base pose as a
        joint angle, and duplicates a value ``base_pos`` already carries in
        full.
        """
        sim = _sim_with(tmp_path, _FREE_BASE_ARM)
        assert "floating_base" in sim._world.robots["arm"].joint_names, "premise: named free joint"

        joints = _publish(sim)["arm"]
        assert "floating_base" not in joints

    def test_each_robot_is_published_with_its_own_joints_and_values(self, tmp_path):
        """In a two-robot scene neither publish carries the other's state.

        Both robots here name their joints ``j1`` / ``j2``, so the compiled
        model holds four joints whose ids and qpos addresses interleave. Each
        observation must carry that robot's joints, at that robot's values.
        """
        sim = _sim_with(tmp_path, _FREE_BASE_ARM, name="alice")
        second = tmp_path / "bob.xml"
        second.write_text(_FIXED_BASE_ARM, encoding="utf-8")
        _ok(sim.add_robot(name="bob", urdf_path=str(second)), "add_robot(bob)")
        assert sim._world.robots["bob"].joint_ids != sim._world.robots["alice"].joint_ids, (
            "premise: the two robots must occupy different joint ids"
        )

        published = _publish(sim)
        assert set(published) == {"alice", "bob"}
        for name, joints in published.items():
            observation = sim.get_observation(name, skip_images=True)
            expected = {
                joint: float(observation[joint])
                for joint in sim._world.robots[name].joint_names
                if isinstance(observation.get(joint), (int, float))
            }
            assert joints == pytest.approx(expected), f"{name} published {joints}, expected {expected}"
        assert published["alice"] == pytest.approx(_ANGLES), "alice keeps the angles it was driven to"


class TestTheAlreadyCorrectAndDegradedPathsAreUnchanged:
    """Controls: a fixed-base arm and a simulation with no observation surface."""

    def test_a_fixed_base_arm_publishes_its_joint_angles(self, tmp_path):
        """With no free joint, ids and qpos addresses coincide.

        This is the shape the publisher was already correct for, so it must
        keep reporting exactly the same values.
        """
        sim = _sim_with(tmp_path, _FIXED_BASE_ARM)
        model = sim._world._model
        robot = sim._world.robots["arm"]
        assert [int(model.jnt_qposadr[jid]) for jid in robot.joint_ids] == list(robot.joint_ids), (
            "premise: this model must not shift the joints"
        )

        joints = _publish(sim)["arm"]
        assert joints == pytest.approx(_ANGLES)

    def test_a_simulation_with_no_observation_surface_publishes_no_joints(self):
        """A wrapped object that cannot answer reports nothing, not an error.

        The publish loop serves the whole fleet, so one robot that cannot be
        read must not raise into it.
        """

        class _Robot:
            policy_running = True
            policy_steps = 0
            policy_instruction = ""
            joint_names = ["j1"]

        class _World:
            robots = {"arm": _Robot()}
            sim_time = 0.0
            step_count = 0

        class _Sim:
            _world = _World()

        captured: dict[str, Any] = {}

        async def record_observation(**kwargs: Any) -> None:
            captured.update(kwargs)

        async def record_state(**kwargs: Any) -> None:
            return None

        driver: Any = SimulationDeviceDriver(_Sim())
        driver.stateUpdate = record_state
        driver.observationUpdate = record_observation
        asyncio.run(driver._publishState())

        assert captured["joints"] == {}


class TestTheReadIsDelegatedRatherThanReimplemented(unittest.TestCase):
    """The driver consumes the observation contract instead of restating it."""

    def test_the_publisher_reads_through_get_observation_without_images(self):
        """One reader owns qpos addressing, the free-joint rule and the lock.

        The 10Hz loop wants joint state only, so the read must not also render
        every camera on the model.
        """
        calls: list[tuple[str, bool]] = []

        class _Robot:
            policy_running = True
            policy_steps = 2
            policy_instruction = "hold"
            joint_names = ["j1", "j2"]

        class _World:
            robots = {"arm": _Robot()}
            sim_time = 0.5
            step_count = 5

        class _Sim:
            _world = _World()

            def get_observation(self, robot_name, *, skip_images=False):
                calls.append((robot_name, skip_images))
                return {"j1": 0.3, "j2": -0.6, "base_pos": [0.0, 0.25, 0.5]}

        captured: dict[str, Any] = {}

        async def record_observation(**kwargs: Any) -> None:
            captured.update(kwargs)

        async def record_state(**kwargs: Any) -> None:
            return None

        driver: Any = SimulationDeviceDriver(_Sim())
        driver.stateUpdate = record_state
        driver.observationUpdate = record_observation
        asyncio.run(driver._publishState())

        self.assertEqual(calls, [("arm", True)])
        self.assertEqual(captured["joints"], {"j1": 0.3, "j2": -0.6})

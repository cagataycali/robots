"""A latched external wrench belongs to the body it was applied to.

``apply_force`` latches a wrench that MuJoCo re-applies on every subsequent
step. These tests pin who that latch belongs to: applying a wrench to one body
must not revoke a wrench already latched on another, a second call on the same
body must replace rather than accumulate onto it, and the latch must keep
describing the world-frame wrench the caller asked for as the body moves.

Pre-fix the wrench went into ``qfrc_applied``, one world-wide generalized-force
vector that every call zeroed in full, so a second ``apply_force`` on a
different body silently cancelled the first while both calls reported success.
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two hinges in series: a wrench on ``link2`` maps onto ``link1``'s DOF as well,
# which is why no slice of ``qfrc_applied`` can be said to belong to one body.
ARM_XML = """
<mujoco model="two_link">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _state(sim, body_name):
    """Return the ``get_body_state`` json block for ``body_name``."""
    result = sim.get_body_state(body_name)
    assert result["status"] == "success", result
    return [block["json"] for block in result["content"] if "json" in block][0]


def _speed(sim, body_name):
    return float(np.linalg.norm(_state(sim, body_name)["linear_velocity"]))


@pytest.fixture
def sim():
    """A weightless world, so the only motion is the wrench under test."""
    engine = Simulation(tool_name="force_latch_sim", mesh=False)
    engine.create_world(gravity=[0.0, 0.0, 0.0])
    yield engine
    engine.cleanup()


@pytest.fixture
def two_pucks(sim):
    """Two free bodies far enough apart that they never touch."""
    for name, x in (("puck_a", 0.0), ("puck_b", 1.0)):
        assert (
            sim.add_object(name=name, shape="box", size=[0.05] * 3, position=[x, 0.0, 0.3], mass=0.1)["status"]
            == "success"
        )
    return sim


@pytest.fixture
def arm_and_puck(sim, tmp_path):
    """A hinge chain plus an unrelated free body."""
    arm_file = tmp_path / "two_link.xml"
    arm_file.write_text(ARM_XML)
    assert sim.add_robot(name="arm", urdf_path=str(arm_file))["status"] == "success"
    assert (
        sim.add_object(name="puck", shape="box", size=[0.05] * 3, position=[1.0, 0.0, 0.3], mass=0.1)["status"]
        == "success"
    )
    return sim


def _hinge_angles(model, data, *joint_names):
    """Read named hinge angles, so an unrelated body's DOFs cannot shift them."""
    angles = []
    for name in joint_names:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"joint {name!r} not in the compiled model"
        angles.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return angles


class TestOneLatchPerBody:
    """A wrench latched on one body survives a wrench latched on another."""

    def test_a_wrench_on_a_second_body_leaves_the_first_one_latched(self, two_pucks):
        """Both pucks were pushed, so both must move.

        Pre-fix the second call zeroed the whole buffer and ``puck_a`` stayed
        exactly at rest while the call that requested its motion had already
        reported success.
        """
        sim = two_pucks
        assert sim.apply_force("puck_a", force=[1.0, 0.0, 0.0])["status"] == "success"
        assert sim.apply_force("puck_b", force=[1.0, 0.0, 0.0])["status"] == "success"

        sim.step(200)

        assert _speed(sim, "puck_a") > 1.0, "the wrench latched on puck_a was revoked by the call targeting puck_b"
        assert _speed(sim, "puck_b") > 1.0
        assert _speed(sim, "puck_a") == pytest.approx(_speed(sim, "puck_b"), rel=1e-6)

    def test_a_wrench_on_a_robot_link_and_on_a_free_object_coexist(self, arm_and_puck):
        """A chain body and a free body hold their wrenches at the same time.

        This is the case no per-DOF clear can express: the chain body's wrench
        reaches its ancestors' DOFs, so there is no slice of the generalized
        buffer that could have been cleared for it alone.
        """
        sim = arm_and_puck
        assert sim.apply_force("arm/link2", force=[0.0, 0.0, -2.0])["status"] == "success"
        assert sim.apply_force("puck", force=[1.0, 0.0, 0.0])["status"] == "success"

        model, data = sim._world._model, sim._world._data
        (joint_before,) = _hinge_angles(model, data, "arm/j1")
        sim.step(200)

        (joint_after,) = _hinge_angles(model, data, "arm/j1")
        assert abs(joint_after - joint_before) > 0.05, "the arm never felt its latched wrench"
        assert _speed(sim, "puck") > 1.0, "the free body never felt its latched wrench"

    def test_a_second_call_on_the_same_body_replaces_its_wrench(self, two_pucks):
        """Re-latching the same wrench is idempotent, not cumulative."""
        sim = two_pucks
        sim.apply_force("puck_a", force=[1.0, 0.0, 0.0])
        sim.apply_force("puck_b", force=[1.0, 0.0, 0.0])
        # puck_b is asked for the same wrench three times over.
        sim.apply_force("puck_b", force=[1.0, 0.0, 0.0])
        sim.apply_force("puck_b", force=[1.0, 0.0, 0.0])

        sim.step(200)

        assert _speed(sim, "puck_b") == pytest.approx(_speed(sim, "puck_a"), rel=1e-6)

    def test_zeroing_one_body_leaves_the_other_pushed(self, two_pucks):
        """A zero wrench stops its own target and only its own target."""
        sim = two_pucks
        sim.apply_force("puck_a", force=[1.0, 0.0, 0.0])
        sim.apply_force("puck_b", force=[1.0, 0.0, 0.0])
        sim.step(100)
        a_coasting, b_accelerating = _speed(sim, "puck_a"), _speed(sim, "puck_b")

        assert sim.apply_force("puck_a", force=[0.0, 0.0, 0.0])["status"] == "success"
        sim.step(100)

        assert _speed(sim, "puck_a") == pytest.approx(a_coasting, rel=1e-6), "puck_a kept accelerating after its stop"
        assert _speed(sim, "puck_b") > b_accelerating * 1.5, "puck_b lost its wrench to the call that stopped puck_a"

    def test_reset_clears_every_latched_wrench(self, two_pucks):
        """``reset()`` is how a caller drops every wrench in the world."""
        sim = two_pucks
        sim.apply_force("puck_a", force=[1.0, 0.0, 0.0])
        sim.apply_force("puck_b", force=[1.0, 0.0, 0.0])

        assert sim.reset()["status"] == "success"
        sim.step(200)

        assert _speed(sim, "puck_a") == pytest.approx(0.0, abs=1e-9)
        assert _speed(sim, "puck_b") == pytest.approx(0.0, abs=1e-9)


class TestLatchedWrenchDirection:
    """The latch keeps describing the wrench the caller asked for."""

    def test_a_latched_force_keeps_its_world_direction_as_the_body_moves(self, arm_and_puck):
        """A constant world-frame force stays world-frame as the arm swings.

        Ground truth is the same force re-mapped through the configuration on
        every step. A generalized force frozen at call time drifts away from it
        by an order of magnitude more than the tolerance below once the arm has
        turned.
        """
        sim = arm_and_puck
        model, data = sim._world._model, sim._world._data
        force = np.array([0.0, 0.0, -2.0])

        truth_model = mj.MjModel.from_xml_string(ARM_XML)
        truth_model.opt.gravity[:] = 0.0
        truth_model.opt.timestep = model.opt.timestep
        truth_model.opt.integrator = model.opt.integrator
        truth_data = mj.MjData(truth_model)
        truth_id = mj.mj_name2id(truth_model, mj.mjtObj.mjOBJ_BODY, "link2")
        for _ in range(300):
            truth_data.qfrc_applied[:] = 0.0
            mj.mj_applyFT(
                truth_model,
                truth_data,
                force,
                np.zeros(3),
                truth_data.xipos[truth_id].copy(),
                truth_id,
                truth_data.qfrc_applied,
            )
            mj.mj_step(truth_model, truth_data)

        assert sim.apply_force("arm/link2", force=force)["status"] == "success"
        sim.step(300)

        assert _hinge_angles(model, data, "arm/j1", "arm/j2") == pytest.approx(
            _hinge_angles(truth_model, truth_data, "j1", "j2"), abs=0.05
        )

    def test_a_force_at_an_offset_point_spins_a_free_body(self, two_pucks):
        """A force off the centre of mass carries its lever-arm torque.

        ``+x`` force applied above the centre of mass is a ``+y`` torque
        (``r x F`` with ``r = +z``), so the puck must spin about ``+y`` while
        the same force at the centre of mass leaves it unspun.
        """
        sim = two_pucks
        assert sim.apply_force("puck_a", force=[2.0, 0.0, 0.0], point=[0.0, 0.0, 0.34])["status"] == "success"
        assert sim.apply_force("puck_b", force=[2.0, 0.0, 0.0], point=[1.0, 0.0, 0.30])["status"] == "success"

        sim.step(200)

        assert _state(sim, "puck_a")["angular_velocity"][1] > 1.0
        assert _state(sim, "puck_b")["angular_velocity"][1] == pytest.approx(0.0, abs=1e-6)
        # Both were pushed with the same force, so both travel the same way.
        assert _speed(sim, "puck_a") > 1.0
        assert _speed(sim, "puck_b") > 1.0

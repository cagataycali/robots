"""Multi-robot ejection: surviving-robot state + fail-soft rebuild contract.

``Simulation.remove_robot`` rebuilds the whole MJCF from the remaining
``world.robots`` and re-attaches every survivor (see
:func:`strands_robots.simulation.mujoco.scene_ops.eject_robot_from_scene`).
The guardrail suite pins the "no compiled world" and unknown-body early
returns; the snapshot/restore round-trip is pinned per joint-type width.

What was NOT pinned - and is pinned here - is the behaviour a scene with more
than one robot depends on:

* a surviving robot keeps its joint state across the rebuild AND has its
  actuator/joint ids re-resolved against the freshly compiled model, so it can
  still be driven after a sibling is removed; and
* the documented fail-soft contract: if the rebuild cannot compile, the eject
  reports failure rather than leaving a half-mutated world, and
  ``remove_robot`` surfaces that as a structured error.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two single-joint arms; attaching both namespaces them as ``armN/...`` so the
# rebuild has to re-resolve ids by fully-qualified name.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def two_arm_sim(tmp_path):
    """A compiled world holding two attached single-joint arms."""
    arm_path = tmp_path / "arm.xml"
    arm_path.write_text(_ARM_XML)
    sim = Simulation(tool_name="devx_multi_eject", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm1", urdf_path=str(arm_path))
    sim.add_robot(name="arm2", urdf_path=str(arm_path))
    try:
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


def _joint_qpos(sim: Simulation, joint_name: str) -> float:
    world = sim._world
    assert world is not None and world._model is not None
    mj = sim._mj
    jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    assert jid >= 0, f"joint {joint_name!r} not in compiled model"
    adr = int(world._model.jnt_qposadr[jid])
    return float(world._data.qpos[adr])


def test_remove_robot_preserves_survivor_state_and_reresolves_actuators(two_arm_sim):
    """Removing one arm keeps the survivor's pose and leaves it drivable.

    The XML round-trip reallocates ``model``/``data`` and shifts every body/
    joint index, so the survivor's cached actuator ids must be rebuilt by name.
    A survivor that lost its actuator ids would silently stop responding to
    ``send_action`` - the regression this pins.
    """
    sim = two_arm_sim
    world = sim._world
    assert world is not None
    mj = sim._mj

    # Put the survivor at a distinct, non-zero pose before the rebuild.
    surv_jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, "arm2/pan")
    surv_adr = int(world._model.jnt_qposadr[surv_jid])
    world._data.qpos[surv_adr] = 0.5
    mj.mj_forward(world._model, world._data)

    result = sim.remove_robot("arm1")
    assert result["status"] == "success", result

    # Registry reflects the removal.
    remaining = sim.list_robots()
    assert "arm1" not in remaining
    assert "arm2" in remaining

    # Survivor kept its pose across the fresh compile.
    assert _joint_qpos(sim, "arm2/pan") == pytest.approx(0.5, abs=1e-6)

    # Actuator ids were re-resolved against the new model, so the survivor is
    # still drivable: a position command moves the joint toward its target.
    survivor = sim._world.robots["arm2"]
    assert survivor.actuator_ids, "survivor lost its actuator ids after rebuild"

    send = sim.send_action({"arm2/pan": 1.2}, robot_name="arm2")
    assert not (isinstance(send, dict) and send.get("status") == "error"), send
    for _ in range(200):
        sim.step()
    assert _joint_qpos(sim, "arm2/pan") > 0.5, "survivor did not track the new command"


def test_remove_robot_reports_error_when_eject_fails(two_arm_sim, monkeypatch):
    """A rebuild that cannot eject surfaces a structured error, not a crash.

    ``remove_robot`` pops the target from the registry before delegating to
    ``eject_robot_from_scene``; if the eject returns ``False`` the caller must
    report failure rather than pretend success on a half-mutated world.
    """
    sim = two_arm_sim
    monkeypatch.setattr(
        "strands_robots.simulation.mujoco.simulation.eject_robot_from_scene",
        lambda *a, **k: False,
    )
    result = sim.remove_robot("arm1")
    assert result["status"] == "error"
    assert "arm1" in result["content"][0]["text"]


def test_eject_from_scene_returns_false_when_recompile_fails(two_arm_sim, monkeypatch):
    """``eject_robot_from_scene`` fails soft (returns ``False``) on a bad compile.

    A rebuilt spec that will not compile must not install a broken
    ``model``/``data`` pair; the helper returns ``False`` and leaves the prior
    world intact so the caller can report the failure.
    """
    sim = two_arm_sim
    world = sim._world
    assert world is not None
    prior_model = world._model

    # Drop every robot so the survivor re-attach loop is a no-op, then force the
    # fresh compile of the rebuilt spec to raise the way MuJoCo would on an
    # invalid model.
    world.robots.clear()
    bad_spec = MagicMock()
    bad_spec.compile.side_effect = ValueError("compile boom")
    monkeypatch.setattr(scene_ops.SpecBuilder, "build", staticmethod(lambda _world: bad_spec))

    assert scene_ops.eject_robot_from_scene(world, "arm1") is False
    # The failed rebuild did not swap in the broken model.
    assert world._model is prior_model


# --- D13: the actuation half of MjData across an eject ------------------------
#
# ``eject_robot_from_scene`` deliberately does a fresh ``spec.compile()`` +
# ``mj.MjData(new_model)`` to dodge the MuJoCo attach/delete segfault, unlike the
# sibling ``spec.recompile(model, data)`` path which carries MjData forward. But
# ``_snapshot_joint_state`` only carries ``qpos``/``qvel``, so ``data.ctrl`` came
# back all-zero from the fresh MjData and every surviving position servo lost its
# commanded target - driven from its held pose toward 0 on the next step.
#
# Measured pre-fix with two 2-joint arms, arm1 commanded to (0.9, -0.6):
#
#     ctrl before remove: [0.9, -0.6, 0.0, 0.0]
#     ctrl AFTER  remove: [0.0, 0.0]           <- survivor's targets wiped
#     arm1 pan after 4s more physics: -0.0000  <- collapsed from 0.9
#
# remove_robot was the ONLY mutation path with this hole (remove_object,
# move_object, add_object, add_camera, remove_camera all preserve ctrl).
#
# The engine's own contract already says actuation is state: PhysicsMixin
# .save_state uses mjSTATE_INTEGRATION precisely because mjSTATE_FULLPHYSICS
# "silently excluded ctrl and qfrc_applied, so the first step after a restore
# drove toward the pre-restore targets".


def _ctrl_by_name(sim: Simulation, actuator_name: str) -> float:
    world = sim._world
    assert world is not None and world._model is not None
    mj = sim._mj
    aid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    assert aid >= 0, f"actuator {actuator_name!r} not in compiled model"
    return float(world._data.ctrl[aid])


class TestEjectPreservesActuationState:
    def test_survivor_keeps_its_commanded_ctrl(self, two_arm_sim):
        """The regression: ctrl came back [0.0] from the fresh MjData."""
        sim = two_arm_sim
        sim.send_action({"pan": 0.9}, robot_name="arm2")
        sim.step(n_steps=50)
        before = _ctrl_by_name(sim, "arm2/pan_act")
        assert before == pytest.approx(0.9, abs=1e-6), f"fixture never commanded the servo (ctrl {before})"

        assert sim.remove_robot("arm1")["status"] == "success"

        after = _ctrl_by_name(sim, "arm2/pan_act")
        assert after == pytest.approx(0.9, abs=1e-6), f"ctrl wiped {before:.4f} -> {after:.4f}"

    def test_survivor_keeps_driving_toward_its_target_not_toward_zero(self, two_arm_sim):
        """The physical consequence: the arm is driven to 0 with no command sent.

        This fixture's servo (kp=50, no damping) oscillates rather than settling,
        so the invariant asserted is the one that actually distinguishes the bug:
        the joint stays on the commanded SIDE of zero and well away from it.
        Pre-fix the survivor's ctrl was zeroed, so it was actively driven to 0.
        """
        sim = two_arm_sim
        sim.send_action({"pan": 0.9}, robot_name="arm2")
        sim.step(n_steps=500)
        held = _joint_qpos(sim, "arm2/pan")
        assert held > 0.3, f"fixture never moved off zero (at {held})"

        assert sim.remove_robot("arm1")["status"] == "success"
        # No send_action here: the arm must keep tracking its held target.
        sim.step(n_steps=1000)

        after = _joint_qpos(sim, "arm2/pan")
        assert after > 0.3, f"survivor was driven back toward zero: {held:.4f} -> {after:.4f}"

    def test_ctrl_is_restored_by_name_not_by_index(self, two_arm_sim):
        """The fresh compile shifts actuator ids; index-based restore corrupts.

        Removing arm1 drops the LOWER-indexed actuators, so arm2's servo moves
        from index 1 to index 0. A flat-index copy would write arm1's value into
        arm2's slot.
        """
        sim = two_arm_sim
        world = sim._world
        mj = sim._mj
        # Distinct values so a mix-up is visible rather than coincidentally equal.
        sim.send_action({"pan": -0.4}, robot_name="arm1")
        sim.send_action({"pan": 0.7}, robot_name="arm2")
        sim.step(n_steps=50)
        arm2_index_before = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_ACTUATOR, "arm2/pan_act")

        assert sim.remove_robot("arm1")["status"] == "success"

        arm2_index_after = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_ACTUATOR, "arm2/pan_act")
        assert arm2_index_after != arm2_index_before, "fixture did not shift the actuator index"
        assert _ctrl_by_name(sim, "arm2/pan_act") == pytest.approx(0.7, abs=1e-6)

    def test_the_ejected_robots_ctrl_is_dropped(self, two_arm_sim):
        """A vanished actuator name must be skipped, not raise."""
        sim = two_arm_sim
        sim.send_action({"pan": -0.4}, robot_name="arm1")
        sim.step(n_steps=50)

        assert sim.remove_robot("arm1")["status"] == "success"

        world = sim._world
        mj = sim._mj
        assert mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_ACTUATOR, "arm1/pan_act") < 0
        assert int(world._model.nu) == 1, "the ejected actuator survived the rebuild"

    def test_latched_qfrc_applied_survives(self, two_arm_sim):
        """apply_force latches a torque in qfrc_applied; the fresh MjData zeroed it."""
        sim = two_arm_sim
        world = sim._world
        mj = sim._mj
        jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, "arm2/pan")
        dof_adr = int(world._model.jnt_dofadr[jid])
        world._data.qfrc_applied[dof_adr] = 0.25

        assert sim.remove_robot("arm1")["status"] == "success"

        new_jid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_JOINT, "arm2/pan")
        new_adr = int(sim._world._model.jnt_dofadr[new_jid])
        assert float(sim._world._data.qfrc_applied[new_adr]) == pytest.approx(0.25, abs=1e-9)

    def test_an_unlatched_qfrc_applied_stays_zero(self, two_arm_sim):
        """Only non-zero entries are carried, so nothing is invented."""
        sim = two_arm_sim

        assert sim.remove_robot("arm1")["status"] == "success"

        assert not (sim._world._data.qfrc_applied != 0).any()


class TestActuationSnapshotHelper:
    def test_snapshot_keys_every_named_actuator(self, two_arm_sim):
        sim = two_arm_sim
        sim.send_action({"pan": 0.3}, robot_name="arm1")
        sim.step(n_steps=10)

        snapshot = scene_ops._snapshot_actuation(sim._world)

        assert set(snapshot["ctrl"]) == {"arm1/pan_act", "arm2/pan_act"}
        assert snapshot["ctrl"]["arm1/pan_act"] == pytest.approx(0.3, abs=1e-6)

    def test_snapshot_of_an_uncompiled_world_is_empty(self):
        sim = Simulation(tool_name="devx_eject_empty", mesh=False)
        try:
            assert scene_ops._snapshot_actuation(sim._world) == {} if sim._world is not None else True
        finally:
            sim.cleanup(policy_stop_timeout=0.5)

    def test_restoring_an_empty_snapshot_is_a_no_op(self, two_arm_sim):
        sim = two_arm_sim
        sim.send_action({"pan": 0.55}, robot_name="arm2")
        sim.step(n_steps=10)

        assert scene_ops._restore_actuation(sim._world, {}) == 0

        assert _ctrl_by_name(sim, "arm2/pan_act") == pytest.approx(0.55, abs=1e-6)

    def test_restore_skips_names_that_no_longer_resolve(self, two_arm_sim):
        sim = two_arm_sim

        restored = scene_ops._restore_actuation(
            sim._world, {"ctrl": {"ghost/nope_act": 1.0}, "act": {}, "qfrc_applied": {}}
        )

        assert restored == 0

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
  still be driven after a sibling is removed;
* the documented fail-soft contract: if the rebuild cannot compile, the eject
  reports failure rather than leaving a half-mutated world, and
  ``remove_robot`` surfaces that as a structured error; and
* a BODY-MOUNTED camera (a wrist camera, ``parent_body`` set) survives the
  rebuild still parented to its body. ``SpecBuilder.build`` does not attach
  robots, so it added such a camera before its parent existed and raised
  ``ValueError`` straight out of ``remove_robot`` - meaning no scene carrying a
  wrist camera could remove a robot at all, and the message blamed the caller
  ("Pass the fully-qualified body name") for the very name ``add_camera`` had
  just accepted. Mounted cameras are now deferred to
  ``SpecBuilder.add_deferred_cameras``, run after every survivor is attached.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder  # noqa: E402

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
def arm_path(tmp_path):
    """The single-joint arm written to disk, ready for ``add_robot``."""
    path = tmp_path / "arm.xml"
    path.write_text(_ARM_XML)
    return path


@pytest.fixture
def two_arm_sim(arm_path):
    """A compiled world holding two attached single-joint arms."""
    sim = Simulation(tool_name="devx_multi_eject", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm1", urdf_path=str(arm_path))
    sim.add_robot(name="arm2", urdf_path=str(arm_path))
    try:
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def three_arm_sim(arm_path):
    """A compiled world holding three attached single-joint arms."""
    sim = Simulation(tool_name="devx_multi_eject_3", mesh=False)
    sim.create_world()
    for name in ("arm1", "arm2", "arm3"):
        sim.add_robot(name=name, urdf_path=str(arm_path))
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


def _camera_parent_body(sim: Simulation, cam_name: str) -> str:
    """Name of the compiled body the camera hangs off (``"world"`` if unmounted)."""
    world = sim._world
    assert world is not None and world._model is not None
    mj = sim._mj
    cid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
    assert cid >= 0, f"camera {cam_name!r} not in compiled model"
    return str(world._model.body(int(world._model.cam_bodyid[cid])).name)


def _camera_xpos(sim: Simulation, cam_name: str) -> np.ndarray:
    """The camera's world position, as MuJoCo resolves it from its parent body."""
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    mj = sim._mj
    cid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
    assert cid >= 0, f"camera {cam_name!r} not in compiled model"
    return np.array(world._data.cam_xpos[cid], dtype=float).copy()


def _mount_wrist(sim: Simulation, robot: str, name: str = "wrist") -> None:
    """Mount a wrist camera on ``robot``'s only body, the way a real rig does."""
    result = sim.add_camera(name=name, parent_body=f"{robot}/link0", position=[0.05, 0.0, 0.02])
    assert result["status"] == "success", result


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


class TestBodyMountedCameraSurvivesTheRebuild:
    """A wrist camera must not make ``remove_robot`` unusable."""

    def test_remove_robot_succeeds_with_a_body_mounted_camera(self, two_arm_sim):
        """The rebuild completes instead of raising out of the tool contract.

        ``SpecBuilder.build`` added the mounted camera before any robot was
        attached, so the parent lookup failed and ``ValueError`` escaped
        ``remove_robot`` entirely - not even a structured error dict.
        """
        sim = two_arm_sim
        _mount_wrist(sim, "arm1")

        result = sim.remove_robot("arm2")

        assert result["status"] == "success", result
        assert "arm2" not in sim.list_robots()

    def test_the_camera_is_still_registered_and_compiled(self, two_arm_sim):
        """The survivor's camera comes back in both the registry and the model."""
        sim = two_arm_sim
        _mount_wrist(sim, "arm1")

        assert sim.remove_robot("arm2")["status"] == "success"

        assert "wrist" in sim.list_cameras()
        # Present in the freshly compiled model, not merely in the registry.
        assert _camera_parent_body(sim, "wrist") == "arm1/link0"

    def test_the_camera_is_still_mounted_not_reparented_to_the_world(self, two_arm_sim):
        """Mounting is the whole point: a world-fixed fallback would be wrong.

        Silently re-adding the camera under ``worldbody`` would keep
        ``remove_robot`` working while quietly turning a wrist camera into a
        static one, so the parent body is asserted explicitly.
        """
        sim = two_arm_sim
        _mount_wrist(sim, "arm1")
        before = _camera_parent_body(sim, "wrist")
        assert before == "arm1/link0"

        assert sim.remove_robot("arm2")["status"] == "success"

        assert _camera_parent_body(sim, "wrist") == before

    def test_the_camera_still_tracks_its_body_after_the_rebuild(self, two_arm_sim):
        """Driving the survivor moves its camera - the observable effect of mounting."""
        sim = two_arm_sim
        _mount_wrist(sim, "arm1")
        assert sim.remove_robot("arm2")["status"] == "success"

        at_rest = _camera_xpos(sim, "wrist")
        send = sim.send_action({"arm1/pan": 1.2}, robot_name="arm1")
        assert not (isinstance(send, dict) and send.get("status") == "error"), send
        for _ in range(200):
            sim.step()

        assert _joint_qpos(sim, "arm1/pan") > 0.5, "survivor did not track the command"
        travelled = float(np.linalg.norm(_camera_xpos(sim, "wrist") - at_rest))
        assert travelled > 1e-3, f"camera did not follow its body (moved {travelled:.6f} m)"

    def test_the_camera_still_renders_after_the_rebuild(self, two_arm_sim):
        """The camera is usable as an observation source, not just present."""
        sim = two_arm_sim
        _mount_wrist(sim, "arm1")
        assert sim.remove_robot("arm2")["status"] == "success"

        result = sim.render(camera_name="wrist", width=64, height=48)

        assert result["status"] == "success", result

    def test_a_world_fixed_camera_is_unaffected(self, two_arm_sim):
        """Control: only ``parent_body`` cameras were ever at risk."""
        sim = two_arm_sim
        assert sim.add_camera(name="overhead", position=[1.2, -1.2, 0.9], target=[0.0, 0.0, 0.3])["status"] == "success"

        assert sim.remove_robot("arm2")["status"] == "success"

        assert "overhead" in sim.list_cameras()
        assert _camera_parent_body(sim, "overhead") == "world"

    def test_the_camera_survives_two_consecutive_rebuilds(self, three_arm_sim):
        """Each rebuild re-runs the deferral, so it must be idempotent."""
        sim = three_arm_sim
        _mount_wrist(sim, "arm1")

        assert sim.remove_robot("arm3")["status"] == "success"
        assert sim.remove_robot("arm2")["status"] == "success"

        remaining = sim.list_robots()
        assert "arm1" in remaining and "arm2" not in remaining and "arm3" not in remaining
        assert _camera_parent_body(sim, "wrist") == "arm1/link0"


class TestCameraOnTheRemovedRobot:
    """A camera whose parent is being removed has no parent to keep."""

    def test_it_is_dropped_rather_than_aborting_the_removal(self, two_arm_sim, caplog):
        """Removing the mount point must stay possible, and say what it dropped.

        The alternative - refusing the removal - would make a robot carrying a
        wrist camera unremovable, which is the defect this replaces. Dropping
        matches how the robot's own URDF cameras are already handled.
        """
        sim = two_arm_sim
        _mount_wrist(sim, "arm2", name="doomed")

        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.scene_ops"):
            result = sim.remove_robot("arm2")

        assert result["status"] == "success", result
        assert "doomed" not in sim.list_cameras(), "a camera with no parent stayed registered"
        assert "doomed" in caplog.text and "arm2" in caplog.text

    def test_a_sibling_mounted_camera_is_untouched(self, two_arm_sim):
        """Only the parentless camera goes; the survivor's is mounted as usual."""
        sim = two_arm_sim
        _mount_wrist(sim, "arm1", name="keeper")
        _mount_wrist(sim, "arm2", name="doomed")

        assert sim.remove_robot("arm2")["status"] == "success"

        assert "doomed" not in sim.list_cameras()
        assert _camera_parent_body(sim, "keeper") == "arm1/link0"


class TestDeferralContract:
    """``build`` defers mounted cameras; ``add_deferred_cameras`` mounts them."""

    def test_build_defers_a_mounted_camera_and_reports_it(self, two_arm_sim):
        """A bare ``build`` spec holds no robot bodies, so it can mount nothing."""
        sim = two_arm_sim
        _mount_wrist(sim, "arm1")
        world = sim._world
        assert world is not None

        spec = SpecBuilder.build(world)

        assert [c.name for c in spec.cameras] == ["default"]
        # Nothing to mount on yet, so the camera is reported, not raised.
        assert SpecBuilder.add_deferred_cameras(spec, world) == ["wrist"]

    def test_it_mounts_once_the_parent_is_attached(self, two_arm_sim):
        """With the robot attached, the same call mounts and reports nothing."""
        sim = two_arm_sim
        _mount_wrist(sim, "arm1")
        world = sim._world
        assert world is not None
        robot = world.robots["arm1"]

        spec = SpecBuilder.build(world)
        SpecBuilder.attach_robot(spec, robot, robot.urdf_path)

        assert SpecBuilder.add_deferred_cameras(spec, world) == []
        assert "wrist" in [c.name for c in spec.cameras]

    def test_a_parent_body_that_exists_nowhere_is_still_refused(self, two_arm_sim):
        """The deferral must not weaken the guard on a genuinely bad mount point."""
        sim = two_arm_sim

        result = sim.add_camera(name="bad", parent_body="nosuchrobot/link0", position=[0.0, 0.0, 0.1])

        assert result["status"] == "error", result
        assert "nosuchrobot/link0" in result["content"][0]["text"]
        assert "bad" not in sim.list_cameras()

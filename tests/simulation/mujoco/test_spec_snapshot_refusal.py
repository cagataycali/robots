"""The ``_snapshot_spec`` refusal contract: no way back means no mutation.

Every scene mutation that is only validated by the recompile it precedes takes a
deep copy of the live ``MjSpec`` first, via
:func:`strands_robots.simulation.mujoco.scene_ops._snapshot_spec`. That snapshot
is the only way back: the spec can carry state that exists ONLY on the spec and
is absent from the ``_world`` registries - weld equalities from
``attach_bodies``, actuators from ``actuate_robot``, bodies from
``patch_scene_mjcf`` - so a rebuild-from-registry rollback would silently drop
them.

The helper's docstring states the consequence for a caller: "A caller that
cannot snapshot must refuse its mutation rather than proceed with no way back."
Proceeding anyway is the failure this contract exists to prevent - a mutation
applied to the live spec with nothing to restore leaves an orphan behind, and an
orphan makes every LATER scene mutation fail to recompile, bricking the whole
world after one bad call.

Five callers share the helper and each refuses through its OWN channel, which
is why one test per surface is not one test five times:

* ``add_robot`` -> ``inject_robot_into_scene`` returns ``False``, and the caller
  also unwinds the ``_world.robots`` registry entry it had already made,
* ``actuate_robot`` -> ``actuate_robot_in_scene`` returns ``False``,
* ``patch_scene_mjcf`` -> raises ``RuntimeError``, which the facade converts,
* ``detach_bodies`` -> ``remove_equality_constraint`` returns ``False``, and the
  weld it was about to delete is still holding the pair, so the attachment
  ``attach_bodies`` recorded stays true,
* ``remove_camera`` -> ``eject_camera_from_scene`` returns ``False``, and the
  caller keeps the ``_world.cameras`` entry it had not dropped yet.

Each test asserts the whole cost of the refusal rather than only its status: the
compiled model is unchanged, the mutation's own element is absent, the scene is
still mutable, and - because a copy failure is transient - the identical call
succeeds once it clears. A caller that proceeded without a snapshot would fail
that last assertion, not the first.
"""

from __future__ import annotations

import logging

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_actuate_robot import MINI_ARM_URDF  # noqa: E402

_LOGGER_NAME = "strands_robots.simulation.mujoco.scene_ops"

# The two failures the helper's ``except`` names. ``MjSpec.copy`` is a pybind11
# binding, so both are plausible for a spec the copy cannot represent.
_REFUSED_COPIES = [
    pytest.param(RuntimeError("MjSpec.copy unavailable"), id="RuntimeError"),
    pytest.param(ValueError("MjSpec.copy rejected the spec"), id="ValueError"),
]


class _SpecWithNoCopy:
    """Stands in for a live spec whose deep copy cannot be taken."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def copy(self) -> object:
        raise self._exc


def _refuse_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the snapshot every scene mutation takes fail.

    Patches the binding rather than the helper so the refusal enters through the
    same call the production path makes. ``monkeypatch.undo()`` in the tests
    models the failure clearing, which is what lets the retry assertion show the
    refusal cost the caller nothing permanent.
    """

    def _boom(_self: object) -> object:
        raise RuntimeError("simulated MjSpec.copy failure")

    monkeypatch.setattr(mj.MjSpec, "copy", _boom)


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_spec_snapshot_refusal", mesh=False)
    s.create_world()
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def arm_sim(sim, tmp_path):
    """A world holding an actuator-less URDF arm, ready for ``actuate_robot``."""
    urdf = tmp_path / "mini_arm.urdf"
    urdf.write_text(MINI_ARM_URDF)
    assert sim.add_robot(name="arm", urdf_path=str(urdf))["status"] == "success"
    assert sim._world is not None
    assert sim._world._model.nu == 0, "the arm must load actuator-less for this test"
    return sim


def _body_id(sim: Simulation, name: str) -> int:
    assert sim._world is not None
    return int(mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, name))


class TestTheSnapshotHelperReportsAFailedCopy:
    @pytest.mark.parametrize("exc", _REFUSED_COPIES)
    def test_a_refused_copy_reports_no_snapshot(self, exc: BaseException, caplog) -> None:
        """A copy the binding refuses is reported as "no snapshot", with the reason."""
        with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
            assert scene_ops._snapshot_spec(_SpecWithNoCopy(exc), context="inject_robot 'arm'") is None

        logged = caplog.text
        assert "cannot snapshot the scene spec" in logged
        # The caller's context is what tells an operator WHICH mutation refused.
        assert "inject_robot 'arm'" in logged
        assert str(exc) in logged

    def test_an_unexpected_failure_propagates(self) -> None:
        """Only the two named failures mean "no snapshot"; anything else propagates.

        Folding an unknown failure into ``None`` would let a caller report a
        clean refusal for a fault it has not diagnosed, so the narrow ``except``
        is part of the contract.
        """
        with pytest.raises(TypeError, match="not a spec"):
            scene_ops._snapshot_spec(_SpecWithNoCopy(TypeError("not a spec")), context="ctx")

    def test_a_working_copy_hands_back_an_independent_spec(self, sim: Simulation) -> None:
        """Non-vacuity: on a real live spec the helper returns a distinct copy."""
        assert sim._world is not None
        spec = sim._world._backend_state["spec"]
        backup = scene_ops._snapshot_spec(spec, context="ctx")
        assert backup is not None
        assert backup is not spec


class TestAddRobotRefusesWithoutAWayBack:
    def test_the_robot_is_refused_and_the_refusal_costs_nothing(
        self, sim: Simulation, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        urdf = tmp_path / "mini_arm.urdf"
        urdf.write_text(MINI_ARM_URDF)
        assert sim._world is not None
        nbody_before = sim._world._model.nbody

        _refuse_snapshots(monkeypatch)
        refused = sim.add_robot(name="arm", urdf_path=str(urdf))
        monkeypatch.undo()

        assert refused["status"] == "error"
        assert "arm" in refused["content"][0]["text"]
        # Nothing was attached, and the registry entry the caller's name claimed
        # was unwound - so the name is free rather than half-taken.
        assert sim._world._model.nbody == nbody_before
        assert "arm" not in sim._world.robots

        # The scene is still mutable, and the identical add succeeds once the
        # copy failure clears. A mutation applied with no way back would have
        # left an orphan that makes both of these fail.
        assert (
            sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])["status"]
            == "success"
        )
        assert sim.add_robot(name="arm", urdf_path=str(urdf))["status"] == "success"
        assert sim._world._model.nbody > nbody_before
        assert "arm" in sim._world.robots


class TestActuateRobotRefusesWithoutAWayBack:
    def test_the_surgery_is_refused_and_no_actuator_is_added(
        self, arm_sim: Simulation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sim = arm_sim
        assert sim._world is not None

        _refuse_snapshots(monkeypatch)
        refused = sim.actuate_robot(robot_name="arm", kp=30.0)
        monkeypatch.undo()

        assert refused["status"] == "error"
        assert "arm" in refused["content"][0]["text"]
        # actuate_robot rewrites option, joints, geoms and actuators together,
        # which is why it needs a snapshot at all: none of it was applied.
        assert sim._world._model.nu == 0
        assert list(sim._world.robots["arm"].actuator_ids) == []

        assert (
            sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])["status"]
            == "success"
        )
        assert sim.actuate_robot(robot_name="arm", kp=30.0)["status"] == "success"
        assert sim._world._model.nu > 0


class TestPatchSceneMjcfRefusesWithoutAWayBack:
    def test_the_batch_is_refused_with_the_reason_and_nothing_is_patched(
        self, sim: Simulation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert sim._world is not None
        nbody_before = sim._world._model.nbody
        ops = [{"op": "add_body", "name": "marker", "pos": [0.3, 0, 0.2]}]

        _refuse_snapshots(monkeypatch)
        refused = sim.patch_scene_mjcf(ops)
        monkeypatch.undo()

        assert refused["status"] == "error"
        message = refused["content"][0]["text"]
        # This caller refuses by raising, so the reason has to survive the
        # facade's conversion to the tool envelope rather than being flattened
        # into a bare "patch failed".
        assert "snapshot" in message.lower()
        assert sim._world._model.nbody == nbody_before
        assert _body_id(sim, "marker") == -1

        assert (
            sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])["status"]
            == "success"
        )
        assert sim.patch_scene_mjcf(ops)["status"] == "success"
        assert _body_id(sim, "marker") >= 0


class TestDetachBodiesRefusesWithoutAWayBack:
    def test_the_weld_removal_is_refused_and_the_pair_stays_attached(
        self, sim: Simulation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert sim._world is not None
        for name, z in (("carrier", 0.6), ("cube", 0.45)):
            assert (
                sim.add_object(name=name, shape="box", size=[0.08, 0.08, 0.08], position=[0.0, 0.0, z])["status"]
                == "success"
            )
        assert sim.attach_bodies("carrier", "cube", mode="weld")["status"] == "success"
        neq_before = int(sim._world._model.neq)
        assert neq_before == 1

        _refuse_snapshots(monkeypatch)
        refused = sim.detach_bodies("carrier", "cube")
        monkeypatch.undo()

        assert refused["status"] == "error"
        assert "cube" in refused["content"][0]["text"]
        # Nothing was deleted, so the recorded attachment is still true of the
        # scene: the weld is on the live spec and in the compiled model.
        spec = scene_ops._get_spec(sim._world)
        assert spec is not None
        assert "attach_weld_carrier__cube" in [eq.name for eq in spec.equalities]
        assert int(sim._world._model.neq) == neq_before
        assert sim.attachment_involving("cube") == "cube"

        # The scene is still mutable and the identical detach succeeds once the
        # copy failure clears. A delete applied with no way back would instead
        # be refused here as "not found", and the add_object would apply it.
        assert (
            sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])["status"]
            == "success"
        )
        assert int(sim._world._model.neq) == neq_before
        assert sim.detach_bodies("carrier", "cube")["status"] == "success"
        assert int(sim._world._model.neq) == 0
        assert sim.attachment_involving("cube") is None


class TestRemoveCameraRefusesWithoutAWayBack:
    def test_the_camera_is_kept_and_the_refusal_costs_nothing(
        self, sim: Simulation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert sim._world is not None
        assert sim.add_camera(name="watch", position=[0.7, -0.7, 0.5], target=[0, 0, 0.1])["status"] == "success"
        ncam_before = int(sim._world._model.ncam)

        _refuse_snapshots(monkeypatch)
        refused = sim.remove_camera("watch")
        monkeypatch.undo()

        assert refused["status"] == "error"
        assert "watch" in refused["content"][0]["text"]
        # This caller had not dropped its registry entry yet, so refusing before
        # the delete leaves nothing to unwind - the camera is simply still there.
        assert "watch" in sim._world.cameras
        assert int(sim._world._model.ncam) == ncam_before

        assert (
            sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])["status"]
            == "success"
        )
        assert sim.remove_camera("watch")["status"] == "success"
        assert "watch" not in sim._world.cameras
        assert int(sim._world._model.ncam) == ncam_before - 1

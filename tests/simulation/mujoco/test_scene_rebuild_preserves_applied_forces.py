"""A force latched before a scene rebuild is still applied after it.

Both operations that rebuild a scene's ``MjData`` used to discard the two
externally-applied-force buffers, and each did it silently -- the call returned
``"success"`` and the push simply stopped:

* growing the scene (``add_robot``, ``add_object``, ``add_camera`` ->
  ``_recompile_preserving_state``). ``spec.recompile`` transfers ``qpos``,
  ``qvel``, ``ctrl``, ``act`` and the clock but not ``qfrc_applied`` or
  ``xfrc_applied``, and nothing put them back.
* shrinking it (``remove_robot`` -> ``eject_robot_from_scene``). A fresh
  ``MjData`` zero-fills both, and the name-keyed restore carried only the
  joint-space half.

Three places in the package already state the contract this pins, which is why
it is a defect rather than an open question:
:meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.apply_force` documents
that MuJoCo never clears a latched wrench, so it persists "until the next
apply_force call for this body (or a ``reset()``)" -- a list a scene rebuild is
not on; :meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.add_robot`
promises that "a latched ``apply_force`` wrench persists" across the very
recompile it performs; and ``save_state`` checkpoints both buffers so a
``load_state`` restores "latched external forces, not just positions".

Every preservation test measures the force by its EFFECT from rest, not by
reading the buffer back. A body that was already moving keeps coasting in the
absence of gravity and damping, so displacement alone stays positive after the
wrench is gone: measured on the pre-fix tree, a drifter that had run 50 steps
under a 10 N push still advanced 0.098 m over the next 50 with an all-zero
``xfrc_applied`` row, purely on accumulated velocity, while the same 50 steps
FROM REST advanced 0.000 m. Zeroing the velocity first is what makes the
assertion about the force.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A single free body in a gravity-free world: with no gravity, no damping and no
# actuator, the ONLY thing that can accelerate it is a latched wrench, so the
# distance it covers from rest is a direct readout of that wrench.
_DRIFTER_XML = """
<mujoco model="drifter">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hull" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# A hinge with no actuator: only ``qfrc_applied`` can drive it, which is the
# joint-space half of the same concept.
_HINGE_XML = """
<mujoco model="hinge_only">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.1">
      <joint name="spin" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_PUSH = [10.0, 0.0, 0.0]
_TORQUE = [0.0, 0.0, 0.25]
_COAST_STEPS = 50


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_rebuild_forces", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _write(tmp_path, name: str, xml: str) -> str:
    path = tmp_path / f"{name}.xml"
    path.write_text(xml)
    return str(path)


def _handles(sim: Simulation, body: str):
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    mj = sim._mj
    model, data = world._model, world._data
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, f"body {body!r} missing from the compiled model"
    return mj, model, data, bid


def _wrench(sim: Simulation, body: str) -> list[float]:
    _, _, data, bid = _handles(sim, body)
    return [float(v) for v in data.xfrc_applied[bid]]


def _travel_from_rest(sim: Simulation, body: str, axis: int = 0) -> float:
    """Distance ``body`` covers in ``_COAST_STEPS`` starting from zero velocity.

    Zeroing the velocity first is the whole point: it removes the accumulated
    momentum that would otherwise keep a body moving after its wrench was lost,
    so a non-zero result can only come from a force that is still applied.
    """
    mj, model, data, bid = _handles(sim, body)
    jid = int(model.body_jntadr[bid])
    dof = int(model.jnt_dofadr[jid])
    data.qvel[dof : dof + 6] = 0.0
    mj.mj_forward(model, data)
    start = float(data.xpos[bid][axis])
    assert sim.step(_COAST_STEPS)["status"] == "success"
    _, _, data, bid = _handles(sim, body)
    return float(data.xpos[bid][axis]) - start


def _joint_angle(sim: Simulation, joint: str) -> float:
    mj = sim._mj
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jid >= 0, f"joint {joint!r} missing from the compiled model"
    return float(world._data.qpos[int(world._model.jnt_qposadr[jid])])


def _pushed_drifter(sim: Simulation, tmp_path) -> tuple[str, float]:
    """A world holding one pushed drifter. Returns ``(body, travel_from_rest)``."""
    assert sim.create_world(gravity=[0.0, 0.0, 0.0])["status"] == "success"
    assert sim.add_robot(name="probe", urdf_path=_write(tmp_path, "drifter", _DRIFTER_XML))["status"] == "success"
    body = "probe/hull"
    assert sim.apply_force(body_name=body, force=_PUSH, torque=_TORQUE)["status"] == "success"
    baseline = _travel_from_rest(sim, body)
    assert baseline > 1e-4, f"premise: the wrench moves it before any rebuild (got {baseline})"
    return body, baseline


class TestALatchedWrenchSurvivesAnyRebuild:
    """``apply_force`` outlives the rebuild, whichever direction the scene changes."""

    def test_it_still_pushes_after_another_robot_is_removed(self, sim, tmp_path):
        body, baseline = _pushed_drifter(sim, tmp_path)
        assert (
            sim.add_robot(name="doomed", urdf_path=_write(tmp_path, "hinge_only", _HINGE_XML), position=[2, 0, 0])[
                "status"
            ]
            == "success"
        )

        assert sim.remove_robot("doomed")["status"] == "success"

        assert _travel_from_rest(sim, body) == pytest.approx(baseline, rel=1e-6), (
            "removing some other robot revoked the wrench: the drifter no longer "
            f"accelerates (was {baseline} m from rest), yet remove_robot reported success"
        )
        assert _wrench(sim, body) == pytest.approx([*_PUSH, *_TORQUE])

    def test_it_still_pushes_after_another_robot_is_added(self, sim, tmp_path):
        body, baseline = _pushed_drifter(sim, tmp_path)

        assert (
            sim.add_robot(name="newcomer", urdf_path=_write(tmp_path, "hinge_only", _HINGE_XML), position=[2, 0, 0])[
                "status"
            ]
            == "success"
        )

        assert _travel_from_rest(sim, body) == pytest.approx(baseline, rel=1e-6), (
            "add_robot documents that 'a latched apply_force wrench persists', but "
            f"the drifter stopped accelerating (was {baseline} m from rest)"
        )

    def test_it_still_pushes_after_an_object_is_added(self, sim, tmp_path):
        body, baseline = _pushed_drifter(sim, tmp_path)

        assert sim.add_object(name="crate", shape="box", position=[2, 0, 0.1])["status"] == "success"

        assert _travel_from_rest(sim, body) == pytest.approx(baseline, rel=1e-6), (
            "add_object rebuilds the scene through the same recompile, so it must not revoke a wrench either"
        )


class TestTheJointSpaceHalfSurvivesToo:
    """``qfrc_applied`` is the joint-keyed sibling of the same concept."""

    def test_a_joint_torque_survives_a_scene_grow(self, sim, tmp_path):
        assert sim.create_world(gravity=[0.0, 0.0, 0.0])["status"] == "success"
        assert (
            sim.add_robot(name="spinner", urdf_path=_write(tmp_path, "hinge_only", _HINGE_XML))["status"] == "success"
        )
        mj, model, data, _ = _handles(sim, "spinner/link")
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "spinner/spin")
        assert jid >= 0
        data.qfrc_applied[int(model.jnt_dofadr[jid])] = 0.5

        assert sim.add_object(name="crate", shape="box", position=[2, 0, 0.1])["status"] == "success"

        _, model, data, _ = _handles(sim, "spinner/link")
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "spinner/spin")
        assert float(data.qfrc_applied[int(model.jnt_dofadr[jid])]) == pytest.approx(0.5), (
            "spec.recompile drops qfrc_applied, so a joint torque a caller latched "
            "stopped driving the joint as soon as anything entered the scene"
        )
        before = _joint_angle(sim, "spinner/spin")
        assert sim.step(_COAST_STEPS)["status"] == "success"
        assert _joint_angle(sim, "spinner/spin") > before + 1e-6


class TestTheRestoreDoesNotReachPastTheDefect:
    """Controls: these hold on the pre-fix tree too."""

    def test_the_departing_robots_wrench_is_not_resurrected(self, sim, tmp_path):
        """A wrench on the ejected robot must vanish with it, not land elsewhere."""
        body, _ = _pushed_drifter(sim, tmp_path)
        hinge = _write(tmp_path, "hinge_only", _HINGE_XML)
        assert sim.add_robot(name="doomed", urdf_path=hinge, position=[2, 0, 0])["status"] == "success"
        assert sim.apply_force(body_name="doomed/link", force=[0.0, 7.0, 0.0])["status"] == "success"

        assert sim.remove_robot("doomed")["status"] == "success"

        _, model, data, _ = _handles(sim, body)
        # No surviving body may carry the departed body's distinctive +7 N on y.
        for bid in range(int(model.nbody)):
            assert float(data.xfrc_applied[bid][1]) != pytest.approx(7.0), (
                "the ejected robot's wrench was re-keyed onto a surviving body"
            )

    def test_a_scene_nobody_pushed_on_is_unchanged_by_a_rebuild(self, sim, tmp_path):
        """The no-op case: with no force latched, nothing is written back."""
        assert sim.create_world(gravity=[0.0, 0.0, 0.0])["status"] == "success"
        assert sim.add_robot(name="probe", urdf_path=_write(tmp_path, "drifter", _DRIFTER_XML))["status"] == "success"

        assert sim.add_object(name="crate", shape="box", position=[2, 0, 0.1])["status"] == "success"

        _, model, data, _ = _handles(sim, "probe/hull")
        assert not data.xfrc_applied.any(), "a scene nobody pushed on gained a wrench"
        assert not data.qfrc_applied.any()

    def test_apply_force_still_replaces_rather_than_accumulates(self, sim, tmp_path):
        """Per-body replacement is the documented contract; restoring must not stack."""
        body, _ = _pushed_drifter(sim, tmp_path)
        assert sim.add_object(name="crate", shape="box", position=[2, 0, 0.1])["status"] == "success"

        assert sim.apply_force(body_name=body, force=[1.0, 0.0, 0.0])["status"] == "success"

        assert _wrench(sim, body) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (
            "the restored wrench was added to the new one instead of replaced"
        )

    def test_a_reset_still_revokes_the_wrench(self, sim, tmp_path):
        """``reset()`` is on apply_force's documented revocation list; keep it there."""
        body, _ = _pushed_drifter(sim, tmp_path)

        assert sim.reset()["status"] == "success"

        assert not any(_wrench(sim, body)), "reset must still clear a latched wrench"

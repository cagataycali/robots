"""Spec rollback when an object/camera/robot scene injection recompile fails.

``add_object`` / ``add_camera`` / ``add_robot`` mutate the live ``MjSpec``
(insert the body or camera, or attach the robot's subtree) *before* the
recompile that validates the result. If that recompile is
refused - e.g. an object references a mesh asset that was never registered -
the just-inserted element must be rolled back out of the spec, not merely
popped from the Python-side ``_world`` registry. Otherwise the orphan element
lingers in the spec and every subsequent scene mutation keeps failing to
recompile (``repeated name`` collisions), bricking the whole scene after a
single bad add.

The observable proof of correct rollback is that the *same name* can be added
successfully right after a failed attempt: a leaked orphan would make the retry
collide on the duplicate name at recompile time. These tests fail before the
rollback fix (the retry errors with ``repeated name``) and pass after it.

The robot path is the same contract with a wider blast radius: a robot's subtree
carries its own assets, so a model whose mesh file cannot be opened attaches
fine and is only refused at recompile. Leaving that subtree behind bricked the
whole world - every later ``add_object`` / ``add_camera`` / ``add_robot``
recompiled the same broken spec and reported "spec recompile refused" with
nothing wrong with it, and each failed retry leaked another subtree. Rolling the
attach back out is what keeps a refused robot costing exactly one refused add.

``add_object`` / ``add_camera`` also carry an ``except (ValueError,
RuntimeError)`` around the injection call. It surfaces a raise from the
scene-injection layer as a structured ``{'status': 'error'}`` (never re-raised
past tool dispatch, per the tool contract) while rolling the half-added element
out of the ``_world`` registry. For objects this is a live path, not a purely
defensive one: ``inject_object_into_scene`` rolls its spec mutation back and
re-raises so the reason reaches the caller (see
``test_add_object_mass_contract.py`` for the unsupported-shape case). The
camera injector still folds a raise into ``False``, so its guard fires only on
an unexpected error. The raise-path tests force the injector to raise and pin
that contract; they fail if the guard is removed (the exception escapes) or if
the cleanup ``pop`` is dropped (the ghost element leaks).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_inject_cleanup", mesh=False)
    s.create_world()
    s.add_robot("so100")  # a compiled world the injectors can recompile against
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestAddObjectInjectionRollback:
    def test_bad_mesh_rolls_back_and_same_name_is_reusable(self, sim):
        # shape="mesh" with a non-existent file -> recompile refused.
        result = sim.add_object(
            name="widget",
            shape="mesh",
            mesh_path="/nonexistent/does-not-exist.stl",
            position=[0.3, 0.0, 0.1],
        )
        assert result["status"] == "error"
        assert "widget" not in sim._world.objects

        # The spec was rolled back, so the same name is free: a valid object
        # under that name now compiles. (Pre-fix this errored with a
        # "repeated name 'widget' in body" recompile failure.)
        retry = sim.add_object(name="widget", shape="box", position=[0.3, 0.0, 0.1])
        assert retry["status"] == "success"
        assert "widget" in sim._world.objects

    def test_failed_add_does_not_brick_other_objects(self, sim):
        bad = sim.add_object(name="bad", shape="mesh", mesh_path="/nope.stl", position=[0.3, 0.0, 0.1])
        assert bad["status"] == "error"
        # A completely different valid object still compiles - the scene is
        # not bricked by the earlier failure.
        ok = sim.add_object(name="cube", shape="box", position=[0.2, 0.1, 0.1])
        assert ok["status"] == "success"
        assert "cube" in sim._world.objects

    def test_injector_raising_is_caught_cleaned_up_not_reraised(self, sim, monkeypatch):
        # The injectors normally swallow (ValueError, RuntimeError) and return
        # False, so this defensive branch only fires if the scene-injection
        # layer raises unexpectedly. Force that: add_object must surface a
        # structured error (not re-raise past dispatch) and roll the
        # half-added object back out of the registry.
        from strands_robots.simulation.mujoco import simulation as sim_mod

        def boom(_world, _obj):
            raise RuntimeError("spec.recompile blew up")

        monkeypatch.setattr(sim_mod, "inject_object_into_scene", boom)

        result = sim.add_object(name="ghost", shape="box", position=[0.2, 0.0, 0.1])
        assert result["status"] == "error"
        assert "ghost" not in sim._world.objects
        text = result["content"][0]["text"]
        assert "into live scene" in text
        assert "spec.recompile blew up" in text


# A robot model whose mesh asset cannot be opened: MjSpec.from_file parses it and
# spec.attach merges it, and only the recompile that loads the asset is refused.
# That is the narrowest way to reach the rollback path with no asset download.
_UNLOADABLE_MESH_MJCF = """<mujoco model="badbot">
  <asset><mesh name="ghost" file="definitely-absent.stl"/></asset>
  <worldbody>
    <body name="base">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="g" type="mesh" mesh="ghost"/>
    </body>
  </worldbody>
  <actuator><position name="a1" joint="j1"/></actuator>
</mujoco>
"""

_VALID_MJCF = """<mujoco model="goodbot">
  <worldbody>
    <body name="base">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="g" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <actuator><position name="a1" joint="j1"/></actuator>
</mujoco>
"""


def _spec_body_names(sim) -> list[str]:
    return [b.name for b in sim._world._backend_state["spec"].bodies]


class TestAddRobotInjectionRollback:
    """A robot whose attach compiles into a model MuJoCo refuses is rolled back."""

    def test_refused_robot_leaves_the_scene_mutable(self, sim, tmp_path):
        bad = tmp_path / "badbot.xml"
        bad.write_text(_UNLOADABLE_MESH_MJCF)

        refused = sim.add_robot(name="badbot", urdf_path=str(bad))
        assert refused["status"] == "error"
        assert "badbot" not in sim._world.robots

        # Every kind of later scene mutation still compiles. Pre-fix all three
        # failed with "spec recompile refused" / "Failed to inject robot",
        # because they recompiled the spec the refused attach was left in.
        assert sim.add_object(name="marker", shape="box", position=[0.3, 0.0, 0.1])["status"] == "success"
        assert sim.add_camera(name="look", position=[1.0, -1.0, 0.8], target=[0.0, 0.0, 0.1])["status"] == "success"

        good = tmp_path / "goodbot.xml"
        good.write_text(_VALID_MJCF)
        assert sim.add_robot(name="goodbot", urdf_path=str(good))["status"] == "success"

    def test_refused_robot_name_is_reusable(self, sim, tmp_path):
        bad = tmp_path / "badbot.xml"
        bad.write_text(_UNLOADABLE_MESH_MJCF)
        assert sim.add_robot(name="arm", urdf_path=str(bad))["status"] == "error"

        # The rolled-back subtree freed the whole "arm/" namespace, so a
        # corrected model under the same name compiles.
        good = tmp_path / "goodbot.xml"
        good.write_text(_VALID_MJCF)
        retry = sim.add_robot(name="arm", urdf_path=str(good))
        assert retry["status"] == "success"
        assert "arm" in sim._world.robots

    def test_refused_robot_leaves_no_orphan_in_the_live_spec(self, sim, tmp_path):
        bad = tmp_path / "badbot.xml"
        bad.write_text(_UNLOADABLE_MESH_MJCF)
        before = _spec_body_names(sim)

        assert sim.add_robot(name="badbot", urdf_path=str(bad))["status"] == "error"

        after = _spec_body_names(sim)
        assert not [n for n in after if n.startswith("badbot/")], f"orphan subtree left in spec: {after}"
        # The spec is back to exactly the scene the world still describes.
        assert after == before

    def test_repeated_refusals_do_not_accumulate_orphans(self, sim, tmp_path):
        bad = tmp_path / "badbot.xml"
        bad.write_text(_UNLOADABLE_MESH_MJCF)
        before = _spec_body_names(sim)

        for attempt in range(3):
            assert sim.add_robot(name=f"badbot{attempt}", urdf_path=str(bad))["status"] == "error"

        assert _spec_body_names(sim) == before

    def test_rollback_preserves_the_surviving_robot_pose(self, sim, tmp_path):
        # The refused recompile never replaces world._model / _data, so putting
        # the spec back is the whole rollback and physics state is untouched. A
        # robot already in the scene must not be disturbed because a different
        # robot was refused.
        joints = sim._world.robots["so100"].joint_names
        pose = {name: 0.15 + 0.05 * i for i, name in enumerate(joints)}
        assert sim.set_joint_positions(positions=pose, robot_name="so100")["status"] == "success"
        before = {name: sim.get_observation(robot_name="so100")[name] for name in joints}

        bad = tmp_path / "badbot.xml"
        bad.write_text(_UNLOADABLE_MESH_MJCF)
        assert sim.add_robot(name="badbot", urdf_path=str(bad))["status"] == "error"

        after = {name: sim.get_observation(robot_name="so100")[name] for name in joints}
        assert after == pytest.approx(before)


class TestRollbackKeepsSpecOnlyState:
    """The rollback restores the scene, including state only the spec holds.

    A scene's live ``MjSpec`` can carry mutations that the ``_world`` registry
    never records: weld equalities from ``attach_bodies``, actuators from
    ``actuate_robot_in_scene``, bodies authored by ``patch_scene_mjcf``, whole
    scenes from ``replace_scene_mjcf``. A rollback that rebuilt the spec from
    the registry would silently drop every one of them, turning a correctly
    refused add into corruption of a scene that was healthy before it - which is
    why ``remove_robot`` refuses outright while an attachment is live instead of
    rebuilding over it. Restoring a snapshot of the spec keeps them.

    The snapshot has to be a spec copy rather than a ``spec.to_xml()`` round
    trip: the emitted MJCF loses the asset search paths the attached robot's
    mesh references were resolved against (``meshdir``) and re-declares its
    keyframes, so a restored round trip no longer compiles - a rollback that
    leaves the scene as broken as the orphan it removed. The ``sim`` fixture's
    ``so100`` is a mesh-bearing attached robot, so these tests fail against a
    round-trip snapshot as well as against a registry rebuild.
    """

    def test_a_live_weld_survives_a_refused_robot_add(self, sim, tmp_path):
        assert (
            sim.add_object(name="cube", shape="box", size=[0.04, 0.04, 0.04], position=[0.2, 0.0, 0.3])["status"]
            == "success"
        )
        assert sim.attach_bodies(parent="so100/Moving_Jaw", child="cube", mode="weld")["status"] == "success"
        welds_before = sim._world._model.neq

        bad = tmp_path / "badbot.xml"
        bad.write_text(_UNLOADABLE_MESH_MJCF)
        assert sim.add_robot(name="badbot", urdf_path=str(bad))["status"] == "error"

        # The weld is still enforced by physics, and still matches what the
        # attachment registry says is active - so detaching works. A rebuild
        # dropped the equality while leaving the registry entry behind, which
        # made detach_bodies fail and left "cube" unremovable.
        assert sim._world._model.neq == welds_before
        assert sim.detach_bodies(parent="so100/Moving_Jaw", child="cube")["status"] == "success"

    def test_an_agent_authored_body_survives_a_refused_robot_add(self, sim, tmp_path):
        assert (
            sim.patch_scene_mjcf(ops=[{"op": "add_body", "name": "rig", "position": [0.0, 0.3, 0.5]}])["status"]
            == "success"
        )

        bad = tmp_path / "badbot.xml"
        bad.write_text(_UNLOADABLE_MESH_MJCF)
        assert sim.add_robot(name="badbot", urdf_path=str(bad))["status"] == "error"

        # "rig" exists only on the spec - SpecBuilder.build has no idea it was
        # ever authored - so a registry rebuild is where it disappeared.
        assert "rig" in _spec_body_names(sim)
        assert sim.add_object(name="marker", shape="box", position=[0.3, 0.0, 0.1])["status"] == "success"

    def test_a_refused_patch_batch_leaves_the_scene_mutable(self, sim):
        # patch_scene_mjcf rolls its own batch back the same way. Restoring a
        # to_xml() round trip put an uncompilable spec back on a scene holding a
        # mesh-bearing attached robot, so the next mutation failed for a reason
        # that had nothing to do with it.
        result = sim.patch_scene_mjcf(
            ops=[
                {"op": "add_body", "name": "rig", "position": [0.0, 0.3, 0.5]},
                {"op": "add_geom", "body": "no_such_body", "shape": "box"},
            ]
        )
        assert result["status"] == "error"

        assert "rig" not in _spec_body_names(sim)
        assert sim.add_object(name="marker", shape="box", position=[0.3, 0.0, 0.1])["status"] == "success"


class TestAddCameraInjectionRollback:
    def test_recompile_refusal_rolls_back_and_same_name_is_reusable(self, sim, monkeypatch):
        # Camera-injection recompile failures are hard to trigger with valid
        # inputs, so force the recompile to refuse once. The real add_camera
        # spec mutation still runs, exercising the production rollback path
        # (SpecBuilder.remove_camera) - the fix under test, not a stub.
        real_recompile = scene_ops._recompile_preserving_state
        calls = {"n": 0}

        def flaky_recompile(world, spec):
            calls["n"] += 1
            if calls["n"] == 1:
                return False  # simulate a refused recompile on the first inject
            return real_recompile(world, spec)

        monkeypatch.setattr(scene_ops, "_recompile_preserving_state", flaky_recompile)

        result = sim.add_camera(name="wrist", position=[0.5, 0.0, 0.5], target=[0.0, 0.0, 0.1])
        assert result["status"] == "error"
        assert "wrist" not in sim._world.cameras

        # Second inject uses the real recompile; it succeeds only if the first
        # attempt's camera was rolled back out of the spec.
        retry = sim.add_camera(name="wrist", position=[0.5, 0.0, 0.5], target=[0.0, 0.0, 0.1])
        assert retry["status"] == "success"
        assert "wrist" in sim._world.cameras

    def test_injector_raising_is_caught_cleaned_up_not_reraised(self, sim, monkeypatch):
        # Mirror of the object raise-path: an unexpected raise from the camera
        # injection layer must be caught, surfaced as a structured error, and
        # the half-added camera rolled back out of the registry.
        from strands_robots.simulation.mujoco import simulation as sim_mod

        def boom(_world, _cam):
            raise ValueError("camera spec exploded")

        monkeypatch.setattr(sim_mod, "inject_camera_into_scene", boom)

        result = sim.add_camera(name="phantom", position=[0.5, 0.0, 0.5], target=[0.0, 0.0, 0.1])
        assert result["status"] == "error"
        assert "phantom" not in sim._world.cameras
        text = result["content"][0]["text"]
        assert "into live scene" in text
        assert "camera spec exploded" in text

"""Spec rollback when an object/camera scene injection recompile fails.

``add_object`` / ``add_camera`` mutate the live ``MjSpec`` (insert the body or
camera) *before* the recompile that validates the result. If that recompile is
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

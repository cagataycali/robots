"""``remove_object`` drops the cameras mounted on the body it removes.

A camera added with ``add_camera(parent_body=<object>)`` expresses its pose in
that body's frame, so MuJoCo makes it a child element of the body. Deleting the
body therefore deletes the camera at recompile time - the compiled model loses
it whether or not anyone asked. The Python-side ``world.cameras`` registry did
not follow, so ``remove_object`` reported success and left an entry naming a
camera the renderer could no longer resolve. Every camera consumer then
contradicted itself in one breath: ``list_cameras`` offered ``'watch'`` while
``render`` / ``get_camera_params`` refused it with

    Camera 'watch' not found. Available: ['default', 'watch']

naming the missing camera as an available alternative. The robot path already
does the right thing - :func:`~strands_robots.simulation.mujoco.scene_ops.eject_robot_from_scene`
drops both the robot's own URDF cameras and any camera whose parent body
belonged to it, each with a warning, because (in its own words) "stale entries
would linger in the registry and confuse observation code". These tests pin the
same cascade for the object path and the invariant behind it: the registry
never advertises a camera the compiled model does not have.

The two over-reach controls matter as much as the cascade: a camera mounted on a
*different* body and a world-fixed camera must both survive, so the drop is
scoped to the body actually being removed.

``patch_scene_mjcf(delete_body=...)`` is deliberately not covered - it is the
raw-MJCF escape hatch and says so in its own result text ("world.robots /
world.objects / world.cameras registries were NOT updated"), so registry drift
there is its documented contract rather than a defect.

GL-free: ``mesh=False`` and no rendering, so this runs without a GPU.
``get_camera_params`` resolves a camera from ``model``/``data`` alone and is the
consumer used to prove the message no longer offers what it cannot resolve.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_SCENE_OPS_LOGGER = scene_ops.__name__

# A two-body arm with no meshes and no actuators, so the robot-parity test needs
# no asset download and runs anywhere the rest of this file does.
_ARM_XML = """
<mujoco model="pcam_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.04 0.04 0.05"/>
      <body name="link" pos="0 0 0.1">
        <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _camera_names(sim: Any) -> list[str]:
    """The registry's own view of which cameras exist."""
    return list(sim.list_cameras())


def _model_camera_count(sim: Any) -> int:
    """How many cameras the compiled model actually carries."""
    return int(sim._world._model.ncam)


@pytest.fixture
def sim():
    """A world with a mounted camera, a camera on another body, and a fixed one."""
    s = Simulation(tool_name="test_remove_object_camera_cascade_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    s.add_object("crate", shape="box", size=[0.12, 0.12, 0.12], position=[0.3, 0.0, 0.06], mass=0.5)
    s.add_object("plate", shape="box", size=[0.2, 0.2, 0.02], position=[-0.3, 0.0, 0.01], is_static=True)
    s.add_camera(name="watch", position=[0.2, -0.2, 0.2], target=[0.0, 0.0, 0.0], parent_body="crate")
    s.add_camera(name="plate_cam", position=[0.0, -0.2, 0.2], target=[0.0, 0.0, 0.0], parent_body="plate")
    s.add_camera(name="fixed", position=[0.8, -0.8, 0.6], target=[0.0, 0.0, 0.1])
    yield s
    s.cleanup()


class TestTheMountedCameraGoesWithItsBody:
    def test_the_camera_mounted_on_the_removed_object_leaves_the_registry(self, sim) -> None:
        assert "watch" in _camera_names(sim)
        assert sim.remove_object("crate")["status"] == "success"
        assert "watch" not in _camera_names(sim)

    def test_the_registry_agrees_with_the_compiled_model(self, sim) -> None:
        # The premise: before the removal both views already agree.
        assert len(_camera_names(sim)) == _model_camera_count(sim)
        sim.remove_object("crate")
        assert len(_camera_names(sim)) == _model_camera_count(sim)

    def test_no_consumer_is_offered_the_camera_it_cannot_resolve(self, sim) -> None:
        sim.remove_object("crate")
        with pytest.raises(KeyError) as excinfo:
            sim.get_camera_params(camera_name="watch")
        message = str(excinfo.value)
        assert "'watch'" in message, message
        # The available list is the actionable half of that message, so it must
        # not name the camera the same sentence reports as missing.
        available = message.split("Available:", 1)[1]
        assert "watch" not in available, message

    def test_the_dropped_name_is_reusable(self, sim) -> None:
        sim.remove_object("crate")
        result = sim.add_camera(name="watch", position=[0.6, -0.3, 0.3], target=[0.0, 0.0, 0.1])
        assert result["status"] == "success", result
        assert sim.get_camera_params(camera_name="watch") is not None

    def test_the_drop_is_reported_with_the_camera_and_the_body(self, sim, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=_SCENE_OPS_LOGGER):
            sim.remove_object("crate")
        dropped = [r.getMessage() for r in caplog.records if "dropping camera" in r.getMessage()]
        assert len(dropped) == 1, [r.getMessage() for r in caplog.records]
        assert "'watch'" in dropped[0] and "'crate'" in dropped[0], dropped[0]


class TestTheDropIsScopedToTheRemovedBody:
    def test_a_camera_mounted_on_a_surviving_body_still_resolves(self, sim) -> None:
        sim.remove_object("crate")
        assert "plate_cam" in _camera_names(sim)
        assert sim.get_camera_params(camera_name="plate_cam") is not None

    def test_a_world_fixed_camera_still_resolves(self, sim) -> None:
        sim.remove_object("crate")
        assert "fixed" in _camera_names(sim)
        assert sim.get_camera_params(camera_name="fixed") is not None

    def test_removing_an_object_that_carries_no_camera_drops_nothing(self, sim, caplog) -> None:
        before = _camera_names(sim)
        with caplog.at_level(logging.WARNING, logger=_SCENE_OPS_LOGGER):
            assert sim.remove_object("plate")["status"] == "success"
        assert "plate_cam" not in _camera_names(sim), "the plate's own camera goes with it"
        assert "watch" in _camera_names(sim) and "fixed" in _camera_names(sim)
        assert len(_camera_names(sim)) == len(before) - 1
        assert len(_camera_names(sim)) == _model_camera_count(sim)

    def test_ejecting_a_body_absent_from_the_spec_drops_nothing(self, sim) -> None:
        # eject_body_from_scene returns True for a body it cannot find (the
        # caller has already popped the registry entry). Nothing was removed
        # from the scene, so no camera may be dropped either.
        before = _camera_names(sim)
        assert scene_ops.eject_body_from_scene(sim._world, "no_such_body") is True
        assert _camera_names(sim) == before


class TestTheObjectPathMatchesTheRobotPath:
    def test_both_removals_leave_the_registry_agreeing_with_the_model(self, tmp_path) -> None:
        urdf = tmp_path / "arm.xml"
        urdf.write_text(_ARM_XML)
        s = Simulation(tool_name="test_remove_object_camera_cascade_parity", mesh=False)
        try:
            s.create_world(gravity=[0, 0, -9.81])
            assert s.add_robot(name="arm", urdf_path=str(urdf))["status"] == "success"
            s.add_object("crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.3, 0.0, 0.05])
            assert (
                s.add_camera(name="wrist", position=[0.05, 0.0, 0.05], target=[0.1, 0.0, 0.0], parent_body="arm/link")[
                    "status"
                ]
                == "success"
            )
            assert (
                s.add_camera(name="watch", position=[0.2, -0.2, 0.2], target=[0.0, 0.0, 0.0], parent_body="crate")[
                    "status"
                ]
                == "success"
            )
            assert {"wrist", "watch"} <= set(_camera_names(s))

            assert s.remove_robot("arm")["status"] == "success"
            assert "wrist" not in _camera_names(s), "the robot path drops its body-mounted camera"
            assert len(_camera_names(s)) == _model_camera_count(s)

            assert s.remove_object("crate")["status"] == "success"
            assert "watch" not in _camera_names(s), "the object path must do the same"
            assert len(_camera_names(s)) == _model_camera_count(s)
        finally:
            s.cleanup()

"""The 3D view comes from the MuJoCo model itself: ``/tf`` per step, the meshes once.

A small MJCF built from a string stands in for a registry robot: one robot
namespace ``arm/`` with a mesh geom, a collision copy in group 3, a hinge
joint, and one un-namespaced box as the task object. Skipped without
``mujoco`` or the ``[foxglove]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("foxglove")
mujoco = pytest.importorskip("mujoco")

from strands_robots.foxglove import FoxgloveBridge, FoxgloveOptions, mcap_info  # noqa: E402
from strands_robots.foxglove import scene as scene_mod  # noqa: E402

_MJCF = """
<mujoco>
  <asset>
    <mesh name="arm/link" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="arm/base" pos="0 0 0.05">
      <geom type="mesh" mesh="arm/link" group="2" rgba="0.2 0.4 0.6 1"/>
      <geom type="mesh" mesh="arm/link" group="3" rgba="0.5 0.5 0.5 1"/>
      <body name="arm/link1" pos="0 0 0.1">
        <joint name="arm/elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.01 0.05" group="2" rgba="0.5 0.5 0.5 1"/>
      </body>
    </body>
    <body name="cube" pos="0.3 0 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_and_data() -> tuple[Any, Any]:
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm/elbow")]] = 0.5
    mujoco.mj_forward(model, data)
    return model, data


class TestTransforms:
    def test_every_body_but_world_gets_a_transform_from_world(self, model_and_data: tuple[Any, Any]) -> None:
        model, data = model_and_data
        message = scene_mod.frame_transforms(model, data, 1_700_000_000_123_456_789)
        encoded = message.encode()
        for name in ("arm/base", "arm/link1", "cube"):
            assert name.encode() in encoded
        assert b"world" in encoded
        assert encoded.count(b"world") == model.nbody - 1

    def test_quaternions_are_reordered_from_wxyz_to_xyzw(self, model_and_data: tuple[Any, Any]) -> None:
        model, data = model_and_data
        from foxglove.messages import Quaternion, Vector3

        encoded = scene_mod.frame_transforms(model, data, 10**18).encode()
        link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm/link1")
        w, x, y, z = (float(v) for v in data.xquat[link])
        px, py, pz = (float(v) for v in data.xpos[link])
        assert abs(y) > 1e-6, "the elbow is bent, so the rotation is not the identity"
        assert Quaternion(x=x, y=y, z=z, w=w).encode() in encoded
        assert Quaternion(x=w, y=x, z=y, w=z).encode() not in encoded, "the MuJoCo wxyz order must not leak through"
        assert Vector3(x=px, y=py, z=pz).encode() in encoded


class TestScene:
    def test_the_robot_scene_holds_its_visual_geoms_only(self, model_and_data: tuple[Any, Any]) -> None:
        model, _ = model_and_data

        message = scene_mod.scene_update(model, 10**18, robot="arm")
        assert message is not None
        encoded = message.encode()
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm/base")
        visual, collision = (g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == base)
        assert encoded.count(scene_mod._mesh_triangles(model, visual, base).encode()) == 1
        assert scene_mod._mesh_triangles(model, collision, base).encode() not in encoded, "group 3 is not drawn"
        assert b"arm/base" in encoded and b"arm/link1" in encoded
        assert b"cube" not in encoded, "the task object is not part of the robot's scene"
        link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm/link1")
        (capsule,) = (g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == link)
        primitive = scene_mod._primitive(model, capsule, link)
        assert primitive is not None and primitive[0] == "cylinders"
        assert primitive[1].encode() in encoded

    def test_the_world_scene_holds_the_task_objects(self, model_and_data: tuple[Any, Any]) -> None:
        model, _ = model_and_data
        from foxglove.messages import Color, Vector3

        message = scene_mod.scene_update(model, 10**18, robot=None)
        assert message is not None
        encoded = message.encode()
        assert b"cube" in encoded and b"arm/" not in encoded
        assert Vector3(x=0.04, y=0.04, z=0.04).encode() in encoded, "a box is drawn at twice its MuJoCo half-size"
        assert Color(r=1.0, g=0.0, b=0.0, a=1.0).encode() in encoded
        assert b"floor" not in encoded, "planes are left to the panel's own grid"

    def test_a_robot_with_no_visual_geoms_has_no_scene(self, model_and_data: tuple[Any, Any]) -> None:
        model, _ = model_and_data
        assert scene_mod.scene_update(model, 10**18, robot="nobody") is None

    def test_joint_states_resolve_namespaced_names(self, model_and_data: tuple[Any, Any]) -> None:
        model, data = model_and_data
        from foxglove.messages import JointState

        encoded = scene_mod.joint_states(model, data, ["elbow", "missing"], 10**18, robot="arm").encode()
        assert JointState(name="elbow", position=0.5, velocity=0.0).encode() in encoded
        assert b"missing" not in encoded

    def test_the_digest_ignores_time_and_other_robots(self, model_and_data: tuple[Any, Any]) -> None:
        model, _ = model_and_data
        again = mujoco.MjModel.from_xml_string(_MJCF)
        bigger_cube = mujoco.MjModel.from_xml_string(_MJCF.replace('size="0.02 0.02 0.02"', 'size="0.05 0.05 0.05"'))
        assert scene_mod.scene_digest(model, robot="arm") == scene_mod.scene_digest(again, robot="arm")
        assert scene_mod.scene_digest(model, robot="arm") == scene_mod.scene_digest(bigger_cube, robot="arm")
        assert scene_mod.scene_digest(model, robot=None) != scene_mod.scene_digest(bigger_cube, robot=None)


class _Engine:
    """Only what the bridge reads off a MuJoCo engine."""

    def __init__(self, model: Any, data: Any) -> None:
        self.mj_model = model
        self.mj_data = data


class _FakeServer:
    def __init__(self, **kwargs: Any) -> None:
        self.port = 43210

    def stop(self) -> None:
        return None


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import foxglove

    monkeypatch.setattr(foxglove, "start_server", lambda **kwargs: _FakeServer(**kwargs))


class TestTheBridgeWritesTheSceneOnce:
    def _bridge(self, tmp_path: Path, engine: _Engine, monkeypatch: pytest.MonkeyPatch) -> FoxgloveBridge:
        clock = {"t": 0.0}

        def _tick() -> float:
            clock["t"] += 1.0
            return clock["t"]

        from strands_robots.foxglove import bridge as bridge_mod

        monkeypatch.setattr(bridge_mod.time, "monotonic", _tick)
        return FoxgloveBridge(FoxgloveOptions(port=0, mcap=tmp_path / "run.mcap"), name="probe", engine=engine)

    def test_tf_every_step_and_the_scene_exactly_once(
        self, fake_server: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_and_data: tuple[Any, Any]
    ) -> None:
        model, data = model_and_data
        bridge = self._bridge(tmp_path, _Engine(model, data), monkeypatch)
        for _ in range(5):
            bridge.publish_joint_states("arm", ["elbow"], [0.5])
        bridge.shutdown()
        channels = mcap_info(tmp_path / "run.mcap")["channels"]
        assert channels["/tf"]["messages"] == 5
        assert channels["/arm/joint_states"]["messages"] == 5
        assert channels["/arm/scene"]["messages"] == 1
        assert channels["/scene"]["messages"] == 1

    def test_a_new_subscriber_gets_the_scene_live_without_a_second_file_copy(
        self, fake_server: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_and_data: tuple[Any, Any]
    ) -> None:
        import foxglove

        model, data = model_and_data
        bridge = self._bridge(tmp_path, _Engine(model, data), monkeypatch)
        # A second sink on the live context stands in for the WebSocket clients.
        live_copy = foxglove.open_mcap(str(tmp_path / "live.mcap"), context=bridge._live)
        bridge.publish_joint_states("arm", ["elbow"], [0.5])
        bridge._on_subscribe("/arm/scene")
        bridge.publish_joint_states("arm", ["elbow"], [0.5])
        live_copy.close()
        bridge.shutdown()
        assert mcap_info(tmp_path / "run.mcap")["channels"]["/arm/scene"]["messages"] == 1
        assert mcap_info(tmp_path / "live.mcap")["channels"]["/arm/scene"]["messages"] == 2

    def test_a_recompile_that_keeps_the_robot_meshes_writes_no_second_copy(
        self, fake_server: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_and_data: tuple[Any, Any]
    ) -> None:
        model, data = model_and_data
        engine = _Engine(model, data)
        bridge = self._bridge(tmp_path, engine, monkeypatch)
        bridge.publish_joint_states("arm", ["elbow"], [0.5])
        # The same MJCF compiled again is a new model object with the same bytes,
        # which is what add_object leaves behind for the robot's own scene.
        engine.mj_model = mujoco.MjModel.from_xml_string(_MJCF)
        engine.mj_data = mujoco.MjData(engine.mj_model)
        mujoco.mj_forward(engine.mj_model, engine.mj_data)
        bridge.publish_joint_states("arm", ["elbow"], [0.5])
        bridge.shutdown()
        channels = mcap_info(tmp_path / "run.mcap")["channels"]
        assert channels["/arm/scene"]["messages"] == 1
        assert channels["/tf"]["messages"] == 2

    def test_a_recompile_that_changes_the_scene_writes_the_new_one(
        self, fake_server: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_and_data: tuple[Any, Any]
    ) -> None:
        model, data = model_and_data
        engine = _Engine(model, data)
        bridge = self._bridge(tmp_path, engine, monkeypatch)
        bridge.publish_joint_states("arm", ["elbow"], [0.5])
        engine.mj_model = mujoco.MjModel.from_xml_string(
            _MJCF.replace('size="0.02 0.02 0.02"', 'size="0.05 0.05 0.05"')
        )
        engine.mj_data = mujoco.MjData(engine.mj_model)
        mujoco.mj_forward(engine.mj_model, engine.mj_data)
        bridge.publish_joint_states("arm", ["elbow"], [0.5])
        bridge.shutdown()
        channels = mcap_info(tmp_path / "run.mcap")["channels"]
        assert channels["/scene"]["messages"] == 2, "the cube grew, so the world scene is written again"
        assert channels["/arm/scene"]["messages"] == 1, "the arm did not change, so its scene is not"
        assert np is not None

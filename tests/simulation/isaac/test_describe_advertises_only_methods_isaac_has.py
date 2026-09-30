"""Every method ``describe()`` advertises exists on Isaac, with MuJoCo's envelope.

``SimEngine.describe()`` tells agents to call it first "instead of guessing
method names", and Isaac inherits it unchanged. It advertises
``get_robot_state`` and ``get_features``; Isaac had neither, so an agent that did
as told got ``AttributeError`` (measured on a live GPU run).
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState


def _advertised() -> list[str]:
    import inspect
    import re

    from strands_robots.simulation.base import SimEngine

    src = inspect.getsource(SimEngine.describe)
    block = src.split("methods: dict[str, str] = {", 1)[1].split("\n        }\n", 1)[0]
    return sorted(set(re.findall(r'^\s{12}"([a-z_]+)": ', block, re.M)))


class _Robot:
    def __init__(self) -> None:
        self.name = "arm"
        self.joint_names = ["a", "b"]
        self.data_config = "so100"
        self.description_path = "/x/so100.xml"


def _engine(obs: dict[str, float] | None = None, stale: bool = False) -> Any:
    engine: Any = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig()
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._physics_view_stale = stale
    engine._sim_time = 0.25
    engine._robots = {"arm": _Robot()}
    engine._objects = {"cube": types.SimpleNamespace(shape="box", is_static=False, prim_path="/World/Objects/cube")}
    engine._cameras = {"front": _CameraState("front", "/World/Cameras/front", 64, 48)}
    engine.get_observation = lambda name, skip_images=False: dict(
        obs if obs is not None else {"a": 0.1, "a.vel": 0.5, "b": -0.2, "b.vel": 0.0}
    )
    engine.get_body_state = lambda body_name: {
        "status": "success",
        "content": [{"text": ""}, {"json": {"position": [0.1, 0.2, 0.3]}}],
    }
    engine.physics_timestep = lambda: 1 / 120
    return engine


@pytest.mark.parametrize("method", _advertised())
def test_every_advertised_method_exists(method: str) -> None:
    assert callable(getattr(IsaacSimulation, method, None)), f"describe() advertises {method!r}; Isaac lacks it"


def test_get_robot_state_has_the_mujoco_shape() -> None:
    result = _engine().get_robot_state()
    assert result["status"] == "success"
    state = next(b["json"] for b in result["content"] if "json" in b)["state"]
    assert state == {"a": {"position": 0.1, "velocity": 0.5}, "b": {"position": -0.2, "velocity": 0.0}}


def test_get_robot_state_refuses_a_stale_view_and_an_unknown_robot() -> None:
    assert _engine(stale=True).get_robot_state()["status"] == "error"
    assert "not found" in _engine().get_robot_state("nope")["content"][0]["text"]
    assert _engine(obs={}).get_robot_state()["status"] == "error"


def test_get_features_has_the_mujoco_schema() -> None:
    result = _engine().get_features()
    features = next(b["json"] for b in result["content"] if "json" in b)["features"]
    assert features["joint_names"] == ["a", "b"]
    assert features["camera_names"] == ["front"]
    assert features["timestep"] == pytest.approx(1 / 120)
    assert features["robots"]["arm"]["n_joints"] == 2


def test_list_objects_reports_the_live_pose_and_list_cameras_the_names() -> None:
    engine = _engine()
    listing = next(b["json"] for b in engine.list_objects()["content"] if "json" in b)["objects"]
    assert listing["cube"]["position"] == [0.1, 0.2, 0.3]
    assert engine.list_cameras() == ["front"]

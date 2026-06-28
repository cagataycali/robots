"""Regression tests for ``start_recording(cameras=...)`` camera scoping.

By default every scene camera is recorded into the LeRobotDataset, which
silently includes the implicit ``default`` free camera and any view the trained
policy never declared. ``cameras=`` lets a caller record exactly the views the
policy expects (e.g. the three SmolVLA cameras) so the dataset schema matches
the policy's ``input_features`` and is not bloated by stray cameras.

Covers:
* default (``cameras=None``) records every camera including ``default``;
* ``cameras=[subset]`` declares exactly that subset in the dataset schema;
* an unknown camera name fails loudly (no silent drop);
* the ``_drop_unrecorded_cameras`` frame filter keeps state, drops excluded
  cameras, and is a no-op when nothing is scoped.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

os.environ.setdefault("MUJOCO_GL", "egl")

_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_with_cameras():
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=path)
    s.add_camera(name="cam_a", position=[0.5, 0.0, 0.3], target=[0.0, 0.0, 0.1], width=64, height=64)
    s.add_camera(name="cam_b", position=[0.0, 0.0, 0.7], target=[0.0, 0.0, 0.0], width=64, height=64)
    yield s
    s.destroy()


def _recorder_image_features(sim) -> set[str]:
    rec = sim._world._backend_state["dataset_recorder"]
    feats = rec.dataset.features
    return {k for k in feats if k.startswith("observation.images.")}


def test_default_records_every_camera_including_default(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(repo_id="local/scope_all", root=str(tmp_path / "all"), overwrite=True)
    assert res["status"] == "success"
    img = _recorder_image_features(sim)
    assert img == {
        "observation.images.default",
        "observation.images.cam_a",
        "observation.images.cam_b",
    }
    assert sim._world._backend_state["recording_cameras"] is None


def test_cameras_subset_scopes_dataset_schema(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(
        repo_id="local/scope_subset",
        root=str(tmp_path / "subset"),
        cameras=["cam_a", "cam_b"],
        overwrite=True,
    )
    assert res["status"] == "success"
    # The stray implicit ``default`` camera must NOT be in the schema.
    assert _recorder_image_features(sim) == {
        "observation.images.cam_a",
        "observation.images.cam_b",
    }
    assert sim._world._backend_state["recording_cameras"] == {"cam_a", "cam_b"}


def test_cameras_single_camera(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(
        repo_id="local/scope_one",
        root=str(tmp_path / "one"),
        cameras=["cam_a"],
        overwrite=True,
    )
    assert res["status"] == "success"
    assert _recorder_image_features(sim) == {"observation.images.cam_a"}


def test_unknown_camera_fails_loudly(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(
        repo_id="local/scope_bad",
        root=str(tmp_path / "bad"),
        cameras=["cam_a", "nope"],
        overwrite=True,
    )
    assert res["status"] == "error"
    text = res["content"][0]["text"]
    assert "nope" in text
    assert "cam_a" in text  # available cameras are listed
    # A failed start must not leave the engine half-armed.
    assert sim._world._backend_state.get("recording") is False


def test_drop_unrecorded_cameras_helper():
    from strands_robots.simulation.mujoco.simulation import _drop_unrecorded_cameras

    obs = {
        "shoulder_pan": 0.1,
        "shoulder_pan.vel": 0.0,
        "cam_a": np.zeros((8, 8, 3), dtype=np.uint8),
        "cam_b": np.zeros((8, 8, 3), dtype=np.uint8),
        "default": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    # None -> unchanged (legacy default: record all).
    assert _drop_unrecorded_cameras(obs, None) is obs

    out = _drop_unrecorded_cameras(obs, {"cam_a"})
    assert set(out) == {"shoulder_pan", "shoulder_pan.vel", "cam_a"}
    # Scalars preserved, excluded cameras dropped.
    assert out["shoulder_pan"] == 0.1
    assert "cam_b" not in out
    assert "default" not in out

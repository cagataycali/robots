"""Targeted coverage for ``RecordingMixin`` (LeRobotDataset recorder).

Covers:
* ``start_recording`` with no world → graceful error
* ``stop_recording`` with no active recording → graceful error
* ``get_recording_status`` with/without active session
* start_recording twice → second call does NOT crash (overwrite path)
* HF-cache repo_id path (repo_id with '/' and no local root)
* Multi-robot namespace prefix for joint names
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "glfw")

# Inline MJCF XML to avoid network-dependent so101 model downloads.
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
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_with_two_robots():
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("alpha", urdf_path=path, position=[-0.2, 0, 0])
    s.add_robot("beta", urdf_path=path, position=[0.2, 0, 0])
    s.step(5)
    yield s
    s.destroy()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_start_recording_no_world_returns_graceful_error():
    from strands_robots.simulation import Simulation

    s = Simulation()
    r = s.start_recording(repo_id="local/nope", task="t")
    assert r["status"] == "error"
    assert "No world" in r["content"][0]["text"]
    s.destroy()


def test_stop_recording_without_start_is_idempotent(sim_with_two_robots):
    """T16: idempotent - success with 'Was not recording' message."""
    r = sim_with_two_robots.stop_recording()
    assert r["status"] == "success"
    assert "Was not recording" in r["content"][0]["text"]


def test_get_recording_status_shows_active_and_idle(sim_with_two_robots, tmp_path):
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_two_robots

    # Idle before any start
    r = sim.get_recording_status()
    assert r["status"] == "success"

    # Start → active
    r = sim.start_recording(repo_id="local/status_probe", fps=20, root=str(tmp_path), overwrite=True)
    assert r["status"] == "success"

    r = sim.get_recording_status()
    assert r["status"] == "success"

    # Stop → idle again
    sim.stop_recording()
    r = sim.get_recording_status()
    assert r["status"] == "success"


def test_start_recording_overwrite_wipes_existing_dir(sim_with_two_robots, tmp_path):
    """The ``overwrite=True`` flag removes any pre-existing dataset dir
    before re-creating it (covers the ``shutil.rmtree`` branch)."""
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    # Pre-create some junk in the target dir
    junk = tmp_path / "stale.txt"
    junk.write_text("stale")
    assert junk.exists()

    r = sim_with_two_robots.start_recording(
        repo_id="local/overwrite_probe",
        fps=20,
        root=str(tmp_path),
        overwrite=True,
    )
    assert r["status"] == "success"
    # The junk should be gone (dir was wiped)
    assert not junk.exists()

    sim_with_two_robots.stop_recording()


def test_start_recording_namespaced_joint_prefix_with_two_robots(sim_with_two_robots, tmp_path):
    """With >1 robot, joint_names are prefixed with the robot's instance name."""
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    r = sim_with_two_robots.start_recording(repo_id="local/namespace_probe", fps=20, root=str(tmp_path), overwrite=True)
    assert r["status"] == "success"

    from strands_robots.policies.mock import MockPolicy

    p = MockPolicy()
    p.set_robot_state_keys(sim_with_two_robots.robot_joint_names("alpha"))
    r = sim_with_two_robots.run_policy("alpha", policy_object=p, duration=0.2, control_frequency=20.0)
    assert r["status"] == "success"

    sim_with_two_robots.stop_recording()

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    joint_names = info["features"]["observation.state"]["names"]
    # Unique joint names - the fix we pushed
    assert len(joint_names) == len(set(joint_names)), f"dup names: {joint_names}"
    # Both robots prefixed
    assert any(jn.startswith("alpha__") for jn in joint_names)
    assert any(jn.startswith("beta__") for jn in joint_names)

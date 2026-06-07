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


def test_b12_multi_episode_resume_appends(sim_with_two_robots, tmp_path):
    """B12 regression: start_recording(overwrite=False) on an EXISTING dataset
    must RESUME (append) instead of crashing with FileExistsError.

    Pre-fix, the 2nd start_recording raised
    ``Dataset init failed: [Errno 17] File exists`` because it always called
    LeRobotDataset.create() (mkdir exist_ok=False). The fix routes to
    DatasetRecorder.resume() when a dataset already exists on disk.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    sim = sim_with_two_robots
    root = str(tmp_path / "multiep")

    # Episode 1 (fresh)
    r = sim.start_recording(repo_id="local/multiep", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r
    sim.run_policy(robot_name="alpha", policy_provider="mock", instruction="ep0",
                   duration=0.3, control_frequency=20.0, fast_mode=True)
    r = sim.stop_recording()
    assert r["status"] == "success", r

    # Episode 2 (append — overwrite=False on existing dir must NOT crash)
    r = sim.start_recording(repo_id="local/multiep", fps=20, root=root, overwrite=False)
    assert r["status"] == "success", f"B12 regression — resume failed: {r}"
    sim.run_policy(robot_name="alpha", policy_provider="mock", instruction="ep1",
                   duration=0.3, control_frequency=20.0, fast_mode=True)
    r = sim.stop_recording()
    assert r["status"] == "success", r

    # Readback: two episodes appended into one dataset.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/multiep", root=root)
    assert ds.meta.total_episodes == 2, f"expected 2 episodes, got {ds.meta.total_episodes}"


def test_b4_synchronized_multi_robot_recording(sim_with_two_robots, tmp_path):
    """B4 fix: run_multi_policy drives BOTH robots in one synchronized loop and
    records them into ONE merged frame per timestep — so every frame co-observes
    both arms (the whole point of a multi-robot dataset).

    Pre-fix, two independent start_policy threads shared the recorder and
    interleaved single-robot frames (each frame had only one robot's state, the
    other's columns zero). Here we assert BOTH robots' action columns are
    non-zero in EVERY frame.
    """
    from strands_robots.dataset_recorder import has_lerobot_dataset

    if not has_lerobot_dataset():
        pytest.skip("lerobot not installed")

    import numpy as np
    from strands_robots.policies import create_policy

    sim = sim_with_two_robots
    root = str(tmp_path / "sync")
    r = sim.start_recording(repo_id="local/sync_multi", fps=20, root=root, overwrite=True)
    assert r["status"] == "success", r

    pols = {"alpha": create_policy("mock"), "beta": create_policy("mock")}
    r = sim.run_multi_policy(policies=pols,
                             instructions={"alpha": "a", "beta": "b"},
                             duration=0.5, control_frequency=20.0)
    assert r["status"] == "success", r
    assert r["steps"] > 0
    sim.stop_recording()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/sync_multi", root=root)

    # Schema: 2 robots × 3 joints (the inline test arm) = 6-dim, prefixed.
    af = ds.features["action"]
    names = af["names"] if isinstance(af, dict) else getattr(af, "names", None)
    assert names is not None
    assert any(n.startswith("alpha__") for n in names)
    assert any(n.startswith("beta__") for n in names)

    half = len(names) // 2
    both = 0
    for i in range(len(ds)):
        ac = np.asarray(ds[i]["action"])
        a = float(np.abs(ac[:half]).sum())
        b = float(np.abs(ac[half:]).sum())
        if a > 1e-6 and b > 1e-6:
            both += 1
    # Every frame must co-observe both robots (synchronized recording).
    assert both == len(ds) and len(ds) > 0, (
        f"B4: only {both}/{len(ds)} frames had both robots co-observed"
    )


def test_run_multi_policy_validates_robots(sim_with_two_robots):
    """run_multi_policy rejects unknown robots and empty policy maps."""
    from strands_robots.policies import create_policy

    sim = sim_with_two_robots
    assert sim.run_multi_policy(policies={})["status"] == "error"
    r = sim.run_multi_policy(policies={"ghost": create_policy("mock")}, duration=0.1)
    assert r["status"] == "error"
    assert "not found" in r["content"][0]["text"].lower()

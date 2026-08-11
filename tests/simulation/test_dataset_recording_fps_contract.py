"""A dataset recording must be refused at a frame rate it cannot be written at.

``start_recording`` takes an ``fps`` that becomes the LeRobotDataset's frame
rate. LeRobot itself only rejects ``fps <= 0``, so every other unusable value
was accepted by all three backends and cost the caller the episode *after*
``status="success"`` had been returned:

* ``fps=2.7`` (or ``nan``) created the dataset, then killed the per-camera video
  encoder thread on the first frame; the rollout aborted with "on_frame hook
  failed 5 times in a row" and ``stop_recording`` could not save the pending
  frames, so the recording was lost.
* ``fps=True`` - an ``int`` subclass - silently recorded a 1 fps dataset, giving
  every frame a 1-second timestamp a policy would then train on.
* ``fps="30"`` dead-ended in a raw ``TypeError: '<=' not supported between
  instances of 'str' and 'int'`` that never named the parameter to fix.

The domain is now the same positive-whole-number one the plain-MP4 recorders and
the ``run_policy(video=...)`` dict already enforce, checked before any recorder
is created, and shared by the MuJoCo / Newton / Isaac ``start_recording``
implementations so the three surfaces cannot drift.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

import strands_robots.simulation as simulation_pkg
from strands_robots.simulation.recording import dataset_recording_option_error
from strands_robots.tools.run_policy import run_policy as run_policy_tool

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# One actuated hinge plus a camera: enough for run_policy to drive something and
# for the dataset schema to declare an image column, with no asset download.
_ARM_XML = """
<mujoco model="fps_contract_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="0 -1 0.4" xyaxes="1 0 0 0 0.3 1"/>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom name="link" type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""

# Values no dataset can be recorded at. Each one previously returned
# status="success" from start_recording (except the bare-string TypeError).
UNUSABLE_FPS = [0, -5, 2.7, float("nan"), float("inf"), True, "30", None, [30]]


@pytest.fixture
def sim(tmp_path):
    model = tmp_path / "fps_contract_arm.xml"
    model.write_text(_ARM_XML)
    s = Simulation(tool_name="fps_contract", mesh=False)
    s.create_world()
    s.add_robot("arm", urdf_path=str(model))
    s.add_camera(name="view", position=[0.6, -0.6, 0.4], target=[0.0, 0.0, 0.1], width=64, height=64)
    yield s
    s.cleanup()


class TestFpsDomain:
    """The shared accepted domain for a dataset frame rate."""

    @pytest.mark.parametrize("fps", UNUSABLE_FPS)
    def test_unusable_rate_reports_the_parameter_and_the_method(self, fps):
        error = dataset_recording_option_error("start_recording", fps)
        assert error is not None
        assert error["status"] == "error"
        text = error["content"][0]["text"]
        assert "start_recording" in text
        assert "fps" in text
        assert repr(fps) in text

    @pytest.mark.parametrize("fps", [1, 30, 30.0, np.int64(60), np.float64(30.0)])
    def test_positive_whole_rate_is_accepted(self, fps):
        assert dataset_recording_option_error("start_recording", fps) is None


class TestStartRecordingRefusesUnusableFps:
    """An unusable rate must be refused before a recorder exists."""

    @pytest.mark.parametrize("fps", UNUSABLE_FPS)
    def test_no_session_is_opened_and_nothing_is_written(self, sim, tmp_path, fps):
        root = tmp_path / "dataset"
        result = sim.start_recording(repo_id="local/fps_contract", task="t", fps=fps, root=str(root))

        assert result["status"] == "error"
        assert "fps" in result["content"][0]["text"]
        # No half-open session: status must still read idle, and a later
        # stop_recording must not claim to have saved anything.
        status = sim.get_recording_status()
        assert "idle" in status["content"][0]["text"].lower()
        # Nothing on disk either - the refusal precedes dataset creation.
        assert not root.exists() or not any(root.iterdir())


class TestStartRecordingHonorsUsableFps:
    """A usable rate still records a complete, reopenable episode."""

    def test_episode_round_trips_at_the_requested_rate(self, sim, tmp_path):
        pytest.importorskip("lerobot")
        root = tmp_path / "dataset"
        started = sim.start_recording(repo_id="local/fps_contract", task="wave", fps=30, root=str(root))
        assert started["status"] == "success", started

        rollout = sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            n_steps=6,
            control_frequency=30.0,
        )
        assert rollout["status"] == "success", rollout
        stopped = sim.stop_recording()
        assert stopped["status"] == "success", stopped

        # Round-trip: the dataset reopens, carries the requested rate, and holds
        # the frames the rollout produced (with the per-camera MP4 on disk).
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local/fps_contract", root=str(root))
        assert dataset.fps == 30
        assert dataset.num_frames == 6
        assert list(root.rglob("*.mp4"))


class TestRunPolicyToolForwardsTheGuard:
    """The agent-facing tool reports the rate before it starts a rollout."""

    def test_unusable_dataset_fps_is_reported_before_the_rollout(self, sim, tmp_path):
        root = tmp_path / "dataset"
        result = run_policy_tool(
            simulation=sim,
            robot_name="arm",
            policy_provider="mock",
            n_steps=4,
            control_frequency=30.0,
            dataset_root=str(root),
            dataset_fps=2.5,
        )
        assert result["status"] == "error"
        assert "fps" in result["content"][0]["text"]
        assert not root.exists() or not any(root.iterdir())


def _start_recording_calls_the_shared_guard(module_path: Path) -> bool:
    """True when the module's ``start_recording`` calls the shared fps guard.

    Parsed by AST so backends whose optional dependencies (Isaac Sim, Newton)
    are not installed are still checked. It proves the guard is *called*, never
    that its refusal is *returned* - a copy that keeps the call and drops the
    ``return`` satisfies it - so the returned refusal is driven per backend in
    ``test_recording_preflight_refusals_across_backends.py``.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "start_recording":
            return any(
                isinstance(call.func, ast.Name) and call.func.id == "dataset_recording_option_error"
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
    return False


@pytest.mark.parametrize("backend", ["mujoco", "newton", "isaac"])
def test_every_backend_start_recording_shares_the_guard(backend):
    """No backend may accept a dataset rate the others refuse."""
    module_path = Path(simulation_pkg.__file__).parent / backend / "recording.py"
    assert _start_recording_calls_the_shared_guard(module_path), (
        f"{backend}/recording.py start_recording must call dataset_recording_option_error"
    )

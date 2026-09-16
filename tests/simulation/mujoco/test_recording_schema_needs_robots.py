"""The dataset recorder's schema is frozen from the robots present at start.

``start_recording`` on a scene with no robots opens a camera-only dataset
("0 joints" in the success text); ``add_robot`` afterwards succeeded; and the
first recorded rollout then failed inside lerobot ("Feature mismatch ...
Extra features: {'action', 'observation.state'}") with an empty dataset shell
left on disk. ``add_robot`` now refuses while a recording is active, naming the
sequence that works, and the camera-only success text says what it is.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


@pytest.fixture
def empty_world():
    sim = MuJoCoSimEngine(tool_name="rec_no_robots", mesh=False)
    sim.create_world()
    try:
        yield sim
    finally:
        sim.cleanup()


def test_start_recording_with_no_robots_says_it_is_camera_only(empty_world, tmp_path):
    r = empty_world.start_recording(repo_id="lab/zero", root=str(tmp_path / "zero"), fps=30)
    try:
        assert r["status"] == "success"
        text = _text(r)
        assert "0 joints" in text
        assert (
            "No robots are attached, so this dataset has no observation.state/action columns and records cameras only"
            " (one frame per step). To record a robot, stop_recording, add_robot, then start_recording again."
        ) in text
        # The hazard the note prevents: adding the robot INTO this recording.
        assert empty_world.add_robot(name="so101", data_config="so101")["status"] == "error"
        assert empty_world.list_robots() == []
    finally:
        empty_world._world._backend_state["recording"] = False
        empty_world._world._backend_state.pop("dataset_recorder", None)


def test_add_robot_while_recording_is_refused_and_sequence_works(empty_world, tmp_path):
    sim = empty_world
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    assert sim.start_recording(repo_id="lab/one", root=str(tmp_path / "one"), fps=30)["status"] == "success"
    try:
        r = sim.add_robot(name="arm2", data_config="so101", position=[0.5, 0, 0])
        assert r["status"] == "error"
        text = _text(r)
        assert text.startswith(
            "add_robot: a dataset recording is active and its schema was frozen from the 1 robot(s) present at start_recording"
        )
        assert text.endswith("stop_recording first, add the robot, then start_recording again.")
        assert sim.list_robots() == ["so101"]
        assert sim.run_policy(policy_provider="mock", duration=0.2, control_frequency=30)["status"] == "success"
    finally:
        stopped = sim.stop_recording()
    assert stopped["status"] == "success"
    assert sim.add_robot(name="arm2", data_config="so101", position=[0.5, 0, 0])["status"] == "success"

"""``discard_episode`` drops the open take, so a bad episode never reaches disk.

Every episode boundary a recording session offered SAVED: ``reset`` flushes the
open episode as its own, ``save_episode`` and ``stop_recording`` write it. Asked
"that take was bad, discard it", an agent on the MuJoCo tool had no action to
reach for; it saved the take as its own episode and told the user to delete it
with LeRobot's dataset tools. ``discard_episode`` clears the recorder's open
buffer (un-counting the frames) and leaves the saved episodes alone.

Pinned here: a session that records, discards one take and records again writes
exactly the kept episodes and frames; the action is published to agents; it
refuses outside a recording and during a running policy; an empty buffer is a
no-op; and a recorder that cannot clear says the frames are still buffered.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_FPS = 30


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _ok(result: dict, what: str) -> dict:
    if result["status"] != "success":
        raise AssertionError(f"{what} refused: {_text(result)}")
    return result


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine(tool_name="discard_episode", mesh=False)
    _ok(engine.create_world(), "create_world")
    _ok(engine.add_robot(name="so101", data_config="so101"), "add_robot")
    try:
        yield engine
    finally:
        engine.cleanup()


def _take(sim, seconds: float) -> None:
    _ok(
        sim.run_policy(robot_name="so101", policy_provider="mock", duration=seconds, control_frequency=_FPS),
        "run_policy",
    )


def test_a_discarded_take_never_reaches_the_dataset(sim, tmp_path):
    root = tmp_path / "ds"
    # Action-only (cameras=[]): the verdict is episode lengths, not pixels.
    _ok(sim.start_recording(repo_id="lab/takes", root=str(root), fps=_FPS, cameras=[]), "start_recording")
    _take(sim, 0.4)
    _ok(sim.reset(), "reset")  # closes take 1 as episode 0
    _take(sim, 0.3)  # the bad take

    result = _ok(sim.discard_episode(), "discard_episode")

    assert _json(result) == {"discarded_frames": 9, "saved_episodes": 1}
    _ok(sim.reset(), "reset")  # nothing open: must not write an empty episode
    _take(sim, 0.2)
    _ok(sim.stop_recording(), "stop_recording")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="lab/takes", root=str(root))
    assert dataset.num_episodes == 2
    lengths = [int(n) for n in dataset.meta.episodes["length"]]
    assert lengths == [12, 6], f"the discarded 9-frame take reached the dataset: {lengths}"
    assert dataset.num_frames == 18


def test_an_agent_can_reach_it(sim, tmp_path):
    enum = sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    assert "discard_episode" in enum
    _ok(sim(action="start_recording", repo_id="lab/agent", root=str(tmp_path / "a"), fps=_FPS, cameras=[]), "start")
    _take(sim, 0.2)

    result = sim(action="discard_episode")

    assert result["status"] == "success", _text(result)
    assert _json(result)["discarded_frames"] == 6
    status = _text(sim(action="get_recording_status"))
    assert "0 steps buffered" in status and "discard_episode" in status


def test_an_empty_buffer_is_a_no_op(sim, tmp_path):
    _ok(sim.start_recording(repo_id="lab/empty", root=str(tmp_path / "e"), fps=_FPS, cameras=[]), "start")

    result = _ok(sim.discard_episode(), "discard_episode")

    assert _json(result) == {"discarded_frames": 0, "saved_episodes": 0}


def test_it_refuses_outside_a_recording(sim):
    result = sim.discard_episode()

    assert result["status"] == "error"
    assert "not recording" in _text(result)


def test_it_refuses_while_a_policy_is_writing_the_buffer(sim, tmp_path):
    _ok(sim.start_recording(repo_id="lab/busy", root=str(tmp_path / "b"), fps=_FPS, cameras=[]), "start")
    _take(sim, 0.2)
    sim._world.robots["so101"].policy_running = True
    try:
        result = sim.discard_episode()
    finally:
        sim._world.robots["so101"].policy_running = False

    assert result["status"] == "error", _text(result)
    assert sim._world._backend_state["dataset_recorder"].episode_frame_count == 6, "nothing may be dropped"


def test_a_recorder_that_cannot_clear_keeps_the_frames_and_says_so(sim, tmp_path, monkeypatch):
    _ok(sim.start_recording(repo_id="lab/old", root=str(tmp_path / "o"), fps=_FPS, cameras=[]), "start")
    _take(sim, 0.2)
    recorder = sim._world._backend_state["dataset_recorder"]
    monkeypatch.setattr(recorder, "clear_episode_buffer", lambda: False)

    result = sim.discard_episode()

    assert result["status"] == "error"
    assert "the 6 buffered frames are still there" in _text(result)

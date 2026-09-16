"""``replay_episode`` reads back what this session recorded; a miss says where it looked.

Recorded under ``root=`` with an ``owner/name`` id, then replayed by id alone,
the dataset was resolved by LeRobot's rule (``$HF_LEROBOT_HOME/<id>``), missed,
and fell through to a Hugging Face Hub lookup whose raw multi-line 404 ("make
sure you specified the correct repo_id and repo_type") came back as the whole
answer - for a dataset that was on disk the whole time. And after that
``stop_recording``, ``get_recording_status`` reported "last episode: 0 steps"
because it read the buffer the save had just cleared.

Pinned: ``stop_recording`` remembers what it saved; ``get_recording_status``
reports it (repo_id, frames, episodes, root); ``replay_episode`` of the same
repo_id with no root adopts that root and says so; a Hub miss becomes one
sentence naming the local directory that was checked and the Hub; a root the
caller passed is never overridden.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.policy_runner import _dataset_load_failure_text


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    s = MuJoCoSimEngine(tool_name="replay_finds", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def _record(sim, root) -> None:
    assert sim.start_recording(repo_id="lab/so101_ep", root=str(root), task="wave", fps=30)["status"] == "success"
    assert sim.run_policy("so101", policy_provider="mock", duration=0.2, control_frequency=30.0)["status"] == "success"
    assert sim.stop_recording()["status"] == "success"


def test_status_before_any_save_says_so(sim):
    r = sim.get_recording_status()
    assert "nothing saved in this session" in _text(r)
    assert _json(r) == {"recording": False, "steps": 0, "last_dataset": None}


def test_status_after_stop_reports_what_was_saved_not_a_cleared_buffer(sim, tmp_path):
    _record(sim, tmp_path / "ds")
    r = sim.get_recording_status()
    text = _text(r)
    assert "last episode: 0 steps" not in text
    assert "Last saved: lab/so101_ep - 6 frames, 1 episode(s)" in text, text
    assert str(tmp_path / "ds") in text
    last = _json(r)["last_dataset"]
    assert last["repo_id"] == "lab/so101_ep"
    assert last["frame_count"] == 6
    assert last["episode_count"] == 1
    assert last["root"] == str(tmp_path / "ds")


def test_replay_by_id_alone_adopts_the_root_this_session_recorded_to(sim, tmp_path):
    _record(sim, tmp_path / "ds")
    r = sim.replay_episode("lab/so101_ep", robot_name="so101", speed=10.0)
    assert r["status"] == "success", _text(r)
    assert "Replayed episode 0 from lab/so101_ep" in _text(r)
    assert f"root taken from this session's recording: {tmp_path / 'ds'}" in _text(r)


def test_a_root_the_caller_passed_is_never_overridden(sim, tmp_path):
    _record(sim, tmp_path / "ds")
    r = sim.replay_episode("lab/so101_ep", robot_name="so101", root=str(tmp_path / "elsewhere"))
    assert r["status"] == "error"
    text = _text(r)
    assert f"not under root='{tmp_path / 'elsewhere'}'" in text
    assert "root taken from" not in text


def test_a_hub_miss_is_one_sentence_naming_where_was_looked(sim):
    r = sim.replay_episode("nobody/does_not_exist_xyz_strands", robot_name="so101")
    assert r["status"] == "error"
    text = _text(r)
    assert text.startswith("replay: dataset 'nobody/does_not_exist_xyz_strands' was not found")
    assert "nobody/does_not_exist_xyz_strands (LeRobot's local home for that id)" in text
    assert "Hugging Face Hub (404)" in text
    assert "recorded with root= must be replayed with the same root=" in text
    assert "Client Error" not in text
    assert "repo_type" not in text


def test_a_failure_that_is_not_a_hub_miss_keeps_the_library_text():
    assert _dataset_load_failure_text("a/b", None, ValueError("Episode 5 out of range (0-0)")) == (
        "Episode 5 out of range (0-0)"
    )


def test_the_helper_names_the_root_when_one_was_passed():
    class RepositoryNotFoundError(Exception):
        pass

    text = _dataset_load_failure_text("a/b", "/data/x", RepositoryNotFoundError("404 Client Error ..."))
    assert "not under root='/data/x'" in text

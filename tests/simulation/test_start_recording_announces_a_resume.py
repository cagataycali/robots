"""``start_recording`` on a ``repo_id``/``root`` that already holds a dataset says it is resuming.

The resume was a ``logger.info`` only. ``overwrite`` is not a published tool
parameter, so an agent that re-recorded the same ``repo_id`` got the same
fresh-dataset sentence as the first time and saw its "first" episode land at
``episode_index 2``. The answer now carries a RESUMING line with the episodes and
frames already on disk, the index the next episode gets, and how to start fresh;
a json block reports the same. A first recording says nothing about resuming.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    s = MuJoCoSimEngine(tool_name="resume_told", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def test_first_recording_is_fresh_and_says_so_in_json(sim, tmp_path):
    root = str(tmp_path / "ds")
    r = sim.start_recording(repo_id="lab/again", root=root, fps=30)
    assert r["status"] == "success"
    assert "RESUMING" not in _text(r)
    assert _json(r) == {
        "repo_id": "lab/again",
        "dataset_dir": root,
        "fps": 30,
        "resumed": False,
        "episodes_on_disk": 0,
        "frames_on_disk": 0,
    }
    sim.stop_recording()


def test_recording_the_same_repo_again_announces_the_resume(sim, tmp_path):
    root = str(tmp_path / "ds")
    sim.start_recording(repo_id="lab/again", root=root, fps=30)
    r = sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0, n_episodes=2)
    assert r["status"] == "success", _text(r)
    assert "2 episode(s)" in _text(sim.stop_recording())

    again = sim.start_recording(repo_id="lab/again", root=root, fps=30)
    assert again["status"] == "success", _text(again)
    assert (
        f"RESUMING the existing dataset at {root}: 2 episode(s) / 30 frames already on disk - new episodes "
        "append after them (the next is episode_index 2). For a fresh dataset record to a new repo_id or root."
    ) in _text(again)
    assert _json(again)["resumed"] is True
    assert _json(again)["episodes_on_disk"] == 2
    assert _json(again)["frames_on_disk"] == 30

    # And the appended episode really is index 2.
    r = sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0, n_episodes=1)
    assert r["status"] == "success", _text(r)
    assert "45 frames, 3 episode(s)" in _text(sim.stop_recording())
    assert sim.replay_episode("lab/again", root=root, episode=2, speed=20.0)["status"] == "success"

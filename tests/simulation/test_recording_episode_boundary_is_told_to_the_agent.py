"""``get_recording_status`` reports the dataset's episodes, not just the open buffer.

The status text read the ``trajectory`` mirror alone, which holds the OPEN
episode and empties at every flush: right after ``run_policy(n_episodes=3)``
saved 45 frames it said "0 steps captured", while ``stop_recording`` on the very
next call reported "45 frames, 3 episode(s)". The saved counts now stand beside
the open buffer, and the two recipes that still promised "run_policy (once per
episode)" name the published boundaries instead.
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
    s = MuJoCoSimEngine(tool_name="boundary_told", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def test_status_reports_saved_episodes_beside_the_open_buffer(sim, tmp_path):
    sim.start_recording(repo_id="lab/b", root=str(tmp_path / "ds"), fps=30)
    r = sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0, n_episodes=3)
    assert r["status"] == "success", _text(r)

    status = sim.get_recording_status()
    assert "0 steps buffered in the open episode (episode_index 3)" in _text(status)
    assert "3 episode(s) / 45 frames saved so far" in _text(status)
    assert _json(status)["episodes_saved"] == 3
    assert _json(status)["frames_saved"] == 45
    assert _json(status)["open_episode_index"] == 3

    # A further rollout buffers into the open episode: the saved counts do not
    # move until it is flushed, and the buffer is reported separately.
    sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0)
    status = sim.get_recording_status()
    assert "15 steps buffered in the open episode (episode_index 3)" in _text(status)
    assert "3 episode(s) / 45 frames saved so far" in _text(status)
    assert _json(status)["steps"] == 15
    assert _json(status)["frames_saved"] == 45

    # The status agreed with the dataset all along.
    assert "60 frames, 4 episode(s)" in _text(sim.stop_recording())


def test_status_off_a_recording_is_unchanged(sim):
    assert "[idle] Not recording" in _text(sim.get_recording_status())
    assert "episodes_saved" not in _json(sim.get_recording_status())


@pytest.mark.parametrize(
    "call, expected",
    [
        ("stop_recording", "run_policy(n_episodes=N), or run_policy then reset per episode"),
        ("save_episode", "run_policy -> save_episode per episode"),
    ],
)
def test_the_recipes_name_a_published_boundary(sim, tmp_path, call, expected):
    if call == "stop_recording":
        sim.start_recording(repo_id="lab/b", root=str(tmp_path / "ds"), fps=30)
    r = getattr(sim, call)()
    assert r["status"] == "error"
    assert "once per episode" not in _text(r)
    assert expected in _text(r)

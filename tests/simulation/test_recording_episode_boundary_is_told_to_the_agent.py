"""The agent-facing recording texts tell the truth about episode boundaries.

A single ``run_policy`` under an open recording buffers into the OPEN episode and
flushes nothing - the json block said so (``episode_flush_deferred``) but the
prose did not, and ``start_recording`` promised "run_policy (one call per
episode)". Two ``run_policy`` calls with two instructions produced ONE dataset
episode of 30 frames. ``get_recording_status`` read the open-episode buffer alone,
so right after ``run_policy(n_episodes=3)`` had saved 45 frames it said
"0 steps captured".

Pinned: the ``run_policy`` answer names the open episode and the three ways to
close it (reset / n_episodes=N / stop_recording); ``get_recording_status`` reports
saved episodes and frames beside the open buffer, with a json block;
``start_recording`` and the empty-``stop_recording`` recipe no longer say "one
call per episode"; the boundary itself is unchanged (two calls, one episode).
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


def _rollout(sim, instruction: str) -> dict:
    return sim.run_policy(
        "so101", policy_provider="mock", duration=0.5, control_frequency=30.0, instruction=instruction
    )


def test_start_recording_no_longer_promises_one_call_per_episode(sim, tmp_path):
    r = sim.start_recording(repo_id="lab/b", root=str(tmp_path / "ds"), fps=30)
    assert r["status"] == "success"
    assert "one call per episode" not in _text(r)
    assert "run_policy(n_episodes=N) for N distinct episodes" in _text(r)
    assert "consecutive run_policy calls WITHOUT a reset join ONE episode" in _text(r)


def test_a_single_run_policy_names_the_open_episode_and_how_to_close_it(sim, tmp_path):
    sim.start_recording(repo_id="lab/b", root=str(tmp_path / "ds"), fps=30)
    r = _rollout(sim, "pick")
    assert r["status"] == "success", _text(r)
    text = _text(r)
    assert "OPEN episode (episode_index 0, 15 frames in it so far) - not yet a saved dataset episode" in text
    assert "reset closes it as its own episode" in text
    assert "run_policy(n_episodes=N)" in text
    assert "stop_recording saves it" in text
    assert _json(r)["episode_flush_deferred"] is True

    # The second call joins the same open episode - the boundary is unchanged, the note says so.
    r2 = _rollout(sim, "place")
    assert "episode_index 0, 30 frames in it so far" in _text(r2)
    assert "30 frames, 1 episode(s)" in _text(sim.stop_recording())


def test_recording_status_reports_saved_episodes_beside_the_open_buffer(sim, tmp_path):
    sim.start_recording(repo_id="lab/b", root=str(tmp_path / "ds"), fps=30)
    r = sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0, n_episodes=3)
    assert r["status"] == "success", _text(r)
    status = sim.get_recording_status()
    assert (
        "0 steps buffered in the open episode (episode_index 3); 3 episode(s) / 45 frames saved so far to lab/b"
        in _text(status)
    )
    assert _json(status) == {
        "recording": True,
        "open_episode_steps": 0,
        "open_episode_index": 3,
        "episodes_saved": 3,
        "frames_saved": 45,
        "repo_id": "lab/b",
    }
    _rollout(sim, "one more")
    status = sim.get_recording_status()
    assert "15 steps buffered in the open episode (episode_index 3); 3 episode(s) / 45 frames saved" in _text(status)
    assert _json(status)["open_episode_steps"] == 15
    assert "60 frames, 4 episode(s)" in _text(sim.stop_recording())


def test_reset_between_rollouts_yields_one_episode_each_and_the_note_follows(sim, tmp_path):
    sim.start_recording(repo_id="lab/b", root=str(tmp_path / "ds"), fps=30)
    _rollout(sim, "pick")
    assert "Episode 1 saved" in _text(sim.reset())
    r = _rollout(sim, "place")
    assert "episode_index 1, 15 frames in it so far" in _text(r)
    assert "30 frames, 2 episode(s)" in _text(sim.stop_recording())


def test_no_recording_no_note(sim):
    r = _rollout(sim, "pick")
    assert r["status"] == "success"
    assert "Recording:" not in _text(r)
    assert "[idle] Not recording" in _text(sim.get_recording_status())


def test_the_empty_stop_recording_recipe_no_longer_says_once_per_episode(sim, tmp_path):
    sim.start_recording(repo_id="lab/b", root=str(tmp_path / "ds"), fps=30)
    r = sim.stop_recording()
    assert r["status"] == "error"
    assert "once per episode" not in _text(r)
    assert "run_policy(n_episodes=N), or run_policy then reset per episode" in _text(r)

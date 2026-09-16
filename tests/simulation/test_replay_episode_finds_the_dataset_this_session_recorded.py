"""``replay_episode`` with ``root`` omitted finds the dataset this session recorded.

``start_recording(repo_id="lab/demo", root=<dir>)`` → ``stop_recording`` (reports
"Local: <dir>") → ``replay_episode(repo_id="lab/demo", episode=1)`` answered a raw
huggingface_hub 404 ("Repository Not Found … /api/datasets/lab/demo/refs") for a
dataset sitting in a directory the sim itself had just written. Now the sim
remembers where each ``repo_id`` was recorded this session and replays from there
when ``root`` is omitted (saying so); a genuinely unknown id gets a one-line cause
plus the remedy instead of six lines of Hub boilerplate.
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
def recorded(tmp_path):
    s = MuJoCoSimEngine(tool_name="replay_root", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    root = str(tmp_path / "ds")
    assert s.start_recording(repo_id="lab/demo", root=root, fps=30)["status"] == "success"
    r = s.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0, n_episodes=2)
    assert r["status"] == "success", _text(r)
    stop = s.stop_recording()
    assert "2 episode(s)" in _text(stop) and f"Local: {root}" in _text(stop)
    yield s, root
    s.cleanup()


def test_root_omitted_replays_from_the_directory_this_session_recorded(recorded):
    sim, root = recorded
    assert sim.recorded_dataset_dir("lab/demo") == root
    r = sim.replay_episode("lab/demo", episode=1, speed=20.0)
    assert r["status"] == "success", _text(r)
    assert f"(root omitted - using the directory this session recorded 'lab/demo' to: {root})" in _text(r)
    assert _json(r)["root"] == root
    assert _json(r)["frames_applied"] == 15


def test_an_explicit_root_is_never_overridden_and_carries_no_note(recorded):
    sim, root = recorded
    r = sim.replay_episode("lab/demo", episode=0, root=root, speed=20.0)
    assert r["status"] == "success", _text(r)
    assert "root omitted" not in _text(r)
    assert _json(r)["root"] == root


def test_an_id_this_session_never_recorded_gets_a_one_line_cause_and_the_remedy(recorded, monkeypatch):
    sim, _ = recorded
    assert sim.recorded_dataset_dir("lab/never") is None

    def _boom(repo_id, episode=0, root=None):
        raise RuntimeError(
            "404 Client Error. (Request ID: Root=1-abc)\n\n"
            "Repository Not Found for url: https://huggingface.co/api/datasets/lab/never/refs.\n"
            "Please make sure you specified the correct `repo_id` and `repo_type`.\n"
            "If you are trying to access a private or gated repo, make sure you are authenticated.\n"
            "For more details, see https://huggingface.co/docs/..."
        )

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", _boom)
    r = sim.replay_episode("lab/never", episode=0)
    assert r["status"] == "error"
    text = _text(r)
    assert text.startswith("replay_episode: could not open episode 0 of 'lab/never': Repository Not Found for url:")
    assert "Request ID" not in text and "If you are trying" not in text
    assert "root was omitted, so 'lab/never' was looked up on the Hub" in text
    assert "pass root=<its directory> - stop_recording reports it as 'Local:'" in text


def test_out_of_range_keeps_the_dataset_error_and_no_hub_hint(recorded):
    sim, _ = recorded
    r = sim.replay_episode("lab/demo", episode=7, speed=20.0)
    assert r["status"] == "error"
    assert "could not open episode 7 of 'lab/demo': Episode 7 out of range (0-1)." in _text(r)
    assert "looked up on the Hub" not in _text(r)

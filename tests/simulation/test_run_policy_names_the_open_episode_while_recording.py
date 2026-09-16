"""A single-rollout ``run_policy`` while recording says how many frames sit in the open episode and that it closed none.

Through the agent tool, ``start_recording`` -> ``run_policy`` -> ``run_policy``
-> ``stop_recording`` saves ONE 20-frame episode where the caller meant two.
``run_policy`` closes no episode; the merge was documented in its docstring
and in a ``logger.info`` the agent never sees, and the result text said only
"Policy complete … 10 steps". The two remedies an agent can reach are
``reset`` between rollouts and ``n_episodes=N`` in one call
(``save_episode`` is Python-only: not in the published action enum).

Pinned: the result text carries a ``Recorder:`` line with the open-episode
frame count; a second rollout without a boundary says APPENDED and names the
prior count; json gains ``episode_open_frames`` / ``episode_merged_prior_frames``;
the line names ``reset`` and ``n_episodes``; not recording -> no line and no
fields. ``start_recording``'s advice names a boundary an agent can dial
instead of the false "one call per episode" - a call is not an episode.
"""

from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import recording as _recording_mod
from strands_robots.simulation.mujoco.simulation import Simulation


class _FakeRecorder:
    """Counts frames like the LeRobot recorder does: monotonic total + open episode."""

    def __init__(self) -> None:
        self.frame_count = 0
        self.episode_frame_count = 0
        self.dataset = None

    def add_frame(self, *_a, **_k) -> None:
        self.frame_count += 1
        self.episode_frame_count += 1

    def save_episode(self) -> None:
        self.episode_frame_count = 0


@pytest.fixture
def sim():
    s = Simulation(tool_name="open_episode_line", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _recording(sim: Simulation, monkeypatch, recorder: _FakeRecorder) -> None:
    monkeypatch.setattr(sim, "_is_recording", lambda: True)
    monkeypatch.setattr(sim, "_active_recorder", lambda: recorder)
    # The real hook feeds the recorder one frame per control step; drive the
    # fake the same way so the counts are the rollout's, not the test's.
    real_hook = sim._make_run_policy_hook

    def hook(robot_name, instruction):
        inner = real_hook(robot_name, instruction)

        def on_frame(*a, **k):
            recorder.add_frame()
            if inner is not None:
                return inner(*a, **k)
            return None

        return on_frame

    monkeypatch.setattr(sim, "_make_run_policy_hook", hook)


def test_first_rollout_reports_the_open_episode_and_that_it_closed_none(sim, monkeypatch):
    rec = _FakeRecorder()
    _recording(sim, monkeypatch, rec)
    r = sim.run_policy("so101", policy_provider="mock", n_steps=10, control_frequency=10)
    assert r["status"] == "success", r
    text = _text(r)
    assert "Recorder: +10 frames buffered in the open episode (10 unsaved)" in text, text
    assert "closes no episode" in text and "reset" in text and "n_episodes=N" in text, text
    assert "APPENDED" not in text
    j = _json(r)
    assert j["episode_open_frames"] == 10 and j["episode_merged_prior_frames"] == 0, j
    assert j["episode_flush_deferred"] is True and j["episodes_saved"] == 0


def test_second_rollout_without_a_boundary_says_appended_and_names_the_prior_frames(sim, monkeypatch):
    rec = _FakeRecorder()
    _recording(sim, monkeypatch, rec)
    sim.run_policy("so101", policy_provider="mock", n_steps=10, control_frequency=10)
    r = sim.run_policy("so101", policy_provider="mock", n_steps=5, control_frequency=10)
    text = _text(r)
    assert "+5 frames APPENDED to the open episode, which already held 10" in text, text
    assert "now 15 unsaved frames in ONE episode" in text, text
    j = _json(r)
    assert j["episode_open_frames"] == 15 and j["episode_merged_prior_frames"] == 10, j


def test_a_closed_episode_starts_the_count_over(sim, monkeypatch):
    rec = _FakeRecorder()
    _recording(sim, monkeypatch, rec)
    sim.run_policy("so101", policy_provider="mock", n_steps=10, control_frequency=10)
    rec.save_episode()
    r = sim.run_policy("so101", policy_provider="mock", n_steps=5, control_frequency=10)
    text = _text(r)
    assert "+5 frames buffered in the open episode (5 unsaved)" in text, text
    assert _json(r)["episode_merged_prior_frames"] == 0


def test_not_recording_adds_no_line_and_no_fields(sim):
    r = sim.run_policy("so101", policy_provider="mock", n_steps=5, control_frequency=10)
    assert r["status"] == "success", r
    assert "Recorder:" not in _text(r)
    j = _json(r)
    assert "episode_open_frames" not in j and "episode_merged_prior_frames" not in j


def test_the_remedies_named_are_ones_an_agent_can_call():
    from strands_robots.simulation.mujoco.simulation import _PUBLISHED_ACTIONS, _PUBLISHED_PARAMS

    line = Simulation._open_episode_line(20, 10)
    assert "reset" in line and "reset" in _PUBLISHED_ACTIONS
    assert "n_episodes=N" in line and "n_episodes" in _PUBLISHED_PARAMS
    # save_episode is Python-only; the line may mention it but must say so.
    assert "save_episode" not in _PUBLISHED_ACTIONS
    assert "From Python, save_episode" in line


def _start_recording_advice() -> str:
    """``start_recording``'s success text, read from source so no dataset is needed.

    Located by the marker it opens with, as
    ``test_recording_advice_names_every_route_that_captures`` does: the body
    also holds refusal texts, and joining every literal would let a check pass
    on prose that says nothing about capture.
    """
    tree = ast.parse(inspect.getsource(_recording_mod))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_recording")
    for node in ast.walk(fn):
        if isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if "Recording to LeRobotDataset" in text:
                return text
    raise AssertionError("start_recording has no success text naming the dataset it opened")


def test_start_recording_advice_names_a_boundary_an_agent_can_dial():
    """The advice must not promise that one run_policy call is one episode.

    "run_policy (one call per episode)" is false - a rollout closes no episode -
    and it is the sentence a caller acts on for the rest of the session, so it
    has to name a boundary instead: ``reset`` between rollouts or
    ``n_episodes=N`` in one call, both of which the tool publishes.
    """
    from strands_robots.simulation.mujoco.simulation import _PUBLISHED_ACTIONS, _PUBLISHED_PARAMS

    advice = _start_recording_advice()
    assert "one call per episode" not in advice, advice
    assert "closes NO episode" in advice, advice
    assert "merge into one" in advice, advice
    assert "reset" in advice and "reset" in _PUBLISHED_ACTIONS
    assert "n_episodes=N in one call" in advice and "n_episodes" in _PUBLISHED_PARAMS
    # The advice may not send an agent to a verb the tool refuses.
    for unreachable in ("save_episode", "verify_dataset_episodes"):
        assert unreachable not in advice, f"advice names unpublished {unreachable}: {advice}"
        assert unreachable not in _PUBLISHED_ACTIONS

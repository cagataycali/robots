"""``step`` under an active dataset recording says, on that call, that it records nothing.

The recorder is fed by ``run_policy``'s per-step hook and by nothing else. A
caller scripting a demonstration with ``set_joint_positions`` + ``step`` while a
recording is active captures zero frames, and used to learn that only from
``stop_recording``'s empty-dataset refusal - after the whole motion had run.
The note lands on the ``step`` result instead, while the motion is still ahead,
and stays silent when a rollout (which does record) is in flight.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import Simulation  # noqa: E402

NOTE = "NOT RECORDED"


@pytest.fixture
def sim():
    s = Simulation(tool_name="step_recording_note_test", mesh=False)
    s.create_world()
    s.add_robot("so101")
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def test_step_without_a_recording_carries_no_note(sim):
    result = sim.step(5)
    assert result["status"] == "success"
    assert NOTE not in _text(result)
    assert _text(result).startswith("+5 steps | t=")


def test_step_under_an_active_recording_names_what_feeds_the_recorder(sim):
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["dataset_recorder"] = object()
    result = sim.step(5)
    assert result["status"] == "success", result
    text = _text(result)
    assert text.startswith(NOTE), "the note leads the line - a trailing note was read past"
    assert "+5 steps | t=" in text, "the step summary itself is still there"
    assert "captures no frames" in text
    assert "run_policy" in text
    assert "stop_recording" in text


def test_step_while_a_rollout_is_recording_stays_quiet(sim):
    """A rollout in flight feeds the recorder, so the note would be wrong."""
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["dataset_recorder"] = object()
    sim._world.robots["so101"].policy_running = True
    try:
        result = sim.step(5)
    finally:
        sim._world.robots["so101"].policy_running = False
    assert result["status"] == "success", result
    assert NOTE not in _text(result)


def test_zero_step_noop_is_unchanged(sim):
    sim._world._backend_state["recording"] = True
    result = sim.step(0)
    assert result["status"] == "success"
    assert "no-op" in _text(result)
    assert NOTE not in _text(result)


def test_start_recording_success_text_says_only_run_policy_captures():
    """The rule is stated where the recording begins, not only where it fails."""
    import ast
    import inspect

    from strands_robots.simulation.mujoco import recording

    src = inspect.getsource(recording)
    joined = " ".join(
        node.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert "Frames are captured by run_policy only" in joined
    assert "do not feed the recorder" in joined

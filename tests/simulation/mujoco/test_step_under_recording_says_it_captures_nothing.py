"""``step`` under an active dataset recording says, on that call, that it records nothing.

The recorder is fed by a policy rollout's per-step hook and by nothing else. A
caller scripting a demonstration with ``set_joint_positions`` + ``step`` while a
recording is active captures zero frames, and used to learn that only from
``stop_recording``'s empty-dataset refusal - after the whole motion had run.
The note lands on the ``step`` result instead, while the motion is still ahead,
and stays silent when a rollout (which does record) is in flight.

Two public actions launch a rollout - ``run_policy`` and ``start_policy`` - and
``_announce_rollout`` raises ``policy_running`` for both, which is the flag the
note's guard reads. So the advertised rule names both: telling a caller that
``run_policy`` is the only thing that records is wrong about ``start_policy``,
whose rollout does feed the recorder.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import Simulation  # noqa: E402

NOTE = "NOT RECORDED"
# Both raise ``policy_running`` through ``_announce_rollout``, and a rollout
# from either one feeds the recorder through the same per-step hook.
LAUNCHERS = frozenset({"run_policy", "start_policy"})


@pytest.fixture
def sim():
    s = Simulation(tool_name="step_recording_note_test", mesh=False)
    s.create_world()
    s.add_robot("so101")
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _recording_advice() -> str:
    """The text ``start_recording`` can emit, from its body alone.

    Scoped to that one function and with its docstring dropped: joining every
    literal in the module made a single-word check vacuous, since the module
    prose says "per-step capture" whatever the advice says.
    """
    import ast
    import inspect

    from strands_robots.simulation.mujoco import recording

    tree = ast.parse(inspect.getsource(recording))
    body = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "start_recording"
    ).body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]  # the docstring is documentation, not advice
    return " ".join(
        node.value
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


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
    assert "stop_recording" in text
    for launcher in LAUNCHERS:
        assert launcher in text, f"{launcher} launches a rollout that records"


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


def test_start_recording_text_names_every_launcher_that_captures():
    """The rule is stated where the recording begins, not only where it fails.

    And it names both launchers. ``start_policy``'s rollout feeds the recorder
    through the same hook ``run_policy``'s does, so "``run_policy`` only" sends
    a caller who used ``start_policy`` looking for a defect that is not there.
    """

    joined = _recording_advice()
    assert "Frames are captured by a policy rollout only" in joined
    assert "do not feed the recorder" in joined
    for launcher in LAUNCHERS:
        assert launcher in joined, f"{launcher} launches a rollout that records"
    assert "captured by run_policy only" not in joined, "start_policy records too"


def test_every_launcher_the_rule_names_is_a_dialable_action(sim):
    """Advice a caller cannot dial is worse than no advice.

    Both launchers are named in the ``step`` note and in ``start_recording``'s
    text, so both must be reachable through the tool's own action enum.
    """
    actions = set(sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"])
    assert LAUNCHERS <= actions, f"not dialable: {LAUNCHERS - actions}"


# Every rollout that records, measured under one active recording on a so101
# scene: run_multi_policy saved 15 frames and start_policy 30. Naming run_policy
# alone was wrong about both.
RECORDS = frozenset({"run_policy", "start_policy", "run_multi_policy"})


def test_the_advice_names_every_rollout_that_raises_the_flag_the_guard_reads():
    """``policy_running`` defines "a rollout is in flight", so its raisers are
    the paths that record - and the advice has to name each of them.

    Pinned as the whole set rather than one name per case: the advice named
    ``run_policy`` alone while three sites raise the flag, and a fourth raiser
    added later would quietly make the advertised rule false again. This fails
    when the set changes, which is the moment to decide what the advice says.
    """
    import ast
    import inspect

    from strands_robots.simulation.mujoco import simulation as mujoco_sim

    raisers: set[str] = set()
    stack: list[str] = []

    class Walk(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "policy_running"
                    and getattr(node.value, "value", None) is True
                ):
                    raisers.add(stack[-1])
            self.generic_visit(node)

    Walk().visit(ast.parse(inspect.getsource(mujoco_sim)))
    # ``_make_run_policy_hook`` is run_policy's own hook, not a separate entry
    # point; ``_announce_rollout`` is shared by run_policy and start_policy.
    assert raisers == {"_announce_rollout", "_make_run_policy_hook", "run_multi_policy"}, raisers

    advice = _recording_advice()
    for method in sorted(RECORDS):
        assert method in advice, f"{method} launches a rollout that feeds the recorder"


def test_the_advice_still_names_what_does_not_record():
    advice = _recording_advice()
    for method in ("step", "set_joint_positions", "teleoperate", "replay_episode"):
        assert method in advice

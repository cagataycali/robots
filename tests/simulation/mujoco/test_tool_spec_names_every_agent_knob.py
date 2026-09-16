"""Every knob a sim action accepts from JSON is a property of the tool schema.

The unknown-parameter refusal lists an action's accepted names from its Python
signature; the agent plans from the JSON schema. Where the two disagree the
agent is told ``Valid: [... 'overwrite' ...]`` for a knob the schema never
showed it, and knobs it was never told about run only at their defaults.
Measured on ``Robot("so101", mode="sim")``: 13 agent-usable parameters were
accepted at runtime and absent from the schema - ``randomize``'s
``color_range`` / ``friction_range`` / ``mass_range``, ``start_recording``'s
``overwrite``, ``replay_episode``'s ``action_key_map``,
``start_cameras_recording``'s ``max_frames_per_camera``, and the seven rollout
knobs of ``run_policy`` / ``eval_policy`` / ``evaluate_benchmark`` /
``start_policy`` (``control_substeps``, ``reset_between``, ``async_rtc``,
``rtc_inference_timeout_s``, ``wbc_install_torque_control``,
``max_onframe_failures``, ``policy_kwargs``) - and the four ``randomize``
booleans carried no description, so nothing said ``randomize_physics`` is
destructive.

Three kinds of parameter are exempt, each with its reason recorded below, and
:func:`test_no_json_expressible_knob_is_exempted_as_unexpressible` grades the
first kind against the signatures so a plain ``int``/``bool``/``dict`` knob
cannot be hidden there again - which is how the seven rollout knobs stayed
invisible while the table claimed no agent could express them.
"""

from __future__ import annotations

import ast
import inspect
import re
from typing import Any

import pytest

from strands_robots import Robot

# JSON has no way to spell these: a callable or an already-built Policy.
_JSON_CANNOT_EXPRESS: dict[str, str] = {
    "policy_object": "an already-constructed Policy instance",
    "on_frame": "a per-frame callback",
    "observer": "a rollout-observer callback",
}
# Published under other spellings on purpose. ``video=`` is a dict the router
# folds from the flat keys the schema does publish (see
# MuJoCoSimEngine._fold_flat_video_keys), the way ``torque`` is published as
# ``torque_vec``.
_SPELLED_FLAT_IN_THE_SCHEMA: dict[str, str] = {
    "video": "folded from output_path / fps / camera_name",
}
# Accepted from Python without being published, by a decision that
# tests/simulation/mujoco/test_router_refusals_name_a_published_spelling.py
# owns and pins ("the dispatcher is deliberately wider than the schema").
_DELIBERATELY_UNPUBLISHED: dict[str, str] = {
    "bucket": "stop_recording: HF Storage Bucket sync target",
    "run_id": "stop_recording: subpath inside that bucket",
}
_EXEMPT = (
    frozenset(_JSON_CANNOT_EXPRESS) | frozenset(_SPELLED_FLAT_IN_THE_SCHEMA) | frozenset(_DELIBERATELY_UNPUBLISHED)
)

# Method parameters the schema reaches under another name.
_REMAPPED = {"torque": "torque_vec"}

# An annotation JSON cannot construct names one of these; anything else is a
# scalar, a list or a mapping a model can emit.
_UNEXPRESSIBLE_IN_JSON = re.compile(r"Callable|Policy\b|Observer")


@pytest.fixture(scope="module")
def sim():
    robot = Robot("so101", mode="sim")
    yield robot
    robot.cleanup()


def _actions(sim) -> list[str]:
    return list(sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"])


def _plain_parameters(sim, action: str) -> dict[str, inspect.Parameter]:
    """The parameters of *action*'s method, minus ``self`` and ``*args``/``**kwargs``."""
    fn = getattr(sim, action, None)
    if fn is None:
        return {}
    return {
        name: param
        for name, param in inspect.signature(fn).parameters.items()
        if name not in ("self", "kwargs") and param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
    }


def _refusal_valid_list(sim, action: str) -> list[str]:
    """The names an unknown-parameter refusal advertises for *action*."""
    result = sim(action=action, definitely_not_a_field=1)
    assert result["status"] == "error", result
    text = " ".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))
    assert "Valid:" in text, text
    return list(ast.literal_eval(text.split("Valid:", 1)[1].strip().rstrip(".")))


def test_every_agent_knob_is_in_the_schema(sim) -> None:
    props = sim.tool_spec["inputSchema"]["json"]["properties"]
    missing: dict[str, list[str]] = {}
    for action in _actions(sim):
        for name in _plain_parameters(sim, action):
            if name in _EXEMPT or _REMAPPED.get(name, name) in props:
                continue
            missing.setdefault(action, []).append(name)
    assert missing == {}, f"accepted at runtime but invisible to the agent: {missing}"


def test_no_refusal_advertises_a_knob_the_schema_hides(sim) -> None:
    """What the agent is told it may send is what the schema lets it send.

    Read from the refusals themselves rather than the signatures, so the two
    surfaces are graded against each other instead of both against one list.
    """
    props = sim.tool_spec["inputSchema"]["json"]["properties"]
    hidden: dict[str, list[str]] = {}
    for action in _actions(sim):
        offered = _refusal_valid_list(sim, action)
        unseen = [n for n in offered if n not in props and n not in _EXEMPT]
        if unseen:
            hidden[action] = sorted(unseen)
    assert hidden == {}, f"advertised by a refusal, absent from the schema: {hidden}"


def test_no_json_expressible_knob_is_exempted_as_unexpressible(sim) -> None:
    """The exemption for callables holds only names whose annotation is one.

    A plain ``int`` / ``bool`` / ``dict`` parameter listed there would be
    hidden from the agent by a reason that is not true of it - the state the
    seven rollout knobs were in.
    """
    annotations: dict[str, set[str]] = {}
    for action in _actions(sim):
        for name, param in _plain_parameters(sim, action).items():
            if name in _JSON_CANNOT_EXPRESS:
                annotations.setdefault(name, set()).add(str(param.annotation))
    assert set(annotations) == set(_JSON_CANNOT_EXPRESS), sorted(annotations)
    for name, seen in annotations.items():
        for annotation in seen:
            assert _UNEXPRESSIBLE_IN_JSON.search(annotation), f"{name}: {annotation} is expressible in JSON"


def test_the_video_exemption_publishes_its_flat_spellings(sim) -> None:
    """``video`` is exempt because the keys the router folds into it are published."""
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    props = sim.tool_spec["inputSchema"]["json"]["properties"]
    flat = MuJoCoSimEngine._RUN_POLICY_RESIDUAL_VIDEO_KEYS
    assert flat, "premise: the router names the flat video keys it accepts"
    assert [key for key in flat if key not in props] == []


def test_randomize_knobs_are_described(sim) -> None:
    props = sim.tool_spec["inputSchema"]["json"]["properties"]
    for name in (
        "randomize_colors",
        "randomize_lighting",
        "randomize_physics",
        "randomize_positions",
        "position_noise",
        "color_range",
        "friction_range",
        "mass_range",
    ):
        assert props[name].get("description", "").startswith("randomize:"), name
    for name in ("color_range", "friction_range", "mass_range"):
        assert props[name]["type"] == "array" and props[name]["minItems"] == props[name]["maxItems"] == 2


def test_the_new_rollout_knobs_reach_the_rollout(sim, monkeypatch) -> None:
    """A published knob is one that reaches the runner, not one merely accepted.

    Recorded at :meth:`MuJoCoSimEngine._drive_rollout`, the seam the action
    hands the rollout to, so a knob dropped either by the dispatcher or by the
    action body is caught rather than a schema property that binds to nothing.
    """
    sent: dict[str, Any] = {
        "control_substeps": 3,
        "reset_between": False,
        "async_rtc": True,
        "rtc_inference_timeout_s": 0.25,
        "wbc_install_torque_control": False,
        "max_onframe_failures": 2,
        "policy_kwargs": {"target_pose": [0.1, 0.2, 0.3]},
    }
    seen: dict[str, Any] = {}

    def record(self, robot_name: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"status": "success", "content": [{"text": "recorded"}]}

    monkeypatch.setattr(type(sim), "_drive_rollout", record)
    result = sim(action="run_policy", policy_provider="mock", **sent)
    assert result["status"] == "success", result
    assert {key: seen.get(key) for key in sent} == sent
